"""Minimal asyncio HTTP/1.1 client with an explicit keep-alive connection pool.

Standard library only, so the load generator image is tiny and every
connection decision is visible:

  * max_conns caps open TCP connections. A Kubernetes ClusterIP/NodePort
    Service balances per *connection* (L4), not per request, so this number
    decides how evenly load spreads across pods. Try 4 vs 64 in Stage 9.
  * A response carrying "Connection: close" closes that connection; the next
    request opens a new one, which kube-proxy routes to a live endpoint.
    That is how draining pods shed keep-alive clients.
  * Idle connections older than idle_timeout are discarded before reuse,
    staying under the server's keep-alive timeout (uvicorn default 5 s).
  * A request that fails on a *reused* connection before any response byte
    arrives is retried once on a fresh connection (what browsers and most
    HTTP libraries do); such retries are counted, never hidden.
"""
from __future__ import annotations

import asyncio
import time
import urllib.parse
from collections import deque


class HTTPError(Exception):
    def __init__(self, kind: str, msg: str = ""):
        super().__init__(f"{kind}: {msg}" if msg else kind)
        self.kind = kind


class _Conn:
    __slots__ = ("reader", "writer", "last_used", "reused")

    def __init__(self, reader, writer):
        self.reader, self.writer = reader, writer
        self.last_used = time.monotonic()
        self.reused = False

    def close(self):
        try:
            self.writer.close()
        except Exception:
            pass


class HTTPPool:
    def __init__(self, base_url: str, max_conns: int = 32, idle_timeout: float = 2.0, connect_timeout: float = 2.0):
        u = urllib.parse.urlparse(base_url)
        if u.scheme != "http":
            raise ValueError("only http:// targets are supported")
        self.host = u.hostname
        self.port = u.port or 80
        self.prefix = u.path.rstrip("/")
        self.max_conns = max_conns
        self.idle_timeout = idle_timeout
        self.connect_timeout = connect_timeout
        self._idle: deque[_Conn] = deque()
        self._open = 0
        self._cond = asyncio.Condition()
        self.stats = {"connects": 0, "retried": 0, "server_closed": 0}

    async def _acquire(self) -> _Conn:
        async with self._cond:
            while True:
                while self._idle:
                    conn = self._idle.pop()
                    if time.monotonic() - conn.last_used <= self.idle_timeout:
                        conn.reused = True
                        return conn
                    conn.close()
                    self._open -= 1
                if self._open < self.max_conns:
                    self._open += 1
                    break
                await self._cond.wait()
        try:
            r, w = await asyncio.wait_for(asyncio.open_connection(self.host, self.port), self.connect_timeout)
        except asyncio.TimeoutError:
            await self._discard(None)
            raise HTTPError("connect_timeout") from None
        except OSError as exc:
            await self._discard(None)
            raise HTTPError("connect_refused" if isinstance(exc, ConnectionRefusedError) else "connect_error", str(exc)) from None
        except asyncio.CancelledError:
            await self._discard(None)
            raise
        self.stats["connects"] += 1
        return _Conn(r, w)

    async def _release(self, conn: _Conn) -> None:
        conn.last_used = time.monotonic()
        async with self._cond:
            self._idle.append(conn)
            self._cond.notify()

    async def _discard(self, conn: _Conn | None) -> None:
        if conn is not None:
            conn.close()
        async with self._cond:
            self._open -= 1
            self._cond.notify()

    async def request(self, method: str, path: str, body: bytes = b"", headers: dict | None = None):
        """Return (status, headers_lowercase, body_bytes). Raises HTTPError."""
        for attempt in (1, 2):
            conn = await self._acquire()
            try:
                return await self._roundtrip(conn, method, path, body, headers or {})
            except HTTPError as exc:
                if attempt == 1 and conn.reused and exc.kind == "closed_before_response":
                    self.stats["retried"] += 1
                    continue
                raise
        raise HTTPError("unreachable")

    async def _roundtrip(self, conn: _Conn, method: str, path: str, body: bytes, headers: dict):
        head = [f"{method} {self.prefix}{path} HTTP/1.1", f"Host: {self.host}", f"Content-Length: {len(body)}"]
        if body:
            head.append("Content-Type: application/json")
        head += [f"{k}: {v}" for k, v in headers.items()]
        got_bytes = False
        try:
            conn.writer.write(("\r\n".join(head) + "\r\n\r\n").encode() + body)
            await conn.writer.drain()
            status_line = await conn.reader.readline()
            if not status_line:
                raise HTTPError("closed_before_response")
            got_bytes = True
            parts = status_line.split(b" ", 2)
            status = int(parts[1])
            resp_headers: dict[str, str] = {}
            while True:
                line = await conn.reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
                k, _, v = line.decode("latin-1").partition(":")
                resp_headers[k.strip().lower()] = v.strip()
            if "content-length" in resp_headers:
                data = await conn.reader.readexactly(int(resp_headers["content-length"]))
            elif resp_headers.get("transfer-encoding", "").lower() == "chunked":
                chunks = []
                while True:
                    size = int((await conn.reader.readline()).split(b";")[0], 16)
                    if size == 0:
                        await conn.reader.readline()
                        break
                    chunks.append(await conn.reader.readexactly(size))
                    await conn.reader.readline()
                data = b"".join(chunks)
            else:
                data = await conn.reader.read()
                resp_headers["connection"] = "close"
        except HTTPError:
            await self._discard(conn)
            raise
        except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError, OSError, ValueError, IndexError) as exc:
            await self._discard(conn)
            kind = "reset_mid_response" if got_bytes else "closed_before_response"
            if isinstance(exc, ConnectionResetError) and not got_bytes:
                kind = "closed_before_response"
            raise HTTPError(kind, type(exc).__name__) from None
        except asyncio.CancelledError:
            await self._discard(conn)
            raise
        if resp_headers.get("connection", "").lower() == "close":
            self.stats["server_closed"] += 1
            await self._discard(conn)
        else:
            await self._release(conn)
        return status, resp_headers, data

    async def close(self) -> None:
        async with self._cond:
            while self._idle:
                self._idle.pop().close()
                self._open -= 1
