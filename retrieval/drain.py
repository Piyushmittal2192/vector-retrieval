"""preStop hook: start draining, then hold the pod open while endpoints update.

Order of events when a pod is deleted:
  1. Pod marked Terminating; endpoint controller and kube-proxy start removing
     it from the Service (asynchronous, can take a few seconds).
  2. kubelet runs this hook. We flip /ready to 503 and start sending
     "Connection: close" so keep-alive clients reconnect elsewhere, then
     sleep so requests that still arrive during step 1 are served.
  3. Only after the hook returns does the container get SIGTERM; uvicorn
     stops accepting and finishes in-flight requests.
terminationGracePeriodSeconds must exceed wait + graceful shutdown time.
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.request


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--wait", type=float, default=15.0)
    args = ap.parse_args()
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{args.port}/admin/drain", data=b"", method="POST")
        urllib.request.urlopen(req, timeout=3).read()
        print(f"drain started; holding {args.wait}s", flush=True)
    except Exception as exc:  # never block shutdown on the hook itself
        print(f"drain call failed: {exc}", file=sys.stderr, flush=True)
    time.sleep(args.wait)
    return 0


if __name__ == "__main__":
    sys.exit(main())
