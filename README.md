# production-vector-retrieval

A production-style semantic search lab: versioned HNSW indexes, a micro-batched
retrieval service, blue/green index promotion on a 3-node kind cluster, an
open-loop load generator, failure drills, and benchmarks you measure yourself.

The step-by-step lab guide lives in the "Production Vector Retrieval Lab" doc.
This README is the map of the repo.

## Quick start (laptop, ~10 minutes)

```bash
git clone https://github.com/Piyushmittal2192/vector-retrieval && cd vector-retrieval
make venv                      # .venv with CPU-only torch
make test                      # 18 tests, no model or network needed
make model                     # download all-MiniLM-L6-v2 into ./models (once)
make data TIER=xs              # 20k products, 2k labeled queries
make index V=v1 && make validate V=v1
make serve V=v1                # terminal 1
make load-local RATE=50 DURATION=30 NAME=first-run   # terminal 2
```

No internet for the model? Use `ENCODER=hash` on every make command: a lexical
stand-in encoder that exercises every infrastructure path (relevance numbers
are then meaningless).

## Cluster (after the quick start)

```bash
make image image-loadgen       # service image (TAG=1.0) + load generator image
make cluster-up load-images    # kind: 1 control plane + 2 workers, artifact server, Prometheus
make deploy COLOR=blue V=v1    # 3 replicas pinned to index v1
make loadtest NAME=steady RATE=100 DURATION=120
```

## Layout

| Path | What it is |
| --- | --- |
| `data/` | Catalog, labeled-query and change-log generators (`taxonomy.py` holds the product world) |
| `retrieval/build_index.py` | Catalog at a watermark -> immutable, checksummed index version |
| `retrieval/validate_index.py` | Promotion gate: integrity, encoder fingerprint, golden queries, recall, nDCG, freshness, baseline diff |
| `retrieval/server.py` | Starlette API: `/search`, `/ready`, `/health`, `/version`, `/metrics`, `/admin/drain` |
| `retrieval/engine.py` | Startup validation and batched encode + search |
| `retrieval/batcher.py` | Micro-batching, bounded queue, load shedding, deadlines |
| `retrieval/fetch_index.py` | Init container: download + verify an index version |
| `retrieval/drain.py` | preStop hook |
| `loadgen/` | Open-loop load generator, shadow comparator (standard library only) |
| `bench/` | Encoder and ANN benchmarks |
| `deploy/` | Dockerfiles and Kubernetes manifests |
| `ops/` | Cluster, deploy, promote, load test, chaos, plotting scripts |
| `tests/` | Unit and integration tests |

Generated, not committed: `data/*.jsonl`, `artifacts/`, `cache/`, `models/`, `reports/`.

## Dependency versions

`requirements/serve.in` holds compatible ranges. After the first image build,
`make lock` freezes the exact versions resolved inside the image into
`requirements/serve.lock`; every later build installs from the lock.
