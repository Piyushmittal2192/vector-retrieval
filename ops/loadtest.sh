#!/usr/bin/env bash
# Run the load generator as an in-cluster Job against Service "retrieval".
#   ops/loadtest.sh steady-200 --rate 200 --duration 120
#   ops/loadtest.sh sweep-3pods --mode sweep --rates 50,100,200,300,400 --step-duration 30
# Results land in ./reports/<name>/ on your machine; a plot is drawn at the end.
source "$(dirname "$0")/common.sh"
NAME="${1:?usage: loadtest.sh NAME [loadgen args...]}"; shift
TARGET="${TARGET:-http://retrieval.$NS.svc.cluster.local}"
run_job loadgen.loadgen "$NAME" --target "$TARGET" --queries /artifacts/queries/queries.jsonl \
  --out "/reports/$NAME" --name "$NAME" "$@"
if [ -f "reports/$NAME/timeline.csv" ]; then
  $PY ops/plot_run.py "reports/$NAME" || echo "(plot skipped: matplotlib missing?)"
fi
