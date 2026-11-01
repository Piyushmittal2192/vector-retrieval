#!/usr/bin/env bash
# Replay the labeled queries against both colors and compare (in-cluster).
#   ops/shadow.sh                 # a = active color, b = the other
source "$(dirname "$0")/common.sh"
A="${A:-$(active_color)}"; B="${B:-$(other_color "$A")}"
NAME="shadow-$A-vs-$B-$(date +%H%M%S)"
run_job loadgen.shadow_compare "$NAME" --a "http://retrieval-$A.$NS.svc.cluster.local" \
  --b "http://retrieval-$B.$NS.svc.cluster.local" --queries /artifacts/queries/queries.jsonl --out "/reports/$NAME" "$@"
