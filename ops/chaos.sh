#!/usr/bin/env bash
# Failure drills. Run them while a load test is running in another terminal.
#   ops/chaos.sh crash [COLOR]           process exits with no drain (like an OOM kill or segfault)
#   ops/chaos.sh crash-encoder           one encoder service pod exits with no drain (Stage 9A)
#   ops/chaos.sh delete [COLOR]          graceful pod deletion (preStop drain runs)
#   ops/chaos.sh drain-node [NODE]       planned maintenance: evicts pods, respects the PDB
#   ops/chaos.sh uncordon [NODE]
#   ops/chaos.sh stop-node [NODE]        unplanned node loss (docker stop the kind node)
#   ops/chaos.sh start-node [NODE]
#   ops/chaos.sh corrupt SRC DST         copy index SRC to DST with a flipped byte (checksum drill)
#   ops/chaos.sh wrong-encoder SRC DST   copy index SRC to DST with a different encoder fingerprint
source "$(dirname "$0")/common.sh"
cmd="${1:-help}"; shift || true
NODE_DEFAULT="$CLUSTER-worker"

pick_pod() {
  local color="${1:-$(active_color)}"
  k get pods -l "app=retrieval,color=$color" --field-selector=status.phase=Running \
    -o jsonpath='{.items[*].metadata.name}' | tr ' ' '\n' | shuf -n1
}

case "$cmd" in
  crash)
    pod=$(pick_pod "${1:-}"); [ -n "$pod" ] || die "no running pod"
    echo "$(date -Iseconds) crashing $pod (exit without drain)"
    k exec "$pod" -c retrieval -- python -c "import urllib.request as u; u.urlopen(u.Request('http://127.0.0.1:8000/admin/crash', data=b'', method='POST'), timeout=3)" || true
    ;;
  crash-encoder)
    pod=$(k get pods -l app=encoder --field-selector=status.phase=Running -o jsonpath='{.items[*].metadata.name}' | tr ' ' '\n' | shuf -n1)
    [ -n "$pod" ] || die "no running encoder pod"
    echo "$(date -Iseconds) crashing encoder $pod"
    k exec "$pod" -c encoder -- python -c "import urllib.request as u; u.urlopen(u.Request('http://127.0.0.1:8001/admin/crash', data=b'', method='POST'), timeout=3)" || true
    ;;
  delete)
    pod=$(pick_pod "${1:-}"); [ -n "$pod" ] || die "no running pod"
    echo "$(date -Iseconds) deleting $pod (graceful)"
    k delete pod "$pod" --wait=false
    ;;
  drain-node)
    node="${1:-$NODE_DEFAULT}"
    echo "$(date -Iseconds) draining $node"
    kubectl drain "$node" --ignore-daemonsets --delete-emptydir-data --timeout=300s
    echo "bring it back with: ops/chaos.sh uncordon $node"
    ;;
  uncordon)
    kubectl uncordon "${1:-$NODE_DEFAULT}"
    ;;
  stop-node)
    node="${1:-$NODE_DEFAULT}"
    echo "$(date -Iseconds) stopping node container $node (no drain, no warning)"
    docker stop "$node"
    echo "watch: kubectl get nodes -w ; kubectl -n $NS get pods -o wide -w ; bring back: ops/chaos.sh start-node $node"
    ;;
  start-node)
    docker start "${1:-$NODE_DEFAULT}"
    ;;
  corrupt|wrong-encoder)
    src="${1:?SRC version}"; dst="${2:?DST version}"
    [ -d "artifacts/indexes/$src" ] || die "no artifacts/indexes/$src"
    [ ! -e "artifacts/indexes/$dst" ] || die "artifacts/indexes/$dst exists"
    cp -r "artifacts/indexes/$src" "artifacts/indexes/.$dst.tmp"
    $PY - "$cmd" "artifacts/indexes/.$dst.tmp" "$dst" <<'PY'
import json, os, random, sys
mode, path, dst = sys.argv[1:]
m = json.load(open(f"{path}/manifest.json"))
m["index_version"] = dst
if mode == "corrupt":
    size = os.path.getsize(f"{path}/index.bin")
    with open(f"{path}/index.bin", "r+b") as f:
        f.seek(size // 2)
        b = f.read(1)
        f.seek(size // 2)
        f.write(bytes([b[0] ^ 0xFF]))
    print("flipped one byte in index.bin; manifest checksum left unchanged")
else:
    rng = random.Random(0)
    m["encoder"]["fingerprint"]["vectors"] = [[x + rng.gauss(0, 0.05) for x in row] for row in m["encoder"]["fingerprint"]["vectors"]]
    print("perturbed the encoder fingerprint: files verify, but the model no longer matches")
json.dump(m, open(f"{path}/manifest.json", "w"), indent=2)
PY
    chmod -R a+rX "artifacts/indexes/.$dst.tmp"
    mv "artifacts/indexes/.$dst.tmp" "artifacts/indexes/$dst"
    echo "created artifacts/indexes/$dst; deploy it with: FORCE=1 ops/deploy.sh green $dst"
    ;;
  *)
    sed -n '2,13p' "$0"
    ;;
esac
