# shellcheck shell=bash
# Shared settings for ops scripts. Override any of these in your environment.
set -euo pipefail
LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LAB_ROOT"
NS="${NS:-search}"
CLUSTER="${CLUSTER:-retrieval-lab}"
IMAGE="${IMAGE:-retrieval-api:1.0}"
LOADGEN_IMAGE="${LOADGEN_IMAGE:-retrieval-loadgen:1.0}"
if [ -x .venv/bin/python ]; then PY="${PY:-.venv/bin/python}"; else PY="${PY:-python3}"; fi
k() { kubectl -n "$NS" "$@"; }
active_color() { k get svc retrieval -o jsonpath='{.spec.selector.color}' 2>/dev/null || true; }
other_color() { if [ "$1" = blue ]; then echo green; else echo blue; fi; }
die() { echo "ERROR: $*" >&2; exit 1; }

# run_job MODULE NAME ARGS... : run python -m MODULE as a Job, stream logs, return its status.
run_job() {
  local module="$1" name="$2"; shift 2
  local job; job="lg-$(echo "$name" | tr 'A-Z_.' 'a-z--' | tr -cd 'a-z0-9-' | cut -c1-50)"
  local args; args=$($PY -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$@")
  k delete job "$job" --ignore-not-found --wait=true >/dev/null
  $PY ops/render.py deploy/k8s/loadgen-job.tmpl.yaml JOB_NAME="$job" LOADGEN_IMAGE="$LOADGEN_IMAGE" \
    MODULE="$module" ARGS="$args" HOST_UID="$(id -u)" HOST_GID="$(id -g)" LOADGEN_CPU="${LOADGEN_CPU:-2}" \
    | kubectl apply -f - >/dev/null
  echo "job/$job started; streaming logs (Ctrl+C stops streaming, not the job)"
  for _ in $(seq 1 120); do
    phase=$(k get pods -l "job-name=$job" -o jsonpath='{.items[0].status.phase}' 2>/dev/null || true)
    [[ "$phase" =~ ^(Running|Succeeded|Failed)$ ]] && break
    sleep 1
  done
  k logs -f "job/$job" || true
  k wait --for=condition=complete "job/$job" --timeout=30s >/dev/null 2>&1 && return 0
  echo "job/$job did not complete successfully: kubectl -n $NS describe job/$job" >&2
  return 1
}
