#!/usr/bin/env bash
# Deploy one color pinned to one index version (and wait for it to be ready).
#   ops/deploy.sh blue v1
#   REPLICAS=4 IMAGE=retrieval-api:1.1 ops/deploy.sh blue v1     # rolling update of code
#   DRAIN=off ops/deploy.sh blue v1                              # drill: no preStop drain
#   ENCODER=remote ops/deploy.sh blue v1                         # use the separate encoder service
#   EXEC_MODE=semaphore COMPUTE_SLOTS=4 ops/deploy.sh blue v1    # another execution strategy
source "$(dirname "$0")/common.sh"
COLOR="${1:?usage: deploy.sh blue|green VERSION}"
VERSION="${2:?usage: deploy.sh blue|green VERSION}"
[[ "$COLOR" =~ ^(blue|green)$ ]] || die "color must be blue or green"
manifest="artifacts/indexes/$VERSION/manifest.json"
[ -f "$manifest" ] || die "no $manifest; build the index first"

report="artifacts/validation/$VERSION.json"
if [ "${FORCE:-0}" != 1 ]; then
  [ -f "$report" ] || die "$VERSION has no validation report. Run: make validate V=$VERSION  (or FORCE=1 for drills)"
  $PY -c "import json,sys; sys.exit(0 if json.load(open('$report'))['passed'] else 1)" \
    || die "$VERSION FAILED validation ($report). Refusing to deploy (FORCE=1 overrides for drills)."
fi

backend=$($PY -c "import json; print(json.load(open('$manifest'))['encoder']['backend'])")
ENCODER_URL=""
if [ "${ENCODER:-local}" = remote ]; then
  backend=remote
  ENCODER_URL="${ENCODER_URL_OVERRIDE:-http://encoder.$NS.svc.cluster.local}"
fi
est_mib=$($PY -c "import json; print(json.load(open('$manifest'))['index']['estimated_pod_memory_mib'])")
# With a remote encoder the pod no longer loads torch and the model (~1 GiB less).
[ "$backend" = remote ] && est_mib=$(( est_mib > 1280 ? est_mib - 768 : 512 ))
index_mib=$($PY -c "import json; m=json.load(open('$manifest')); print(int(sum(f['bytes'] for f in m['files'].values())/2**20)+1)")
MEMORY="${MEMORY:-${est_mib}Mi}"
INDEX_DISK="${INDEX_DISK:-$(( index_mib * 2 + 256 ))Mi}"
if [ "${DRAIN:-on}" = on ]; then
  PRESTOP_CMD="[\"python\", \"-m\", \"retrieval.drain\", \"--wait\", \"${DRAIN_WAIT:-15}\"]"
else
  PRESTOP_CMD='["true"]'
fi

echo "deploying retrieval-$COLOR: index=$VERSION image=$IMAGE replicas=${REPLICAS:-3} memory=$MEMORY drain=${DRAIN:-on} encoder=$backend exec_mode=${EXEC_MODE:-batcher}"
$PY ops/render.py deploy/k8s/retrieval.tmpl.yaml \
  COLOR="$COLOR" INDEX_VERSION="$VERSION" IMAGE="$IMAGE" REPLICAS="${REPLICAS:-3}" \
  ARTIFACT_SOURCE="${ARTIFACT_SOURCE:-http://artifact-server.$NS.svc.cluster.local/indexes}" \
  CPU_REQUEST="${CPU_REQUEST:-1}" CPU_LIMIT="${CPU_LIMIT:-2}" TORCH_THREADS="${TORCH_THREADS:-2}" \
  MEMORY="$MEMORY" INDEX_DISK="$INDEX_DISK" ENCODER_BACKEND="$backend" ENCODER_URL="$ENCODER_URL" \
  EXEC_MODE="${EXEC_MODE:-batcher}" COMPUTE_SLOTS="${COMPUTE_SLOTS:-4}" \
  MAX_BATCH="${MAX_BATCH:-32}" MAX_WAIT_MS="${MAX_WAIT_MS:-2}" MAX_QUEUE="${MAX_QUEUE:-256}" \
  REQUEST_TIMEOUT_MS="${REQUEST_TIMEOUT_MS:-1000}" QUERY_CACHE_SIZE="${QUERY_CACHE_SIZE:-0}" EF="${EF:-0}" \
  PRESTOP_CMD="$PRESTOP_CMD" SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-default}" \
  | kubectl apply -f -

if [ "${WAIT:-1}" = 1 ]; then
  k rollout status "deployment/retrieval-$COLOR" --timeout="${TIMEOUT:-600s}"
fi
k get pods -l "app=retrieval,color=$COLOR" -L index-version -o wide
