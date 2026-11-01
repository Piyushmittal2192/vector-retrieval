#!/usr/bin/env bash
# Deploy the separate embedding service (Stage 9A).
#   ops/deploy_encoder.sh                       # 2 replicas, micro-batching on
#   MAX_BATCH=1 ops/deploy_encoder.sh           # batching off (for the 2x2 comparison)
#   REPLICAS=3 ops/deploy_encoder.sh            # scale encoders without touching retrieval
source "$(dirname "$0")/common.sh"
echo "deploying encoder: replicas=${REPLICAS:-2} max_batch=${MAX_BATCH:-32} backend=${ENCODER_BACKEND:-st} device=${DEVICE:-cpu}"
$PY ops/render.py deploy/k8s/encoder.tmpl.yaml \
  IMAGE="$IMAGE" REPLICAS="${REPLICAS:-2}" ENCODER_BACKEND="${ENCODER_BACKEND:-st}" DEVICE="${DEVICE:-cpu}" \
  TORCH_THREADS="${TORCH_THREADS:-2}" CPU_REQUEST="${CPU_REQUEST:-1}" CPU_LIMIT="${CPU_LIMIT:-2}" MEMORY="${MEMORY:-1536Mi}" \
  MAX_BATCH="${MAX_BATCH:-32}" MAX_WAIT_MS="${MAX_WAIT_MS:-2}" MAX_QUEUE="${MAX_QUEUE:-512}" REQUEST_TIMEOUT_MS="${REQUEST_TIMEOUT_MS:-1000}" \
  | kubectl apply -f -
k rollout status deployment/encoder --timeout="${TIMEOUT:-300s}"
k get pods -l app=encoder -o wide
