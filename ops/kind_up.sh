#!/usr/bin/env bash
# Create the 3-node kind cluster and the shared infrastructure.
source "$(dirname "$0")/common.sh"
for bin in docker kind kubectl; do command -v "$bin" >/dev/null || die "$bin not installed"; done
mkdir -p artifacts/indexes artifacts/queries reports
chmod 755 artifacts artifacts/indexes artifacts/queries
cfg=reports/.kind-cluster.yaml
$PY ops/render.py deploy/k8s/kind-cluster.yaml LAB_ROOT="$LAB_ROOT" > "$cfg"
if kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
  echo "cluster $CLUSTER already exists"
else
  kind create cluster --config "$cfg" --wait 120s
fi
kubectl apply -f deploy/k8s/namespace.yaml
kubectl apply -f deploy/k8s/artifact-server.yaml
kubectl apply -f deploy/k8s/prometheus.yaml
kubectl apply -f deploy/k8s/service.yaml

echo "installing metrics-server (needed by kubectl top and the HPA)"
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
kubectl -n kube-system patch deployment metrics-server --type=json \
  -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]' || true

k rollout status deploy/artifact-server --timeout=180s
k rollout status deploy/prometheus --timeout=180s
kubectl get nodes -o wide
echo
echo "Taints (retrieval pods must land only on workers):"
kubectl get nodes -o custom-columns=NAME:.metadata.name,TAINTS:.spec.taints[*].key
echo
echo "Prometheus UI: http://localhost:9090   Service 'retrieval': http://localhost:8080"
