#!/usr/bin/env bash
# One-screen view of what is serving.
source "$(dirname "$0")/common.sh"
echo "active color (Service retrieval selector): $(active_color)"
echo
k get deploy -l app=retrieval -o custom-columns=NAME:.metadata.name,INDEX:.spec.template.metadata.labels.index-version,IMAGE:.spec.template.spec.containers[0].image,READY:.status.readyReplicas,WANT:.spec.replicas
echo
k get pods -l app=retrieval -L color,index-version -o wide
echo
echo "endpoints behind Service retrieval:"
k get endpointslices -l kubernetes.io/service-name=retrieval -o jsonpath='{range .items[*].endpoints[*]}{"  "}{.targetRef.name}{"  ready="}{.conditions.ready}{"  terminating="}{.conditions.terminating}{"\n"}{end}'
k get hpa 2>/dev/null || true
kubectl top pods -n "$NS" -l app=retrieval 2>/dev/null || true
