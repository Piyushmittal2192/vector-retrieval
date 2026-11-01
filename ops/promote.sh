#!/usr/bin/env bash
# Switch user traffic (Service "retrieval") to a color. Instant and reversible.
#   ops/promote.sh green      # promote
#   ops/promote.sh blue       # roll back
source "$(dirname "$0")/common.sh"
TARGET="${1:?usage: promote.sh blue|green}"
current="$(active_color)"
want=$(k get deploy "retrieval-$TARGET" -o jsonpath='{.spec.replicas}' 2>/dev/null) || die "retrieval-$TARGET does not exist"
ready=$(k get deploy "retrieval-$TARGET" -o jsonpath='{.status.readyReplicas}')
[ "${ready:-0}" -ge 1 ] && [ "${ready:-0}" -ge "$want" ] || die "retrieval-$TARGET has ${ready:-0}/$want ready pods; not promoting"
version=$(k get deploy "retrieval-$TARGET" -o jsonpath='{.spec.template.metadata.labels.index-version}')
k patch service retrieval -p "{\"spec\":{\"selector\":{\"app\":\"retrieval\",\"color\":\"$TARGET\"}}}"
mkdir -p reports
echo "$(date -Iseconds) promote ${current:-none} -> $TARGET (index $version)" | tee -a reports/promotions.log
sleep 2
echo "Service retrieval endpoints now:"
k get endpointslices -l kubernetes.io/service-name=retrieval -o jsonpath='{range .items[*].endpoints[*]}{.targetRef.name}{"  ready="}{.conditions.ready}{"\n"}{end}'
