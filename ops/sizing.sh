#!/usr/bin/env bash
# Look at this machine and suggest a lab tier and replica count.
set -euo pipefail
cores=$(nproc)
mem_gib=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
disk_gib=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
echo "cores=$cores  memory=${mem_gib}GiB  free disk=${disk_gib}GiB"
# Leave 2 cores and 4 GiB for the OS, Docker, the control-plane node, Prometheus and the load generator.
usable_cores=$(( cores > 4 ? cores - 2 : 2 ))
usable_mem=$(( mem_gib > 8 ? mem_gib - 4 : 4 ))
echo "budget for retrieval pods: ~${usable_cores} cores, ~${usable_mem} GiB"
echo
if   [ "$mem_gib" -ge 30 ] && [ "$cores" -ge 12 ]; then tier=l
elif [ "$mem_gib" -ge 16 ] && [ "$cores" -ge 8 ];  then tier=m
elif [ "$mem_gib" -ge 12 ];                         then tier=s
else tier=xs; fi
# Each pod: 1 CPU request / 2 CPU limit. Blue+green coexist during promotion, so budget 2x.
replicas=$(( usable_cores / 2 / 2 )); [ "$replicas" -lt 2 ] && replicas=2; [ "$replicas" -gt 4 ] && replicas=4
echo "suggested: TIER=$tier  REPLICAS=$replicas  (both colors running = $(( replicas * 2 )) pods during promotion)"
[ "$disk_gib" -lt 30 ] && echo "WARNING: under 30 GiB free disk; images (~1.5 GB x 3 kind nodes) and indexes need room"
cat <<'T'

tier  products  filler  total vectors  index size   pod memory   build time (8 cores, MiniLM)
xs    20k       0       20k            ~30 MB       ~1.1 GiB     a minute or two
s     100k      0       100k           ~160 MB      ~1.3 GiB     several minutes
m     300k      0       300k           ~0.5 GB      ~2 GiB       tens of minutes (embedding-bound)
l     300k      1.7M    2M             ~3.3 GB      ~5.5 GiB     m + HNSW build of 2M (long)
Sizes are rough guides; the builder prints the measured index size and a suggested pod memory.
T
