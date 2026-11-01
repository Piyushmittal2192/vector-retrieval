#!/usr/bin/env bash
# Copy local images into every kind node (kind nodes cannot see your Docker images).
source "$(dirname "$0")/common.sh"
for img in "$@"; do
  echo "loading $img"
  kind load docker-image "$img" --name "$CLUSTER"
done
