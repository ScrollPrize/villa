#!/usr/bin/env bash
# New bank only: 12–32 voxel search band, 80–160 voxel paths, 5 low-priority workers.
set -euo pipefail
export BANK_NAME=neighbor_negatives_bulk_r12_32_l80_160_v1
exec bash "$(dirname "$0")/launch_neighbor_bank_128.sh" \
    --min-path-length 80 --max-path-length 160 \
    --min-distance 12 --max-distance 32 --workers 5 "$@"
