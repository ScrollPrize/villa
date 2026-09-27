#!/usr/bin/env bash
# Retain validated paths of 80–160 trace-grid voxels, using 12 workers.
set -euo pipefail
export BANK_NAME=neighbor_negatives_bulk_80_160_v1
exec bash "$(dirname "$0")/launch_neighbor_bank_128.sh" \
    --min-path-length 80 --max-path-length 160 "$@"
