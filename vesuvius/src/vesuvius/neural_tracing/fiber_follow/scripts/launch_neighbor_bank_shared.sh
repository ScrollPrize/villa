#!/usr/bin/env bash
# Fresh v2 bank: nearby and outer relationships, 80–160 voxel paths.
# Training can start once run.json exists and discovers subsequent publications.
set -euo pipefail
export BANK_NAME=${BANK_NAME:-neighbor_samples_r0_32_l80_160_v2}
exec bash "$(dirname "$0")/launch_neighbor_bank_128.sh" \
    --min-path-length 80 --max-path-length 160 \
    --min-distance 0 --max-distance 32 --workers 5 "$@"
