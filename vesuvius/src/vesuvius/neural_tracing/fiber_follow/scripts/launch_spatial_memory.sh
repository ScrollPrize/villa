#!/usr/bin/env bash
# Separate v3 run; all existing v2 recipes/checkpoints remain supported.
set -euo pipefail
FF="$(cd "$(dirname "$0")/.." && pwd)"
export RUN_NAME=${RUN_NAME:-axial_spatial_memory_v3_run1}
exec bash "$FF/scripts/launch_memory.sh" \
    --memory-version 3 --route-grid-step 2 --route-transition-radius 1 \
    --route-transition-cost 0.25 --route-loss-weight 1 --route-sequence-weight 0.5 \
    --compile "$@"
