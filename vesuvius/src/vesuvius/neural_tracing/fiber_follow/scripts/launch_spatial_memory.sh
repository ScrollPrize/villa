#!/usr/bin/env bash
# Separate v3 run using the axial_memory_seq_run4 training/sampling recipe.
set -euo pipefail
FF="$(cd "$(dirname "$0")/.." && pwd)"
export RUN_NAME=${RUN_NAME:-axial_spatial_memory_v3_run1}
exec bash "$FF/scripts/launch_memory.sh" \
    --n-commit 8 --presence-dropout 0 \
    --fresh-fraction 0.7 --decision-fraction 0.1 --bank-following-probability 0.2 \
    --memory-version 3 --route-grid-step 2 --route-transition-radius 1 \
    --route-transition-cost 0.25 --route-loss-weight 1 --route-sequence-weight 0.5 \
    --compile "$@"
