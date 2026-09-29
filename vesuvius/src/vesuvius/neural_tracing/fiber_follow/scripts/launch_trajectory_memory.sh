#!/usr/bin/env bash
# Main-encoder feature memory; two decisions x two traces per microbatch.
set -euo pipefail
FF="$(cd "$(dirname "$0")/.." && pwd)"
export RUN_NAME=${RUN_NAME:-axial_trajectory_memory_v4_run1}
exec bash "$FF/scripts/launch_memory.sh" \
    --n-commit 8 --presence-dropout 0 \
    --fresh-fraction 0.7 --decision-fraction 0.1 --bank-following-probability 0.2 \
    --memory-version 4 --no-correction --trajectory-sequence-weight 0 \
    --feature-sequence-length 2 --feature-memory-grid 2 4 4 \
    --compile "$@"
