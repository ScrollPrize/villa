#!/usr/bin/env bash
# Observation memory; two decisions x two traces per microbatch.
set -euo pipefail
FF="$(cd "$(dirname "$0")/.." && pwd)"
export RUN_NAME=${RUN_NAME:-axial_observation_memory_v4_run1}
exec bash "$FF/scripts/launch_memory.sh" \
    --n-commit 16 --presence-dropout 0 --direction-inputs \
    --batch 8 --microbatch 4 --workers 8 \
    --fresh-fraction 0.7 --decision-fraction 0.3 --bank-following-probability 0.2 \
    --memory-switch-probability 0.3 --memory-stride 8 --memory-probe-weight 0 \
    --memory-version 4 --no-correction --trajectory-sequence-weight 0 \
    --feature-memory-revision 2 --recurrent-refinement-steps 1 \
    --feature-sequence-length 2 --feature-memory-grid 2 4 4 \
    --compile "$@"
