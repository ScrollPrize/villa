#!/usr/bin/env bash
# Optional v5: full fine proposal map, connected route, two refinement passes.
set -euo pipefail
FF="$(cd "$(dirname "$0")/.." && pwd)"
export RUN_NAME=${RUN_NAME:-axial_candidate_memory_v5_run1}
exec bash "$FF/scripts/launch_memory.sh" \
    --n-commit 16 --presence-dropout 0 \
    --fresh-fraction 0.7 --decision-fraction 0.1 --bank-following-probability 0.2 \
    --memory-version 5 --no-correction --trajectory-sequence-weight 0 \
    --feature-memory-revision 2 --feature-detail-tokens 16 --feature-stream-steps 128 \
    --feature-replay-weight 0.5 --feature-sequence-length 2 --feature-memory-grid 2 4 4 \
    --memory-stride 8 --proposal-step 0.5 --proposal-candidates 32 \
    --proposal-suppression 0.5 --recurrent-refinement-steps 2 \
    --proposal-warmup-steps 500 --proposal-inherited-lr-scale 0.1 \
    --direction-inputs --batch 8 --microbatch 4 --workers 8 --compile "$@"
