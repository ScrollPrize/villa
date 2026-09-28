#!/usr/bin/env bash
# One encoder across current/seed/history crops; fresh architecture and optimizer.
set -euo pipefail
FF="$(cd "$(dirname "$0")/.." && pwd)"
export RUN_NAME=${RUN_NAME:-axial_unified_run1}
exec bash "$FF/scripts/launch_axial.sh" \
    --memory-version 3 --memory-slots 16 --memory-steps 64 --memory-grad-steps 32 \
    --memory-stride 4 --memory-probe-weight 0.5 \
    --memory-switch-probability 0.15 --memory-switch-tail 16 96 --dagger-after 96 \
    --microbatch 2 --workers 2 --activation-checkpointing --no-compile \
    --log-every 10 "$@"
