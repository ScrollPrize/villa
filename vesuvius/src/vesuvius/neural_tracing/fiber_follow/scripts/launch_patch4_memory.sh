#!/usr/bin/env bash
# Compiled patch/shuffle follower.
set -euo pipefail
FF="$(cd "$(dirname "$0")/.." && pwd)"
export RUN_NAME=${RUN_NAME:-axial_patch4_overlap_tokens_slabs_v11_run1}
# Edit these defaults here, or override them through environment variables.
BATCH_SIZE=${BATCH_SIZE:-16}
GRAD_STEPS=${GRAD_STEPS:-1}
WORKERS=${WORKERS:-8}
# Effective batch per optimizer update is BATCH_SIZE * GRAD_STEPS.
# regression.train always compiles its training operations; no --compile flag.
exec bash "$FF/scripts/launch_memory.sh" \
    --encoder patch4 --token-only --batch "$BATCH_SIZE" --grad-steps "$GRAD_STEPS" \
    --workers "$WORKERS" "$@"
