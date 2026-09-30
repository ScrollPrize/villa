#!/usr/bin/env bash
# Compiled patch/shuffle follower.
set -euo pipefail
FF="$(cd "$(dirname "$0")/.." && pwd)"
export RUN_NAME=${RUN_NAME:-axial_patch4_tokens_slabs_v10_run1}
# Edit these defaults here, or override them through environment variables.
BATCH_SIZE=${BATCH_SIZE:-16}
MICROBATCH_SIZE=${MICROBATCH_SIZE:-166}
WORKERS=${WORKERS:-8}
# Batch is a minimum decision budget; whole loader chunks can exceed it.
# regression.train always compiles its training operations; no --compile flag.
exec bash "$FF/scripts/launch_memory.sh" \
    --encoder patch4 --token-only --batch "$BATCH_SIZE" --microbatch "$MICROBATCH_SIZE" \
    --workers "$WORKERS" "$@"
