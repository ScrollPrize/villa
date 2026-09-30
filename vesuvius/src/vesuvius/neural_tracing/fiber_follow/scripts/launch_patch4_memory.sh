#!/usr/bin/env bash
# Compiled patch/shuffle follower.
set -euo pipefail
FF="$(cd "$(dirname "$0")/.." && pwd)"
export RUN_NAME=${RUN_NAME:-axial_patch4_memory_v9_run1}
# Edit these defaults here, or override them through environment variables.
BATCH_SIZE=${BATCH_SIZE:-1}
MICROBATCH_SIZE=${MICROBATCH_SIZE:-16}
WORKERS=${WORKERS:-8}
# Batch is a minimum decision budget; whole loader chunks can exceed it.
# regression.train always compiles its training operations; no --compile flag.
exec bash "$FF/scripts/launch_memory.sh" \
    --encoder patch4 --batch "$BATCH_SIZE" --microbatch "$MICROBATCH_SIZE" \
    --workers "$WORKERS" --feature-sequence-length 2 "$@"
