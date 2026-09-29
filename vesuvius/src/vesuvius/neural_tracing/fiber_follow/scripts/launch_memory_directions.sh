#!/usr/bin/env bash
# Run4's checkpoint-32000 recipe, adding unsigned direction inputs everywhere.
# Starts a NEW run from EMA weights; old optimizer/step state is not resumed.
set -euo pipefail
FF="$(cd "$(dirname "$0")/.." && pwd)"
export RUN_NAME="${RUN_NAME:-axial_memory_directions_run1}"
INIT_TRACER="${INIT_TRACER:-$FF/output/axial_memory_seq_run4/ckpt_032000.pt}"
if [[ ! -f "$INIT_TRACER" ]]; then
    echo "Initialization checkpoint not found: $INIT_TRACER" >&2
    exit 1
fi
command=(bash "$FF/scripts/launch_memory.sh"
    --init-tracer "$INIT_TRACER" --direction-inputs
    --n-commit 8 --presence-dropout 0
    --fresh-fraction 0.7 --bank-following-probability 0.2 --decision-fraction 0.1)
if [[ ${1:-} == --dry-run ]]; then
    shift
    printf 'RUN_NAME=%q ' "$RUN_NAME"
    printf '%q ' "${command[@]}" "$@"
    printf '\n'
    exit 0
fi
exec "${command[@]}" "$@"
