#!/usr/bin/env bash
# Observation-memory regression with causal segment survival scoring.
set -euo pipefail
FF="$(cd "$(dirname "$0")/.." && pwd)"
VES="$(cd "$FF/../../../.." && pwd)"
cd "$FF"
export PYTHON="${PYTHON:-$VES/.venv/bin/python}"
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-4}
export AGENTS_AGENT_MODE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
RUN_NAME=${RUN_NAME:-axial_survival_memory_v8_run1}
BANK_PATH=${BANK_PATH:-$FF/output/neighbor_samples_r0_32_l80_160_v2}
if [[ -e "$FF/output/$RUN_NAME" || -e "$FF/output/logs/$RUN_NAME.log" ]]; then
    echo "Fresh run destination already exists: $RUN_NAME" >&2
    exit 1
fi
# Training settings live in regression.train.build_parser.
exec bash "$FF/scripts/launch_regression.sh" "$RUN_NAME" \
    --negative-bank "$BANK_PATH" "$@"
