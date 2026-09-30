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
exec bash "$FF/scripts/launch_regression.sh" "$RUN_NAME" \
    --out-root "$FF/output" --device cuda --steps 100000 \
    --batch 8 --microbatch 4 --workers 8 --worker-cache-gb 0.5 --threads 4 \
    --lr 0.0003 --warmup 500 --ema-decay 0.999 \
    --memory-grad-clip 5 --rest-grad-clip 20 \
    --confidence-weight 0.5 --tolerance 1.5 --n-commit 16 \
    --channels 32 --hidden 128 --axial-layers 4 --decoder-layers 4 \
    --no-activation-checkpointing --recurrent-refinement-steps 2 \
    --no-history-prob 0.15 --short-history-prob 0.4 \
    --decision-fraction 0.3 --decision-choice-fraction 0.75 --candidate-weight 1.0 --fresh-fraction 0.7 \
    --presence-dropout 0 --direction-inputs \
    --blur-probability 0.25 --blur-sigma 0.5 1.25 \
    --contacts "$FF/output/direct_ct_spatial_run1/contacts.json" \
    --hard-spans "$FF/output/hard_spans_8a0bb01095fa.json" \
    --contact-fraction 0.2 --hard-span-fraction 0.1 --lateral-fraction 0.1 \
    --memory-slots 16 --memory-steps 64 --memory-stride 8 \
    --feature-history-loss-fraction 0.25 --feature-switch-crop-fraction 0.15 \
    --memory-switch-probability 0.3 --memory-switch-tail 16 96 --dagger-after 96 \
    --compile --seed 0 --val-z 45000 48500 \
    --log-every 50 --ckpt-every 1000 --diag-every 5000 --diag-max-len 400 \
    --long-diag-every 0 --long-diag-max-len 1200 \
    --recovery-every 1000 --recovery-seeds 8 --recovery-length 32 \
    --dagger-every 1000 --dagger-seeds 64 --dagger-trace-len 6000 --replay-keep 4 \
    --negative-bank "$BANK_PATH" --negative-bank-refresh-seconds 30 --negative-bank-cache-mb 64 \
    --bank-coverage-probability 0.2 --bank-following-probability 0.2 \
    --bank-hard-fraction 0.5 --replay-failure-fraction 0.5 \
    --bank-switch-tolerance 0.75 --bank-own-tolerance 1.5 \
    --bank-wrong-continuation-probability 0 --bank-wrong-continuation-tail 4 16 "$@"
