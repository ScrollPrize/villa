#!/usr/bin/env bash
# One live bank for identity negatives, wrong continuations and extra following.
# Run: bash scripts/launch_identity_shared_bank.sh [additional training options]
set -euo pipefail
FF="$(cd "$(dirname "$0")/.." && pwd)"
VES="$(cd "$FF/../../../.." && pwd)"
cd "$FF"
export PYTHON="${PYTHON:-$VES/.venv/bin/python}"
export AGENTS_AGENT_MODE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
RUN_NAME=${RUN_NAME:-direct_identity_shared_bank_run1}
if [[ -e "$FF/output/$RUN_NAME" || -e "$FF/output/logs/$RUN_NAME.log" ]]; then
    echo "Fresh run destination already exists: $RUN_NAME" >&2
    exit 1
fi
exec bash "$FF/scripts/launch_regression.sh" "$RUN_NAME" \
    --out-root "$FF/output" --device cuda --steps 100000 \
    --batch 16 --microbatch 4 --workers 12 --worker-cache-gb 0.5 --threads 4 \
    --lr 0.0003 --warmup 500 --ema-decay 0.999 \
    --confidence-weight 0.5 --tolerance 1.5 --n-commit 4 \
    --channels 32 --decoder-layers 4 \
    --correction --correction-steps 2 --correction-limit 1.0 \
    --no-history-prob 0.15 --short-history-prob 0.4 \
    --identity --identity-weight 0.5 --identity-temperature 0.1 \
    --appearance-channels 32 --embedding 32 --negative-threshold 0.7 \
    --presence-dropout 0.25 --anchor-prob 0.75 \
    --contacts "$FF/output/direct_ct_spatial_run1/contacts.json" \
    --hard-spans "$FF/output/hard_spans_8a0bb01095fa.json" \
    --contact-fraction 0.2 --hard-span-fraction 0.1 --lateral-fraction 0.1 \
    --compile --seed 0 --val-z 45000 48500 \
    --log-every 50 --ckpt-every 1000 --diag-every 5000 --diag-max-len 400 \
    --long-diag-every 0 --long-diag-max-len 1200 \
    --recovery-every 1000 --recovery-seeds 8 --recovery-length 32 \
    --dagger-every 1000 --dagger-seeds 64 --dagger-trace-len 6000 --replay-keep 4 \
    --negative-bank "$FF/output/neighbor_negatives_bulk_r12_32_l80_160_v1" \
    --negative-bank-refresh-seconds 30 --negative-bank-cache-mb 64 \
    --identity-query-patches --negative-lateral-max 32 \
    --bank-following-probability 0.1 \
    --bank-wrong-continuation-probability 0.75 --bank-wrong-continuation-tail 4 128 "$@"
