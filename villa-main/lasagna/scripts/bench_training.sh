#!/bin/bash
# Benchmark training with cProfile.
# Usage: bash lasagna/scripts/bench_training.sh
set -uo pipefail

VENV="${VENV:-$HOME/hendrik/train_venv}"
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"

SRC="${SRC:-$(cd "$(dirname "$0")/../.." && pwd)}"
export PYTHONPATH="$SRC/vesuvius/src/:$SRC/lasagna/"
export CUDA_VISIBLE_DEVICES=7
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 BLOSC_NTHREADS=4 OPENBLAS_NUM_THREADS=4

CONFIG="${CONFIG:-$SRC/lasagna/configs/tifxyz_train_s3_srv.json}"
WEIGHTS="${WEIGHTS:-runs/tifxyz3d/20260418_070618_lasagna_scratch_all_1e-5_dev6,7/model_best.pt}"
DURATION="${DURATION:-60}"

OUTDIR="$SRC/lasagna/tmp/bench_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

echo "=== Bench: duration=${DURATION}s, output=$OUTDIR ==="

# Start training under cProfile
python -m cProfile -o "$OUTDIR/profile.prof" \
    "$SRC/lasagna/train_tifxyz.py" \
    --config "$CONFIG" --weights "$WEIGHTS" \
    --patch-size 192 --label-patch-size 192 --model-patch-size 320 \
    --batch-size 4 --num-workers 4 --epochs 1 --verbose \
    --w-cos 1.0 --w-mag 100.0 --w-dir 100.0 --w-smooth 100.0 \
    --lr 3.2e-4 --no-himag-filter --run-name bench \
    2>&1 | tee "$OUTDIR/train.log" &
PID=$!

trap "kill $PID 2>/dev/null; pkill -P $PID 2>/dev/null; wait 2>/dev/null" EXIT

# Run for DURATION then stop
sleep "$DURATION"
echo ""
echo "[bench] ${DURATION}s elapsed, stopping..."
kill $PID 2>/dev/null || true
pkill -P $PID 2>/dev/null || true
sleep 2
kill -9 $PID 2>/dev/null || true
wait $PID 2>/dev/null || true

# Print top 30 functions by cumulative time
echo ""
echo "=== Top 30 by cumulative time ==="
python -c "
import pstats
s = pstats.Stats('$OUTDIR/profile.prof')
s.sort_stats('cumulative')
s.print_stats(30)
" 2>/dev/null || echo "(profile incomplete — training may not have saved stats)"

echo ""
echo "=== Top 30 by total time ==="
python -c "
import pstats
s = pstats.Stats('$OUTDIR/profile.prof')
s.sort_stats('tottime')
s.print_stats(30)
" 2>/dev/null || echo "(profile incomplete)"

echo ""
echo "--- Iteration speed ---"
grep -oP '\d+\.\d+s/it|\d+\.\d+it/s' "$OUTDIR/train.log" | tail -10 || true

echo ""
echo "Profile: $OUTDIR/profile.prof"
echo "View interactively: snakeviz $OUTDIR/profile.prof"
