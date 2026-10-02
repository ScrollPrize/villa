#!/usr/bin/env bash
set -euo pipefail
# The coordinate regression training/tracing run (plans/consolidated_training_tracing_alignment.md).
# A new run initialized from matching model and EMA tensors of the 81k path-geometry checkpoint:
# fresh AdamW with a short warmup, empty replay recollected under the current rules, the
# single task budget, and the paris50 source weights. Nothing else is inherited.
# Duration and stopping are fixed here: 40,000 updates, then evaluate the last checkpoint
# against 81k with evaluation/evaluate.py (2,000-voxel Paris limit, same frozen seeds).
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_python="${FIBER_PYTHON:-$task_root/../../../../.venv/bin/python}"
task_run="${COORDINATE_RUN_NAME:-mixed_ct_afv_coordinate_run1}"
source_checkpoint="${COORDINATE_INIT:-$task_root/output/mixed_ct_afv_stem32_run3_pathgeom_paris50/coordinate_regression_weights.pt}"
cd "$task_root"
exec "$task_python" -u -m vesuvius.neural_tracing.fiber_follow.train.train \
  --name "$task_run" --init-weights "$source_checkpoint" \
  --dataset-config "$task_root/configs/mixed_ct_datasets_paris50.json" \
  --stem-channels 32 --stem-blocks 2 \
  --decoder-layers 6 --decoder-ffn 2048 --scorer-layers 4 --axial-layers 2 --hidden 256 --encoder-ffn 256 \
  --recurrent-refinement-steps 3 \
  --steps 40000 --lr 0.0001 --warmup 2000 --rest-grad-clip 60 --tolerance 3.0 --n-commit 16 \
  --batch 4 --grad-steps 3 --workers 10 --threads 4 --worker-cache-gb 0.5 \
  --remote-prefetch-connections 48 --afv-length-power 3 \
  --dagger-every 1000 --dagger-fibers 64 --dagger-batch 8 --dagger-trace-len 768 \
  --replay-max-age 12000 --replay-event-cap 64 --terminal-fallback-cap 0.5 \
  --out-root "$task_root/output" "$@"
