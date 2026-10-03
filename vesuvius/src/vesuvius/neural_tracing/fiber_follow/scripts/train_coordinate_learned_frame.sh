#!/usr/bin/env bash
set -euo pipefail
# Fresh 100k-step follower run from the converted 81k model/EMA weights.
# Reuse the existing coordinate launcher; restore the source run's 5k warmup,
# 6000-voxel collection limit and 96-voxel post-failure collection window.
# Current task/sampling rules apply. No optimizer, scheduler, or replay is resumed.
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export COORDINATE_RUN_NAME="${COORDINATE_RUN_NAME:-mixed_ct_afv_coordinate_learned_frame_100k}"
task_frame="${FRAME_CHECKPOINT:-$task_root/output/heading_model_l0_w16_centered_frame/ckpt_064000.pt}"
exec bash "$task_root/scripts/train_coordinate_regression.sh" \
  --steps 100000 --warmup 5000 \
  --dagger-trace-len 6000 --dagger-after 96 \
  --frame-checkpoint "$task_frame" "$@"
