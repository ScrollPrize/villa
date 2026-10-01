#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_run="${STEM_RUN_NAME:-mixed_ct_afv_stem32_fresh_run1}"
cd "$task_root"
# Fresh initialization; use the same mixed datasets, volume cache and workers.
exec bash "$task_root/scripts/train_mixed_ct.sh" \
  --name "$task_run" --stem-channels 32 --stem-blocks 2 \
  --steps 100000 --lr 0.0001 --warmup 5000 \
  --batch 10 --grad-steps 1 --prefer-replay-for-light-gt \
  --clean-fraction 0.7 --gt-perturb-probability 0.25 \
  --gt-perturb-max-offset 0.5 --gt-perturb-max-angle-deg 2 \
  --decision-fraction 0.2 --fresh-fraction 0.75 --memory-switch-probability 0.3333333333333333 \
  --bank-following-probability 0 --replay-continuation-fraction 0.8 --prefer-real-wrong-turns \
  --remote-prefetch-connections 48 "$@"
