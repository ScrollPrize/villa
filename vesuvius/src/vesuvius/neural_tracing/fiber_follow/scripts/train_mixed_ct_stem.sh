#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_run="${STEM_RUN_NAME:-mixed_ct_afv_stem32_fresh_run1}"
cd "$task_root"
# Fresh initialization; use the same mixed datasets, volume cache and workers.
exec bash "$task_root/scripts/train_mixed_ct.sh" \
  --name "$task_run" --stem-channels 32 --stem-blocks 2 \
  --steps 100000 --lr 0.0001 --warmup 5000 \
  --fresh-fraction 0.9 --bank-following-probability 0 \
  --remote-prefetch-connections 48 "$@"
