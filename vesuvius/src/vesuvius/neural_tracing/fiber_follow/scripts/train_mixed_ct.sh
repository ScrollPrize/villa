#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_python="${FIBER_PYTHON:-$task_root/../../../../.venv/bin/python}"
exec "$task_python" -m vesuvius.neural_tracing.fiber_follow.regression.train \
  --name mixed_ct_afv_run1 \
  --dataset-config "$task_root/configs/mixed_ct_datasets.json" \
  --input-mode ct --no-direction-inputs --presence-dropout 0 \
  --encoder patch4 --token-only --recurrent-refinement-steps 3 \
  --batch 16 --microbatch 16 --workers 10 --threads 4 --worker-cache-gb 0.5 \
  --lr 0.0003 --warmup 1000 --steps 100000 \
  --out-root "$task_root/output" "$@"
