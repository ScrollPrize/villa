#!/usr/bin/env bash
set -euo pipefail
# Settings captured directly from mixed_ct_afv_stem32_fresh_run1/ckpt_031000.pt.
# Checkpoint SHA256: 9ce8501427e73674b591e2244ac975f6149534406bc3a9681054cf39bd4c6efa
# The checkpoint is provenance only; this script does not load it.
# Batch 6 with two accumulation steps gives 12 examples per optimizer update.
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_python="${FIBER_PYTHON:-$task_root/../../../../.venv/bin/python}"
task_run="${STEM_RUN_NAME:-mixed_ct_afv_stem32_fresh_run2}"
cd "$task_root"
exec "$task_python" -u -m vesuvius.neural_tracing.fiber_follow.regression.train \
  --name "$task_run" \
  --fiber-zarrs /mnt/raid_nvme/spiral_dataset_working/fiber_zarrs \
  --fibers /mnt/raid_nvme/spiral_dataset_working/fibers \
  --ct /mnt/raid_nvme/volpkgs/s1_2um_ds2.volpkg/volumes/s1_ds2.zarr \
  --manifest "$task_root/output/single_path_v11_preparation/seeds.json" \
  --dataset-config "$task_root/configs/mixed_ct_datasets.json" \
  --input-mode ct \
  --onpolicy \
  --out-root "$task_root/output" \
  --device cuda \
  --steps 100000 \
  --batch 6 \
  --grad-steps 2 \
  --workers 10 \
  --worker-cache-gb 0.5 \
  --remote-prefetch-connections 48 \
  --remote-prefetch-queue-size 512 \
  --remote-prefetch-lookahead 16 \
  --remote-prefetch-timeout 120.0 \
  --threads 4 \
  --lr 0.0001 \
  --warmup 5000 \
  --ema-decay 0.999 \
  --history-grad-clip 5.0 \
  --rest-grad-clip 100.0 \
  --confidence-weight 0.5 \
  --tolerance 3.0 \
  --n-commit 16 \
  --channels 32 \
  --stem-channels 32 \
  --stem-blocks 2 \
  --encoder patch4 \
  --history-encoder fine \
  --token-only \
  --no-direction-inputs \
  --decoder-layers 4 \
  --axial-layers 4 \
  --hidden 128 \
  --memory-switch-probability 0.3333333333333333 \
  --memory-switch-tail 16.0 96.0 \
  --no-activation-checkpointing \
  --no-history-prob 0.15 \
  --short-history-prob 0.4 \
  --decision-fraction 0.2 \
  --decision-choice-fraction 0.75 \
  --candidate-weight 1.0 \
  --fresh-fraction 0.75 \
  --clean-fraction 0.7 \
  --gt-perturb-probability 0.25 \
  --gt-perturb-max-offset 0.5 \
  --gt-perturb-max-angle-deg 2.0 \
  --live-continuation \
  --live-continuation-steps 4 8 \
  --replay-continuation-fraction 0.8 \
  --prefer-real-wrong-turns \
  --prefer-replay-for-light-gt \
  --negative-bank "$task_root/output/neighbor_samples_r0_32_l80_160_v2" \
  --bank-coverage-probability 0.2 \
  --no-prefer-long-continuations \
  --negative-bank-refresh-seconds 30.0 \
  --negative-bank-cache-mb 64.0 \
  --bank-wrong-continuation-probability 0.0 \
  --bank-wrong-continuation-tail 4.0 16.0 \
  --bank-following-probability 0.0 \
  --bank-hard-fraction 0.5 \
  --replay-failure-fraction 0.2 \
  --bank-switch-tolerance 0.75 \
  --bank-own-tolerance 1.5 \
  --presence-dropout 0.0 \
  --blur-probability 0.25 \
  --blur-sigma 0.5 1.25 \
  --lateral-fraction 0.1 \
  --seed 0 \
  --val-z 45000.0 48500.0 \
  --log-every 50 \
  --ckpt-every 1000 \
  --diag-every 5000 \
  --batch-diag-every 1000 \
  --diag-max-len 400.0 \
  --long-diag-every 0 \
  --long-diag-max-len 1200.0 \
  --recovery-every 1000 \
  --recovery-seeds 8 \
  --recovery-length 32.0 \
  --dagger-every 1000 \
  --dagger-seeds 64 \
  --dagger-trace-len 6000.0 \
  --dagger-after 96.0 \
  --replay-keep 4 \
  --recurrent-refinement-steps 3
