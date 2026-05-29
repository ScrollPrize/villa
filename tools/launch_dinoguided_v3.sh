#!/usr/bin/env bash
# Launch the dinoguided v3 ink-detection finetune on GPUs 4-7.
#
# v3 differences vs v2:
#   - dynamic_label.kind="self_distill": pseudo-labels come from running
#     v2 ckpt_077000 (and conditionally ckpt_064000) with TTA, then thresholding
#     into binary; replaces the DINO + UNet union pipeline.
#   - Resumes weights+optimizer+scheduler+EMA from
#     /ephemeral/3d_ink_checkpoints/ps256_bcedice_dinoguided_v2/ckpt_077000.pth
#   - Adds extra_patches sampler: 25% of training samples come from random
#     +/-1024 jittered crops around 8 user-supplied Paris4 coords.
#   - Output dir: /ephemeral/3d_ink_checkpoints/ps256_bcedice_dinoguided_v3/
#   - Log: /ephemeral/logs/ink_dinoguided_v3.log
#   - master_port 29519 (29518 used by v2; v2 must be stopped before launch)

set -euo pipefail

WORKTREE="/home/ubuntu/sean_ink/villa-dinoguided"
VENV="/home/ubuntu/sean_ink/villa/vesuvius/.venv"
CONFIG="/home/ubuntu/sean_ink/configs/ps256_3d_bcedice_dinoguided_v3.json"
LOG="/ephemeral/logs/ink_dinoguided_v3.log"

mkdir -p "$(dirname "$LOG")"
cd "$WORKTREE"

nohup env \
    CUDA_VISIBLE_DEVICES=4,5,6,7 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    PYTHONPATH="$WORKTREE/vesuvius/src:$WORKTREE/ink-detection" \
    "$VENV/bin/torchrun" \
        --nproc_per_node=4 --standalone --master_port=29519 \
        ink-detection/koine_machines/training/train.py \
        "$CONFIG" \
    > "$LOG" 2>&1 &
disown

echo "launched. log: $LOG"
echo "tail -F $LOG | tr '\\r' '\\n'"
