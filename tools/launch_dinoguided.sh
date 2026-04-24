#!/usr/bin/env bash
# Launch the dinoguided ink-detection training on GPUs 4-7.
#
# Assumes M0/M1 have been run:
#   - DINO ckpt at /ephemeral/dinov2_ckpts/checkpoint_step_352500_paris4.pt
#   - UNet snapshot at /ephemeral/dinov2_ckpts/teacher_unet_ckpt_*.pth
#   - Reference embedding at /ephemeral/dinov2_ckpts/avg_ref_embedding.npy
#   - Worktree at /home/ubuntu/sean_ink/villa-dinoguided
#   - Config at /home/ubuntu/sean_ink/configs/ps256_3d_bcedice_dinoguided.json
#
# The training writes to /ephemeral/3d_ink_checkpoints/ps256_bcedice_dinoguided/
# and logs to /ephemeral/logs/ink_dinoguided.log.

set -euo pipefail

WORKTREE="/home/ubuntu/sean_ink/villa-dinoguided"
VENV="/home/ubuntu/sean_ink/villa/vesuvius/.venv"
CONFIG="/home/ubuntu/sean_ink/configs/ps256_3d_bcedice_dinoguided.json"
LOG="/ephemeral/logs/ink_dinoguided.log"

mkdir -p "$(dirname "$LOG")"
cd "$WORKTREE"

nohup env \
    CUDA_VISIBLE_DEVICES=4,5,6,7 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    PYTHONPATH="$WORKTREE/vesuvius/src:$WORKTREE/ink-detection" \
    "$VENV/bin/torchrun" \
        --nproc_per_node=4 --standalone --master_port=29517 \
        ink-detection/koine_machines/training/train.py \
        "$CONFIG" \
    > "$LOG" 2>&1 &
disown

echo "launched. log: $LOG"
echo "tail -F $LOG | tr '\\r' '\\n'"
