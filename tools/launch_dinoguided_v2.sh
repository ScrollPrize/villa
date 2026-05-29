#!/usr/bin/env bash
# Launch the dinoguided v2 ink-detection training on GPUs 4-7.
#
# v2 differences vs v1:
#   - Resumes from /ephemeral/3d_ink_checkpoints/ps256_bcedice_dinoguided/ckpt_063000.pth
#     for both the student (weights+optimizer+scheduler+EMA) and the frozen
#     UNet teacher inside the dino-guided label generator.
#   - Uses the new input-mask gating in DinoGuidedLabelGenerator (any voxel
#     with raw value <= input_mask_threshold is forced to 0 in unet_prob).
#   - Output dir: /ephemeral/3d_ink_checkpoints/ps256_bcedice_dinoguided_v2/
#   - Log: /ephemeral/logs/ink_dinoguided_v2.log
#   - master_port 29518 (29517 used by v1; v1 must be stopped before launch)

set -euo pipefail

WORKTREE="/home/ubuntu/sean_ink/villa-dinoguided"
VENV="/home/ubuntu/sean_ink/villa/vesuvius/.venv"
CONFIG="/home/ubuntu/sean_ink/configs/ps256_3d_bcedice_dinoguided_v2.json"
LOG="/ephemeral/logs/ink_dinoguided_v2.log"

mkdir -p "$(dirname "$LOG")"
cd "$WORKTREE"

nohup env \
    CUDA_VISIBLE_DEVICES=4,5,6,7 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    PYTHONPATH="$WORKTREE/vesuvius/src:$WORKTREE/ink-detection" \
    "$VENV/bin/torchrun" \
        --nproc_per_node=4 --standalone --master_port=29518 \
        ink-detection/koine_machines/training/train.py \
        "$CONFIG" \
    > "$LOG" 2>&1 &
disown

echo "launched. log: $LOG"
echo "tail -F $LOG | tr '\\r' '\\n'"
