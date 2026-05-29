#!/usr/bin/env bash
# Launch the dinoguided v3-fullsup variant on GPUs 0-3 in parallel with v3.
#
# Identical to v3 except `force_full_supervision: true`, which makes the loss
# weight every voxel (including pure-air background) instead of masking out
# voxels where supervision_mask <= 0. The intent: penalize the model for any
# positive prediction outside the scroll material.
#
#   - Output dir: /ephemeral/3d_ink_checkpoints/ps256_bcedice_dinoguided_v3_fullsup/
#   - Log: /ephemeral/logs/ink_dinoguided_v3_fullsup.log
#   - master_port 29520 (29519 used by v3 on GPUs 4-7)

set -euo pipefail

WORKTREE="/home/ubuntu/sean_ink/villa-dinoguided"
VENV="/home/ubuntu/sean_ink/villa/vesuvius/.venv"
CONFIG="/home/ubuntu/sean_ink/configs/ps256_3d_bcedice_dinoguided_v3_fullsup.json"
LOG="/ephemeral/logs/ink_dinoguided_v3_fullsup.log"

mkdir -p "$(dirname "$LOG")"
cd "$WORKTREE"

nohup env \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="$WORKTREE/vesuvius/src:$WORKTREE/ink-detection" \
    "$VENV/bin/torchrun" \
        --nproc_per_node=4 --standalone --master_port=29520 \
        ink-detection/koine_machines/training/train.py \
        "$CONFIG" \
    > "$LOG" 2>&1 &
disown

echo "launched. log: $LOG"
echo "tail -F $LOG | tr '\\r' '\\n'"
