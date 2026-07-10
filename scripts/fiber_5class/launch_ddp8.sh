#!/usr/bin/env bash
# 8-GPU DDP launcher for the 4-class fiber/ink self-distillation trainer.
#
# All paths are derived from this script's location, so the package is
# relocatable. Override the config by passing a path as $1.
#
#   bash launch_ddp8.sh                          # uses configs/train.json
#   bash launch_ddp8.sh configs/train.smoke.json # smoke run
#
# Environment knobs:
#   SESSION       tmux session name        (default f5c_ddp8)
#   NPROC         processes / GPUs         (default 8)
#   CUDA_DEVICES  CUDA_VISIBLE_DEVICES     (default 0,1,2,3,4,5,6,7)
#   VESUVIUS_DIR  path to the vesuvius uv project (default: repo's vesuvius/)
#   WANDB_API_KEY optional; falls back to ~/.netrc

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="${1:-${SCRIPT_DIR}/configs/train.json}"
TRAIN="${SCRIPT_DIR}/train.py"
# vesuvius uv project: default to <repo>/vesuvius (three levels up: fiber_5class -> scripts -> repo).
VESUVIUS_DIR="${VESUVIUS_DIR:-$(cd "${SCRIPT_DIR}/../../vesuvius" && pwd)}"
SESSION="${SESSION:-f5c_ddp8}"
NPROC="${NPROC:-8}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WARNING: WANDB_API_KEY not set; relying on ~/.netrc for auth." >&2
fi
if [[ ! -f "$CFG" ]]; then
  echo "missing config: $CFG" >&2
  exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session $SESSION already exists; kill it first: tmux kill-session -t $SESSION" >&2
  exit 1
fi

echo "launching $SESSION on GPUs $CUDA_DEVICES (nproc=$NPROC) cfg=$CFG"
tmux new-session -d -s "$SESSION" "bash -lc '
  set -euo pipefail
  cd \"${VESUVIUS_DIR}\"
  export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}
  export WANDB_API_KEY=${WANDB_API_KEY:-}
  export TORCH_NCCL_BLOCKING_WAIT=0
  # --no-sync: keep the manually-installed CUDA-12 cupy/cucim/cuws in the venv
  # (the resolver would otherwise revert them). See README.
  exec uv run --no-sync torchrun \
    --standalone --nproc-per-node=${NPROC} \
    \"${TRAIN}\" --config \"${CFG}\"
'"

echo "attach with: tmux attach -t $SESSION"
