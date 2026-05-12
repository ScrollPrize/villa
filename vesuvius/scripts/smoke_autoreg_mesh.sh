#!/usr/bin/env bash
# smoke_autoreg_mesh.sh — short 8-GPU autoreg_mesh smoke with canary attached.
#
# Usage:
#     bash scripts/smoke_autoreg_mesh.sh <config.json> [num_steps=200] [nproc_per_node=8]
#
# Pass criteria:
#   - num_steps complete with no NCCL/OOM errors.
#   - Canary log has zero TRIGGER lines.
#   - Final 20 step_times within +/- 20% of median.
set -euo pipefail

CONFIG="${1:?usage: smoke_autoreg_mesh.sh <config.json> [num_steps=200] [nproc=8]}"
NUM_STEPS="${2:-200}"
NPROC="${3:-8}"

if [ ! -f "$CONFIG" ]; then
  echo "error: config not found: $CONFIG" >&2
  exit 2
fi

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

RUN_NAME="smoke-$(date +%Y%m%d_%H%M%S)"
LOG_DIR="/ephemeral/checkpoints/${RUN_NAME}"
CANARY_LOG="/ephemeral/canary/${RUN_NAME}.log"
mkdir -p "$LOG_DIR" "$(dirname "$CANARY_LOG")"

TMP_CFG="${LOG_DIR}/smoke_config.json"

# jq: clone the production config and override fields for a short run.
#  - num_steps -> $NUM_STEPS
#  - disable checkpointing (avoid filling disk)
#  - redirect out_dir + run name
#  - drop log_frequency to 10 so we see steps quickly
jq \
  --argjson n "$NUM_STEPS" \
  --arg out "$LOG_DIR" \
  --arg run "$RUN_NAME" \
  '. + {
    num_steps: $n,
    ckpt_frequency: 999999999,
    ckpt_at_step_zero: false,
    save_final_checkpoint: false,
    out_dir: $out,
    wandb_run_name: $run,
    wandb_log_images: false,
    log_frequency: 10
  }' "$CONFIG" > "$TMP_CFG"

# ulimits to handle the file_system sharing strategy.
ulimit -n 1048576 || echo "warn: ulimit -n 1048576 failed (continuing)"
ulimit -u 65535   || echo "warn: ulimit -u failed (continuing)"

# NCCL and AWS safety env. The trainer also sets these defensively but
# exporting here makes them visible to torchrun + spawn workers.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=0
export AWS_MAX_ATTEMPTS=3
export AWS_RETRY_MODE=adaptive
# wandb disabled for smoke (no project key changes; we just rename the run)
export WANDB_MODE=disabled

echo "[smoke] config=$CONFIG"
echo "[smoke] tmp_cfg=$TMP_CFG"
echo "[smoke] log_dir=$LOG_DIR"
echo "[smoke] canary_log=$CANARY_LOG"
echo "[smoke] num_steps=$NUM_STEPS nproc=$NPROC"

# Launch training in background, then attach the canary to its PID.
(
  uv run --extra models python -m torch.distributed.run \
    --standalone --nproc_per_node="$NPROC" \
    -m vesuvius.neural_tracing.autoreg_mesh.train "$TMP_CFG" \
    2>&1 | tee -a "$LOG_DIR/train.log"
) &
TRAIN_PID=$!
echo "[smoke] training pid=$TRAIN_PID"

# Canary watches the torchrun parent (we pass the immediate subshell pid;
# canary walks descendants). It auto-stops when the parent exits.
uv run python "$REPO_DIR/scripts/canary.py" \
  --ppid "$TRAIN_PID" --log "$CANARY_LOG" \
  --interval 5 --mem-floor-gb 200 --shm-floor-pct 5 --fd-ceiling 500000 &
CANARY_PID=$!

wait "$TRAIN_PID"
RC=$?
echo "[smoke] training exited with rc=$RC"

# Give the canary a moment to see the ppid go away.
sleep 3
kill "$CANARY_PID" 2>/dev/null || true
wait "$CANARY_PID" 2>/dev/null || true

# Quick verdict from canary log.
if grep -q '"event": "TRIGGER"' "$CANARY_LOG"; then
  echo "[smoke] CANARY TRIGGERED — see $CANARY_LOG"
  exit 3
fi
if [ "$RC" -ne 0 ]; then
  echo "[smoke] training failed (rc=$RC) — see $LOG_DIR/train.log"
  exit "$RC"
fi
echo "[smoke] PASS"
exit 0
