#!/usr/bin/env bash
# Runs flat ink-detection inference on a Colab GPU session: runs
# colab_bootstrap.sh for environment setup (session, Drive mount, vesuvius
# install), then launches vesuvius.ink_detection.inference.infer detached on
# the remote session against a training checkpoint on Drive, writes the
# prediction TIFF to local VM disk and copies it onto Drive when done. By
# default it then waits for completion (polling Drive, not the colab CLI's
# own session status — see colab_resume_watchdog.sh for why) and stops the
# session afterwards so it doesn't keep holding a GPU slot.
#
# The model, normalization and input geometry are all rebuilt from the config
# embedded in the checkpoint, so no training config is needed here — only
# the input surface volume, which defaults to the first one listed in
# CONFIG_LOCAL so it matches what the checkpoint was trained on.
#
# Usage:
#   REPO_URL=https://github.com/<you>/villa.git \
#   REPO_BRANCH=<branch with your ink_detection changes> \
#   RCLONE_CONF_LOCAL=~/.config/rclone/rclone_vesuvius.conf \
#   CHECKPOINT_REMOTE=/content/drive/vesuvius/runs/ink_tutorial_s3/ckpt_020000.pth \
#   ./colab_infer.sh
#
# Env vars (all optional except REPO_URL, RCLONE_CONF_LOCAL, CHECKPOINT_REMOTE):
#   CHECKPOINT_REMOTE   Checkpoint path on the mounted Drive (/content/drive/...)
#   CONFIG_LOCAL        Training config to read INPUT_ZARR from (default: ../configs/ink_tutorial_s3.json)
#   INPUT_ZARR          Surface volume zarr, local path or s3:// URL
#                       (default: first surface_volume_paths entry in CONFIG_LOCAL)
#   OUTPUT_NAME         Basename for the prediction            (default: <run-dir>_<checkpoint-name>)
#   OUTPUT_DIR_REMOTE   Drive folder for TIFF, log and status  (default: /content/drive/vesuvius/predictions)
#   INFER_ARGS          Extra args passed to infer, e.g. "--tta-mirror --batch-size 4"
#   WAIT                1 to wait for completion locally       (default: 1)
#   STOP_SESSION        1 to stop the session once finished    (default: 1, only applies with WAIT=1)
#   POLL_SECONDS        How often to check Drive for the status file (default: 60)
#   All colab_bootstrap.sh env vars (REPO_BRANCH, VESUVIUS_COLAB_SESSION, VESUVIUS_COLAB_GPU, etc.) are passed through.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=colab_lib.sh
source "$SCRIPT_DIR/colab_lib.sh"

SESSION="${VESUVIUS_COLAB_SESSION:-vesuvius-infer}"
RCLONE_CONF_LOCAL="${RCLONE_CONF_LOCAL:?Inference reads the checkpoint from Drive — set RCLONE_CONF_LOCAL to your rclone.conf}"
CHECKPOINT_REMOTE="${CHECKPOINT_REMOTE:?Set CHECKPOINT_REMOTE to a checkpoint on the mounted Drive, e.g. /content/drive/vesuvius/runs/<run>/ckpt_020000.pth}"
CONFIG_LOCAL="${CONFIG_LOCAL:-$SCRIPT_DIR/../configs/ink_tutorial_s3.json}"
OUTPUT_DIR_REMOTE="${OUTPUT_DIR_REMOTE:-/content/drive/vesuvius/predictions}"
INFER_ARGS="${INFER_ARGS:-}"
WAIT="${WAIT:-1}"
STOP_SESSION="${STOP_SESSION:-1}"
POLL_SECONDS="${POLL_SECONDS:-60}"

if [[ -z "${INPUT_ZARR:-}" ]]; then
    [[ -f "$CONFIG_LOCAL" ]] || { echo "[infer] INPUT_ZARR unset and config not found: $CONFIG_LOCAL" >&2; exit 1; }
    INPUT_ZARR="$(python3 -c "
import json, sys
paths = json.load(open(sys.argv[1]))['datasets'][0].get('surface_volume_paths') or {}
print(next(iter(paths.values()), ''))
" "$CONFIG_LOCAL")"
    [[ -n "$INPUT_ZARR" ]] || { echo "[infer] no surface_volume_paths in $CONFIG_LOCAL — set INPUT_ZARR explicitly" >&2; exit 1; }
fi

CKPT_NAME="$(basename "$CHECKPOINT_REMOTE" .pth)"
RUN_DIR_NAME="$(basename "$(dirname "$CHECKPOINT_REMOTE")")"
OUTPUT_NAME="${OUTPUT_NAME:-${RUN_DIR_NAME}_${CKPT_NAME}}"
OUTPUT_LOCAL="/content/predictions/$OUTPUT_NAME.tif"
OUTPUT_REMOTE="$OUTPUT_DIR_REMOTE/$OUTPUT_NAME.tif"
LOG_REMOTE="$OUTPUT_DIR_REMOTE/$OUTPUT_NAME.log"
STATUS_REMOTE="$OUTPUT_DIR_REMOTE/$OUTPUT_NAME.status"
PROGRESS_REMOTE="$OUTPUT_DIR_REMOTE/$OUTPUT_NAME.progress"
LOG_LOCAL="/content/predictions/$OUTPUT_NAME.log"
# The same Drive paths, as seen through the local rclone remote ("gdrive:"
# is the remote name colab_bootstrap.sh mounts at /content/drive).
STATUS_RCLONE="gdrive:${STATUS_REMOTE#/content/drive/}"
PROGRESS_RCLONE="gdrive:${PROGRESS_REMOTE#/content/drive/}"

log "input:      $INPUT_ZARR"
log "checkpoint: $CHECKPOINT_REMOTE"
log "output:     $OUTPUT_REMOTE"

log "running colab_bootstrap.sh for environment setup (session: $SESSION)"
VESUVIUS_COLAB_SESSION="$SESSION" RCLONE_CONF_LOCAL="$RCLONE_CONF_LOCAL" "$SCRIPT_DIR/colab_bootstrap.sh"

log "launching inference in the background"
# Written to local VM disk first: the Drive mount's VFS write cache is capped
# at 3G (colab_bootstrap.sh), and a tiled TIFF is written with seeks, so it's
# safer to produce it locally and copy the finished file over in one go. The
# status file is written last, so its presence means the TIFF is complete.
# --no-sync for the same reason as colab_train.sh: a plain `uv run` would try
# to rebuild volume-cartographer instead of using the installed wheel.
#
# The log also goes to local disk: the Drive mount only uploads a file once
# it's closed, so a log written straight to Drive stays invisible (and is
# lost) if the session dies mid-run — observed on the first T4 attempt, which
# died ~5-25 min in with nothing on Drive to show for it. Instead a heartbeat
# loop rewrites a small, closed <name>.progress file on Drive every minute
# (log tail + RAM + GPU), so there's always a recent snapshot to diagnose a
# dead session from; the full log is copied over at the end either way.
remote_bash "
export PATH=\"\$HOME/.local/bin:\$PATH\"
mkdir -p /content/predictions '$OUTPUT_DIR_REMOTE'
rm -f '$STATUS_REMOTE'
cd \$HOME/villa/vesuvius
nohup bash -c '
PYTHONUNBUFFERED=1 uv run --no-sync --extra models python -m vesuvius.ink_detection.inference.infer \
    \"$INPUT_ZARR\" \"$CHECKPOINT_REMOTE\" \"$OUTPUT_LOCAL\" $INFER_ARGS \
  && cp \"$OUTPUT_LOCAL\" \"$OUTPUT_REMOTE\" \
  && status=ok || status=\"failed (exit \$?)\"
cp \"$LOG_LOCAL\" \"$LOG_REMOTE\"
echo \"\$status\" > \"$STATUS_REMOTE\"
' > '$LOG_LOCAL' 2>&1 &
pid=\$!
disown
nohup bash -c \"
while kill -0 \$pid 2>/dev/null; do
  { date -u; free -m; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader;
    echo ---; tail -c 4000 '$LOG_LOCAL' | tr '\\\\r' '\\\\n' | grep -v '^[[:space:]]*\\\$' | tail -20; } > /tmp/infer.progress 2>&1
  cp /tmp/infer.progress '$PROGRESS_REMOTE'
  sleep 60
done
\" > /dev/null 2>&1 &
disown
echo \"launched inference, pid \$pid\"
" 120

log "inference launched. Progress snapshot (every 60s): $PROGRESS_REMOTE"
if [[ "$WAIT" != "1" ]]; then
    log "not waiting (WAIT=$WAIT). Done when $STATUS_REMOTE exists; stop the session afterwards with: colab stop -s $SESSION"
    exit 0
fi

# Completion is a file on Drive, but a dead session never writes one, so also
# watch for the session disappearing. `colab status` is not trusted on a
# single reading (see colab_resume_watchdog.sh), only after several
# consecutive polls in a row without the session listed.
log "waiting for $STATUS_RCLONE (polling every ${POLL_SECONDS}s)"
STATUS=""
MISSING=0
while [[ -z "$STATUS" ]]; do
    sleep "$POLL_SECONDS"
    STATUS="$(rclone cat --config "$RCLONE_CONF_LOCAL" "$STATUS_RCLONE" 2>/dev/null || true)"
    [[ -n "$STATUS" ]] && break
    if colab status 2>/dev/null | grep -q "^\[$SESSION\]"; then
        MISSING=0
    else
        MISSING=$((MISSING + 1))
        log "session $SESSION not listed by colab status ($MISSING/3)"
        if (( MISSING >= 3 )); then
            log "session $SESSION is gone and no status file was written — inference died with it."
            log "last progress snapshot ($PROGRESS_RCLONE):"
            rclone cat --config "$RCLONE_CONF_LOCAL" "$PROGRESS_RCLONE" 2>/dev/null || echo "(none)"
            exit 1
        fi
    fi
done
log "inference finished: $STATUS"

if [[ "$STOP_SESSION" == "1" ]]; then
    log "stopping session $SESSION"
    colab stop -s "$SESSION" || log "WARNING: could not stop $SESSION — stop it manually with: colab stop -s $SESSION"
fi

[[ "$STATUS" == "ok" ]] || { log "see the log for details: $LOG_REMOTE"; exit 1; }
log "prediction: $OUTPUT_REMOTE"
