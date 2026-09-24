#!/usr/bin/env bash
# Trains a vesuvius model on a Colab GPU session: runs colab_bootstrap.sh for
# environment setup (session, Drive mount, vesuvius install), uploads a local
# training config onto the mounted Drive, then launches training detached on
# the remote session so a multi-hour run survives client disconnects and the
# colab exec reliability issues documented in docs/colab.md. Credentials are
# never embedded here — RCLONE_CONF_LOCAL is a path, passed through to
# colab_bootstrap.sh exactly as it already handles it.
#
# Usage:
#   REPO_URL=https://github.com/<you>/villa.git \
#   RCLONE_CONF_LOCAL=~/.config/rclone/rclone_vesuvius.conf \
#   ./colab_train.sh
#
# Env vars (all optional except REPO_URL and RCLONE_CONF_LOCAL — training
# needs Drive mounted, so unlike colab_bootstrap.sh this one requires it):
#   CONFIG_LOCAL             Local training config           (default: ../configs/ink_tutorial.json)
#   CONFIG_REMOTE            Where it lands on Drive          (default: /content/drive/vesuvius/configs/ink_tutorial.json)
#   TRAIN_MODULE             Python module to run             (default: vesuvius.ink_detection.training.train)
#   TRAIN_LOG_REMOTE         Training log path on Drive        (default: /content/drive/vesuvius/runs/<config-name>.log)
#   WANDB_API_KEY            W&B API key (wandb.ai/authorize) — only needed if the config sets wandb_project;
#                            passed by reference at invocation time, never embedded in this script.
#   All colab_bootstrap.sh env vars (REPO_BRANCH, VESUVIUS_COLAB_SESSION, VESUVIUS_COLAB_GPU, etc.) are passed through.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=colab_lib.sh
source "$SCRIPT_DIR/colab_lib.sh"

SESSION="${VESUVIUS_COLAB_SESSION:-vesuvius}"
RCLONE_CONF_LOCAL="${RCLONE_CONF_LOCAL:?Training needs Drive mounted — set RCLONE_CONF_LOCAL to your rclone.conf}"
CONFIG_LOCAL="${CONFIG_LOCAL:-$SCRIPT_DIR/../configs/ink_tutorial.json}"
CONFIG_NAME="$(basename "$CONFIG_LOCAL")"
CONFIG_REMOTE="${CONFIG_REMOTE:-/content/drive/vesuvius/configs/$CONFIG_NAME}"
TRAIN_MODULE="${TRAIN_MODULE:-vesuvius.ink_detection.training.train}"
TRAIN_LOG_REMOTE="${TRAIN_LOG_REMOTE:-/content/drive/vesuvius/runs/${CONFIG_NAME%.json}.log}"

[[ -f "$CONFIG_LOCAL" ]] || { echo "[train] config not found: $CONFIG_LOCAL" >&2; exit 1; }

WANDB_API_KEY="${WANDB_API_KEY:-}"
WANDB_EXPORT=""
if [[ -n "$WANDB_API_KEY" ]]; then
    WANDB_EXPORT="export WANDB_API_KEY='$WANDB_API_KEY';"
elif grep -q '"wandb_project"' "$CONFIG_LOCAL"; then
    log "WARNING: config sets wandb_project but no WANDB_API_KEY was provided — training will fail at wandb.init() unless you've already run 'uv run wandb login' on this session yourself."
fi

log "running colab_bootstrap.sh for environment setup (session: $SESSION)"
VESUVIUS_COLAB_SESSION="$SESSION" RCLONE_CONF_LOCAL="$RCLONE_CONF_LOCAL" "$SCRIPT_DIR/colab_bootstrap.sh"

log "uploading config: $CONFIG_LOCAL -> $CONFIG_REMOTE"
remote_bash "mkdir -p '$(dirname "$CONFIG_REMOTE")' '$(dirname "$TRAIN_LOG_REMOTE")'" 60
colab upload -s "$SESSION" "$CONFIG_LOCAL" "$CONFIG_REMOTE"

log "launching training in the background (module: $TRAIN_MODULE)"
# --no-sync matters here for the same reason as in colab_bootstrap.sh: a
# plain `uv run` re-syncs against uv.lock's editable volume-cartographer
# source and would try to rebuild it from scratch instead of using whatever
# colab_bootstrap.sh already installed (wheel-cache or from-source build).
remote_bash "
export PATH=\"\$HOME/.local/bin:\$PATH\"
$WANDB_EXPORT
cd \$HOME/villa/vesuvius
nohup uv run --no-sync --extra models python -m $TRAIN_MODULE '$CONFIG_REMOTE' > '$TRAIN_LOG_REMOTE' 2>&1 &
echo \"launched training, pid \$!\"
disown
sleep 5
echo '--- first log output ---'
tail -n 30 '$TRAIN_LOG_REMOTE' || true
" 120

log "training launched. It runs detached, so it survives this script exiting and any client-side disconnects."
log "Tail the log:    colab exec -s $SESSION --timeout 30 <<< \"import subprocess; print(subprocess.run(['tail','-n','50','$TRAIN_LOG_REMOTE'],capture_output=True,text=True).stdout)\""
log "Or reconnect:    colab console -s $SESSION"
log "Checkpoints/previews land under the config's out_dir on Drive, so they persist even if the session is later reclaimed."
