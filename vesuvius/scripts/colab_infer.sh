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
# Resumable: inference runs with --resume-dir on Drive, snapshotting its
# progress every RESUME_EVERY blocks. Colab sessions were observed to die
# after ~40 min, while a full segment takes hours on a T4 (e.g. 85,750
# patches at ~2.2 patches/s, ~9-10h), so when a session dies (no heartbeat
# for STALE_MINUTES) this script starts a fresh one and reruns the identical
# command, which continues from the last snapshot. The result is
# bit-identical to an uninterrupted run (see inference/resume.py).
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
#   STALE_MINUTES       Declare the session dead after this long without a heartbeat (default: 10)
#   RESUME_EVERY        Blocks between resume snapshots        (default: 500, ~4 min on a T4)
#   KEEPALIVE_SECONDS   Interval of a no-op `colab exec` while waiting (default: 300; 0 disables)
#   RESUME_DIR_REMOTE   Durable resume directory on Drive      (default: <OUTPUT_DIR_REMOTE>/<OUTPUT_NAME>.resume;
#                       deleted after a successful run)
#   MAX_ATTEMPTS        Sessions to try before giving up       (default: 40; quota refusals don't count)
#   QUOTA_BACKOFF_MAX   Longest wait between quota-refused session requests, seconds (default: 1800)
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
STALE_MINUTES="${STALE_MINUTES:-10}"
RESUME_EVERY="${RESUME_EVERY:-500}"
KEEPALIVE_SECONDS="${KEEPALIVE_SECONDS:-300}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-40}"
QUOTA_BACKOFF_MAX="${QUOTA_BACKOFF_MAX:-1800}"

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
RESUME_DIR_REMOTE="${RESUME_DIR_REMOTE:-$OUTPUT_DIR_REMOTE/$OUTPUT_NAME.resume}"
LOG_LOCAL="/content/predictions/$OUTPUT_NAME.log"
# The same Drive paths, as seen through the local rclone remote ("gdrive:"
# is the remote name colab_bootstrap.sh mounts at /content/drive).
STATUS_RCLONE="gdrive:${STATUS_REMOTE#/content/drive/}"
PROGRESS_RCLONE="gdrive:${PROGRESS_REMOTE#/content/drive/}"

log "input:      $INPUT_ZARR"
log "checkpoint: $CHECKPOINT_REMOTE"
log "output:     $OUTPUT_REMOTE"
log "resume dir: $RESUME_DIR_REMOTE (snapshot every $RESUME_EVERY blocks)"

# One attempt = bootstrap a fresh session, launch (or resume) inference,
# and wait. Returns 0 when finished ok, 1 on a real inference failure, 2
# when the session died or could not be set up (worth another attempt), and
# 3 when Colab refused to assign a runtime at all (quota; wait, then retry).
# remote_bash/colab_bootstrap.sh call `exit 1` on exec-level failures, so
# every remote step runs in a subshell or child process to keep that from
# killing the retry loop.
run_attempt() {
    log "running colab_bootstrap.sh for environment setup (session: $SESSION)"
    local rc=0
    RCLONE_CONF_LOCAL="$RCLONE_CONF_LOCAL" run_bootstrap || rc=$?
    if (( rc == 3 )); then
        log "Colab refused to assign a runtime for $SESSION (TooManyAssignmentsError: usage/session quota)"
        return 3
    elif (( rc != 0 )); then
        log "bootstrap failed for $SESSION"
        return 2
    fi

    if ! (
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
            --resume-dir \"$RESUME_DIR_REMOTE\" --resume-every $RESUME_EVERY \
          && cp \"$OUTPUT_LOCAL\" \"$OUTPUT_REMOTE\" \
          && rm -rf \"$RESUME_DIR_REMOTE\" \
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
    ); then
        log "launch failed for $SESSION (connection issue)"
        return 2
    fi

    log "inference launched. Progress snapshot (every 60s): $PROGRESS_REMOTE"
    if [[ "$WAIT" != "1" ]]; then
        log "not waiting (WAIT=$WAIT). Done when $STATUS_REMOTE exists; rerun this script to resume if the session dies."
        exit 0
    fi

    # Completion is a file on Drive, but a dead session never writes one, so
    # also watch the heartbeat: <name>.progress is rewritten every minute
    # while inference runs, and a session counts as dead once it stops
    # changing for STALE_MINUTES. `colab status` is NOT used for this —
    # observed live, it listed no sessions at all while this one was healthy
    # and mid-inference (see also colab_resume_watchdog.sh).
    log "waiting for $STATUS_RCLONE (polling every ${POLL_SECONDS}s, dead after ${STALE_MINUTES} min without a heartbeat)"
    # Keep-alive: sessions were reclaimed ~20 min after the last `colab exec`
    # reached them, regardless of the detached job still running on the GPU
    # (seen on training and inference sessions alike — every death came
    # ~20 min after the launch exec or the last manual probe). A no-op exec
    # every KEEPALIVE_SECONDS shows the session as in use. Failures are
    # ignored (subshell: remote_py exits on error); the heartbeat decides.
    STATUS=""
    local last_beat="" beat last_change=$SECONDS last_keepalive=$SECONDS
    while [[ -z "$STATUS" ]]; do
        sleep "$POLL_SECONDS"
        if (( KEEPALIVE_SECONDS > 0 && SECONDS - last_keepalive >= KEEPALIVE_SECONDS )); then
            ( remote_py "print('keepalive')" 60 ) >/dev/null 2>&1 || log "keep-alive exec to $SESSION failed (ignored)"
            last_keepalive=$SECONDS
        fi
        STATUS="$(rclone cat --config "$RCLONE_CONF_LOCAL" "$STATUS_RCLONE" 2>/dev/null || true)"
        [[ -n "$STATUS" ]] && break
        beat="$(rclone lsl --config "$RCLONE_CONF_LOCAL" "$PROGRESS_RCLONE" 2>/dev/null || true)"
        if [[ -n "$beat" && "$beat" != "$last_beat" ]]; then
            last_beat="$beat"
            last_change=$SECONDS
        elif (( SECONDS - last_change > STALE_MINUTES * 60 )); then
            log "no heartbeat for ${STALE_MINUTES} min — session $SESSION died with inference unfinished."
            log "last progress snapshot ($PROGRESS_RCLONE):"
            rclone cat --config "$RCLONE_CONF_LOCAL" "$PROGRESS_RCLONE" 2>/dev/null | tail -5 || echo "(none)"
            return 2
        fi
    done
    log "inference finished: $STATUS"
    if [[ "$STOP_SESSION" == "1" ]]; then
        log "stopping session $SESSION"
        colab stop -s "$SESSION" || log "WARNING: could not stop $SESSION — stop it manually with: colab stop -s $SESSION"
    fi
    [[ "$STATUS" == "ok" ]] || return 1
    return 0
}

BASE_SESSION="$SESSION"
# Quota refusals (rc 3) don't use up an attempt: observed live, after a ~75
# min session Colab refused every new runtime for 30+ min, and a fixed 30s
# retry burned all 40 attempts on refusals alone. Back off exponentially
# instead (2 min doubling up to QUOTA_BACKOFF_MAX) and retry indefinitely —
# the resume state on Drive makes waiting free.
quota_wait=120
requests=0
for (( attempt = 1; attempt <= MAX_ATTEMPTS; )); do
    requests=$((requests + 1))
    SESSION="${BASE_SESSION}-r${requests}"
    log "=== attempt $attempt/$MAX_ATTEMPTS: session '$SESSION' ==="
    rc=0
    run_attempt || rc=$?
    if (( rc == 3 )); then
        log "waiting ${quota_wait}s before requesting a runtime again"
        sleep "$quota_wait"
        quota_wait=$(( quota_wait * 2 > QUOTA_BACKOFF_MAX ? QUOTA_BACKOFF_MAX : quota_wait * 2 ))
        continue
    fi
    quota_wait=120
    attempt=$((attempt + 1))
    if (( rc == 0 )); then
        log "prediction: $OUTPUT_REMOTE"
        exit 0
    elif (( rc == 1 )); then
        log "inference failed (not a session death) — see the log: $LOG_REMOTE"
        exit 1
    fi
    # A dead or half-set-up session can linger and hold the GPU quota
    # (TooManyAssignmentsError); it is useless either way, so stop it.
    colab stop -s "$SESSION" >/dev/null 2>&1 || true
    sleep 30
done
log "giving up after $MAX_ATTEMPTS attempts; rerun this script to continue from the last snapshot in $RESUME_DIR_REMOTE"
exit 1
