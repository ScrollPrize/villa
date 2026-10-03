#!/usr/bin/env bash
# Watches a detached colab_train.sh run for stalls (no new checkpoint within
# STALL_TIMEOUT) and automatically resumes it on a fresh session, patching in
# the local uncommitted/unpushed source files each time. Built live after a
# single ink_tutorial_s3.json full-training run needed 4 manual resumes due to
# recurring, undiagnosed `colab exec`/session-death reliability issues (see
# docs/colab.md "colab exec reliability"). Ground truth for "is it alive" is
# always a new checkpoint file landing on Drive, never the colab CLI's own
# session-status reporting, which was repeatedly observed to be wrong in both
# directions (false "session lost" while still training; true death not
# always reflected promptly).
#
# Usage:
#   REPO_URL=https://github.com/<you>/villa.git \
#   RCLONE_CONF_LOCAL=~/.config/rclone/rclone_vesuvius.conf \
#   CONFIG_LOCAL=../configs/ink_tutorial_s3.json \
#   RUN_NAME=ink_tutorial_s3 \
#   TARGET_ITERATIONS=20000 \
#   ./colab_resume_watchdog.sh
#
# Env vars (all optional except REPO_URL, RCLONE_CONF_LOCAL, CONFIG_LOCAL):
#   RUN_NAME            Basename used for out_dir/log naming   (default: derived from CONFIG_LOCAL)
#   TARGET_ITERATIONS   Stop once a checkpoint >= this exists  (default: read from config's num_iterations)
#   SAVE_EVERY          Expected checkpoint cadence, iters     (default: read from config's save_every)
#   STEP_SECONDS        Expected wall-seconds per iteration     (default: 1.3, used only to size the stall timeout)
#   STALL_GRACE         Extra seconds added on top of the expected per-checkpoint time (default: 900)
#   SESSION_PREFIX      Colab session name prefix               (default: RUN_NAME)
#   VESUVIUS_COLAB_GPU  GPU tier                                (default: l4)
#   POLL_SECONDS        How often to check Drive for a new checkpoint (default: 60)
#
# All colab_bootstrap.sh env vars are passed through as-is.
#
# Modified source files under src/vesuvius/ink_detection/ are pushed to
# Drive (gdrive:vesuvius/configs/<basename>.py) and pulled onto each
# fresh session — the same manual pattern used throughout this session,
# needed because these commits are not yet pushed to the remote the session
# clones from. Remove PATCH_FILES (or this whole block) once they are.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=colab_lib.sh
source "$SCRIPT_DIR/colab_lib.sh"

REPO_URL="${REPO_URL:?Set REPO_URL}"
RCLONE_CONF_LOCAL="${RCLONE_CONF_LOCAL:?Set RCLONE_CONF_LOCAL}"
CONFIG_LOCAL="${CONFIG_LOCAL:?Set CONFIG_LOCAL}"
[[ -f "$CONFIG_LOCAL" ]] || { echo "[watchdog] config not found: $CONFIG_LOCAL" >&2; exit 1; }

RUN_NAME="${RUN_NAME:-$(basename "$CONFIG_LOCAL" .json)}"
CONFIG_REMOTE="/content/drive/vesuvius/configs/$(basename "$CONFIG_LOCAL")"
OUT_DIR_REMOTE="/content/drive/vesuvius/runs/$RUN_NAME"
SESSION_PREFIX="${SESSION_PREFIX:-$RUN_NAME}"
GPU="${VESUVIUS_COLAB_GPU:-l4}"
POLL_SECONDS="${POLL_SECONDS:-60}"
STEP_SECONDS="${STEP_SECONDS:-1.3}"
STALL_GRACE="${STALL_GRACE:-900}"

TARGET_ITERATIONS="${TARGET_ITERATIONS:-$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('num_iterations',0))" "$CONFIG_LOCAL")}"
SAVE_EVERY="${SAVE_EVERY:-$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('save_every',1000))" "$CONFIG_LOCAL")}"
STALL_TIMEOUT=$(python3 -c "print(int(float('$SAVE_EVERY') * float('$STEP_SECONDS') + float('$STALL_GRACE')))")

log "target: $TARGET_ITERATIONS iterations, checkpoint every $SAVE_EVERY, stall timeout ${STALL_TIMEOUT}s"

PATCH_FILES=(
    "src/vesuvius/ink_detection/config.py"
    "src/vesuvius/ink_detection/types.py"
    "src/vesuvius/ink_detection/volume_io.py"
    "src/vesuvius/ink_detection/data/segment.py"
    "src/vesuvius/ink_detection/data/patch_finding_default.py"
)
VESUVIUS_LOCAL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

latest_checkpoint() {
    # Sort by filename, not by mtime: checkpoint numbers are zero-padded, so
    # plain lexicographic sort on "ckpt_NNNNNN.pth" is equivalent to numeric
    # sort by iteration and is the only correct definition of "latest" here.
    # Sorting by mtime (the previous implementation) breaks the moment two
    # training processes are ever writing to the same out_dir concurrently —
    # observed live: a presumed-dead session turned out to still be alive,
    # training from its own earlier resume point, and rewrote a lower-
    # numbered checkpoint (ckpt_009000.pth) with a newer timestamp than the
    # real latest one (ckpt_010000.pth). An mtime sort would have handed that
    # stale, lower checkpoint back as "latest" on the next resume, silently
    # discarding 1000 real iterations of progress.
    rclone lsf --config "$RCLONE_CONF_LOCAL" "gdrive:vesuvius/runs/$RUN_NAME/" 2>/dev/null \
        | grep '^ckpt_' | sort | tail -1
}

checkpoint_iter() {
    # ckpt_004000.pth -> 4000
    sed -E 's/^ckpt_0*([0-9]+)\.pth$/\1/' <<<"$1"
}

n=0
CURRENT_CKPT="$(latest_checkpoint)"
if [[ -n "$CURRENT_CKPT" ]]; then
    log "found existing checkpoint on Drive: $CURRENT_CKPT (iter $(checkpoint_iter "$CURRENT_CKPT"))"
fi

while true; do
    n=$((n + 1))
    SESSION="${SESSION_PREFIX}-w${n}"
    export VESUVIUS_COLAB_SESSION="$SESSION"

    log "=== attempt $n: session '$SESSION' ==="
    rc=0
    REPO_URL="$REPO_URL" RCLONE_CONF_LOCAL="$RCLONE_CONF_LOCAL" VESUVIUS_COLAB_GPU="$GPU" \
        run_bootstrap || rc=$?
    if (( rc != 0 )); then
        # A 10s retry is fine for a one-off transient failure, but useless
        # against TooManyAssignmentsError (the concurrent-session quota) —
        # that won't clear in 10s, and retrying that fast just hammers the
        # API with more session-creation requests while already over quota.
        # Observed live: 20+ back-to-back failures in under a minute.
        if (( rc == 3 )); then
            log "bootstrap failed for $SESSION (concurrent-session quota) — backing off 120s"
            sleep 120
        else
            log "bootstrap failed for $SESSION, retrying with a new session"
            sleep 10
        fi
        continue
    fi

    CFG_TMP="$(mktemp /tmp/watchdog_config_XXXX.json)"
    if [[ -n "$CURRENT_CKPT" ]]; then
        python3 -c "
import json, sys
p = sys.argv[1]
cfg = json.load(open(p))
cfg['checkpoint'] = '$OUT_DIR_REMOTE/$CURRENT_CKPT'
json.dump(cfg, open(sys.argv[2], 'w'), indent=2)
" "$CONFIG_LOCAL" "$CFG_TMP"
        log "resuming from $OUT_DIR_REMOTE/$CURRENT_CKPT"
    else
        cp "$CONFIG_LOCAL" "$CFG_TMP"
        log "no checkpoint yet — starting fresh"
    fi
    rclone copyto --config "$RCLONE_CONF_LOCAL" "$CFG_TMP" "gdrive:vesuvius/configs/$(basename "$CONFIG_LOCAL")"
    rm -f "$CFG_TMP"

    PATCH_CMDS=""
    for f in "${PATCH_FILES[@]}"; do
        name="$(basename "$f")"
        rclone copyto --config "$RCLONE_CONF_LOCAL" "$VESUVIUS_LOCAL_ROOT/$f" "gdrive:vesuvius/configs/$name"
        PATCH_CMDS+="rclone copyto --config /root/.config/rclone/rclone.conf 'gdrive:vesuvius/configs/$name' /root/villa/vesuvius/$f
"
    done
    # $SESSION (not just $n) must be in this path: $n resets to 1 for every
    # fresh watchdog *instance*, so two concurrent instances' attempts can
    # collide on the same filename. Observed live: a presumed-dead session
    # that was actually still alive kept writing to the same
    # ..._attempt9.log path a much later instance's attempt 9 also used,
    # silently interleaving two unrelated training runs' log output in one
    # file. $SESSION already embeds the instance prefix, so this is unique
    # per attempt regardless of how many instances are running.
    TRAIN_LOG_REMOTE="/content/drive/vesuvius/runs/${RUN_NAME}_${SESSION}.log"

    # remote_bash calls `exit 1` internally (colab_lib.sh's check_exec_rc /
    # check_remote_ok) on any exec-level failure, e.g. a client-side
    # connection hang. Called as a bare statement, that `exit 1` would kill
    # this entire watchdog script, not just the current attempt — observed
    # live: patching/launch hit "Connection was lost" after a long, healthy
    # stretch of bootstrap-quota retries, and the whole process was simply
    # gone with no further attempts, silently ending unattended training.
    # Running both calls in a subshell confines that `exit 1` to the
    # subshell, so a failure here is just another reason to retry with a
    # fresh session, exactly like a bootstrap failure already is.
    if ! (
        remote_bash "
mkdir -p /content/vesuvius_volume_cache '$(dirname "$CONFIG_REMOTE")'
$PATCH_CMDS
rclone copyto --config /root/.config/rclone/rclone.conf 'gdrive:vesuvius/configs/$(basename "$CONFIG_LOCAL")' '$CONFIG_REMOTE'
echo patched
" 120
        remote_bash "
export PATH=\"\$HOME/.local/bin:\$PATH\"
export WANDB_API_KEY='${WANDB_API_KEY:-}'
cd \$HOME/villa/vesuvius
nohup uv run --no-sync --extra models python -m vesuvius.ink_detection.training.train '$CONFIG_REMOTE' > '$TRAIN_LOG_REMOTE' 2>&1 &
echo launched \$!
disown
" 60
    ); then
        log "patch/launch failed for $SESSION (connection issue) — retrying with a new session"
        sleep 10
        continue
    fi
    log "launched attempt $n, log at $TRAIN_LOG_REMOTE"

    VOLUME_CACHE_DIR="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('volume_cache_dir',''))" "$CONFIG_LOCAL")"
    if [[ -n "$VOLUME_CACHE_DIR" ]]; then
        launch_disk_janitor "$VOLUME_CACHE_DIR" 100
        log "launched disk janitor for $VOLUME_CACHE_DIR (100GB safety-net cap)"
    fi

    # Poll Drive for progress; declare a stall if STALL_TIMEOUT passes with
    # no new checkpoint (generous enough to cover normal patch-discovery
    # startup time on a fresh session, not just steady-state step rate).
    last_progress=$(date +%s)
    while true; do
        sleep "$POLL_SECONDS"
        latest="$(latest_checkpoint)"
        if [[ -n "$latest" && "$latest" != "$CURRENT_CKPT" ]]; then
            CURRENT_CKPT="$latest"
            last_progress=$(date +%s)
            iter="$(checkpoint_iter "$CURRENT_CKPT")"
            log "progress: $CURRENT_CKPT (iter $iter)"
            if [[ "$iter" -ge "$TARGET_ITERATIONS" ]]; then
                log "reached target ($iter >= $TARGET_ITERATIONS) — done."
                exit 0
            fi
            continue
        fi
        now=$(date +%s)
        if (( now - last_progress >= STALL_TIMEOUT )); then
            # Deliberately does NOT auto-stop "$SESSION" here: the stall
            # heuristic (no new checkpoint within STALL_TIMEOUT) has been
            # observed to false-positive on a session that was merely slow or
            # unresponsive to colab exec while still training correctly
            # underneath — auto-stopping on that verdict would actively kill
            # real, confirmed progress, which is worse than the cost of
            # leaving a possibly-still-alive session running. The tradeoff is
            # accepted: a truly-dead session's slot may sit unstopped for a
            # while, but a merely-slow one is never killed by this script.
            log "no new checkpoint in ${STALL_TIMEOUT}s — resuming on a new session. NOTE: '$SESSION' is presumed dead but was not stopped, since that verdict has been wrong before (a session can look stalled for a long time yet still be training). Check 'colab sessions' / the Colab billing dashboard and stop it manually once you've confirmed it's not still making progress."
            break
        fi
    done
done
