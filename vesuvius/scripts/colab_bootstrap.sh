#!/usr/bin/env bash
# Bootstraps a vesuvius Colab GPU session end-to-end via google-colab-cli,
# driven entirely from this local machine. See ../docs/colab.md for the
# manual walkthrough this automates and the reasoning behind each step.
#
# One-time manual prerequisites this script does NOT do for you (see
# docs/colab.md for each):
#   - Create your own rclone Google Drive OAuth client and publish its
#     consent screen to "In production" (avoids the 7-day token expiry).
#   - Run `rclone config` once inside any Colab session to produce a working
#     rclone.conf, save it locally, and point RCLONE_CONF_LOCAL at it below.
#     Without it, this script skips Drive mounting and prints how to do it.
#
# Usage:
#   REPO_URL=https://github.com/<you>/villa.git ./colab_bootstrap.sh
#
# Env vars (all optional except REPO_URL):
#   REPO_BRANCH              Branch to clone/checkout      (default: main)
#   VESUVIUS_COLAB_SESSION   Session name                  (default: vesuvius)
#   VESUVIUS_COLAB_GPU       GPU tier: t4|l4|a100           (default: t4)
#   VESUVIUS_COLAB_HIGH_MEM  1 to request --high-mem        (default: 0)
#   RCLONE_CONF_LOCAL        Local rclone.conf to push      (default: ./rclone.conf)
#   EXEC_TIMEOUT             Seconds for slow remote steps  (default: 1800)
#   MEM_PER_JOB_GB           RAM budget per compile job     (default: 2)
#   SKIP_JOB_CAP             1 to disable the RAM-based cap (default: 0)

set -euo pipefail

SESSION="${VESUVIUS_COLAB_SESSION:-vesuvius}"
GPU="${VESUVIUS_COLAB_GPU:-t4}"
HIGH_MEM="${VESUVIUS_COLAB_HIGH_MEM:-0}"
REPO_URL="${REPO_URL:?Set REPO_URL to your villa fork, e.g. https://github.com/<you>/villa.git}"
RCLONE_CONF_LOCAL="${RCLONE_CONF_LOCAL:-./rclone.conf}"
EXEC_TIMEOUT="${EXEC_TIMEOUT:-1800}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=colab_lib.sh
source "$SCRIPT_DIR/colab_lib.sh"

log "checking google-colab-cli"
if ! command -v colab >/dev/null 2>&1; then
    uv tool install google-colab-cli
fi

log "ensuring session '$SESSION' exists"
if ! colab sessions 2>&1 | grep -q "\[$SESSION\]"; then
    gpu_args=(--gpu "$GPU")
    [[ "$HIGH_MEM" == "1" ]] && gpu_args+=(--high-mem)
    colab new -s "$SESSION" "${gpu_args[@]}"
else
    log "session '$SESSION' already running, reusing it"
fi

# Not routed through remote_py/remote_bash: this probe's output needs
# inspecting for the KernelClient bug *before* check_remote_ok would treat
# that traceback as fatal. Still needs the same client-side timeout guard
# though (a raw, unwrapped call here was observed hanging for 90+ minutes).
home_probe() {
    timeout "$(client_timeout 30)" bash -c "printf '%s' \"\$1\" | colab exec -s \"\$2\" --timeout 30" _ \
        "import os; print(os.environ['HOME'])" "$SESSION" 2>&1
}

log "resolving remote \$HOME (do not hardcode /root — see colab.md)"
HP_RC=0
HOME_PROBE=$(home_probe) || HP_RC=$?
# google-colab-cli 0.6.0 depends on jupyter-kernel-client unpinned, which
# crossed into 1.0.x and renamed the class the CLI expects (KernelClient ->
# JupyterKernelClient), breaking exec/repl/drivemount/install. See colab.md.
if grep -q "has no attribute 'KernelClient'" <<<"$HOME_PROBE"; then
    log "applying jupyter-kernel-client<1.0.0 compatibility pin"
    uv tool install google-colab-cli --with "jupyter-kernel-client<1.0.0" --force
    HP_RC=0
    HOME_PROBE=$(home_probe) || HP_RC=$?
fi
check_exec_rc "$HP_RC" "$HOME_PROBE" 30
check_remote_ok "$HOME_PROBE"
REMOTE_HOME="$HOME_PROBE"
REMOTE_HOME="${REMOTE_HOME//$'\r'/}"
log "remote home: $REMOTE_HOME"

if [[ -f "$RCLONE_CONF_LOCAL" ]]; then
    log "pushing rclone.conf and mounting Google Drive"
    remote_bash "mkdir -p '$REMOTE_HOME/.config/rclone' && command -v rclone >/dev/null || (apt-get update -y && apt-get install -y --no-install-recommends rclone)" 300
    colab upload -s "$SESSION" "$RCLONE_CONF_LOCAL" "$REMOTE_HOME/.config/rclone/rclone.conf"
    remote_bash "mkdir -p /content/drive && (mountpoint -q /content/drive || (rclone mount gdrive: /content/drive --vfs-cache-mode writes --daemon && sleep 3))" 60
else
    log "no local rclone.conf at '$RCLONE_CONF_LOCAL' — skipping Drive mount."
    log "See docs/colab.md 'Mounting Google Drive' to create one, then rerun with RCLONE_CONF_LOCAL set."
fi

log "cloning/updating vesuvius (branch: ${REPO_BRANCH:-<default>})"
remote_bash "
if [ -d '$REMOTE_HOME/villa/.git' ]; then
    git -C '$REMOTE_HOME/villa' fetch origin
    git -C '$REMOTE_HOME/villa' checkout '${REPO_BRANCH:-main}'
    git -C '$REMOTE_HOME/villa' pull --ff-only
else
    git clone ${REPO_BRANCH:+-b "$REPO_BRANCH"} '$REPO_URL' '$REMOTE_HOME/villa'
fi
" 300

log "ensuring uv is installed remotely"
remote_bash "export PATH=\"\$HOME/.local/bin:\$PATH\"; command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh" 120

# ---- volume-cartographer wheel caching -------------------------------------
# uv sync's editable path dependency rebuilds volume-cartographer from source
# every session, even though the underlying VM image (Ubuntu noble, same
# Python, portable -march=x86-64-v3 build — see docs/colab.md) is effectively
# static run to run. Cache a real wheel on Drive instead, keyed by the git
# tree hash of volume-cartographer/ (not the whole repo's commit, so unrelated
# monorepo changes don't invalidate it) so a stale wheel from before a source
# change is never silently reused. Best-effort: only active when
# RCLONE_CONF_LOCAL points at a real config (Drive mounted in step 5 above);
# otherwise falls back to a session-local dir with no cross-session reuse.
WHEEL_CACHE_DIR="/root/.cache/vc-wheel"
CACHED_WHEEL=""
if [[ -f "$RCLONE_CONF_LOCAL" ]]; then
    VC_TREE_HASH=$(remote_bash "git -C '$REMOTE_HOME/villa' rev-parse HEAD:volume-cartographer" 30 | tail -1)
    WHEEL_CACHE_DIR="/content/drive/vesuvius/vc-wheel-cache/$VC_TREE_HASH"
    log "checking Drive wheel cache for volume-cartographer @ $VC_TREE_HASH"
    # rclone's Drive VFS mount can be slow to resolve a path, especially one
    # that doesn't exist yet (first-ever cache check) — 30s was observed to
    # be too tight and timed out; give it real headroom.
    CACHED_WHEEL=$(remote_bash "ls '$WHEEL_CACHE_DIR'/*.whl 2>/dev/null | head -1 || true" 90 | tail -1)
fi

log "uv sync --extra models, excluding volume-cartographer (fast — no compile)"
remote_bash "export PATH=\"\$HOME/.local/bin:\$PATH\"; cd '$REMOTE_HOME/villa/vesuvius' && uv sync --extra models --no-install-package volume-cartographer" "$EXEC_TIMEOUT"

if [[ -n "$CACHED_WHEEL" ]]; then
    log "cache hit: installing $CACHED_WHEEL (skipping the volume-cartographer compile)"
    # --no-deps: `uv sync --no-install-package volume-cartographer` above
    # already resolved every other package (numpy, numcodecs, etc.) as one
    # consistent set satisfying the whole project, including numba's
    # numpy<=2.4 constraint. Without --no-deps, `uv pip install` resolves
    # the wheel's own (unpinned) numpy requirement in isolation and happily
    # grabs the latest release instead, silently breaking numba at import
    # time (`ImportError: Numba needs NumPy 2.4 or less`) — confirmed live.
    remote_bash "export PATH=\"\$HOME/.local/bin:\$PATH\"; cd '$REMOTE_HOME/villa/vesuvius' && uv pip install --python .venv/bin/python --force-reinstall --no-deps '$CACHED_WHEEL'" 300
else
    log "cache miss — building volume-cartographer from source"

    # install_build_deps.sh already no-ops the LLVM/CMake/GCC compatibility
    # shims wherever the VM's own archives are new enough, so this is safe to
    # run on any Ubuntu release, not just Jammy. This step is only needed on
    # a cache miss (build toolchain); the compiled wheel's runtime shared
    # library deps (Qt6/OpenCV/CGAL/Ceres/etc.) are also installed by it, so
    # a *future* cache-hit run on a fresh VM still needs it re-run once for
    # those runtime libs even though it won't recompile anything.
    log "installing volume-cartographer build deps on the VM (slow — up to \$EXEC_TIMEOUT=${EXEC_TIMEOUT}s)"
    remote_bash "bash '$REMOTE_HOME/villa/volume-cartographer/scripts/install_build_deps.sh'" "$EXEC_TIMEOUT"

    # Ninja (this project's generator everywhere) already parallelizes by
    # default using all visible CPUs — no explicit -j needed, unlike Make.
    # The real risk on a GPU VM is the opposite of single-threading:
    # CGAL/Ceres/OpenCV translation units are multi-GB-per-job, so an
    # uncapped job count can OOM. Cap it by available RAM instead of
    # trusting Ninja's raw core count. SKIP_JOB_CAP=1 disables this (leaves
    # CMAKE_BUILD_PARALLEL_LEVEL unset) — for A/B timing comparisons only.
    JOB_CAP_EXPORT=""
    if [[ "${SKIP_JOB_CAP:-0}" == "1" ]]; then
        log "SKIP_JOB_CAP=1 — leaving build parallelism at Ninja's own default"
    else
        MEM_PER_JOB_GB="${MEM_PER_JOB_GB:-2}"
        log "computing a memory-safe build job count (~${MEM_PER_JOB_GB}GB/job budget)"
        BUILD_JOBS=$(remote_bash "
cores=\$(nproc)
mem_gb=\$(awk '/MemAvailable/{printf \"%d\", \$2/1024/1024}' /proc/meminfo)
jobs=\$(( mem_gb / $MEM_PER_JOB_GB ))
[ \"\$jobs\" -lt 1 ] && jobs=1
[ \"\$jobs\" -gt \"\$cores\" ] && jobs=\$cores
echo \$jobs
" 30 | tail -1)
        log "using CMAKE_BUILD_PARALLEL_LEVEL=$BUILD_JOBS on the remote VM"
        JOB_CAP_EXPORT="export CMAKE_BUILD_PARALLEL_LEVEL=$BUILD_JOBS;"
    fi

    log "building a volume-cartographer wheel (this is the slow step — up to \$EXEC_TIMEOUT=${EXEC_TIMEOUT}s)"
    remote_bash "export PATH=\"\$HOME/.local/bin:\$PATH\"; $JOB_CAP_EXPORT mkdir -p '$WHEEL_CACHE_DIR' && uv build --wheel -o '$WHEEL_CACHE_DIR' '$REMOTE_HOME/villa/volume-cartographer'" "$EXEC_TIMEOUT"

    BUILT_WHEEL=$(remote_bash "ls '$WHEEL_CACHE_DIR'/*.whl | head -1" 30 | tail -1)
    log "installing freshly built wheel: $BUILT_WHEEL"
    # --no-deps: see the cache-hit branch above for why this matters.
    remote_bash "export PATH=\"\$HOME/.local/bin:\$PATH\"; cd '$REMOTE_HOME/villa/vesuvius' && uv pip install --python .venv/bin/python --force-reinstall --no-deps '$BUILT_WHEEL'" 300
fi

log "verifying GPU visibility"
# --no-sync is required here: volume-cartographer is declared as an editable
# path dependency in pyproject.toml/uv.lock, and a plain `uv run` re-syncs
# the environment against the lockfile before running — which would try to
# rebuild it as editable and silently undo the wheel we just installed
# (failing outright on a cache hit, since install_build_deps.sh was skipped).
remote_bash "export PATH=\"\$HOME/.local/bin:\$PATH\"; cd '$REMOTE_HOME/villa/vesuvius' && uv run --no-sync --extra models python -c \"import torch; print(torch.__version__, '| cuda:', torch.cuda.is_available())\"" 120

log "done. Reconnect any time with: colab console -s $SESSION"
