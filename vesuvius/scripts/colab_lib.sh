#!/usr/bin/env bash
# Shared helpers for driving a google-colab-cli session, used by both
# colab_bootstrap.sh and colab_train.sh. Source this after setting SESSION;
# EXEC_TIMEOUT is optional (defaults to 1800 if remote_bash is called without
# an explicit timeout).
#
# Extracted rather than duplicated per this repo's AGENTS.md guidance: don't
# copy an implementation across callers, share it.

: "${EXEC_TIMEOUT:=1800}"

log() { printf '\n[bootstrap] %s\n' "$*"; }

# Extracts a single printed value from captured `colab exec` output (stdin).
# google-colab-cli intermittently mixes "[colab] ..." update-check lines into
# exec output, both before AND after the real value (observed: "[colab] To
# silence this check, ..." landing last, so a bare `tail -1` handed that
# banner text to `uv pip install` as the wheel path). Drop [colab]-prefixed,
# blank, and CR characters, then take the last remaining line. Always exits
# 0 (empty output is a legitimate answer, e.g. a wheel-cache miss), since
# grep's exit 1 on no matches would otherwise abort `set -e -o pipefail`.
last_value() { { grep -v '^\[colab\]' || true; } | { grep -v '^[[:space:]]*$' || true; } | tr -d '\r' | tail -1; }

# `colab exec` always exits 0 even when the executed code raises a traceback
# (verified against a live session), so failures are detected by scanning
# the captured output instead of relying on the process exit code.
check_remote_ok() {
    if grep -q "Traceback (most recent call last)\|CalledProcessError" <<<"$1"; then
        echo "[bootstrap] remote step failed:" >&2
        printf '%s\n' "$1" >&2
        exit 1
    fi
}

# `colab exec --timeout` only bounds execution time *inside the remote
# kernel* — it does nothing if the client's connection to the backend itself
# hangs (observed directly: a call sat blocked for 2+ hours against a
# session that `colab status` showed as IDLE the whole time). Wrap every
# call in a hard client-side `timeout` (declared timeout + 60s grace for
# real network latency) so a stuck connection fails loudly instead of
# hanging the whole script indefinitely.
client_timeout() { echo $(( $1 + 60 )); }

# Checks the exit code from a wrapped `colab exec` call: 124 means our
# client-side `timeout` killed a hung connection; any other nonzero means
# `colab exec` itself failed at the CLI level (e.g. "Session lost 404/401")
# rather than the remote code raising (which exits 0 — see check_remote_ok).
# Both cases must be printed and fatal, not silently swallowed.
check_exec_rc() {
    local rc="$1" out="$2" secs="$3"
    if [[ "$rc" -eq 124 ]]; then
        echo "[bootstrap] client-side connection hung (>${secs}s + 60s grace) — session may need a fresh reconnect" >&2
        exit 1
    elif [[ "$rc" -ne 0 ]]; then
        echo "[bootstrap] colab exec itself failed (exit $rc):" >&2
        printf '%s\n' "$out" >&2
        exit 1
    fi
}

# Runs a Python snippet against the session via stdin. Requires $SESSION set.
remote_py() {
    local secs out rc=0
    secs="${2:-30}"
    out=$(timeout "$(client_timeout "$secs")" bash -c "printf '%s' \"\$1\" | colab exec -s \"\$2\" --timeout \"\$3\"" _ "$1" "$SESSION" "$secs" 2>&1) || rc=$?
    printf '%s\n' "$out"
    check_exec_rc "$rc" "$out" "$secs"
    check_remote_ok "$out"
}

# Runs a real shell script against the session. `colab exec` sends a file's
# content straight to the session's Jupyter kernel, so a leading `%%bash`
# line is treated as a cell magic and everything after it runs as bash with
# output streamed back normally — this is how `exec` runs non-Python code.
# Requires $SESSION set.
remote_bash() {
    local tmp out secs rc=0
    secs="${2:-$EXEC_TIMEOUT}"
    tmp="$(mktemp)"
    { printf '%%%%bash\n'; printf '%s\n' "$1"; } > "$tmp"
    out=$(timeout "$(client_timeout "$secs")" colab exec -s "$SESSION" -f "$tmp" --timeout "$secs" 2>&1) || rc=$?
    rm -f "$tmp"
    printf '%s\n' "$out"
    check_exec_rc "$rc" "$out" "$secs"
    check_remote_ok "$out"
}

# Launches a detached background loop on the session that keeps a zarr
# volume_cache_dir (an S3 chunk cache — see docs/ink_detection.md "Volume
# paths and disk cache") under a total size bound for as long as training
# runs, not just at each volume's open time. Added after a long S3-backed
# training run repeatedly died mid-run, correlated with steadily rising disk
# usage (W&B system/disk./.usageGB). A dedicated 15-minute diagnostic run
# later disproved disk exhaustion as the actual cause on this VM tier (only
# 24%->26% of a 236GB disk over the full run, ~174GB free throughout), so the
# default here is a generous last-resort backstop, not an active space-saver
# — the real per-volume bound is volume_cache_max_gb in the training config,
# enforced by zarr's own CacheStore LRU.
#
# Caveat, not fully resolved: the safety check below assumes a file's atime
# reflects recent reads, but this VM's root filesystem mounts with `relatime`
# (confirmed live), which only guarantees an atime update roughly once every
# 24h per file (or on the first read after a write) — not on every read. A
# file read repeatedly within the same day can look "untouched" long before
# it actually is, so -amin here is a soft signal, not a hard guarantee
# against deleting a file an active worker is reading. Keeping the threshold
# high (given the proven disk headroom) is the main mitigation for now.
#   - Only ever operates inside $1 (the cache directory itself), never Drive
#     or S3 — this cache is purely local and re-fetchable, so removing an
#     entry just means the next read is a cache miss instead of a hit.
#   - Deletes oldest-accessed-first, stopping as soon as it's back under
#     budget, so it never over-prunes.
# Requires $SESSION set.
launch_disk_janitor() {
    local cache_dir="$1" max_gb="${2:-100}"
    remote_bash "
nohup bash -c '
CACHE_DIR=\"$cache_dir\"
MAX_BYTES=\$(( $max_gb * 1024 * 1024 * 1024 ))
while true; do
  if [ -d \"\$CACHE_DIR\" ]; then
    total=\$(du -sb \"\$CACHE_DIR\" 2>/dev/null | cut -f1)
    if [ -n \"\$total\" ] && [ \"\$total\" -gt \"\$MAX_BYTES\" ]; then
      find \"\$CACHE_DIR\" -type f -amin +2 -printf \"%A@ %p\n\" 2>/dev/null | sort -n | while read -r _ path; do
        cur=\$(du -sb \"\$CACHE_DIR\" 2>/dev/null | cut -f1)
        [ -n \"\$cur\" ] && [ \"\$cur\" -le \"\$MAX_BYTES\" ] && break
        rm -f \"\$path\"
      done
    fi
  fi
  sleep 120
done
' > /tmp/disk_janitor.log 2>&1 &
disown
echo \"disk janitor launched, pid \$!\"
" 30
}
