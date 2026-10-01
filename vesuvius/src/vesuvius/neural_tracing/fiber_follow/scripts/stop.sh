#!/bin/bash
# Stop this trainer, all descendants, and orphaned collectors scoped to this run.
set -euo pipefail
FF="$(cd "$(dirname "$0")/.." && pwd)"
VES="$(cd "$FF/../../../.." && pwd)"
name=${1:?Usage: stop.sh RUN_NAME}
case "$name" in
    ''|.|..|*/*) echo 'Run name must be a single directory name.' >&2; exit 1 ;;
esac
export PYTHONPATH="$VES/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$VES/.venv/bin/python" -m vesuvius.neural_tracing.fiber_follow.shared.stop_run \
    "$FF/output/logs/$name.pid" "$name"
