#!/bin/bash
# Stop this trainer, all descendants, and orphaned collectors scoped to this run.
set -euo pipefail
FF="$(cd "$(dirname "$0")/.." && pwd)"
VES="$(cd "$FF/../../../.." && pwd)"
name=${1:?Usage: stop.sh RUN_NAME}
case "$name" in
    ''|.|..|*/*) echo 'Run name must be a single directory name.' >&2; exit 1 ;;
esac
pid_file="$FF/output/logs/$name.pid"
[[ -f "$pid_file" ]] || pid_file="$FF/output/$name/trainer.pid"
export PYTHONPATH="$VES/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$VES/.venv/bin/python" -m vesuvius.neural_tracing.fiber_follow.train.stop_run "$pid_file" "$name"
