#!/usr/bin/env bash
# Background training from a run configuration, with a persistent log:
#   launch_train.sh RUN.json [--resume CHECKPOINT]
# The log and PID file are named after the configuration's "name".
set -euo pipefail
FF="$(cd "$(dirname "$0")/.." && pwd)"
VES="$(cd "$FF/../../../.." && pwd)"
PYTHON=${PYTHON:-$VES/.venv/bin/python}
export PYTHONPATH="$VES/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    exec "$PYTHON" "$FF/train/train.py" --help
fi
config=${1:?Usage: launch_train.sh RUN.json [--resume CHECKPOINT]}
shift
config="$(cd "$(dirname "$config")" && pwd)/$(basename "$config")"
name="$("$PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))["name"])' "$config")"
case "$name" in
    ''|.|..|*/*) echo 'Run name must be a single directory name.' >&2; exit 1 ;;
esac
command=("$PYTHON" -u "$FF/train/train.py" --config "$config" "$@")
"$PYTHON" - "$FF/output/logs" "$name" "${command[@]}" <<'PY'
from pathlib import Path
import shlex
import subprocess
import sys

log_dir = Path(sys.argv[1])
name = sys.argv[2]
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / f'{name}.log'
pid_path = log_dir / f'{name}.pid'
# Append so resuming (or a rejected duplicate launch) preserves earlier output.
with log_path.open('ab') as log:
    process = subprocess.Popen(sys.argv[3:], stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
pid_path.write_text(f'{process.pid}\n')
print(f'Launched training process {process.pid}; startup errors will appear in the log.')
print(f'Log: {log_path}')
print(f'PID file: {pid_path}')
print(f'Watch: tail -f {shlex.quote(str(log_path))}')
PY
