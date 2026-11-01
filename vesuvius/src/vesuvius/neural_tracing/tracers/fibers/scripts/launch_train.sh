#!/usr/bin/env bash
# Background training from a run configuration, with a persistent log:
#   launch_train.sh RUN.json [--resume CHECKPOINT]
# TRAINER=PATH selects another trainer with the same arguments, e.g. TRAINER=sheet_strip/train.py launch_train.sh
# configs/sheet_strip/paris4_slim50k.json (single GPU).
# With runtime.gpus > 1, e.g. CUDA_VISIBLE_DEVICES=4,7 launch_train.sh RUN.json (2 processes, one per visible GPU).
# The log and PID file are named after the configuration's "name".
set -euo pipefail
FF="$(cd "$(dirname "$0")/.." && pwd)"
VES="$(cd "$FF/../../../../.." && pwd)"
PYTHON=${PYTHON:-$VES/.venv/bin/python}
TRAINER="$(cd "$FF" && realpath "${TRAINER:-train/train.py}")"
export PYTHONPATH="$VES/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    exec "$PYTHON" "$TRAINER" --help
fi
config=${1:?Usage: launch_train.sh RUN.json [--resume CHECKPOINT]}
shift
config="$(cd "$(dirname "$config")" && pwd)/$(basename "$config")"
name="$("$PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))["name"])' "$config")"
case "$name" in
    ''|.|..|*/*) echo 'Run name must be a single directory name.' >&2; exit 1 ;;
esac
# runtime.gpus > 1: one data-parallel process per GPU (torchrun; shared/distributed.py). The visible GPUs come from
# CUDA_VISIBLE_DEVICES; the PID file then names the torchrun agent, which stops every process with it.
gpus="$("$PYTHON" -c 'import json, sys; print(int((json.load(open(sys.argv[1])).get("runtime") or {}).get("gpus", 1)))' "$config")"
if (( gpus > 1 )); then
    command=("$PYTHON" -u -m torch.distributed.run --standalone --nproc_per_node="$gpus" "$TRAINER" --config "$config" "$@")
else
    command=("$PYTHON" -u "$TRAINER" --config "$config" "$@")
fi
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
