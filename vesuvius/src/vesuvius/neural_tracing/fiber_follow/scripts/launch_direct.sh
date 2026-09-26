#!/usr/bin/env bash
# Background training with a persistent log. Additional CLI options override defaults.
set -euo pipefail
FF="$(cd "$(dirname "$0")/.." && pwd)"
VES="$(cd "$FF/../../../.." && pwd)"
PYTHON=${PYTHON:-$VES/.venv/bin/python}
export PYTHONPATH="$VES/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    exec "$PYTHON" -m vesuvius.neural_tracing.fiber_follow.direct.train --help
fi
name=${1:?Usage: launch_direct.sh NAME [train options...]}
shift
case "$name" in
    ''|.|..|*/*) echo 'Run name must be a single directory name.' >&2; exit 1 ;;
esac
command=("$PYTHON" -u -m vesuvius.neural_tracing.fiber_follow.direct.train \
    --name "$name" \
    --fiber-zarrs /mnt/raid_nvme/spiral_dataset_working/fiber_zarrs \
    --fibers /mnt/raid_nvme/spiral_dataset_working/fibers \
    --ct /mnt/raid_nvme/volpkgs/s1_2um_ds2.volpkg/volumes/s1_ds2.zarr \
    --manifest "$FF/output/single_path_v11_preparation/seeds.json" \
    --fixed-bank "$FF/output/single_path_v11_preparation/fixed_recovery.npz" "$@")
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
