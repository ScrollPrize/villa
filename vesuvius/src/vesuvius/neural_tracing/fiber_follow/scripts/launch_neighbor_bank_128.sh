#!/usr/bin/env bash
# Long bank for wrong-fiber tails up to 128 trace voxels.
# Pilot: bash scripts/launch_neighbor_bank_128.sh --max-shards 6
# Continue: bash scripts/launch_neighbor_bank_128.sh --resume
set -euo pipefail
FF="$(cd "$(dirname "$0")/.." && pwd)"
VES="$(cd "$FF/../../../.." && pwd)"
PYTHON=${PYTHON:-$VES/.venv/bin/python}
BANK_NAME=${BANK_NAME:-neighbor_negatives_bulk_tail128_v1}
# Existing runs can pin the producer sources whose hashes are in run.json.
source_root="$FF/output/$BANK_NAME/source_snapshot"
if [[ ! -d "$source_root" ]]; then source_root="$VES/src"; fi
export PYTHONPATH="$source_root${PYTHONPATH:+:$PYTHONPATH}"
export AGENTS_AGENT_MODE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/neighbor-bank-128-mpl}
cd "$FF"
priority=(nice -n 19)
if command -v ionice >/dev/null 2>&1; then priority+=(ionice -c 3); fi
command=("${priority[@]}" "$PYTHON" -u -m vesuvius.neural_tracing.fiber_follow.data.neighbor_bulk
    --fibers /mnt/raid_nvme/spiral_dataset_working/fibers
    --fiber-zarrs /mnt/raid_nvme/spiral_dataset_working/fiber_zarrs
    --ct /mnt/raid_nvme/volpkgs/s1_2um_ds2.volpkg/volumes/s1_ds2.zarr/1
    --ct-grid-scale 8
    --native-build-python /home/sean/Documents/villa4/volume-cartographer/build-dev/python
    --output "$FF/output/$BANK_NAME"
    --seed-spacing 20 --extrapolation 70 --block-size 192
    --stride 8 --workers 12 "$@")
"$PYTHON" - "$FF/output/logs" "$BANK_NAME" "${command[@]}" <<'PY'
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

folder = Path(sys.argv[1])
folder.mkdir(parents=True, exist_ok=True)
name = sys.argv[2]
pid_path = folder / f'{name}.pid'
if pid_path.exists():
    try:
        os.kill(int(pid_path.read_text().strip()), 0)
    except ProcessLookupError:
        pass
    else:
        raise SystemExit(f'Previous bank process is still running: {pid_path.read_text().strip()}')
log_path = folder / f'{name}.log'
with log_path.open('ab') as log:
    process = subprocess.Popen(sys.argv[3:], stdin=subprocess.DEVNULL, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
pid_path.write_text(f'{process.pid}\n')
record = dict(started_utc=datetime.now(timezone.utc).isoformat(), pid=process.pid,
              cwd=str(Path.cwd()), command=sys.argv[3:], source_root=os.environ['PYTHONPATH'].split(os.pathsep)[0], log=str(log_path))
with (folder / f'{name}_launches.jsonl').open('a') as stream:
    stream.write(json.dumps(record)+'\n')
print(json.dumps(record, indent=2))
PY
