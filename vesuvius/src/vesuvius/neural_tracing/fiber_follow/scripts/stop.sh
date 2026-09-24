#!/bin/bash
# Stop only this run and its descendants, including forkserver DataLoader workers.
set -euo pipefail
FF="$(cd "$(dirname "$0")/.." && pwd)"
VES="$(cd "$FF/../../../.." && pwd)"
name=${1:?Usage: stop.sh RUN_NAME}
"$VES/.venv/bin/python" - "$FF/output/logs/$name.pid" "$name" <<'PY'
from pathlib import Path
import sys
import psutil

pid_file, name = sys.argv[1:]
pid = int(Path(pid_file).read_text().strip())
try:
    parent = psutil.Process(pid)
    args = parent.cmdline()
except psutil.NoSuchProcess:
    print(f'{name}: training process already stopped')
    raise SystemExit(0)
if ('vesuvius.neural_tracing.fiber_follow.train' not in args
        or '--name' not in args or args[args.index('--name')+1] != name):
    raise SystemExit(f'Refusing to stop PID {pid}: it does not match run {name}')
# Freeze the parent before enumerating to prevent it starting another collector.
parent.suspend()
try:
    processes = [parent, *parent.children(recursive=True)]
    for process in processes:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
finally:
    try:
        parent.resume()
    except psutil.NoSuchProcess:
        pass
_, alive = psutil.wait_procs(processes, timeout=5)
for process in alive:
    try:
        process.kill()
    except psutil.NoSuchProcess:
        pass
_, alive = psutil.wait_procs(alive, timeout=5)
alive = [p for p in alive if p.is_running() and p.status() != psutil.STATUS_ZOMBIE]
if alive:
    raise SystemExit(f'Processes still running: {[p.pid for p in alive]}')
print(f'Stopped {name} and its {len(processes)-1} child processes')
PY
