#!/bin/bash
# Launch a fiber_follow training run in the background (own process group).
#   scripts/launch.sh NAME [train.py args...]
# Outputs: output/NAME/ (ckpts, images, log.jsonl), stdout in output/logs/NAME.log
FF="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
VES="$(cd "$FF/../../../.." && pwd)"           # vesuvius/ project root
FIBER_ZARRS=${FIBER_ZARRS:-/mnt/raid_nvme/spiral_dataset_working/fiber_zarrs}
FIBERS=${FIBERS:-/mnt/raid_nvme/spiral_dataset_working/fibers}
PYTHON=${PYTHON:-$VES/.venv/bin/python}
cd "$VES"
export PYTHONPATH="$VES/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MALLOC_MMAP_THRESHOLD_=268435456 MALLOC_TRIM_THRESHOLD_=1073741824 MALLOC_ARENA_MAX=2
name=$1; shift
mkdir -p "$FF/output/logs"
setsid nohup "$PYTHON" -u -m vesuvius.neural_tracing.fiber_follow.train \
    --fiber-zarrs "$FIBER_ZARRS" --fibers "$FIBERS" --name "$name" "$@" \
    > "$FF/output/logs/$name.log" 2>&1 &
echo $! > "$FF/output/logs/$name.pid"
