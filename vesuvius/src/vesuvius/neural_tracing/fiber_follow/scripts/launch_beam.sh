#!/bin/bash
# Launch a learned beam step-scorer training run in the background (own process group).
#   scripts/launch_beam.sh NAME [beam/train.py args...]
# Outputs: output/NAME/ (ckpts, images, log.jsonl), stdout in output/logs/NAME.log
# Stop with scripts/stop.sh NAME.
set -euo pipefail
FF="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
VES="$(cd "$FF/../../../.." && pwd)"           # vesuvius/ project root
FIBER_ZARRS=${FIBER_ZARRS:-/mnt/raid_nvme/spiral_dataset_working/fiber_zarrs}
FIBERS=${FIBERS:-/mnt/raid_nvme/spiral_dataset_working/fibers}
CT_ZARR=${CT_ZARR:-/mnt/raid_nvme/volpkgs/s1_2um_ds2.volpkg/volumes/s1_ds2.zarr}
PREDICTION_MANIFEST=${PREDICTION_MANIFEST:-$FIBER_ZARRS/PHercParis4-20260411134726-las-sd1-7ff0ce6c.lasagna.json}
NORMAL_MANIFEST=${NORMAL_MANIFEST:-/mnt/raid_nvme/volpkgs/s1_2um.volpkg/las_008_s1_full/las_008.lasagna.json}
name=${1:?Usage: launch_beam.sh RUN_NAME [beam/train.py options...]}
shift
cd "$VES"
export LD_LIBRARY_PATH="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MALLOC_MMAP_THRESHOLD_=268435456 MALLOC_TRIM_THRESHOLD_=1073741824 MALLOC_ARENA_MAX=2
export OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-passive}
mkdir -p "$FF/output/logs"
setsid nohup .venv/bin/python -u -m vesuvius.neural_tracing.fiber_follow.beam.train \
    --fiber-zarrs "$FIBER_ZARRS" --fibers "$FIBERS" --ct "$CT_ZARR" --ct-level 1 --ct-grid-scale 8 \
    --prediction-manifest "$PREDICTION_MANIFEST" --normal-manifest "$NORMAL_MANIFEST" \
    --name "$name" "$@" \
    > "$FF/output/logs/$name.log" 2>&1 &
echo $! > "$FF/output/logs/$name.pid"
