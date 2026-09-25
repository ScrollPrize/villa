#!/usr/bin/env bash
# Foreground training in a separate run directory. Additional CLI options override defaults.
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
exec "$PYTHON" -m vesuvius.neural_tracing.fiber_follow.direct.train \
    --name "$name" \
    --fiber-zarrs /mnt/raid_nvme/spiral_dataset_working/fiber_zarrs \
    --fibers /mnt/raid_nvme/spiral_dataset_working/fibers \
    --ct /mnt/raid_nvme/volpkgs/s1_2um_ds2.volpkg/volumes/s1_ds2.zarr \
    --manifest "$FF/output/single_path_v11_preparation/seeds.json" \
    --fixed-bank "$FF/output/single_path_v11_preparation/fixed_recovery.npz" "$@"
