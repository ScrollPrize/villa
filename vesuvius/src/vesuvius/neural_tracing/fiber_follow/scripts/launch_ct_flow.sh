#!/bin/bash
# Future flow with fixed observed history: 32 voxels of visual history, 64-voxel continuations.
set -euo pipefail
FF="$(cd "$(dirname "$0")/.." && pwd)"
name=${1:?Usage: launch_ct_flow.sh RUN_NAME [train.py options...]}
shift
CT_ZARR=${CT_ZARR:-/mnt/raid_nvme/volpkgs/s1_2um_ds2.volpkg/volumes/s1_ds2.zarr}
bash "$FF/scripts/launch.sh" "$name" \
    --inputs ct+presence --ct "$CT_ZARR" --ct-level 0 --ct-grid-scale 4 \
    --crop-depth 208 --crop-width 128 --crop-behind 64 --crop-spacing 0.5 \
    --n-future 64 --future-step 1 --recent-history-points 32 \
    --angle-sigmas 2 5 10 \
    --history-render segments --history-sigma 0.35 --history-jitter 0 \
    --flow-samples 16 --flow-draws 16 --flow-steps 4 --warmup 1000 --norm group \
    --widths 24 64 128 --hidden 128 \
    --steps 50000 --batch 2 --dagger-batch 1 --workers 4 \
    --ckpt-every 500 --diag-every 500 --diag-seeds 32 --diag-batch 1 "$@"
