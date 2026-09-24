#!/bin/bash
# Fresh CT-only run; presence is used for initialization, never as a model channel.
set -euo pipefail
FF="$(cd "$(dirname "$0")/.." && pwd)"
name=${1:?Usage: launch_ct_tube.sh RUN_NAME [train.py options...]}
shift
CT_ZARR=${CT_ZARR:-/home/sean/Documents/volpkgs/s1_ds2.volpkg/volumes/s1_ds2.zarr}
bash "$FF/scripts/launch.sh" "$name" \
    --inputs ct --ct "$CT_ZARR" --ct-level 0 --ct-grid-scale 4 \
    --crop-depth 64 --crop-width 64 --crop-behind 16 --crop-spacing 0.5 \
    --future-step 1 --heat-bins 61 --heat-spacing 0.5 \
    --heatmap-target tube --tube-sigma 0.35 \
    --history-render segments --history-sigma 0.35 --history-jitter 0 \
    --n-candidates 6 --widths 24 64 128 --hidden 128 \
    --steps 50000 --batch 32 --workers 4 \
    --ckpt-every 500 --diag-every 500 --diag-seeds 32 "$@"
