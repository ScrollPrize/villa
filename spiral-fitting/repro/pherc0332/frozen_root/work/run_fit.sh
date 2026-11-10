#!/bin/bash
# PHerc0332 spiral fit in the L2 frame (9.596 um): lasagna normals + grad_mag spacing, tracks,
# graph patches, fiber strips (as patches) and the VC3D fibers.
#   run_fit.sh <overrides.json> <tag> [resume checkpoint]
set -euo pipefail
E=/media/djosey/nvme4/spiral_0332
S=${SPIRAL_ROOT:-/home/djosey/Documents/villa/spiral-fitting}
PY=${SPIRAL_PY:-/home/djosey/Documents/villa/spiral-fitting/.venv/bin/python}
OV=$(readlink -f "$1"); TAG=$2; RESUME=${3:-}
cd $S
export FIT_SPIRAL_CONFIG_OVERRIDES="$(tr -d '\n' < "$OV")"
export FIT_SPIRAL_OUT_DIR=$E/out
export FIT_SPIRAL_RUN_TAG=$TAG
export FIT_SPIRAL_CACHE_DIR=${FIT_CACHE:-$E/cache}
export WANDB_MODE=disabled
if [ -n "$RESUME" ]; then
  export FIT_SPIRAL_RESUME_PATH="$RESUME"
fi
exec $PY fit_spiral.py --dataset ${FIT_DATASET:-$E/dataset}
