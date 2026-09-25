#!/usr/bin/env bash
# Run the benchmark first; launch only with its successful, matching result.
set -euo pipefail
FF="$(cd "$(dirname "$0")/.." && pwd)"
VES="$(cd "$FF/../../../.." && pwd)"
name=${1:?Usage: launch_single_path.sh NAME [train options...]}
shift
PYTHON=${PYTHON:-$VES/.venv/bin/python}
PREPARATION=${PREPARATION:-$FF/output/single_path_v11_preparation}
CT_ZARR=${CT_ZARR:-/mnt/raid_nvme/volpkgs/s1_2um_ds2.volpkg/volumes/s1_ds2.zarr}
MICROBATCH=${MICROBATCH:-2}
"$PYTHON" "$FF/scripts/benchmark_single_path.py" --manifest "$PREPARATION/seeds.json" \
    --microbatch "$MICROBATCH" --out "$PREPARATION/benchmark_b$MICROBATCH.json"
bash "$FF/scripts/launch.sh" "$name" --ct "$CT_ZARR" --microbatch "$MICROBATCH" \
    --fixed-bank "$PREPARATION/fixed_recovery.npz" --manifest "$PREPARATION/seeds.json" \
    --benchmark "$PREPARATION/benchmark_b$MICROBATCH.json" "$@"
