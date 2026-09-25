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
export PYTHONPATH="$VES/src${PYTHONPATH:+:$PYTHONPATH}"
perf_args=()
train_args=("$@")
sampler_mode=""
resume=""
flow_draws=64
manifest="$PREPARATION/seeds.json"
for ((i=0; i<${#train_args[@]}; i++)); do
    arg=${train_args[$i]}
    case "$arg" in
        --cache-training-encoding|--no-cache-training-encoding|--compile|--no-compile)
            perf_args+=("$arg") ;;
        --sampler-mode) sampler_mode=${train_args[$((++i))]} ;;
        --sampler-mode=*) sampler_mode=${arg#*=} ;;
        --microbatch) MICROBATCH=${train_args[$((++i))]} ;;
        --microbatch=*) MICROBATCH=${arg#*=} ;;
        --flow-draws) flow_draws=${train_args[$((++i))]} ;;
        --flow-draws=*) flow_draws=${arg#*=} ;;
        --manifest) manifest=${train_args[$((++i))]} ;;
        --manifest=*) manifest=${arg#*=} ;;
        --resume|--init-from)
            i=$((i+1))
            train_args[$i]=$("$PYTHON" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve(strict=True))' "${train_args[$i]}")
            if [[ "$arg" == --resume ]]; then resume=${train_args[$i]}; fi ;;
        --resume=*|--init-from=*)
            resolved=$("$PYTHON" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve(strict=True))' "${arg#*=}")
            train_args[$i]="${arg%%=*}=$resolved"
            if [[ "$arg" == --resume=* ]]; then resume=$resolved; fi ;;
    esac
done
sampler_mode=$("$PYTHON" -c '
import sys
from vesuvius.neural_tracing.fiber_follow.train import read_checkpoint, resolve_sampler_mode
print(resolve_sampler_mode(sys.argv[2] or None, read_checkpoint(sys.argv[1], "cpu") if sys.argv[1] else None))
' "$resume" "$sampler_mode")
benchmark="$FF/output/preflight/${name}_b${MICROBATCH}_${sampler_mode}.json"
"$PYTHON" "$FF/scripts/benchmark_single_path.py" --manifest "$manifest" \
    --microbatch "$MICROBATCH" --flow-draws "$flow_draws" --sampler-mode "$sampler_mode" \
    "${perf_args[@]}" --out "$benchmark"
PYTHON="$PYTHON" bash "$FF/scripts/launch.sh" "$name" --ct "$CT_ZARR" --microbatch "$MICROBATCH" \
    --fixed-bank "$PREPARATION/fixed_recovery.npz" --manifest "$PREPARATION/seeds.json" \
    --benchmark "$benchmark" --sampler-mode "$sampler_mode" "${train_args[@]}"
