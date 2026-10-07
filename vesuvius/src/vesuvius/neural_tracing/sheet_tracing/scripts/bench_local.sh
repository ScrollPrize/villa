#!/bin/bash
# Benchmark H100-trained checkpoints on this machine (GB10), all models in one run so they share seeds and numerics.
#   scripts/bench_local.sh OUT STEPS name=checkpoint ... [-- extra trace_benchmark args, e.g. --raw NAME]
# Checkpoints are paths relative to output/remote (e.g. m7_fix/last.pt). Defaults: 100 seeds, threshold 0.4.
# Example: scripts/bench_local.sh output/remote/trace_bench/mine_s60_t0.4 60 m7=ctx2_m7_len/last.pt fix=m7_fix/last.pt
set -e
PKG=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(cd "$PKG/../../../.." && pwd)  # vesuvius/
D=/ephemeral/sean/sheet_tracing/data
out=$1; steps=$2; shift 2; ck=(); nm=(); extra=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; extra=("$@"); break; fi
  nm+=("${1%%=*}"); ck+=("$PKG/output/remote/${1#*=}"); shift
done
cd "$REPO"
exec .venv/bin/python -m vesuvius.neural_tracing.sheet_tracing.trace_benchmark --checkpoints "${ck[@]}" --names "${nm[@]}" \
  --threshold 0.4 --seeds 100 --steps "$steps" --out "$out" \
  --path-map $D/s1_ds2.zarr=/home/sean/Documents/volpkgs/s1_ds2.volpkg/volumes/s1_ds2.zarr \
  --path-map $D/m7_ds2.zarr=/home/sean/Documents/volpkgs/s1_ds2.volpkg/volumes/m7_ds2.zarr \
  --path-map $D/fiber_presence.ome.zarr=/home/sean/Desktop/spiral_dataset/fiber_zarrs/PHercParis4-20260411134726-las-sd1-d74cca79_presence.ome.zarr \
  --path-map $D/verified_patches_old=/home/sean/Desktop/spiral_dataset/verified_patches \
  --path-map $D/verified_patches_working=/home/sean/Desktop/spiral_dataset_working/verified_patches "${extra[@]}"
