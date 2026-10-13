#!/bin/bash
# r08: bandcount init + surf layer refine; r05b: gabor init + surf (r05 config) on the extra slabs
cd /ephemeral/sean/experiments/winding_fields
P=/ephemeral/sean/villa/spiral-fitting/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=6
COMMON="--level 3 --normals las --orient aniso --orient_reach 20 --sigma -1 --center 2401.5 1699 --dz 6 --layer_iters 2000"
run() { name=$1; z0=$2; shift 2; mkdir -p runs/$name; $P -u code/solve3d.py $COMMON --z0 $z0 --z1 $((z0+200)) "$@" --out runs/$name/w.npz > runs/$name/log.txt 2>&1; echo "$name exit $?"; }
run r08_bandcount_z4600 4600 --signal surf --init bandcount --iters 0
run r08_bandcount_z4400 4400 --signal surf --init bandcount --iters 0
run r08_bandcount_z4800 4800 --signal surf --init bandcount --iters 0
run r05b_surf_z4400 4400 --signal surf --iters 1500
run r05b_surf_z4800 4800 --signal surf --iters 1500
echo QUEUE DONE
