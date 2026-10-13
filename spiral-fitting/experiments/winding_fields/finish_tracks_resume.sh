#!/bin/bash
# full-band track graph (running) -> voxel targets -> slab fills on GPUs 6+7 -> stitch -> tifxyz -> diag -> combined -> flatten
set -u
cd /ephemeral/sean/experiments/winding_fields
PY=/ephemeral/sean/villa/spiral-fitting/.venv/bin/python
G=runs/tracks_full; R=runs/tracks_field
export PYTHONPATH=pydeps PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# resume: the GPU-6 half is still running; GPU 7 is taken by another job, so the rest runs on GPU 6 after it
while pgrep -f "run_full.py --mode tracks" > /dev/null; do sleep 30; done
CUDA_VISIBLE_DEVICES=6 $PY -u code/run_full.py --mode tracks --root $R --targets $G/targets > $R/driver_tracks_rest.log 2>&1
n=$(ls $R/slab_*/w.npz | wc -l); echo "fields: $n"
grep -h "exit [1-9]" $R/driver_tracks_rest.log && echo "WARNING some slabs failed"
CUDA_VISIBLE_DEVICES=6 $PY -u code/stitch.py $R > $R/stitch.log 2>&1; tail -40 $R/stitch.log
CUDA_VISIBLE_DEVICES=6 $PY -u code/extract_tifxyz.py $R > $R/extract.log 2>&1; grep -E "topology|EXTRACT_DONE|Traceback" $R/extract.log
mkdir -p $R/diag
$PY code/overlap_diag.py $R 8 $R/diag/overlap_08.png > $R/diag/overlap_08.txt 2>&1
$PY code/seam_jumps.py $R > $R/diag/seam_jumps.txt 2>&1
$PY code/render_seams.py runs/joint $R $R/diag/seams_joint_vs_tracks.png > /dev/null 2>&1
for d in runs/joint $R; do echo "== $d"; cat $d/slab_*/eval_mesh.txt | awk '/exact/{split($0,a,"exact\\(<0.5\\) "); split(a[2],b," "); e+=b[1]; split($0,c,"within1 "); w+=c[2]; n++} END {printf "mean over %d slabs: exact %.3f within1 %.3f\n", n, e/n, w/n}'; done
$PY code/combine_tifxyz.py $R/tifxyz $R/tifxyz_combined all_windings_tracks > $R/combine.log 2>&1; tail -3 $R/combine.log
mkdir -p $R/flattened
$PY -u code/flatten_windings.py $R/tifxyz $R/flattened/winding-fields-tracks-lasagna.tifxyz > $R/flattened/flatten.log 2>&1; tail -5 $R/flattened/flatten.log
echo TRACKS_ALL_DONE
