#!/bin/bash
# red-black coupled slab fills from the global track labels:
#   pass 1: even slabs (no two overlap), each with 45 level-3 slices of extra z context, own slices saved
#   pass 2: odd slabs, same padding, slices shared with the even neighbours fixed to their values
# then stitch -> tifxyz -> diag -> combined -> flatten
set -u
cd /ephemeral/sean/experiments/winding_fields
PY=/ephemeral/sean/villa/spiral-fitting/.venv/bin/python
G=runs/afv_full; R=runs/afv_field_rb
export PYTHONPATH=pydeps PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "code/run_full.py" > /dev/null; do sleep 20; done   # resume: let a running pass finish first
EVEN=$($PY -c "print(' '.join(str(z) for i, z in enumerate(range(3750, 8750 - 75, 135)) if i % 2 == 0))")
ODD=$($PY -c "print(' '.join(str(z) for i, z in enumerate(range(3750, 8750 - 75, 135)) if i % 2 == 1))")
for g in 0 1; do CUDA_VISIBLE_DEVICES=$((6+g)) $PY -u code/run_full.py $EVEN --mode tracks --root $R --targets $G/targets --pad 45 --part $g/2 > $R/driver_even_$g.log 2>&1 & done
wait
echo "even fields: $(ls $R/slab_*/w.npz | wc -l)"
for g in 0 1; do CUDA_VISIBLE_DEVICES=$((6+g)) $PY -u code/run_full.py $ODD --mode tracks --root $R --targets $G/targets --pad 45 --bc_root $R --part $g/2 > $R/driver_odd_$g.log 2>&1 & done
wait
n=$(ls $R/slab_*/w.npz | wc -l); echo "fields: $n"
grep -h "exit [1-9]" $R/driver_*.log && echo "WARNING some slabs failed"
CUDA_VISIBLE_DEVICES=6 $PY -u code/stitch.py $R > $R/stitch.log 2>&1; grep -E "absolute|agreement" $R/stitch.log | awk '{print}' | tail -40
CUDA_VISIBLE_DEVICES=6 $PY -u code/extract_tifxyz.py $R > $R/extract.log 2>&1; grep -E "topology|EXTRACT_DONE|Traceback" $R/extract.log
mkdir -p $R/diag
$PY code/overlap_diag.py $R 8 $R/diag/overlap_08.png > $R/diag/overlap_08.txt 2>&1
$PY code/seam_jumps.py $R > $R/diag/seam_jumps.txt 2>&1; echo "seams:"; tail -4 $R/diag/seam_jumps.txt
$PY code/render_seams.py runs/tracks_field_rb2 $R $R/diag/seams_tracks_vs_afv.png > /dev/null 2>&1
for d in runs/joint runs/tracks_field_rb2 $R; do echo "== $d"; cat $d/slab_*/eval_mesh.txt | awk '/exact/{split($0,a,"exact\\(<0.5\\) "); split(a[2],b," "); e+=b[1]; split($0,c,"within1 "); w+=c[2]; n++} END {printf "mean over %d slabs: exact %.3f within1 %.3f\n", n, e/n, w/n}'; done
$PY code/combine_tifxyz.py $R/tifxyz $R/tifxyz_combined all_windings_afv_coupled > $R/combine.log 2>&1; tail -2 $R/combine.log
mkdir -p $R/flattened
$PY -u code/flatten_windings.py $R/tifxyz $R/flattened/winding-fields-afv-coupled-lasagna.tifxyz > $R/flattened/flatten.log 2>&1; tail -2 $R/flattened/flatten.log
$PY code/render_flat.py $R/diag/flattened.png afv-coupled $R/flattened/winding-fields-afv-coupled-lasagna.tifxyz > /dev/null 2>&1
echo AFV_ALL_DONE
