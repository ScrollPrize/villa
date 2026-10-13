#!/bin/bash
# graph pass (running) -> global joint solve -> final fill on GPUs 6+7 -> stitch -> tifxyz -> overlap diag
set -u
cd /ephemeral/sean/experiments/winding_fields
PY=/ephemeral/sean/villa/spiral-fitting/.venv/bin/python
R=runs/joint
export PYTHONPATH=pydeps PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until [ $(grep -c GRAPH_DONE $R/driver_graph_0.log $R/driver_graph_1.log | awk -F: "{s+=\$2} END {print s}") -eq 2 ]; do sleep 30; done
n=$(ls $R/slab_*/graph.npz | wc -l); echo "graphs: $n"
[ "$n" -eq 37 ] || { echo "JOINT_FAIL missing graphs"; exit 1; }
CUDA_VISIBLE_DEVICES=6 $PY -u code/joint_solve.py $R > $R/joint.log 2>&1 || { echo "JOINT_FAIL joint_solve"; exit 1; }
echo "joint solve done"; grep -v IRLS $R/joint.log | tail -8
for g in 0 1; do CUDA_VISIBLE_DEVICES=$((6+g)) $PY -u code/run_full.py --mode final --root $R --part $g/2 > $R/driver_final_$g.log 2>&1 & done
wait
n=$(ls $R/slab_*/w.npz | wc -l); echo "final fields: $n"
grep -h "exit [1-9]" $R/driver_final_*.log && echo "WARNING some final slabs failed"
CUDA_VISIBLE_DEVICES=6 $PY -u code/stitch.py $R > $R/stitch.log 2>&1; tail -40 $R/stitch.log
CUDA_VISIBLE_DEVICES=6 $PY -u code/extract_tifxyz.py $R > $R/extract.log 2>&1; grep -E "topology|EXTRACT_DONE|Traceback" $R/extract.log
mkdir -p $R/diag; $PY code/overlap_diag.py $R 8 $R/diag/overlap_08.png > $R/diag/overlap_08.txt 2>&1
echo JOINT_ALL_DONE
