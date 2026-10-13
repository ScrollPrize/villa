#!/bin/bash
cd /ephemeral/sean/experiments/winding_fields
export PYTHONPATH=pydeps CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/ephemeral/sean/villa/spiral-fitting/.venv/bin/python
for cfg in "r22_ineq_up005|--sg_ineq_up 0.05" "r22_ineq_up02|--sg_ineq_up 0.2"; do
  n=${cfg%%|*}; o=${cfg#*|}; mkdir -p runs/$n
  $P -u code/solve3d.py --init none --orient aniso --signal surf --normals las --sigma -1 --center 2401.5 1699 --z0 4600 --z1 4800 --dz 3 --sheetgraph --sg_aggregate --sg_prior 1e-6 --sg_drop_back 2 --sg_across_ineq --sg_w_across 1 --sg_across_min_t 3 --sg_across_min_par 0.8 --sg_w_wi 5 --sg_w_z 0.3 --wi_root data/winding_inference --layer_iters 0 $o --out runs/$n/w.npz > runs/$n/log.txt 2>&1
  $P code/plot_mesh_rows.py runs/$n/w.npz data/eval_mesh_DO_NOT_USE_AS_INPUT/spiral-output-generation-19-lasagna.tifxyz data/umbilicus.json runs/$n/mesh_row.png >/dev/null 2>&1
  echo "== $n ($o) $(grep -o "total [0-9]*s" runs/$n/log.txt | tail -1)"; grep "off by >0.5 after\|solved step\|Error" runs/$n/log.txt | sed "s/^/   /"; sed -n "4,5p" runs/$n/eval.txt; sed -n "2p;4p;6p" runs/$n/eval_mesh.txt
done
echo QUEUE_DONE
