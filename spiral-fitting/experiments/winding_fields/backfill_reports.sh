#!/bin/bash
# write eval.txt + renders for every run dir that has a w.npz but is missing any of them
cd /ephemeral/sean/experiments/winding_fields
P=/ephemeral/sean/villa/spiral-fitting/.venv/bin/python
for d in runs/*/; do
  f=$d/w.npz; [ -f $f ] || continue
  [ -f $d/eval.txt ] || $P code/eval_windings.py $f data/relative_windings.json > $d/eval.txt
  [ -f $d/r_xy.png ] && [ -f $d/r_zoom.png ] && [ -f $d/r_xz.png ] || $P code/render3d.py $f $d/r > /dev/null
  [ -f $d/ann.png ] || $P code/overlay_ann.py $f data/relative_windings.json $d/ann.png > /dev/null
  echo "$d ok"
done
