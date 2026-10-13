#!/bin/bash
# Stitch + export tifxyz for the slabs completed so far (rerunnable). Output: runs/interim/
cd /ephemeral/sean/experiments/winding_fields
P=/ephemeral/sean/villa/spiral-fitting/.venv/bin/python
MIN=${1:-3}
until [ $(ls runs/full/slab_*/w.npz 2>/dev/null | wc -l) -ge $MIN ]; do sleep 30; done
rm -rf runs/interim && mkdir -p runs/interim
for f in runs/full/slab_*/w.npz; do d=$(basename $(dirname $f)); mkdir -p runs/interim/$d; ln -s $PWD/$f runs/interim/$d/w.npz; done
echo "slabs: $(ls runs/interim | tr "\n" " ")"
$P code/stitch.py runs/interim > runs/interim/stitch.log 2>&1; tail -4 runs/interim/stitch.log
CUDA_VISIBLE_DEVICES=6 $P code/extract_tifxyz.py runs/interim runs/interim/tifxyz 18 > runs/interim/extract.log 2>&1
grep "rows (every\|topology\|EXTRACT_DONE\|Traceback\|Error" runs/interim/extract.log
echo INTERIM_DONE
