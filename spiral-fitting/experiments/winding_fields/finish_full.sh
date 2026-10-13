#!/bin/bash
# after the slab driver: stitch offsets, extract one tifxyz per integer winding
cd /ephemeral/sean/experiments/winding_fields
P=/ephemeral/sean/villa/spiral-fitting/.venv/bin/python
until grep -q FULL_DONE runs/full/driver.log; do sleep 60; done
echo "failed slabs: $(grep -c "exit [1-9]" runs/full/driver.log)"
$P code/stitch.py runs/full > runs/full/stitch.log 2>&1
tail -3 runs/full/stitch.log
CUDA_VISIBLE_DEVICES=6 $P code/extract_tifxyz.py runs/full runs/full/tifxyz 18 > runs/full/extract.log 2>&1
grep "topology\|EXTRACT_DONE\|Traceback\|Error" runs/full/extract.log
echo FINISH_DONE
