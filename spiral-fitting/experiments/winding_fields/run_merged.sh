#!/bin/bash
# tracks + afv fibers in one graph, with and without winding_inference; then full coupled fields
set -u
cd /ephemeral/sean/experiments/winding_fields
PY=/ephemeral/sean/villa/spiral-fitting/.venv/bin/python
export PYTHONPATH=pydeps
M=data/merged/tracks_afv.vctracks
mkdir -p data/merged runs/merged_full runs/merged_nowi_full runs/merged_field_rb runs/merged_nowi_field_rb
[ -f $M.crossings.npz ] || $PY -u code/merge_sources.py data/tracks/2um_ds2_ps256_surf_v2.dbm.vctracks data/tracks/2um_ds2_ps256_surf_v2.dbm.crossings.npz \
    data/fibers_afv/paris4_fibers.vctracks data/fibers_afv/paris4_fibers.vctracks.crossings.npz $M > data/merged/merge.log 2>&1
tail -3 data/merged/merge.log
CUDA_VISIBLE_DEVICES=6 $PY -u code/track_graph.py --z0 7500 --z1 17500 --out runs/merged_full --cg 500 --tracks $M --crossings $M.crossings.npz > runs/merged_full/log.txt 2>&1 &
CUDA_VISIBLE_DEVICES=7 $PY -u code/track_graph.py --z0 7500 --z1 17500 --out runs/merged_nowi_full --cg 500 --tracks $M --crossings $M.crossings.npz --no_wi > runs/merged_nowi_full/log.txt 2>&1 &
wait
for r in merged_full merged_nowi_full; do echo "== $r"; grep -E "crossing level|components|off by|agree|wrote|Error" runs/$r/log.txt; done
for r in merged_full merged_nowi_full; do $PY code/track_targets.py runs/$r/points.npz runs/$r/targets 3 > runs/$r/targets.log 2>&1 & done
wait
for r in merged_full merged_nowi_full; do (setsid $PY code/track_eval.py runs/$r/points.npz > runs/$r/eval_points.txt 2>&1 < /dev/null &); done
bash code/finish_rb_gen.sh runs/merged_full runs/merged_field_rb merged-coupled > runs/merged_field_rb/finish.log 2>&1
NO_WI=1 bash code/finish_rb_gen.sh runs/merged_nowi_full runs/merged_nowi_field_rb merged-nowi-coupled > runs/merged_nowi_field_rb/finish.log 2>&1
echo MERGED_ALL_DONE
