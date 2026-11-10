#!/bin/bash
# Stage-1 comparison (PLAN-S1-final): R0 (legacy anchors), R1 (sheet anchors, patch path), R1b (sheet anchors at
# their own points), each Va (10k from scratch) -> Vb (resume to 25k total), all held-out exclusions on.
#   chain_s1.sh FROZEN_DIR [ARM ...]     (default arms: hub1030h hub1030h_sheet hub1030h_sheetpt)
# Waits for chain_dt to finish. A stage is VALID only when the fitter exited 0, the checkpoint completed its horizon,
# the anchor audit is present with no non-finite evaluation and <= 0.1% rejections at every audit, and scores.json
# exists with this run's manifest hash; a lineage stops at its first invalid stage. Stop file: work/chain_s1.stop.
set -u -o pipefail
echo $$ > /media/djosey/nvme4/spiral_0332/work/chain_s1.pid
D=/media/djosey/nvme4/spiral_0332; F=${1:?frozen dir}; shift
ARMS=${*:-hub1030h hub1030h_sheet hub1030h_sheetpt}
PY=/home/djosey/Documents/villa/spiral-fitting/.venv/bin/python
export SPIRAL_ROOT=$F/spiral-fitting WORK_ROOT=$F/work SPIRAL_PY=$PY
export SPIRAL_SATISFACTION_HOST_QUADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FIT_CACHE=$D/cache_hub1030
until grep -q "chain_dt done" $D/work/chain_dt.log 2>/dev/null; do sleep 120; done
stop() { [ -f $D/work/chain_s1.stop ]; }
rundir() { ls -d $D/out/*_$1 2>/dev/null | tail -1; }

# the frozen code must be what python imports
(cd $SPIRAL_ROOT && $PY -c "
import os, sys, fit_spiral, losses, winding_convention
root = os.path.realpath(os.environ['SPIRAL_ROOT'])
bad = [m.__file__ for m in (fit_spiral, losses, winding_convention) if not os.path.realpath(m.__file__).startswith(root)]
sys.exit('imports outside the frozen root: %s' % bad if bad else 0)") || { echo "$(date) frozen import check FAILED"; exit 1; }
echo "$(date) chain_s1 start, frozen $F, arms: $ARMS"

valid() {  # $1 tag, $2 horizon -> 0 when the stage is valid
  local T=$1 H=$2 R; R=$(rundir $T)
  [ -n "$R" ] && [ -f $R/checkpoint_fitted.ckpt ] || { echo "INVALID $T: no run dir/checkpoint"; return 1; }
  $PY - "$R" "$H" <<'PY' || return 1
import json, sys, torch
run, horizon = sys.argv[1], int(sys.argv[2])
ck = torch.load(f'{run}/checkpoint_fitted.ckpt', map_location='cpu', weights_only=False, mmap=True)
done = int(ck.get('completed_iterations', -1))
problems = [] if done == horizon else [f'completed {done} != horizon {horizon}']
try:
    audits = [json.loads(l) for l in open(f'{run}/abs_winding_audit.jsonl')]
except OSError:
    audits = []
if not any(a.get('final') for a in audits):
    problems.append('no final anchor audit')
for a in audits:
    if a['nonfinite'] > 0 or a['rejected'] > 0.001 * max(a['attempted'], 1):
        problems.append(f"audit at {a['iteration']}: nonfinite {a['nonfinite']}, rejected {a['rejected']}/{a['attempted']}")
try:
    scores = json.load(open(f'{run}/scores.json'))
    import hashlib
    if scores.get('manifest_hash') != hashlib.sha256(open(f'{run}/manifest.json', 'rb').read()).hexdigest():
        problems.append('scores.json manifest hash mismatch')
except OSError:
    problems.append('no scores.json')
if problems:
    print(f'INVALID {run}: ' + '; '.join(problems)); sys.exit(1)
PY
  touch $R/stage.valid; echo "$(date) VALID $T"
}

stage() {  # $1 tag, $2 dataset arm (legacy|sheet), $3 horizon, [$4 parent tag]
  local T=$1 ARM=$2 H=$3 P=${4:-} CK="" R rc
  if [ -n "$P" ]; then
    [ -f "$(rundir $P)/stage.valid" ] || { echo "$(date) skip $T: parent $P not VALID"; return 1; }
    CK=$(rundir $P)/checkpoint_fitted.ckpt
  fi
  echo "$(date) === $T ${P:+from $P}"
  FIT_DATASET=$D/dataset_tta_hub1030h_$ARM $F/work/run_fit.sh $D/overrides_$T.json $T $CK > $D/work/fit_$T.log 2>&1; rc=$?
  echo "$(date) $T exited $rc; tracebacks $(grep -c Traceback $D/work/fit_$T.log)"
  R=$(rundir $T); [ $rc -eq 0 ] && [ -n "$R" ] || { echo "INVALID $T: fitter exit $rc"; return 1; }
  FIT_DATASET=$D/dataset_tta_hub1030h_$ARM $PY $F/work/s1/s1_manifest.py run $R $F $D/overrides_$T.json \
    ${CK:+--parent $CK} --anchors $D/dataset_tta_hub1030h_$ARM/abs_winding.json > /dev/null || return 1
  $PY $F/work/s1/s1_score.py $R > $D/work/s1/score_$T.log 2>&1 || { echo "INVALID $T: scoring failed"; return 1; }
  $PY -c "
import json; s = json.load(open('$R/scores.json'))
a = s['slice']['windows']['all']; tr = s.get('anchors_train') or {}; he = s.get('anchors_held') or {}; hp = s.get('held_patches') or {}
fmt = lambda v: 'n/a' if v is None else f'{v:.4f}'
print('S1SCORE $T f025', fmt(a.get('f025')), 'f025_cens', fmt(a.get('f025_cens')), 'f05', fmt(a.get('f05')),
      'held_patches', fmt(hp.get('fraction')), 'train_off_eqfib', fmt(tr.get('equal_fiber_off_rate')),
      'held_off_eqfib', fmt(he.get('equal_fiber_off_rate')))" 2>&1 | tail -1
  valid $T $H
}

for ARM_TAG in $ARMS; do
  case $ARM_TAG in hub1030h) DS=legacy ;; *) DS=sheet ;; esac
  stop && break
  stage Va_${ARM_TAG}_10k $DS 10000 || continue
  stop && break
  stage Vb_${ARM_TAG}_25k $DS 25000 Va_${ARM_TAG}_10k || continue
done
echo "$(date) chain_s1 done"
