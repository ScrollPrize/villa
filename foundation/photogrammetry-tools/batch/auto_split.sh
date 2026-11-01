#!/bin/bash
# Automatic split-orientation join for one object, after split_recon.sh (run as the data owner, no docker):
# marker scale per half (marker_scale.py), rim alignment, silhouette sweep, split_gate.py, merge.
# Exit 0 when WORK/ID/split/align/sfm_data_merged.json was written and may go to merged_openmvs.sh.
# Environment: PY (python with numpy/scipy/trimesh, default ../../scrollcase/.venv/bin/python),
# MARKER_PY (python with OpenCV aruco, default python3).
# Usage: auto_split.sh WORK ID
set -u
W=$1
ID=$2
B=$(dirname "$(readlink -f "$0")")
TOOLS=$(dirname "$B")
PY=${PY:-$TOOLS/../scrollcase/.venv/bin/python}
MARKER_PY=${MARKER_PY:-python3}
S=$W/$ID/split
mkdir -p "$S/align"
declare -A F SCENE
for o in o1 o2; do
  chosen=$(cat "$S/$o/scale-used.txt" 2>/dev/null) || { echo "$ID $o: no OpenMVS scene"; exit 1; }
  SCENE[$o]=$S/$o/scale-$chosen/sfm_data_scaled.json
  [ -s "${SCENE[$o]}" ] && [ -s "$S/$o/openmvs/mvs/scene_dense.ply" ] || { echo "$ID $o: scene or dense cloud missing"; exit 1; }
  lists=$(ls "$S"/run-record/keep-$o-*.txt | grep -v -- '-all\.txt$')
  "$MARKER_PY" "$TOOLS/marker_scale.py" "${SCENE[$o]}" "$S/pgs-input-$o" $lists --json "$S/$o/marker_scale.json" \
    > "$S/$o/marker_scale.log" 2>&1
  F[$o]=$(python3 -c "import json,sys; c=json.load(open(sys.argv[1]))['choice']; print(c['factor'] if c else '')" "$S/$o/marker_scale.json" 2>/dev/null)
  [ -n "${F[$o]}" ] || { echo "$ID $o: no marker scale"; tail -3 "$S/$o/marker_scale.log"; exit 1; }
  echo "$ID $o: $(tail -1 "$S/$o/marker_scale.log")"
done
"$PY" "$TOOLS/split_align.py" align "$S/o1/openmvs/mvs/scene_dense.ply" "$S/o2/openmvs/mvs/scene_dense.ply" "$S/align" \
  --scale-a "${F[o1]}" --scale-b "${F[o2]}" > "$S/align/align.log" 2>&1 || { echo "$ID: align failed"; tail -3 "$S/align/align.log"; exit 1; }
"$PY" "$TOOLS/split_align.py" silhouette "$S/align" "${SCENE[o1]}" "${SCENE[o2]}" "$S/pgs-input-o1" "$S/pgs-input-o2" \
  --every 8 --shifts -0.9 -0.6 -0.3 0 0.3 0.6 0.9 1.2 1.5 > "$S/align/silhouette.log" 2>&1 || { echo "$ID: silhouette failed"; exit 1; }
python3 "$B/split_gate.py" "$S/align" || exit 1
shift_mm=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['thin_shift_mm'])" "$S/align/auto_decision.json")
"$PY" "$TOOLS/split_align.py" merge "$S/align" "${SCENE[o1]}" "${SCENE[o2]}" "$S/align/sfm_data_merged.json" \
  --thin-shift "$shift_mm" > "$S/align/merge.log" 2>&1 || { echo "$ID: merge failed"; exit 1; }
cat "$S/align/merge.log"
