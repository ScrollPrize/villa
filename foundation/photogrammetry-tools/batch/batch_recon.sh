#!/bin/bash
# Unattended reconstruction of several objects, from prepared inputs to OpenMVS meshes in mm.
#
# Before running (as the data owner, no sudo): mask every acquisition block, then prepare_input.py WORK ID for each
# object, which builds WORK/ID/pgs-input/, the scale lists and the per-orientation split inputs.
#
# Stage 1, every object: joint SfM (global, SIFT HIGH, ratio 0.75, robust), exported to sfm_data.json.
# Stage 2, per object, depending on whether the joint SfM posed every orientation (linked.py, >= 90% of each):
#   linked:   pgs-global-scaler candidates per block and orientation (kept for the record), the scale from
#             marker_scale.py (lens distortion refitted from the markers; median of the consistent blocks), the scene
#             multiplied into mm (scale_scene.py), converted to binary and OpenMVS on all photographs (densify 2,
#             refine 2, 60 GB cap);
#   not linked: the split route of INSTRUCTIONS "Orientations that do not link": ../split_recon.sh, then
#             auto_split.sh (marker scale per half, rim alignment, silhouette, the split_gate.py rules, merge), then
#             ../merged_openmvs.sh. An object the gate rejects stops before the merged OpenMVS run.
# Before trusting a "not linked" result, check the block grouping: blocks the joint SfM posed together with the
# other orientation may belong to it (sncass94's two 24 mm blocks did).
# Steps whose outputs already exist are skipped, so rerunning after a failure only repeats what is missing.
# The marker size is the assumed 31.5 mm; scales are linear in it.
# Containers run one at a time as uid 1000, memory-capped; host-side analysis runs as the owner of WORK.
# Environment: MARKER_PY (python with OpenCV aruco for marker_scale.py, default python3), PY (python with
# numpy/scipy/trimesh, default the scrollcase venv), IMG (default pgs-recon:villa-validated). sudo drops the caller's
# environment, so pass them on the command line: sudo MARKER_PY=/path/to/python bash batch_recon.sh WORK ID ...
#
# Run with: sudo bash batch_recon.sh WORK ID [ID ...]
set -u
W=$1
shift
B=$(dirname "$(readlink -f "$0")")
TOOLS=$(dirname "$B")
IMG=${IMG:-pgs-recon:villa-validated}
OWNER=$(stat -c %U "$W")
MARKER_PY=${MARKER_PY:-python3}
export PY=${PY:-$TOOLS/../scrollcase/.venv/bin/python} MARKER_PY
MARKER_MM=31.5
LOG=$W/batch-$(date -u +%Y%m%dT%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
log() { echo "=== $(date -u +%H:%M:%S) UTC $*"; }
as_owner() { runuser -u "$OWNER" -- env PY="$PY" MARKER_PY="$MARKER_PY" "$@"; }
log "batch_recon.sh $W $*; villa tools at $(git -C "$TOOLS" rev-parse HEAD); image $(docker image inspect -f '{{.Id}}' $IMG)"

convert() {  # convert() DIR IN OUT: openMVG_main_ConvertSfM_DataFormat inside DIR
  docker run --rm --user 1000:1000 -e HOME=/tmp --mount type=bind,src="$1",dst=/r --entrypoint /bin/bash $IMG \
    -c "openMVG_main_ConvertSfM_DataFormat -i /r/$2 -o /r/$3"
}

# ---------------------------------------------------------------- stage 1: joint SfM for every object
for ID in "$@"; do
  R=$W/$ID/recon
  RD=$R/sfm/mvg/recon_dir
  mkdir -p "$W/$ID/run-record/batch"
  cp "$B"/* "$W/$ID/run-record/batch/" 2>/dev/null
  if [ -s "$RD/sfm_data.bin" ]; then
    log "$ID joint SfM: exists"
  else
    mkdir -p "$R/sfm"
    chown -R 1000:1000 "$R"
    log "$ID joint SfM"
    docker run --rm --user 1000:1000 -e HOME=/tmp --memory 48g --memory-swap 48g \
      --mount type=bind,src="$W/$ID/pgs-input",dst=/workspace/photos,readonly \
      --mount type=bind,src="$R/sfm",dst=/workspace/output \
      $IMG pgs-recon \
        --input /workspace/photos --output /workspace/output --name $ID-high-r075-robust \
        --mvg-recon-method global --describer-method SIFT --describer-preset HIGH --matching-ratio 0.75 \
        --threads 8 --file-type ply --mvg-robust --stop-after sfm \
      > "$R/sfm.log" 2>&1
    log "    exit $?"
  fi
  if [ -s "$RD/sfm_data.bin" ] && [ ! -s "$RD/sfm_data.json" ]; then
    convert "$RD" sfm_data.bin sfm_data.json > "$R/convert.log" 2>&1
  fi
done

# ---------------------------------------------------------------- stage 2: per object, joint or split
for ID in "$@"; do
  R=$W/$ID/recon
  RD=$R/sfm/mvg/recon_dir
  if [ ! -s "$RD/sfm_data.json" ]; then
    log "$ID: no SfM scene, skipping"
    continue
  fi
  if python3 "$B/linked.py" "$RD/sfm_data.json" "$R/scale-lists" > "$R/linked.txt"; then
    log "$ID: orientations linked ($(tr '\n' ' ' < "$R/linked.txt")) -> joint route"
    for include_path in "$R"/scale-lists/keep-*.txt; do
      c=$(basename "$include_path" .txt)
      c=${c#keep-}
      out=$R/scale-candidates/$c
      [ -s "$out/scaler.log" ] && continue
      mkdir -p "$out"
      chown 1000:1000 "$R/scale-candidates" "$out"
      docker run --rm --user 1000:1000 -e HOME=/tmp --memory 16g --memory-swap 16g \
        --mount type=bind,src="$W/$ID/pgs-input",dst=/workspace/photos,readonly \
        --mount type=bind,src="$R/sfm",dst=/workspace/sfm,readonly \
        --mount type=bind,src="$out",dst=/workspace/scale-output \
        --mount type=bind,src="$R/scale-lists",dst=/workspace/scale-lists,readonly \
        $IMG pgs-global-scaler \
          --input-scene /workspace/sfm/mvg/recon_dir/sfm_data.bin \
          --output-scene /workspace/scale-output/sfm_data_scaled.bin \
          --sfm-root /workspace/photos --scale-method umeyama --marker-size "$MARKER_MM" --detection-method markers \
          --include-from "/workspace/scale-lists/keep-$c.txt" \
          --histogram-out /workspace/scale-output/histogram.svg \
          --save-landmarks /workspace/scale-output/landmarks.ply \
          --save-scaled-landmarks /workspace/scale-output/landmarks-scaled.ply \
        > "$out/scaler.log" 2>&1
      log "    scaler $c: exit $? ; $(grep -i 'median scale' "$out/scaler.log" | tail -1)"
    done
    lists=$(ls "$R"/scale-lists/keep-*.txt | grep -v -- '-all\.txt$')
    if [ ! -s "$R/marker_scale.json" ]; then
      as_owner "$MARKER_PY" "$TOOLS/marker_scale.py" "$RD/sfm_data.json" "$W/$ID/pgs-input" $lists \
        --json "$R/marker_scale.json" > "$R/marker_scale.log" 2>&1
    fi
    log "    $(tail -1 "$R/marker_scale.log")"
    factor=$(python3 -c "import json,sys; c=json.load(open(sys.argv[1]))['choice']; print(c['factor'] if c else '')" "$R/marker_scale.json" 2>/dev/null)
    if [ -z "$factor" ]; then
      log "$ID: no marker scale, skipping OpenMVS"
      continue
    fi
    mkdir -p "$R/scaled"
    chown "$OWNER" "$R/scaled"
    if [ ! -s "$R/scaled/sfm_data_mm.bin" ]; then
      as_owner python3 "$B/scale_scene.py" "$RD/sfm_data.json" "$R/scaled/sfm_data_mm.json" "$factor"
      chown 1000:1000 "$R/scaled"
      convert "$R/scaled" sfm_data_mm.json sfm_data_mm.bin > "$R/scaled/convert.log" 2>&1
      log "    scaled scene x$factor: $(ls -s "$R/scaled/sfm_data_mm.bin" 2>&1)"
    fi
    run=$R/openmvs/mm-dr2-rl2
    if ls "$run"/mvs/*-dr2-rl2.ply > /dev/null 2>&1; then
      log "$ID OpenMVS: exists"
      continue
    fi
    if [ -d "$run" ]; then mv "$run" "$run-failed-$(date -u +%H%M%S)"; fi
    mkdir -p "$run"
    chown 1000:1000 "$R/openmvs" "$run"
    log "$ID OpenMVS (joint, mm)"
    docker run --rm --user 1000:1000 -e HOME=/tmp --memory 60g --memory-swap 60g \
      --mount type=bind,src="$W/$ID/pgs-input",dst=/workspace/photos,readonly \
      --mount type=bind,src="$R",dst=/workspace/run \
      $IMG pgs-recon \
        --input /workspace/photos --output /workspace/run/openmvs/mm-dr2-rl2 --name $ID-joint-mm-dr2-rl2 \
        --resume-sfm /workspace/run/scaled/sfm_data_mm.bin \
        --threads 8 --file-type ply --mvs-densify --densify-resolution-level 2 --mask-value 0 \
        --refine-resolution-level 2 \
      > "$R/openmvs-mm-dr2-rl2.log" 2>&1
    log "    exit $?"
  else
    S=$W/$ID/split
    log "$ID: orientations NOT linked ($(tr '\n' ' ' < "$R/linked.txt")) -> split route"
    if [ -s "$S/o1/openmvs/mvs/scene_dense.ply" ] && [ -s "$S/o2/openmvs/mvs/scene_dense.ply" ]; then
      log "$ID split halves: exist"
    else
      bash "$TOOLS/split_recon.sh" "$W" "$ID"
    fi
    if [ ! -s "$S/align/sfm_data_merged.json" ]; then
      chown -R "$OWNER" "$S/align" 2>/dev/null
      log "$ID automatic join"
      as_owner bash "$B/auto_split.sh" "$W" "$ID" || { log "$ID: join not accepted, stopping before merged OpenMVS"; continue; }
    fi
    if ls "$S"/merged/openmvs/mvs/*-dr2-rl2.ply > /dev/null 2>&1; then
      log "$ID merged OpenMVS: exists"
    else
      [ -d "$S/merged" ] && mv "$S/merged" "$S/merged-failed-$(date -u +%H%M%S)"
      bash "$TOOLS/merged_openmvs.sh" "$W" "$ID"
    fi
  fi
done
chown "$OWNER" "$LOG" 2>/dev/null
log "done; log $LOG"
