#!/bin/bash
# Split-orientation reconstruction, step 1, for objects whose orientations do not link in a joint SfM (sfm_review.py
# reports 0 verified pairs between orientations): each orientation is reconstructed completely on its own, SfM, marker
# scale candidates and OpenMVS, ready for split_align.py (photogrammetry-docs/INSTRUCTIONS.md, "Orientations that do not
# link").
# Expects per object WORK/<id>/split/: pgs-input-o1/, pgs-input-o2/ with each orientation's photographs and masks (hard
# links into WORK/<id>/pgs-input/), and run-record/keep-o<n>-<height>.txt listing each block's photographs.
# Scale: pgs-global-scaler runs once per camera height of each orientation; OpenMVS resumes from the mid candidate,
# or high, then low, if mid wrote no scene. Its result is provisional: the scaler's 0.1 px inlier threshold leaves the
# corners poorly triangulated on small objects, and a failed candidate can write a scene at scale 1. Measure the scale
# with marker_scale.py on the exported JSON and pass the factor to split_align.py --scale-a/--scale-b (a pure scale
# about the origin, so it applies to the clouds afterwards).
# Time per object: SfM 2 x 15 min, scale 6 x 3-5 min, OpenMVS 2 x 30-40 min. Containers are memory-capped and run one
# at a time.
# Run with: sudo bash split_recon.sh WORK ID [ID ...]
set -u
W=$1
shift
MARKER_MM=31.5
IMG=pgs-recon:villa-validated
for ID in "$@"; do
  S=$W/$ID/split
  for o in o1 o2; do
    mkdir -p "$S/$o/sfm" "$S/$o/openmvs"
    chown -R 1000:1000 "$S/$o"
    echo "=== $ID $o SfM ($(date -u +%H:%M:%S) UTC)"
    docker run --rm --user 1000:1000 -e HOME=/tmp --memory 48g --memory-swap 48g \
      --mount type=bind,src="$S/pgs-input-$o",dst=/workspace/photos,readonly \
      --mount type=bind,src="$S/$o/sfm",dst=/workspace/output \
      $IMG pgs-recon \
        --input /workspace/photos --output /workspace/output --name $ID-$o-high-r075-robust \
        --mvg-recon-method global --describer-method SIFT --describer-preset HIGH --matching-ratio 0.75 \
        --threads 8 --file-type ply --mvg-robust --stop-after sfm \
      > "$S/$o/sfm.log" 2>&1
    echo "    exit $?"
    for h in mid high low; do
      out="$S/$o/scale-$h"
      mkdir -p "$out"
      chown 1000:1000 "$out"
      docker run --rm --user 1000:1000 -e HOME=/tmp --memory 16g --memory-swap 16g \
        --mount type=bind,src="$S/pgs-input-$o",dst=/workspace/photos,readonly \
        --mount type=bind,src="$S/$o/sfm",dst=/workspace/sfm,readonly \
        --mount type=bind,src="$out",dst=/workspace/scale-output \
        --mount type=bind,src="$S/run-record",dst=/workspace/scale-lists,readonly \
        $IMG pgs-global-scaler \
          --input-scene /workspace/sfm/mvg/recon_dir/sfm_data.bin \
          --output-scene /workspace/scale-output/sfm_data_scaled.bin \
          --sfm-root /workspace/photos --scale-method umeyama --marker-size "$MARKER_MM" --detection-method markers \
          --include-from "/workspace/scale-lists/keep-$o-$h.txt" \
          --histogram-out /workspace/scale-output/histogram.svg \
          --save-landmarks /workspace/scale-output/landmarks.ply \
          --save-scaled-landmarks /workspace/scale-output/landmarks-scaled.ply \
        > "$out/scaler.log" 2>&1
      echo "=== $ID $o scale $h: exit $? ; $(grep -i 'median scale' "$out/scaler.log" | tail -1)"
    done
    chosen=""
    for h in mid high low; do
      if [ -s "$S/$o/scale-$h/sfm_data_scaled.bin" ]; then chosen=$h; break; fi
    done
    if [ -z "$chosen" ]; then
      echo "=== $ID $o: no scale candidate produced a scene, skipping OpenMVS"
      continue
    fi
    echo "$chosen" > "$S/$o/scale-used.txt"
    if grep -qiE 'median scale[^0-9]*1(\.0+)?$' "$S/$o/scale-$chosen/scaler.log"; then
      echo "=== $ID $o: scale-$chosen wrote a scene at scale 1 (scaler failed); OpenMVS runs in scene units, fix with marker_scale.py"
    fi
    echo "=== $ID $o OpenMVS from scale-$chosen ($(date -u +%H:%M:%S) UTC)"
    docker run --rm --user 1000:1000 -e HOME=/tmp --memory 60g --memory-swap 60g \
      --mount type=bind,src="$S/pgs-input-$o",dst=/workspace/photos,readonly \
      --mount type=bind,src="$S/$o",dst=/workspace/run \
      $IMG pgs-recon \
        --input /workspace/photos --output /workspace/run/openmvs --name $ID-$o-$chosen-dr2-rl2 \
        --resume-sfm /workspace/run/scale-$chosen/sfm_data_scaled.bin \
        --threads 8 --file-type ply --mvs-densify --densify-resolution-level 2 --mask-value 0 \
        --refine-resolution-level 2 \
      > "$S/$o/openmvs.log" 2>&1
    echo "    exit $?"
    docker run --rm --user 1000:1000 -e HOME=/tmp \
      --mount type=bind,src="$S/$o/scale-$chosen",dst=/r \
      --entrypoint /bin/bash $IMG -c \
        'openMVG_main_ConvertSfM_DataFormat -i /r/sfm_data_scaled.bin -o /r/sfm_data_scaled.json'
    echo "=== $ID $o scene exported to JSON: exit $?"
  done
done
echo "=== done ($(date -u +%H:%M:%S) UTC)"
