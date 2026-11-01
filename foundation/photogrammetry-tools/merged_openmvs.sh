#!/bin/bash
# Split-orientation reconstruction, step 2: one OpenMVS run on all photographs of an object, from the merged scene
# WORK/<id>/split/align/sfm_data_merged.json written by split_align.py merge (orientation 1 as reconstructed alone;
# orientation 2 reconstructed alone, then moved onto orientation 1 by the rim alignment, both at their marker scales).
# The merged scene is first converted to binary by OpenMVG, which validates it, and copied into the run directory:
# pgs-recon writes a colourised copy next to the scene it resumes from, so that directory must be writable.
# OpenMVS settings as split_recon.sh (densify level 2, refine level 2, 60 GB cap); 1-1.5 h per object. Review the
# result with split_align.py check and mesh_qa.py.
# Run with: sudo bash merged_openmvs.sh WORK ID [ID ...]
set -u
W=$1
shift
IMG=pgs-recon:villa-validated
for ID in "$@"; do
  S=$W/$ID/split
  mkdir -p "$S/merged"
  chown 1000:1000 "$S/merged"
  echo "=== $ID validate merged scene ($(date -u +%H:%M:%S) UTC)"
  docker run --rm --user 1000:1000 -e HOME=/tmp \
    --mount type=bind,src="$S/align",dst=/a \
    --entrypoint /bin/bash $IMG -c \
      'openMVG_main_ConvertSfM_DataFormat -i /a/sfm_data_merged.json -o /a/sfm_data_merged.bin' \
    > "$S/merged/convert.log" 2>&1
  rc=$?
  echo "    exit $rc"
  if [ $rc -ne 0 ] || [ ! -s "$S/align/sfm_data_merged.bin" ]; then
    echo "    conversion failed, see $S/merged/convert.log; skipping $ID"
    continue
  fi
  # pgs-recon writes a colourised copy next to the scene it resumes from, so the scene goes into the writable run dir
  cp "$S/align/sfm_data_merged.bin" "$S/merged/sfm_data_merged.bin"
  chown 1000:1000 "$S/merged/sfm_data_merged.bin"
  echo "=== $ID OpenMVS on the merged scene ($(date -u +%H:%M:%S) UTC)"
  docker run --rm --user 1000:1000 -e HOME=/tmp --memory 60g --memory-swap 60g \
    --mount type=bind,src="$W/$ID/pgs-input",dst=/workspace/photos,readonly \
    --mount type=bind,src="$S/merged",dst=/workspace/run \
    $IMG pgs-recon \
      --input /workspace/photos --output /workspace/run/openmvs --name $ID-split-merged-dr2-rl2 \
      --resume-sfm /workspace/run/sfm_data_merged.bin \
      --threads 8 --file-type ply --mvs-densify --densify-resolution-level 2 --mask-value 0 \
      --refine-resolution-level 2 \
    > "$S/merged/openmvs.log" 2>&1
  echo "    exit $?"
done
echo "=== done ($(date -u +%H:%M:%S) UTC)"
