# Task log: full hard-region failure extraction

## 2026-09-21 — task opened

- User requested a quota-free union of all hard regions from the completed
  full-resolution, Classic and best output-stride-4 evaluations.
- Public cases must be GT/reference-only; model identity is retained only in a
  private audit record so the corpus is suitable for future, as-yet-untested
  models.
- Requested context is up to 500 base voxels on both sides of each failure,
  clipped to eligible CP/Lasagna spans. Existing 200-base-voxel merge distance
  is the starting deduplication rule.
- Full-resolution cache namespace and per-source OME-Zarr stores were found;
  coverage will be checked before export and missing chunks will be reported,
  not silently treated as valid.
- The existing extractor is quota/category based, so this task extends its
  shared helpers with an explicit full-union mode rather than copying it.
- Independent implementation review was attempted as required by the workflow,
  but the delegated model was unavailable because of service capacity; local
  consistency review was performed instead.

## 2026-09-21 — extraction completed

- Focused tests passed: `17 passed` for the union helper and runner dispatch.
- The approved runner/config produced
  `/ephemeral/hendrik/fiber3d_train2/evaluations/full_failure_union_20260921_v2`.
- The corpus has 190 deduplicated reference-only pieces and 573.7638 mm total
  length.  Inputs were the full-resolution refinement evaluation, Classic and
  the best recorded output-stride-4 refinement evaluation.  No model name is
  embedded in public case annotations.
- Cache audit used a 64-base-voxel corridor around each reference piece. It
  checked 11,124 channel chunks: 8,421 present and 2,703 absent; 58 pieces are
  fully covered and 132 are partial. Missing chunks are explicit and are not
  treated as zero-valued predictions. The six source labels have 18 stable
  channel symlinks under `vc3d_channels/`.
- The first output directory (`...full_failure_union_20260921_v1`) was an
  interrupted/restarted attempt; the config now points to the complete `v2`
  result and it is the canonical output.
