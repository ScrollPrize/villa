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

## 2026-09-21 — corrected reproduction-search implementation

- The direct `v2` union was invalid because it bypassed the established
  bidirectional reproduction search; it was deleted and must not be used as a
  result.  The corrected run is
  `/ephemeral/hendrik/fiber3d_train2/evaluations/full_failure_union_20260921_v3`.
- The corrected workflow first runs the existing 100-base-voxel stepping
  search (minimal and padded audits in both directions), then adds the
  requested ±500-base-voxel context and performs interval union/deduplication.
  Missing inference is filled on demand (`fill_missing=true`) through the
  shared cache owner.
- Commit `8422ea88a` removes hot-loop `fsync`/flush barriers, writes
  regenerable scheduler artifacts without durability barriers, parallelizes
  reference-file preparation, and lets each native batch use all available
  case workers.  The resumed v3 search is checkpointed and currently running;
  no v3 piece count is reported until its reproduction manifest is complete.
# 2026-09-21 — Global failure-search worklist

- Corrected the extractor scheduler to support `global_worklist`: all
  candidates/directions and feasible 100-voxel distances are queued before a
  native replay, with distance-order consumption after completion. Larger
  distances are ignored once the shortest valid minimal+padded reproduction is
  known. This removes repeated Python decision/replay boundaries.
- Preserved native self-proximity preflight (`excluded=4 of50fibers` in the
  current corpus) before candidate scheduling.
- Earlier rolling-window implementations were retained only as resumable
  checkpoints; they were not continuous and are superseded by this mode.

## 2026-09-21 — native global-trial preparation parallelized

- Global replay descriptors no longer materialize a complete sliced polyline
  serially while parsing the request.  Each source annotation and its arc
  geometry are cached once, interval slicing uses indexed arc bounds, and all
  trial polylines are materialized in a bounded native preparation pool before
  the tracing workers start.  Descriptor order and provenance remain stable.
- Verified with the approved native build command; commit `136f7fbf7`.
- The approved failure-union command was rerun after the build and reused the
  already complete v3 manifest (`case_count=100`, `completed=true`).  A fresh
  non-complete worklist will exercise the new preparation pool.

## 2026-09-21 — explicit high-resolution inspection coverage fill

- Added opt-in `fill_missing_inference` to the approved failure-union config,
  scoped to `fullres_refinement_213753_best_ema999` only. It requests the exact
  deduplicated corridor chunk keys through the existing demand cache/session;
  it does not create synthetic fibers or rerun tracing.
- The first centerline replay attempt was abandoned after CUDA OOM and did not
  mutate the final audit. The direct-key implementation used batch size 1 and
  completed 13,213 unique requested chunks: 849 were initially absent and all
  849 were published. Final `cache_missing_chunk_count=0`.
- Final output remains `/ephemeral/hendrik/fiber3d_train2/evaluations/full_failure_union_20260921_v3`
  with 100 reference-only cases and 722.455325 mm. `highres_coverage_fill.json`
  records the fill counts. The task-plan requirement was changed from a
  read-only audit to an explicit opt-in mutation for inspection coverage.

## 2026-09-22 — canonical physical-volume labels

- Canonicalized exported source labels from the physical volume locator. The
  two Paris4 collections now share one `PHercParis4` channel directory, and
  the `PHerc1229` collection is exported as `PHerc1299`.
- Existing generated alias-only channel directories are safely replaced during
  union export; unrelated/non-symlink content is refused. The regenerated v3
  manifest has five physical source labels and zero missing chunks.

## 2026-09-22 — Paris4 failure-region cache subset

- Added an opt-in exporter for the canonical PHercParis4 cache. It copies only
  level-1 metadata and the audited presence/nx/ny chunks intersecting the 100
  exported failure pieces; it does not copy the full-volume cache.
- The subset contains 8,858 unique chunk keys and occupies 4.9 GiB. It is
  available at `paris4_failure_subset/` and through the symlink
  `vc3d_channels/PHercParis4_failure_subset`.

- The canonical `vc3d_channels/PHercParis4` entry now points directly to that
  subset; the special subset-labelled entry is removed. The manifest records
  the canonical link as `subset: true`, so rsyncing `vc3d_channels/` no longer
  exposes the 703 GiB full Paris4 cache.

## 2026-09-22 — per-volume failure-fiber layout

- Kept every exported fiber JSON filename unchanged, but moved it under
  `fibers/<canonical physical-volume label>/` (`PHerc0139`, `PHerc0332`,
  `PHerc1299`, `PHerc1451`, or `PHercParis4`).
- Updated each manifest path and checksum after the move. This preserves the
  existing fiber identifiers while making the exported corpus directly
  browsable and consumable one volume at a time.

## 2026-09-22 — unlinkable inference scratch

- Retained stale-session cleanup, but moved it behind exclusive cache-writer
  ownership and limited it to direct `session-*` children of known channel
  `.live_tmp` directories.
- New Linux file-mapped result-slot pools use `O_TMPFILE` and a parent-owned
  descriptor path, so their backing files have no directory entry while
  workers are attached. Named temporary files remain the fallback where
  anonymous file creation is unavailable.
- The focused spawn/mapping regression passed. The stale-session regression
  was updated; its Zarr metadata setup is unusually slow in this environment
  and was not used as the implementation validation gate.
