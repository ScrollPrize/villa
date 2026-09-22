# Task plan: full hard-region failure extraction

## Inspection coverage fill (2026-09-21)

Add opt-in `fill_missing_inference.evaluation_name`. Reuse historical model,
cache identity and managed demand session. Deduplicate audited corridor chunk
keys by physical cache, request those keys directly, and drain publication;
do not create synthetic fibers or rerun tracing. Re-audit all channel files.
Use only the approved failure-union runner and test command. Test deduplication,
cached no-op and failure propagation. Spec update: inspection completion requires
zero missing chunks when filling is requested. Docs/log: record exact command,
before/after coverage and abandoned replay-based attempts.

## Active correction: continuous native search (supersedes batch scheduling)

Construct the global trial list once per model, using every unresolved
candidate/direction and all feasible 100-voxel distances. Keep the self-fold
exclusion and historical result/checkpoint semantics. Send this list to one
native evaluator, not successive distance batches. Use compact trial metadata
and native reference slicing to avoid thousands of reference JSON files.
Native workers admit trials in ascending distance order and publish audits
incrementally. After the earliest valid failure and its 100-voxel padding check
are known, cancel longer queued trials for that candidate/direction. Consume
out-of-order results in distance order; never substitute a later failure for an
unfinished earlier result. Reuse the existing demand inference bridge and one
writer owner/model. Persist actual completed audits for resume.

Validation: unit tests for ordered completion, cancellation/padding and resume;
native smoke test/build; resumed real extraction with process start counts and
phase timings. Do not claim continuous scheduling from batch size alone.
Spec update: require global native worklist and native incremental cancellation.
Docs updates: document the execution path and measured timings in task_log.md
and the extraction benchmark docs. Changelog: record the scheduler correction.

## Inputs

Use the completed five-scroll/50-fiber evaluations already recorded in the
benchmark runbook:

1. full-resolution refinement best EMA999 (`5ec278f7...`);
2. current-protocol Classic (`best91.5k`);
3. output-stride-4 refinement best EMA999 u22000 (`67edd3a1...`).

Their failure artifacts are evidence only. The exported dataset must not encode
which model produced a failure; it is a hard-region corpus for future models.

## Extraction algorithm

1. Load each completed evaluation and validate summary completion, annotation
   hashes, source/piece identity, and directed reference arcs.
2. Collect every filtered restart episode from both directions, retaining only
   its source piece and reference arc. Do not run the old quotas or model
   category selection.
3. Convert every episode to a reference interval with 500 base voxels of
   context on either side, clipped to the eligible CP/Lasagna span.
4. Merge intervals per piece when they overlap or are within the existing
   200-base-voxel deduplication distance. Preserve the union geometry and an
   audit list of contributing episodes internally, but omit model identity from
   the public case metadata.
5. Export one forward-oriented GT-only `vc3d_fiber` annotation per merged
   interval. Keep source identity, original piece, reference arcs and endpoint
   provenance; do not export failed trace polylines.
6. Audit full-resolution cache coverage by planning the required corridor around
   every exported GT interval and checking the corresponding presence/normal
   chunk keys. Report covered, missing, and conservatively over-covered keys.
7. Create a stable VC3D view directory containing symlinks to the relevant
   per-source full-resolution `fiber_presence.ome.zarr`, `fiber_nx.ome.zarr`,
   and `fiber_ny.ome.zarr` stores plus a manifest describing the source mapping.

## Implementation

Extend the existing `short_failure_benchmark.py` shared inventory, geometry,
interval and provenance helpers with an explicit full-union extraction mode;
do not copy the extraction logic into a second script. Add a reusable runner
mode/config and a cache-coverage audit that does not infer or mutate data.

## Testing

- unit-test interval clipping, overlap merging, bidirectional arc handling,
  provenance validation and deterministic ordering;
- run the approved native/Python replay tests and the extractor dry-run;
- run the full extraction against the three completed evaluation roots;
- validate the exported manifest, hashes, symlink targets and cache coverage.

## Spec update

Update `planning/specs.md` to define the model-agnostic hard-region union and
GT-only exported annotations. Keep model provenance in the private audit record,
not the public dataset case semantics.

## Docs updates

Add a benchmark document describing the three inputs, deduplication rule,
500-voxel context, cache audit and VC3D symlink view. Link it from the benchmark
runbook.

## Changelog

Add one dated entry for the full hard-region extraction and cache audit.
