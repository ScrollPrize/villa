# Task log: durable Fiberlet crop trace artifacts

## Discovery

- The existing preprocessing route payload cannot exactly represent an
  arbitrary complete crop trace: it stores int16 transverse coordinates on a
  curved endpoint-derived layer sequence. Reusing it would quantize geometry
  and constrain point count/longitudinal placement.
- The existing Fiberlet dataset envelope already provides strict metadata,
  sparse chunk semantics, content identity, atomic chunk publication,
  checksums, structure-of-arrays field blocks, and per-field Zstd. The task can
  extend this shared implementation with one trace payload instead of adding a
  parallel JSON or Zarr implementation.
- Current `FiberletCropTraceLine` has geometry and seed data but no selected
  route cost. Cost must be accumulated while each chosen edge/join is still
  available.
- Existing direction and anchor OBJ generation consumes in-memory lines. It
  must be moved behind a write/reopen boundary.

## Decisions

- Store float64 base XYZ so the durable artifact does not introduce a geometry
  precision change relative to the current `cv::Vec3d` output.
- Store total metric cost and prediction-space path length separately. The
  displayed quality is their density, which is comparable across path lengths.
- Use rank deciles with deterministic ordinal tie-breaking; each trace appears
  in exactly one quality OBJ.
- Use a trace-only `float64_traces` profile and canonical crop-local base-grid
  metadata instead of assigning float64 traces the existing float cache label.
- Publish a fully validated temporary root by atomic rename. The trace metadata
  inventories nonempty chunks and record count so missing sparse chunks remain
  meaningful while an incomplete finite artifact is rejected.
- Rank bin `r` is `floor(10*r/N)` (capped at nine); CSV/table values include
  both total cost and normalized cost density.

## Independent review

- The review required atomic completeness, canonical trace-specific metadata,
  global ordinal/ownership validation, explicit trace contract identity,
  unambiguous CLI syntax, exact decile behavior, and partial/empty/boundary
  tests. All findings were incorporated before implementation.

## Deviations

- The canonical crop exposed existing independent-side behavior in
  `selectInitialPair`: if no joined initial pair exists, it can still trace one
  negative and one positive edge independently. Preserving tracing behavior
  means such a bidirectional output has no central join penalty to store. All
  edge and internal-join costs remain defined and are stored; a central join is
  added only when `transition()` returns one. Tightening acceptance is outside
  this artifact refactor.

## Validation

- GCC Release built `vc_fiber_trace_chunk`, `test_fiberlet_storage`, and
  `test_fiberlet_crop_trace` successfully.
- `test_fiberlet_storage`: 38 test cases passed.
- `test_fiberlet_crop_trace`: 14 test cases passed.
- The canonical Paris4 1024-base-voxel crop produced 500 stored traces in all
  eight crop-local trace chunks. All pre-existing all/direction/anchor OBJ
  files were byte-identical to the pre-refactor run.
- Standalone `visualize` reopened that dataset and regenerated all eight
  existing OBJ files, ten quality-decile OBJs, and the histogram CSV
  byte-for-byte in 0.09 seconds (`user=0.08`, `sys=0.00`, peak RSS 30,980 KiB).
- Clang Debug built the same executable and focused tests. Its
  `test_fiberlet_storage` and `test_fiberlet_crop_trace` runs also passed 38
  and 14 test cases respectively. The only warning was the pre-existing
  ignored `nodiscard` result in `FiberReplay.cpp:1659`, outside this change.
- `git diff --check` passed.
