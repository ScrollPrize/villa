# CPU chunk-based fiber tracing

`vc/fiber_tracer/FiberChunkTrace.hpp` provides reusable native primitives for
building, caching and matching overlapping local fiber paths. It builds on the
existing beam tracer and explicit prediction-region pinning, not a separate
tracing implementation. This API does not launch inference or allocate GPUs.

## Coordinates and construction

Chunk keys are XYZ core indices. Chunk configuration distances, stored paths,
matching history and initial ellipses use base-volume voxels. The builder's
`traceToBaseScale` converts the prediction source's trace coordinates to base
coordinates. `traceFiberChunkHalf` itself takes trace coordinates and bounds.

For each expanded core-plus-halo region, the builder:

1. Samples a regular presence grid and selects deterministic local maxima.
2. Suppresses nearby seeds and checks bidirectional tube continuity, turning
   angles and contrast against a normal-oriented rim.
3. Traces accepted seeds in both directions using the existing beam search.
4. Stores paths in seed order, with independent stop reasons at both ends.

`replayPresenceCutoff` optionally rejects low-presence beam candidates;
`replayPresenceModulated` selects direction-modulated instead of raw presence.
The default cutoff is zero, preserving ordinary tracing behavior. Chunk bounds
stop expansion when a surviving branch exits. The half-trace returns the chosen
path through its first crossing, retaining overlap for downstream matching.

The optional synchronous parallel-for must invoke each job exactly once and
join all jobs before returning or throwing. Parallel construction is used only
when both sampling sources support concurrent access. Paths remain ordered by
seed, independently of completion order. Prediction-field construction can pin
the prediction and normal region explicitly; custom sources need not pin.

## Cache and matching

`FiberChunkTraceCache` shares a single in-flight build per key among callers of
the same cache object. A failed build propagates to its waiters and can be retried.
The optional scheduler wait callback must wait until ready or throw to cancel.
It permits integration with an existing cooperative scheduler without creating
another worker pool.

Use one output directory and cache instance per immutable prediction/normal
dataset combination. Supply a source identity covering both datasets, model
checkpoints and coordinate transforms. Persisted files include that identity
and construction/scoring settings; incompatible files are rejected rather than
silently reused. Writes are atomic. Independent processes writing the same
directory are not coordinated. The in-memory cache retains requested paths for
its lifetime; callers should scope it to their workload.

`matchFiberChunkPaths` selects an oriented path using monotonic geometric
overlap with the history, distance and direction tolerances, and optionally
predicted-width compatibility. Initial matching uses a positive-definite
ellipsoid metric. Continuations require at least 16 base voxels of remaining
path. The result identifies the line, orientation, segment and arc position.

## Routes and reference replay

`traceFiberChunkRoute` follows those matched suffixes across chunk boundaries.
Supply a synchronous chunk provider (typically an adapter to `cache.get`) and
an observer for committed points. It uses the established 16-base-voxel route
policy and rejects other step sizes. Backward context is used only for overlap
matching, not emitted or counted as travelled distance. The initial point is
snapped inside the supplied ellipsoid, not connected by a synthetic segment.

Nonzero continuation gaps are checked at spacing no greater than two base
voxels using direction-modulated presence and optional width compatibility.
Join diagnostics expose the selected path and splice distance. Presence and
invalid-width terminal stops remain stops even beside a chunk face. Cycle,
no-progress, observer, step and distance limits prevent unbounded traversal;
the final point is clipped when necessary to respect the distance budget.

`traceFiberChunkReferenceReplay` connects this route to
`traceReferenceReplayCore`, which remains responsible for matching, metrics,
failure accounting and resets. It constructs the initial ellipsoid from the
same anisotropic acceptance metric and snaps the initial recorded point too.
By default, restart search checks reference positions every 16 base voxels for
a cached path; callers can supply their prediction-derived seed search in
`ReplayConsistencyPolicy::nextSeed`. Absent paths count as skipped reference
length rather than fictitious traces. Reverse-check and acceptance callbacks
are retained, and the reference is never used to guide route continuations.
Tracing normals and evaluation normals have separate explicit coordinate
scales. Progress callbacks report route steps with shared replay context.

Full-volume orchestration, Python bindings and experimental refinement/conflict
filters are not part of this layer.

## Validation

With an existing native CMake build:

```sh
cmake --build <build> --target test_fiber_chunk_trace -j 8
<build>/bin/test_fiber_chunk_trace
```

Tests cover deterministic parallel construction, non-winning boundary branches,
presence stops, shared builds, disk reloads, configuration mismatches, retry,
invalid settings and oriented/width-aware overlap matching. An optional real-data
case uses `VC_TEST_WIDTH_MANIFEST`, `VC_TEST_WIDTH_FIBER` and
`VC_TEST_REVERSE_NORMAL_MANIFEST`: cached predictions and normals plus a fiber
JSON whose first control point selects the test chunk. No GPU inference runs.

Route regressions also cover forward/reverse joins, observer and exact distance
limits, semantic stops, unsupported splices, invalid widths, reference resets,
skipped seed locations and consistency callbacks. A GCC13 Release check on
Paris4 reviewed piece `0039_anon_20260815T025206849_000041_run0.json`, prediction
checkpoint prefix `a56e9dc2`, produced 17 maxima and 10 paths in the seed chunk.
Construction was exactly equal with serial/parallel work and pins on/off.
Reference replay over 256 base voxels selected four chunk paths and returned
one segment with zero failures; disk-cache replay reproduced the points and
termination reasons exactly. This is a correctness check, not a tracing-quality
benchmark or a performance claim. macOS/arm64 have not been validated locally.
