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

Full-volume/live-inference orchestration and experimental refinement/conflict
filters are not part of this layer.

## Python

The same implementation is available in `vc.fiber_trace`. Dataset handles are
retained by `ChunkCache`; it copies both configurations at construction. Native
construction, routing and replay release the GIL. Multiple Python threads may
request chunks from the same cache, sharing in-flight builds. Returned NumPy
arrays own their data and remain valid after the cache or result is destroyed.
Input point/metric arrays must be contiguous CPU float64 arrays.

```python
import numpy as np
from vc import fiber_trace as ft

field = ft.open_prediction_field("predictions.lasagna.json", scaledown_power=0,
                                 enable_width=True)
normals = ft.open_normal_sampler("normals.lasagna.json", field.trace_to_base_scale)
trace = ft.TraceConfig(step_voxels=16 / field.trace_to_base_scale,
                       trace_to_base_scale=field.trace_to_base_scale,
                       replay_presence_cutoff=0.1,
                       replay_presence_modulated=True,
                       width_relative_tolerance=0.4, width_cosine_weight=True)
chunks = ft.ChunkConfig()
chunks.join_width_weight = True
cache = ft.ChunkCache(field, normals, trace, chunks, "chunk-path-cache",
                     "immutable-model-and-normal-dataset-identity")
paths = cache.get([0, 0, 0])  # XYZ key; paths.lines_base is a list of arrays

# reference_xyz is the application's (N,3) polyline in base voxels.
evaluation = ft.ChunkReplayConfig()
result = cache.replay(np.ascontiguousarray(reference_xyz, dtype=np.float64),
                      normals, evaluation, reverse_interval_base=0)
for segment in result.segments:
    print(segment.termination_reason, segment.points_base.shape)
print(len(result.failures), result.seed_search_skipped_base, result.filtered_stops)
```

Width sampling is opt-in; omit `enable_width` and keep width tolerance/join
weight disabled for models without a width channel. The source identity must
identify immutable model, volume, normal data and coordinate transforms, not
just a generic model name. Existing callers of `open_prediction_field` retain
their previous behavior by default.

`cache.route(center_base, direction, metric, maximum_distance_base,
maximum_steps)` traces without GT evaluation. `metric` is the positive-definite
3x3 ellipsoid matrix; the result dictionary contains `points_base`, `reason`
and typed `joins`. Points exclude the initial snap, available as
`joins[0].match.point_base` when a path was found. Native cached routes do not
retain cumulative beam losses; replay's cumulative losses are consequently zero.

Replay accepts separate evaluation normals, optional begin/end base arclengths,
and `reverse_interval_base`. Zero disables reverse checks. A positive interval
uses the existing native reverse verifier and the replay acceptance radii;
rejected spans appear in `filtered_stops` rather than ordinary failures.
The Python wrapper does not add a worker pool, prediction readiness callback or
arbitrary Python scheduler/observer callbacks. Such live orchestration remains
separate from this offline API.

## Validation

With an existing native CMake build:

```sh
cmake --build <build> --target test_fiber_chunk_trace -j 8
<build>/bin/test_fiber_chunk_trace
cmake --build <build> --target vc_fiber_trace -j 8
python -m pytest volume-cartographer/python/tests/test_fiber_trace.py \
  volume-cartographer/python/tests/test_fiber_chunk_trace.py \
  --vc-build-python <build>/python
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
