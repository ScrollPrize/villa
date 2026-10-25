# Chunk-based fiber tracing

`vc/fiber_tracer/FiberChunkTrace.hpp` provides reusable native primitives for
building, caching and matching overlapping local fiber paths. It builds on the
existing beam tracer and explicit prediction-region pinning, not a separate
tracing implementation. It does not launch inference or acquire GPU-manager
grants. CPU construction remains the default; the optional CUDA plugin uses
only the devices already granted to its caller. This guide covers the current
implementation and APIs, not a prescribed benchmark configuration. Model paths,
tuned cohort configurations, results and machine-specific launchers belong in
external experiment archives. Skeleton/BP solvers and model training are separate
from this API.

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

### First step and sampling margin

`traceFiberChunkHalf` sets `preserveInitialDirection=true`. The supplied seed
axis (or refined axis when seed fitting is enabled) initializes the tracing
direction rather than being replaced by the prediction at the seed. It supplies
the initial direction context for scoring; it does not force a straight first
step. The ordinary beam search still chooses that step. Forward and backward
halves start with opposite seed directions.

For native prediction and normal fields, `buildFiberChunkPaths` pins the expanded
core-plus-halo region with an additional sampling margin on every side:

```text
margin_base = stepVoxels * max(1, beamLookaheadSteps) * traceToBaseScale
if online refinement is enabled:
    margin_base += onlineRefineCapBase + onlineRefineEpsilonBase
```

This extra margin is for speculative beam samples and refinement probes, not an
extension of the committed trace bounds. The pinned region is sealed before seed
tracing, allowing read-only sampling without repeated shared-cache lookups.
Custom sources that cannot provide a native pinned region retain their own
sampling behavior. Each CPU seed job uses one internal tracing thread; the
caller-provided parallel-for distributes independent seed jobs instead of
creating nested per-seed thread pools.

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

New construction writes one `x_y_z.msgpack.zst` file per chunk: the existing
document schema serialized losslessly as MessagePack and compressed with Zstd
level 3. Coordinates retain their numerical values; no quantization is added.
Files contain paths, endpoint stop reasons, bounds, seed counts, provenance,
settings and timings, but **not per-point widths or tracing costs**. Width-aware
joins sample the prediction source again. Files retain all paths. When enabled,
conflict filtering runs before publication and records a `conflict_flags` boolean
array plus `conflict_annotation` (versioned settings/source identity and report).
Geometry, stable indices and endpoint reasons are never removed by annotation.

Readers prefer `.msgpack.zst`, falling back to the old `.json` filename only
when the compressed file is absent. Existing JSON caches are not automatically
rewritten or deleted. A corrupt/incompatible compressed file fails explicitly,
even when an older JSON file exists. `loadFiberChunkPaths` accepts either format;
external directory enumerators must include both extensions. Compressed cache
frames must declare their decoded size (at most 1 GiB) and contain exactly one
Zstd frame and one MessagePack document. This is a serialization change, not a
change to the construction identity or tracing algorithm. Legacy import remains
a JSON-preserving provenance operation; new native construction uses the
compressed format.

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

Live-inference orchestration remains separate. The whole-volume command below
consumes completed predictions. Optional native seed refinement, CUDA online fixed-plane refinement and conflict
filtering are described below.

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
skipped seed locations and consistency callbacks. The opt-in real-data test
compares serial/parallel construction, pins on/off, and disk-cache replay for
identical points and termination reasons. Record input identities and measured
results with the external experiment, not in this reusable usage guide.
This is a correctness check, not a tracing-quality benchmark or a performance
claim. macOS/arm64 have not been validated locally.

To additionally validate serialization against an existing real JSON path cache:

```sh
VC_TEST_CHUNK_PATH=/path/to/x_y_z.json <build>/bin/test_fiber_chunk_trace
```

This checks exact decoded geometry, stop reasons and metadata through
MessagePack/Zstd and rejection of a truncated frame. The ordinary tests also
exercise production compressed writes, disk reload, provenance mismatch,
corrupt-file rejection and read-only legacy JSON compatibility.

## Settings and units

All chunk distances, movement caps and filter ellipses are in **base voxels**.
TraceConfig step sizes remain in trace coordinates. Optional matching, refinement and conflict-filter features default off; CPU construction is the default.
`ChunkConfig.to_json()/from_json()` round-trip the settings; unknown keys fail.

- `initial_alignment_power`: 0 preserves nearest matching, 1 cosine, 2 cosine².
  Optional initial normal/tangential radii override only the initial path search.
- `joint_join_score` and `join_weights`: overlap, width, distance and direction.
- `refine_seed`, `seed_movement_base`: bounded location/direction/width fitting
  before CPU tracing. A successful fit initializes that seed's width mean.
- `conflict_normal_base`, `conflict_tangent_base`: both positive enable filtering.
  Shared and separated run lengths are configurable (`conflict_agreement_base`,
  `conflict_separation_base`); sampling 8, angle gate 25°, endpoint guard 16.
  Both entire paths in a detected conflict are flagged; matching skips them
  while raw geometry remains available. It does not trim the conflicting span.
- `cache_only`: reject absent construction chunks; never silently retrace them.
- `presence_peak_restarts`: select the historical aligned local presence peak
  before matching a cached path, including explicit no-initial-path stops.
  Default false retains the generic cached-path seed search. For direct native
  `replay-set`, the corresponding top-level option is `prediction_based_seeding`.
  Peak selection rejects invalid/nonpositive width when width modulation is
  enabled, rather than selecting a seed that the first trace step cannot use.
- `gpu_library`, `gpu_devices`: explicit plugin and granted visible device count.
- `online_refine_iterations`: defaults to 0 (disabled, recommended); positive
  values explicitly enable experimental refinement after each commit.
  Window 0 includes the whole committed path, positive values limit its tail.
  Each point's transverse plane stays fixed from its commit. Epsilon, step and
  displacement cap are explicit. This is experimental, not a default promotion.

Construction cache identity includes construction/trace settings and source
identity; matching/filter changes do not require constructing paths again.
The builder saves **intact** paths and separately versioned conflict flags.
Missing/stale filter annotations are refreshed atomically without retracing.
Older cache specifications are upgraded in memory with disabled defaults
for new construction options; mismatches still fail. Compatible fully annotated
files are not rewritten. Legacy JSON gains a preferred MessagePack+Zstd sibling
when annotation is required; the original JSON is retained.

## Evaluation and cached import

Use the existing [callable evaluation](fiber_evaluation.md) config with
`chunks` settings. It retains the shared queue and live-inference boundary:

```sh
python -m vc.fiber_evaluation evaluation.json results.json
```

For the native CLI, `vc_fiber_trace_metric replay-set --config evaluation.json`
accepts `backend: "chunk-beam"`, `chunk_trace` settings, and per-source
`chunk_cache` and `prediction_identity`. Beam remains supported unchanged.
No GPU is needed to replay existing paths against cached prediction chunks.

Forward replay retains candidate presence filtering inside beam selection;
committed-step checks alone cannot prevent infeasible speculative candidates
from displacing feasible paths. Reverse consistency checks retain width and
direction scoring but do not inherit the forward presence cutoff: they measure
geometric reversibility.

The optional `benchmark_fiber_chunk_cuda --device-tests` runs standalone CUDA
selector and two-stage tracing regressions without a dataset. Run it with
`compute-sanitizer --tool memcheck --error-exitcode 9` inside a granted GPU job.
It must be built with optimization as well as tested with real cold-cache
inputs; host-only scoring tests cannot validate device memory safety.

The CUDA winner reduction synchronizes after all warps snapshot its shared
winner, before the next reduction reuses scratch. Its scoring helper takes the
immutable parameter block by reference: the previous second by-value copy
caused invalid local-memory reads in optimized CUDA 12.8/H100 execution at the
second expansion. Keep the device regression in addition to host math tests.

Old experimental chunk files without the current specification need **explicit
provenance attestation**, not an unchecked compatibility switch:

```sh
python -m vc.fiber_chunk_benchmark import-legacy provenance.json NEW_DIRECTORY
```

The version-1 manifest contains `specification` from
`ft.chunk_cache_specification(chunks, trace, scale, source_identity)` and `files`
entries containing `path` and `sha256`. The source identity must identify the
immutable predictions, normals and transforms. The caller must verify the
attested construction settings against the original run. Import checks hashes,
native geometry and bounds, rejects duplicate keys and existing destinations,
and writes new copies. It cannot establish missing provenance by itself.

For historical native replay documents, keep using
`vesuvius.neural_tracing.fiber_trace_3d.replay_metrics` for minimum-piece pruning,
retained percentage, mean/median and restart-episode accounting. Do not interpret
all `failures` as GT-distance errors: report `distance_above_threshold`,
`chunk_no_initial_path`, `chunk_no_overlap`, limits and other reasons separately.
Filtered stops remain separate. The callable evaluator exposes raw counts and
geometry, not an implicit cohort-specific score.

## Resumable whole-volume construction

`python -m vc.fiber_chunk_volume CONFIG.json OUTPUT_DIRECTORY` constructs and
annotates every occupied tracing core in a **completed local prediction volume**.
It requires no GT fibers and never launches inference. Use the same command to
resume: compatible outputs are validated and skipped without rewriting them.

For an existing workspace build, without modifying an installed Python package:

```sh
python volume-cartographer/scripts/run_fiber_chunk_volume.py CONFIG.json OUTPUT_DIRECTORY \
  --workers 224
```

This wrapper selects this checkout's bindings even if another checkout is
installed editable. `VC_BUILD_DIR` overrides `volume-cartographer/build/fiber-evaluation`.
It does not build or install anything. Relevant build targets are `vc_volume`,
`vc_fiber_trace`, and, for CUDA, `vc_fiber_chunk_cuda`.

```sh
python -m vc.fiber_chunk_volume paris4.json /data/paris4-paths --plan
python -m vc.fiber_chunk_volume paris4.json /data/paris4-paths \
  --workers 224 --vram-gib 8 --priority 100
```

Configuration schema (paths resolve relative to the config; trace/chunk tuning
must be chosen explicitly for the intended baseline, not inferred from a model):

```json
{
  "prediction_manifest": "/data/prediction.zarr",
  "normal_manifest": "/data/normals.lasagna.json",
  "source_identity": "immutable-prediction-and-normal-dataset-identities",
  "scaledown_power": 0,
  "enable_width": true,
  "cache_bytes": 1073741824,
  "loader": {"resource_scaled": true},
  "trace": {
    "parallel_threads": 1,
    "step_voxels": 16,
    "beam_width": 8,
    "beam_lookahead_steps": 2
  },
  "chunks": {
    "channel_hints": true,
    "worker_cache_bytes": 0,
    "core_base": 512,
    "halo_base": 64,
    "grid_base": 4,
    "conflict_normal_base": 10,
    "conflict_tangent_base": 20,
    "conflict_agreement_base": 64,
    "conflict_separation_base": 32,
    "online_refine_iterations": 0,
    "gpu_library": "/path/to/libvc_fiber_chunk_cuda.so"
  }
}
```

`step_voxels` is in trace coordinates, **not base voxels**. Omit it to use
16/base-to-trace-scale automatically. The displayed value 16 is only correct
when trace coordinates equal base coordinates. `scaledown_power` is the existing
prediction-to-trace coordinate adapter, not an extra downsampling operation.
CPU construction uses an empty/omitted `gpu_library`. Omit `normal_manifest` only
when predictions actually contain `sheet_nx/sheet_ny/sheet_grad_mag`; native
normal aliasing preserves packed channel indices. Missing normal heads fail
explicitly. This example illustrates the schema, not a tuned benchmark preset.

Sparse enumeration supports local Zarr v2 ZYX/CZYX arrays, dot/slash chunk keys,
packed channel chunks and partial edge chunks. It maps the **presence** group's
actual physical spacing to tracing cores; it does not assume storage and tracing
chunks match. Missing chunks must mean zero fill in a completed sparse volume.
The command cannot distinguish intentional sparse fill from unfinished inference
or a partially copied volume. Nonzero fill and unsupported layouts fail. Existing
native top2 pooled views enumerate their source presence inventory and retain
the existing C++ pooling implementation. Remote inventory listing/Zarr v3 is not
implemented. `keys_xyz` optionally supplies an explicit unique pilot/job list.

One `FiberCaseQueue` continuously schedules all keys. Native preparation,
beam search, filtering and publication release the GIL; Python only handles
metadata, admission and progress. The existing 128-job GPU preparation admission
budget and per-device batches remain unchanged. Completed chunk geometry is
released immediately—there is no full-volume cache of results or futures.
`--contexts-per-worker` defaults to 8; worker threads are shared with native
within-chunk work. Dataset/pooling caches remain bounded independently.

### Owned ROI tensors for GPU chunk preparation

The native single-option GPU path loads prediction and normal channels once into
an owned `detail::FiberChunkTensor`, using the shared `Volume::readZYX` ROI reader.
It does **not** pin a region and then keep querying the shared chunk cache.
Presence-grid construction, seed/tube validation, width/normal interpolation,
and GPU packing use those immutable local arrays. Cache synchronization and I/O
occur during ROI loading (presence first, companions when needed), not individual
seed/tube samples. Independent chunks run on the existing
shared `FiberCaseQueue`; ROI loading suppresses nested OpenMP teams so it does
not create another full worker pool per chunk.

The shared ROI reader acquires each intersecting storage chunk **once**, then
copies its clipped intersection into the destination. It does not reacquire the
same cache entry for every output row. These intersections have disjoint writes;
the standalone reader can parallelize their copies, while chunk construction
parallelizes across independently owned ROIs. Missing-chunk fill, outside-volume
padding, packed channel offsets and exact copied values retain their prior semantics.

The loaded extent covers the beam lookahead/refinement margin and one additional
native voxel for upper interpolation corners. The GPU grid itself is unchanged.
Scalar samples retain the original double-coordinate tensor interpolation;
batched seed samples retain float-coordinate interpolation. Byte/float width
units, presence-masked width interpolation, and normal validity are unchanged.
The scalar/tensor interpolation helpers are shared with the general sampler,
not separate approximations. Packed-channel ROI reads honor `ChunkResult.byteOffset`.

Direction decoding is skipped for grid samples below the seed threshold; eligible
samples still undergo the original validity checks before NMS. Chunks with no
accepted seeds skip GPU input packing and GPU execution entirely. Timings add
`roi_load_seconds`, `seed_seconds`, and `gpu_pack_seconds`; construction time can
still include queue waits and is not equivalent to CPU time. Empty chunks have
zero GPU times and device `-1`.

This optimized path requires the same aligned native single-option prediction,
width and normal channels as the GPU backend. Generic/synthetic CPU sources keep
their existing sampler path; they are not silently converted. Trace parameters,
cache identity, filter semantics, and stored geometry are unchanged.

Measured on the completed Paris4 packed seven-channel volume (2026-10-25), using
the same settings and a separate RelWithDebInfo build while the original job
continued:512 tracing cores,176 nonempty,1028paths,48 granted CPU workers and six
shared GPUs. Two-run mean wall time fell from25.50s to6.84s (3.73×); all seeds,
coordinates, end reasons and conflict flags matched exactly. Small64-core scaling
tests at1/4/16/48workers improved4.55×/4.89×/4.23×/2.42×. This is not a fresh
full-volume timing or a claim of GPU saturation: average host use was29.5of48cores,
and loading/seed preparation remains the limiting work. Full settings, commands,
raw samples and limitations are in
`/ephemeral/hendrik/fiber3d_train2/evaluations/dense_chunk_preparation_20261025_v1/REPORT.md`.

The subsequent sparse-grid optimization loads presence first, precomputes the
regular grid's axis coordinates/corner offsets/fractional weights, and groups
sites sharing native interpolation corners. A conservative maximum-corner bound
skips cells that cannot reach the seed threshold. Remaining samples use the same
ordered float FMAs and OpenCV reciprocal multiplication/rounding as the point
sampler; the culling slack never changes acceptance thresholds. Direction validity
is checked before NMS. Companion channels load synchronously before any parallel
seed validation, or remain unloaded when no site qualifies (except threshold0,
whose existing zero-valued NMS requires safe companion arrays). Packed storage
may still decompress all channels when reading presence; this is not an I/O
compression change. The presence-only tensor rejects premature companion access.

`seed_grid` timing metadata separates setup, presence, companion loading and
direction validation, with native wall/thread CPU times and pruning counts.
`roi_load_seconds` now measures only the initial presence load; deferred loads
are included in `seed_seconds`. Opt-in `VC_SAMPLING_WAIT_PROFILE=1` also records
the existing cache's wait counters under each grid stage. These nested counters
must not be summed indiscriminately. Thread CPU is -1 if unsupported by the OS.

Matched profiling (12chunks ×3repeats) reduced seed-grid time228.0→25.7ms,
including deferred loads, with exact float-grid equality. Balanced64chunk
end-to-end wall times improved18.66→5.62s with1worker and4.86→2.06s with4.
At48workers, however, sustained512chunk time was unchanged6.20→6.27s; process
CPU use fell28.29→13.86core equivalents. Companion loading is now wait-dominated:
the diagnostic run attributes258.6ms/chunk to payload availability versus0.43ms
to cache-lock acquisition. This does **not** establish physical-disk latency or
a full-volume speedup. Source uses the unchanged1GiB decoded-cache budget;
cache churn/decode scheduling need separate investigation. Exact paths, seeds,
end reasons and flags match; no tracing tuning or resource budgets changed.
Full before/after data, commands and limitations:
`/ephemeral/hendrik/fiber3d_train2/evaluations/sparse_seed_preparation_20261025_v1/REPORT.md`.

Reusable native profiling target: `bench_fiber_seed_grid CONFIG INPUTS OUTPUT
[point]`; default uses sparse preparation, `point` uses eager loading and the
independent per-point reference. This is a preparation microbenchmark, not a
replacement for the manager-controlled end-to-end tracing benchmark.

### Channel hints and cache scheduling

`channel_hints=true` supplies the generic core read hint for the channels used
by this construction. Reading presence schedules the other requested channels
into their **ordinary decoded caches**, so subsequent ordinary reads reuse them.
This is not a tracer-specific retention cache; see
[generic channel-hint and partition APIs](chunk_channel_hints.md).

The batch runner defaults to `loader.resource_scaled=true`: after manager
admission it sizes local reads to `max(16, 2*granted cores)` and decoding to the
granted core count. Metadata preflight exits before those execution pools exist.
`resource_scaled=false` explicitly retains legacy host/benchmark pool settings.

`loader.backend="queued"` selects the experimental generic batch admission
backend; `"direct"` remains the default. `loader.queued_chunk_window=0` sizes the
in-flight unique-key window from the granted read/decode capacities; a positive
value overrides it. It coalesces concurrent ROI/hint requests and refills the
pipeline on completion. Use shared `cache_bytes` (e.g. 10 GiB), with
`chunks.worker_cache_bytes=0`; queued/local partitions cannot be combined.
Window and decoded-cache budget do not cap caller-owned ROI/tensor memory.
See [the queue API and limitations](chunk_channel_hints.md#optional-queued-batch-loader).

`worker_cache_bytes=0` retains the large shared decoded cache. A positive value
selects that many bytes per native preparation worker, using generic decoded
partitions which still share the central I/O/decode pools. On the locked
512-core Paris4 workload, 64 MiB per worker was faster than the equal-aggregate
shared cache; this remains an opt-in experiment, not a global viewer default.
Hints and partitions do not alter construction identity or numerical settings.
Separate companion caches may load even if presence later rejects all seeds;
packed storage already loads those companion bytes together.

Measurements, exact output parity, commands and limitations:
`/ephemeral/hendrik/fiber3d_train2/evaluations/channel_loader_20261025_v1/REPORT.md`.
This extends the earlier sparse-grid timings above; it does not retroactively
claim that sparse seed preparation alone improved sustained throughput.

If new CUDA construction is needed, the command bootstraps/uses the existing GPU
manager and requests a nonexclusive partial grant. By default all GPUs in the
manager inventory are eligible; the manager chooses which available devices to
grant. `--devices` is an optional explicit physical-index restriction, not a
default fixed selection. The worker uses only granted devices/CPU count. The workspace budget is capped to the granted VRAM
minus a 1 GiB context/field reserve. Reuse/filter-only runs request no GPUs.
Cancellation joins admitted work before resource release; blocking filesystem
operations cannot be interrupted immediately. The managed writer owns the
output-directory lock, including during native work.

Resume checks compressed payload integrity, construction/source identity, key,
expected core bounds, geometry/counts and annotation consistency. Corrupt or
construction-incompatible outputs fail without replacement. Missing/changed
filter settings cause annotation only; full geometry is preserved. No done
markers/database are used; atomic `.msgpack.zst` output is the completion record.
The persistent `.writer.lock` is an advisory process lock, not completion state.
Direct API callers must provide equivalent single-writer coordination.

Progress reports completed/total, built, reused and reannotated counts, elapsed
time, completion rate and an extrapolated ETA. A mixed validation/construction
resume has unequal job costs; its early ETA is provisional. Summary JSON is
printed at completion. Construction/filter per-chunk timings live in the outputs.

## Microbenchmarks

Time native construction with caller-owned dataset handles, no path-cache IO:

```python
from vc.fiber_chunk_benchmark import time_construction
report = time_construction(field, normals, trace, chunks, [0, 0, 0],
                           workers=(1, 4, 16, 64), repeats=3)
```

An untimed warmup fills dataset caches. Timed runs rebuild paths and include
queue setup, pinning, seed processing and tracing. Worker-count parity checks
exact paths and stop reasons. Returned stage timings are backend-specific;
do not infer pure tracer time by subtracting unrelated runs.

Time only conflict filtering of already loaded path geometry:

```sh
python -m vc.fiber_chunk_benchmark time-filter \
  --normal-manifest normals.lasagna.json --config filter.json \
  --trace-to-base 1 --workers 1 4 16 64 --repeats 3 \
  --output filter-timing.json chunk-path-cache/0_0_0.json
```

Reports min/median/max/mean, exact rejected-ID parity, and throughput. Normal
data are warmed but not pinned: any subsequent normal-cache miss is included.
Neither helper starts inference or acquires resources on the caller's behalf.

## CUDA build and limitations

CPU builds do not require CUDA. With existing dependencies and a compatible
CUDA host compiler, configure your normal build with:

```sh
cmake -S volume-cartographer -B <build> -DVC_BUILD_FIBER_CHUNK_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build <build> --target vc_fiber_chunk_cuda benchmark_fiber_chunk_cuda -j 8
<build>/bin/benchmark_fiber_chunk_cuda --host-refinement-tests
<build>/bin/benchmark_fiber_chunk_cuda --host-gradient-tests
```

Supply the resulting shared library via `gpu_library`. GPU execution requires a
manager grant owned by the caller; it does not obtain a grant itself. Persistent
device workers are keyed by library and visible device count. Admission covers
**pinning, host preparation and execution**, with 128 admitted chunks, not just
device launches. Wait callbacks cooperate with the shared queue.

The CUDA kernel supports beam8/lookahead2/parent-cap32, 81 angular
candidates, an even step budget up to 160, one prediction option, unfrozen width
and modulated cutoff. Unsupported settings fail explicitly. CPU seed fitting
and embeddings are not implemented in this GPU backend. Online refinement is
GPU-only. Defaults and unsupported settings must not silently fall back.
It uses float arithmetic with FMA disabled, not a CPU/GPU bit-identity promise.

The native C++ `benchmarkExport` option writes an explicit single-chunk fixture
for the standalone CUDA benchmark. Use a distinct directory per chunk; it is
not a shared export destination for concurrent builds. GPU execution is separate
from the host-only refinement tests.

## Native pooling views

`openLasagnaChannelArray` recognizes `.native_pool.json` beside an output
`.zarray`. Recipe revision 1, factor 2, mode `top2_spatial_comp` uses the original
spatial-companion weighting, float widths and axial tensor direction/normal
averaging. `source`, `source_level` and/or a `channels` mapping are explicit;
channel paths may be relative to source. Compatibility default level is 1.
Seven required channels: presence, nx, ny, width, sheet_nx, sheet_ny,
sheet_grad_mag. Output width metadata must be `<f4`; other channels `|u1`.
Output is raw, C-order, slash-separated chunks with half the source chunk size
and ceiling-half shape. Source chunks must be even-sized and aligned.

Views share a bounded 8 GiB RAM bundle cache and coalesce concurrent requests.
No pooled chunk files or scratch files are written. Missing source data raises
an error: readiness/inference belongs to the existing producer boundary.

Eigen is portable. On supported Linux builds with LAPACK, the rare
degenerate/hemisphere-edge fallback uses DSYEVD. Without it, Eigen handles all
cases. **Byte-identical pooling across these solver policies is not promised**;
record the build/solver in prediction provenance when exporting or constructing
path caches. Do not reuse one source identity for different numerical policies.
macOS/arm64 runtime validation remains outstanding.

## Conflict filtering versus reverse retracing

These are independent mechanisms, not two names for the same check:

- `inspectFiberChunkConflicts` compares already constructed paths. A spatial
  index restricts candidate pairs. It looks for a sustained shared run and a
  sustained separated run, using normal-oriented ellipses, direction agreement
  and continuous monotonic correspondence. The cache saves flags for both whole
  conflicting paths, and matching skips flagged paths. The older standalone
  `filterFiberChunkConflicts` API remains an explicitly destructive wrapper for
  callers wanting an emptied copy; both use the same calculation.
- Replay's `reverse_interval_base` / `ReplayConsistencyPolicy` performs actual
  reverse tracing to check geometric reversibility. Rejected spans are filtered
  stops, separate from reference-distance failures. This uses additional tracing
  work, unlike comparing cached paths.

Changing conflict thresholds or join settings does not require rebuilding the
unfiltered path cache. Construct a cache/view with the new configuration; an
existing cache object retains its already loaded view.

The filter's broad phase uses an immutable, chunk-wide **Boost.Geometry R-tree**
of segment bounding boxes (the same spatial-index library already used by
VC3D's point/surface indexes). A candidate path must have box intersections for
enough consecutive query samples to possibly satisfy the shared-run criterion.
This only excludes impossible pairs: nearest-segment projection, tie order,
ellipse/direction acceptance and monotonic correspondence checks are unchanged.
All queries share the read-only index; per-path query state is worker-local.
`index_seconds` and `match_seconds` in the filter report distinguish index
construction from sampling/comparison work.

On three refinement-disabled Paris4 chunks (1,673–2,449 paths), three warmed
repeats in RelWithDebInfo/GCC13 measured 4.2–6.4× faster filtering with one worker,
1.5–2.7× with 16 workers, and 1.1–1.2× with 64 workers, including index construction
and API overhead. Full directed conflict pairs matched the original filter.
Per-path ordered-nearest and bounded-nearest R-tree variants were tested but
were slower on these short paths; neither is used. These are small-sample
filter-only timings, not full-volume or concurrent-chunk throughput measurements.
The remaining cost is mostly sampling/comparison, not tree construction.

## GPU execution and buffer reuse

`FiberChunkGpuBridge.cpp` owns a shared queue and one persistent host worker per
visible, caller-granted GPU. Its 128-chunk admission budget covers host pinning,
preparation and queued/executing work, independently of GPU count. Each device
worker runs one chunk job at a time; host preparation can proceed ahead through
the caller's scheduler. This is not 128 simultaneous kernels per GPU.

The CUDA plugin uploads a chunk's prediction fields once per job, then traces
both directions of its seeds in bounded batches. `gpu_batch_halves` defaults to
8,192; `gpu_batch_memory_mib` defaults to 2,048 MiB. The effective batch is the
minimum of that count, the workspace budget and half the currently free device
memory, accounting for per-half nodes, candidates and output buffers. These
are execution controls, not path-cache identity fields. Each
half-trace gets one CUDA block of 128 threads; blocks process different
half-traces concurrently while threads cooperate on beam candidate scoring and
selection. The chunk fields stay resident across its batches.

Each persistent worker has a reusable device workspace for fields, seeds,
candidates, nodes and outputs. Buffers grow when necessary and are reused for
later jobs; the direction tensor lookup table is initialized once per workspace.
This reuses allocations, not prediction contents across unrelated chunks.
Results are downloaded after each batch. See `vc_chunk_gpu_run` and the `trace`
kernel in `FiberChunkGpu.cu`.

## Direct analytic online refinement

**Experimental; disabled by default and not recommended for normal tracing.**
Keep `online_refine_iterations=0`. The implementation is retained for research,
but the tested variants did not establish a general tracing-quality benefit.
Selecting `online_refine_mode="analytic"` alone does not enable refinement;
an explicit positive iteration count is required.

On the frozen 91-fiber / 182-direction benchmark (last-four-node refinement,
unchanged snapshot and conflict/short-piece filters), the results were:

| Shared-prefix refinement | GT errors | Retained distance | Retained pieces |
|---|---:|---:|---:|
| Disabled | 0 | 81.83% | 296 |
| 1 iteration | 0 | 81.55% | 295 |
| 8 iterations | 1 | 81.39% | 299 |
| 64 iterations | 1 | 81.44% | 302 |

Rolling per-beam refinement also failed to improve these quality results.
A separate one-chunk audit confirmed that refinement genuinely reduced its
paired local objective (9.92%, 17.82%, 24.44% at 1/8/64 iterations); a quarter-sized
trial step at 64 iterations gave 24.39%. Lower objective cost therefore did not
translate into better reference tracing. Backtracking rejects non-improving
proposals; simply increasing iterations is not supported by these results.
This is evidence from the tested data/settings, not a claim that refinement can
never help. Default configuration, cache behavior and explicit opt-in remain
unchanged.

`online_refine_mode` selects `legacy` (compatibility default) or `analytic`.
Both are disabled when `online_refine_iterations=0`. Refinement runs after each
commit, before subsequent beam continuation; it is not an end-of-trace pass.
The supported iteration range is 0–64. Positive values are maximum iterations
per refinement call: optimization stops early when no feasible cost-reducing
update is accepted. Increasing this limit does not change the movement cap.
`online_refine_window=0` moves the full committed prefix except its seed;
positive N moves only the last N vertices. Each vertex's transverse plane and
plane origin are fixed at its first commit. Connectivity never changes.
`online_refine_step_base` bounds a trial update and `online_refine_cap_base`
bounds total movement from that immutable origin. Boundary-exit vertices remain
fixed. CPU construction does not silently substitute another refinement method.

The analytic implementation differentiates trilinear presence, valid-width and
axial direction/normal tensor sampling, then the shared alignment, smoothness
and cumulative-direction score. Point terms are evaluated in parallel within
each half-trace block; many half-traces run concurrently. Adjacent-point
derivatives are gathered, and a short reverse recurrence propagates the
cumulative-direction history derivatives through the movable window only. The
fixed prefix supplies a cached constant boundary state. Gradients are projected
onto the immutable planes. Backtracking accepts only feasible strict cost
reductions, with candidate points scored in parallel.

Unlike the legacy finite-difference implementation, it does not resample/replay
the entire committed prefix for each perturbation. During one refinement solve
the accumulated width reference is fixed; accepted moved-point width deltas
update the global accumulator for future tracing. This intentionally differs
from legacy acceptance, which revalidates the prefix under a recomputed mean.
Therefore enabled analytic refinement has a distinct construction cache identity
and is not promised to produce legacy-identical geometry.

The scalar objective equations are shared in `PortableTraceMath.hpp`.
`TraceDifferential.hpp` provides small forward derivatives for each local term;
`FiberChunkRefinement.cuh` assembles the window gradient and updates. Eigenvector
derivatives use a tangent-plane solve. At tied eigenvalues and the sampler's
diagonal fallback, the derivative is zero; cutoff and other branch boundaries
are piecewise, not globally differentiable. Finite differences occur only in
validation tests, not the analytic optimizer.

The standalone CUDA benchmark additionally emits `refinement.audit` diagnostics
for analytic refinement: paired initial/final window costs, accepted-update
costs, backtracking counts, rejected/invalid proposals and movement-cap hits.
Every accepted update is independently rescored using the beam scorer with the
same fixed prefix and width reference. Tiny float-level differences can occur
between the derivative-value and beam-scorer evaluation paths; raw discrepancies
are reported rather than hidden. These diagnostics are compiled out of the
production GPU library. Audited benchmark timings include their overhead and
must not be treated as production tracing throughput. Compare paired costs
within a solve, not total costs across differently sized output traces.

Remaining serial work within each half-trace includes the cheap direction-history
recurrence/adjoint, acceptance reduction and backtracking attempts. Committed
chain indices are gathered once per commit; the speculative beam suffix is
rebuilt after movement. There is no global synchronization across all fibers
at each commit, and no active-fiber compaction within a batch. Short windows
therefore rely primarily on concurrency across fibers, not hundreds of workers
within one four-point window.

Host gradient checks cover local terms, masked samples and assembled tail
gradients with fixed-prefix boundaries. The standalone fixture benchmark also
runs synthetic device checks for fixed planes, windows, movement caps, disabled
updates and boundary points for both modes before timing. `--device-tests`
separately checks the selector and two-stage trace kernel without requiring a
fixture. Run GPU tests with a caller-owned GPU-manager grant; neither benchmark
command obtains one automatically.

## Experimental rolling beam commits

`beam_commit_mode="rolling"` opts into branch-local GPU tracing;
`"shared_prefix"` remains the compatibility default. Rolling commits have a
separate construction-cache identity even when refinement is disabled. CPU
construction rejects this mode rather than silently using shared-prefix search.
When refinement is enabled, rolling commits require `online_refine_mode="analytic"`.

Each beam owns its committed polyline, cumulative width sum/count, direction
history and immutable per-vertex refinement planes. Search scores a two-step
lookahead (beam8, parent cap32), ranks first-step parents by their best child,
and advances just the first step of each retained parent. Multiple lookahead
children of exactly the same first-step parent cannot occupy separate frontier
slots. Diversity is measured at the lookahead endpoints. The second points are
speculative and discarded; after branch-local refinement the next lookahead is
generated from the moved path. This is intentionally different search semantics,
not a numerically equivalent acceleration of shared-prefix tracing.

Candidate width is scored against its own parent's mean, then added to its
private sum/count. Accepted refinement movements update only that branch's
width contributions. As in shared-prefix analytic refinement, the reference
width is held fixed within one optimization solve. Refined geometry is not
shared between siblings: ping-pong path banks copy the committed prefix before
any child refinement can write it. One CUDA block per seed-half performs search;
one per seed-half/beam performs refinement concurrently across beams and fibers.
Short recurrences within a refinement remain serial. Batching accounts for both
path banks, branch states, candidate workspace and outputs.

Boundary behavior is based on the newly committed point, not speculative
lookahead crossing. A boundary-crossing first point needs no second field
sample. Its *ranking-only* second cost is extrapolated by repeating that first
step's incremental cost, avoiding an artificially cheaper one-term objective.
The final distance-budget advance uses a one-step horizon for every candidate.
The best-ranked valid terminal branch ends the half; nonwinning stopped
branches are excluded from later frontiers, never compared by short-path raw
loss against longer live paths. Refinement-invalid branches are pruned before
choosing that winner. If all second-step lookaheads fail, valid first points are
still advanced/refined and continuation is retried; stale pre-refinement
failure predictions do not end tracing. If all first-step candidates fail,
return the lowest-cost current branch and its own failure witness.

Rolling refinement counters describe the returned winner's ancestry, not work
performed on subsequently pruned beams. GPU timings include all search,
refinement, path copying and rolling-kernel launch work. The standalone CUDA
benchmark's `--rolling-tests` checks width forks, private planes and histories,
per-step progression, boundary/final horizon, invalid winner handling and
refinement-rescued lookahead. Full- and last-N-path optimization use the same
existing refinement controls. See `FiberChunkRolling.cuh` for the implementation.

## Implementation map

Paths below are relative to `volume-cartographer/`:

| Component | Implementation / API |
|---|---|
| Public construction, cache, matching, route and filter interfaces | [`FiberChunkTrace.hpp`](../core/include/vc/fiber_tracer/FiberChunkTrace.hpp) |
| Seeds, refinement, CPU construction, pinning and immutable cache | [`FiberChunkTrace.cpp`](../core/src/fiber_tracer/FiberChunkTrace.cpp) |
| Chunk route traversal and reference replay | [`FiberChunkRoute.cpp`](../core/src/fiber_tracer/FiberChunkRoute.cpp) |
| Normal-oriented path conflict filtering | [`FiberChunkFilter.cpp`](../core/src/fiber_tracer/FiberChunkFilter.cpp) |
| Multi-GPU queue and admission | [`FiberChunkGpuBridge.cpp`](../core/src/fiber_tracer/FiberChunkGpuBridge.cpp) |
| CUDA beam search and online fixed-plane refinement | [`FiberChunkGpu.cu`](../core/src/fiber_tracer/FiberChunkGpu.cu) |
| Native RAM-only prediction pooling | [`PredictionPooling.cpp`](../core/src/lasagna/PredictionPooling.cpp) |
| Python `vc.fiber_trace` bindings | [`fiber_trace.cpp`](../python/vc/fiber_trace.cpp) |
| Callable evaluation and CLI | [`fiber_evaluation.py`](../python/vc/fiber_evaluation.py) |
| Construction/filter benchmarks and legacy cache import | [`fiber_chunk_benchmark.py`](../python/vc/fiber_chunk_benchmark.py) |
