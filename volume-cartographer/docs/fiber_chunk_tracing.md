# Chunk-based fiber tracing

For historical direct-evaluation controls, Python `fiber_trace.training_replay`
calls the same `replayFiberTrainingCase` as the evaluation CLI. It accepts
prediction-based first/restart seeding explicitly (default enabled), modulated
presence cutoff, and the `TraceConfig` reverse interval, full-prefix/endpoint,
history and ellipse settings. `beam_replay` remains the simpler generic replay;
it is not interchangeable with that benchmark protocol.

The callable chunk evaluator `vc.fiber_evaluation.evaluate` accepts an optional
`normal_manifest`. If omitted, it uses the prediction field's fiber-normal
head, matching the volume workflow. `evaluation_normal_manifest` remains an
independent override for ground-truth acceptance; changing tracing normals does
not implicitly change that explicitly configured evaluation source.

Direct CPU candidate scoring batches all candidate positions from the current
beam frontier (all active cones), even when each trace uses one worker. Each
lookahead expansion is a separate batch: later candidates depend on the earlier
scores. Chunk resolvers retain decoded chunks across that batch instead of
reacquiring the shared cache lock for every candidate and channel.

The one-worker path uses `FiberPredictionField::sampleBatchScalarEquivalent`
and `LasagnaNormalSampler::sampleNormalBatchScalarEquivalent` to preserve its
original scalar interpolation arithmetic. The ordinary batch APIs retain their
existing float interpolation, including dense/chunk callers. Width and embedding
sampling use their existing batches. One-worker custom/in-memory sources use the
base batch helper (whole-frontier prefetch plus scalar sampling), preserving their
scalar arithmetic even when their ordinary batch override differs. Pinned
prediction reads still enforce ownership and coverage. The raw, pre-modulation
cutoff retains its separate scalar presence check; post-modulation cutoff reuses
the computed candidate score.

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

### Direct CPU tracing: independent eligibility gates

#### Current Paris4 direct-tracing experiment baseline (2026-10-27)

User-selected baseline: **30% cosine width modulation, joint cutoff0.25,
additional hard actual-width deviation20%**. The joint cutoff includes both
direction and width modulation; width is compared against the cumulative
arithmetic mean before adding the current sample.

```python
width_relative_tolerance = 0.30
width_cosine_weight = True
width_hard_relative_tolerance = 0.20
width_reference_mean = 0.0  # accumulated mean, not a fixed supplied width
replay_presence_modulated = True
replay_presence_ignore_width = False
replay_presence_cutoff = 0.25
replay_reverse_interval_base = 0.0  # FW/BW disabled in this measured baseline
```

Same beam8/lookahead2 exhaustive, step16 base, predicted tracing normals and
reference-seed protocol as the native Paris4 sweep. Native9ef89ac3 predictions,
checkpoint48f7cd9a... at14.272M samples;56fibers/112directions. Min512 results:
0GTerrors,82.841% retained,180pieces,92filtered stops,mean8.301mm,median3.214mm.
Raw GTerrors also0. The extra hard guard is a user preference, not a measured
error improvement over the no-hard control. Full effective settings and report:
`/ephemeral/hendrik/fiber3d_train2/evaluations/paris4_direct_separate_gates_20261027_v1/joint_hard_width/hard15_20/`
(`comparison.json`, key `width30_hard20_joint25`). This selects the baseline for
subsequent direct experiments; it does not alter global API defaults or enable
unsupported hard gates in chunk/GPU tracing.
Machine-readable selected baseline: `paris4_direct_separate_gates_20261027_v1/current_direct_baseline.json`
under the same evaluations root.

Direct tracing also exposes two opt-in `TraceConfig` fields:

- `replay_presence_ignore_width=True`: when `replay_presence_modulated=True`,
  the presence cutoff tests raw presence times the usual six direction-alignment
  factors, without the width factor. The beam objective still includes ordinary
  width modulation (`width_relative_tolerance`, `width_cosine_weight`).
- `width_hard_relative_tolerance=0.05`: reject widths differing by more than5%
  from the current width mean; equality passes. Zero disables this hard gate.
  It requires enabled width sampling/modulation. Candidate eligibility uses the
  current committed mean; pending points are rechecked against the pre-add mean
  as they commit. Cached sampled widths are reused, and rejected points neither
  enter the mean nor reach the replay observer. A fixed `width_reference_mean`
  remains supported.

The objective score and width-free eligibility score are kept separately through
lookahead and commit. `width_cutoff` is a filtered stop in reference replay,
not a GT-distance restart; the existing combined presence-filtered counter
includes it. Invalid-width stops retain their existing treatment. The presence
cutoff also remains the raw-presence floor for the reference seed search.

These options currently apply to **direct CPU tracing only**. Chunk construction
and cache requests reject them explicitly, including GPU chunks. Defaults are
unchanged; existing chunk-cache identities remain unchanged when gates are off.

### Experimental delayed reverse checkpoints and convergence

Direct `training_replay` supports an opt-in reverse policy (ordinary
`beam_replay` and chunk/GPU paths reject these settings):

```python
replay_reverse_interval_base = 512.0
replay_reverse_to_start = True
replay_reverse_all_endpoints = True
replay_reverse_history_steps = 0  # legacy history mode is not used
replay_reverse_delay_steps = 4
replay_reverse_convergence_min_base = 1024.0
replay_reverse_convergence_distance_base = 8.0
replay_reverse_convergence_angle_degrees = 10.0
replay_reverse_convergence_steps = 4
```

A scheduled checkpoint waits for four later committed forward nodes. Their
edges, traversed in reverse chronological order, supply the incoming reverse
direction and cumulative smoothness history. This is not history taken from
earlier forward nodes. Only the checkpoint becomes verified on success; the
future suffix remains pending. Travel accrued during the delay counts toward
the next interval. Delay must be shorter than the interval. Per-segment state
is discarded on rollback/reseed.

Every reverse step still satisfies the broad normal-oriented consistency
ellipse and monotonic backward projection. Once minimum reverse travel is
reached, sustained tight Euclidean distance AND reverse-oriented tangent agreement
inside the already-verified prefix permits explicit early success. Otherwise
reverse tracing continues to the segment start or failure. This is a heuristic:
matching geometry does not prove equal beam frontier/history or frozen width.

At termination, any pending checkpoint uses its available future edges (possibly
fewer than four), then the final endpoint is verified with no future history.
For presence/width stops with delayed verification enabled, instead verify from
four accepted segments before the stop. The final four accepted segments supply
incoming reverse history and are **retained** if verification passes. A failed
check rolls back to the last verified checkpoint and counts as reverse-filtered.
An observer-rejected final point is excluded from both history and the frozen
width reference. A piece entirely inside the four-segment history has nothing
earlier to verify: it is retained as history-only, not counted as a passed check.
At step16 these pieces are at most64base voxels, below the benchmark's512minimum.
Reverse presence cutoff remains disabled; width reference is frozen to the
accepted forward mean and configured hard width eligibility is retained.

`ReplayResult.reverse_diagnostics` / replay JSON contains per-check rows:
`[checkpoint_point_index, history_edges, reverse_travel_base, converged,
terminal, elapsed_seconds]`, ordered across segments; indices reset per segment.
The JSON key is omitted when no delayed-policy checks ran, preserving legacy
outputs. Delay0 and convergence-minimum0 preserve the original policy.
`reverse_stop_tail` adds `checks`, `passed`, and `history_only` counters when
stop-tail processing occurs. Default/FWoff output remains unchanged.

First Paris4 trial (`paris4_direct_separate_gates_20261027_v1/reverse_delayed_v1`):
zero raw/min512 GT errors; min512 retained73.767%,175pieces versus FWoff
82.841%,180pieces. Time605.27s versus98.95s;181/618checks converged early.
These are single-run timings, not a same-settings comparison to legacy full-prefix.
In that first trial, presence/width stop rollback discarded the unverified tail without a
terminal reverse check:162.069mm lost this way, versus10.152mm discarded by
actual reverse failures. The stop-tail policy above replaces that behavior;
the first trial used1536minimum travel and4vx/5degree convergence, not the new
1024/8/10 configuration. Experimental mode is not promoted to the recorded baseline.

Corrected stop-tail trial (`reverse_delayed_tail_v2/REPORT.md` under the same
experiment directory), same112directions, minimum512base output pieces:

|Variant|GT errors|Retained %|Pieces|Presence/width stops|Reverse stops|Mean mm|Median mm|Time s|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|FWoff control|0|82.841|180|92|0|8.301|3.214|103.67|
|Previous1536/4/5, rollback tails|0|73.767|175|84|6|7.554|2.510|605.27|
|Retain stop tails,1024/8/10|0|81.383|181|90|6|7.819|3.122|550.82|

Ordinary-stop rollback loss162.069mm→0.0mm; actual reverse failures discard
14.826mm.243stop-tail checks,236pass/7fail;192short history-only pieces.
861total checks,326early convergences,653072base reverse travel. Raw output
also has zero GTerrors,91.800%retained,490presence/width and15reverse stops;
raw reverse stops increase8→15 while the min512 count remains6.
The measured9%wall-time reduction includes both changed stop policy and relaxed
convergence, one run each; it is not an isolated convergence-speedup measurement.
All112FWoff case JSONs reproduce historical output exactly.112native tests and
16Python tests pass,1optional skip. Baseline/defaults unchanged; no commits.

Single-match follow-up (`reverse_single_convergence_v3/REPORT.md`): only
`replay_reverse_convergence_steps=1` changed. All112 per-case outputs are exactly
equal to the four-match trial after removing reverse-check diagnostics, including
geometry, losses, stops and failures. Same0raw/min512 errors,81.383%retained,
181pieces,90presence/width+6reverse stops,mean7.819/median3.122mm (min512).
Time459.18s versus550.82s (16.6%lower, single timings). Reverse travel falls
653072→624992base (4.3%); early convergence326→411 of861checks. The wall-time
gain is larger than work reduction, so do not attribute all of it to this change.
The approved `direct-reverse-single` runner reused hash-bound controls; no new
inference, native rebuild, or default changes. Report validation passed.

This direct experiment uses32OS case-worker threads in one process, with the
Python GIL released around C++ replay. Each directional trace has
`parallel_threads=1`; its reverse checks run synchronously on the same worker.
Candidate next speed tests (not implemented here):64/112case workers, then
adaptive within-trace candidate work for the tail. Existing multi-thread sampling
can change batch/scalar arithmetic, so validate geometry before using it as a
drop-in speedup. A1024checkpoint interval instead of512 would reduce scheduled
checks but changes filtering granularity and needs a separate quality trial.

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
GPU preparation permits a halo to cross a physical volume boundary: outside
lattice nodes carry zero/invalid presence, normal and width. The CUDA sampler
separately checks the physical source domain (through the last voxel centre,
matching native scalar normal/width sampling), so edge interpolation cannot
invent a path outside the volume. Upper corners at the last voxel are clamped
as in the native sampler. Interior channel/alignment errors still fail. This
repairs the former `GPU export requires six aligned channels` boundary failure;
successful interior caches remain compatible and are validated on resume.

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

### Full-length fibers from completed chunks

The same command has a separate CPU-only continuation/export stage:

```sh
VC_BUILD_DIR=/path/to/native-build python volume-cartographer/scripts/run_fiber_chunk_volume.py \
  CONFIG.json /data/paris4-full-fibers --routes-from /data/paris4-paths --workers 64 --plan
VC_BUILD_DIR=/path/to/native-build python volume-cartographer/scripts/run_fiber_chunk_volume.py \
  CONFIG.json /data/paris4-full-fibers --routes-from /data/paris4-paths --workers 64
```

Use the construction configuration (including prediction identity, scale,
normals and filtering settings). No GPU reservation, inference, beam reconstruction,
or reference fibers are needed. The Python layer handles configuration, directory
locks and progress only; `exportFiberChunkRoutes` runs the shared native
`FiberCaseQueue`. Existing chunks **must be complete and correctly annotated**.
The route stage defaults to one active context per CPU worker (no GPU waits to
overlap); construction retains its eight-context default.
Missing expected files, corrupt frames, mismatched construction identity, and
stale conflict flags fail rather than silently truncating routes or rebuilding.
The source cache is locked against concurrent construction/filter changes.

For each unflagged local path intersecting its half-open owning core, consider
only its still-unclaimed arc intervals. Prefer the vertex nearest the core center
when it lies inside that interval; otherwise use an interpolated midpoint (also
covering segments crossing a core without an interior vertex). Start both directions
on that exact path; do not reselect a nearby candidate. Each half uses
`traceFiberChunkRoute`, just like the benchmark, preserving overlap/direction/width
ranking, splice evidence checks, cached endpoint stops, and cycle detection.
The emitted fiber is reverse(backward half), seed, forward half. The seed appears
once. Full geometry is in **base XYZ voxels**, not tracing or stored-grid units.
This exporter does not add reference restarts, short-piece filtering, or new
reverse checks; existing chunk conflict annotations still exclude flagged paths.

For local/on-demand replay without a complete-volume prefilter, qualified alternatives are checked using the existing in-chunk
conflict inspector and its normal-aligned ellipse, agreement and separation settings.
Equivalent overlapping copies can still match; conflicting continuations stop with
`chunk_ambiguous_continuation`, before splicing or claiming a target. The incoming
history participates in this check. Invalid normal evidence is conservative.

Full-volume export additionally **pre-filters every possible cached-end connection
before tracing**. The shared matcher enumerates all candidates passing its spatial,
direction, overlap and width tests; the shared splice check validates the bridge.
It uses the oriented cached source path as canonical overlap history. Endpoints are
indexed with the existing segment R-tree and work is parallelized by target chunk.

Each connection links a source exit end to a target entry end. The same connection
discovered in reverse is counted once. An edge is allowed only if **both ends have
exactly one partner in the original graph**. Thus both one-to-many forks and
many-to-one merges are blocked, before ownership or tracing order can choose a
winner. Deleting rejected edges does not turn another ambiguous edge into a valid
one. Distinct surviving cached IDs count as distinct choices after the in-chunk
reduction described below.
Runtime history-sensitive matching/splice checks remain additional vetoes. If
stitched history discovers a match absent from the graph, it stops with
`chunk_unqualified_continuation`; it cannot resurrect an omitted connection.
The graph does not assert reciprocal match quality merely because an edge exists:
reverse traversal still has to pass the ordinary runtime checks.

This global prefilter requires the complete declared route inventory and is enabled
by the full-volume exporter. The generic on-demand benchmark does not silently
enumerate/build neighboring chunks to emulate a complete graph; it retains its
local rule unless a caller supplies `FiberChunkContinuationGate`.

**In-chunk reduction before full-volume routing:** construction retains raw paths;
the exporter creates a reduced, immutable view of each loaded chunk. The shared
`findFiberChunkMatches` checks both ends against other paths using the configured
cross-chunk distance, angle, overlap and width gates (production: 12 base voxels,
25 degrees, 32 minimum overlap, 128 maximum terminal history, 40% cosine width
compatibility). Only the 16-base-voxel target runway is removed for this comparison.
The existing normal-ellipse conflict flags remain active; this is an additional
endpoint-compatible rule, not a replacement for those flags.

`reduceFiberChunkPaths` fuses a related group only if **every pair** passes a
reciprocal, complete-history equivalence proof, including paired endpoints within
the same distance gate and consistent orientation. Full-history proof has no
128-voxel cap and permits projection clamping only at the target's actual endpoints,
with all samples still tested. Endpoint arcs need not coincide exactly. Missing or
incompatible width evidence cannot prove equivalence. A chain A~B~C does not fuse
A with C unless they also agree directly. Internal divergence, unmatched tails,
and uncertain groups cause all related paths to be excluded conservatively when
`routes.reject_ambiguous_paths` is true (the historical default).
In that strict mode no two distinct surviving eligible paths qualify against each
other's terminal history. An unrelated incoming path may still match both and
be stopped by the two-sided continuation prefilter.

Fusion selects the longest original path (stable original ID tie-break); it does
not average coordinates or extend tails. Member IDs remain recorded. Endpoint
stops are reoriented and combined conservatively: a member's blocked end cannot
become traversable through fusion (`fused_member_stop:*`). Decisions are computed
once per chunk and reapplied after cache eviction. Original chunk files are never
rewritten. This reduction is currently enabled in full-volume export, not
silently added to the generic on-demand benchmark. Set `reduce_duplicate_paths`
false only for an explicit comparison/control export.

For a **fusion-only** export, keep `reduce_duplicate_paths: true` and set
`reject_ambiguous_paths: false`. Positive fusion decisions and conservative fused
endpoint stops are identical; unresolved groups keep their original IDs, geometry
and stops. This does not attempt further subgroup fusion inside an unresolved
component. Original conflict flags remain exclusions. Cross-chunk ambiguity
still stops routing, and ownership still prevents repeated use of a path interval.
The resulting cache view can contain related/non-equivalent paths; it no longer
claims the strict endpoint separation invariant. Reports distinguish rejected
`ambiguous_paths` from `retained_ambiguous_paths`. Mode is bound into resume;
switching it requires a fresh output directory. Omitted mode means strict, preserving
existing strict exports' resume compatibility.

The full-volume exporter maintains ownership of **cached path arc intervals**.
Already exported intervals stop a subsequent route with `chunk_piece_already_owned`.
Matched overlap behind a handoff is recorded as represented by the incoming emitted
geometry; synthetic seed history does not confer ownership. Unvisited prefixes and
budget-limited tails remain available to subsequent seeds. No complete long route is
repeated merely because another seed lies on one of its already-owned pieces.

A conservative endpoint-to-path R-tree graph groups all possible handoffs into
connected components. Components run in parallel; each component has a deterministic
seed order and one ownership writer. This avoids schedule-dependent first-claimer
races and per-step ownership locks. A large component can limit parallelism; the
component report records its size and graph construction time.

Optional top-level `routes` settings:

```json
{
  "routes": {
    "cache_bytes": 1073741824,
    "maximum_distance_base": 1000000,
    "maximum_steps": 100000,
    "page_points": 50000,
    "reduce_duplicate_paths": true,
    "reject_ambiguous_paths": true
  }
}
```

Distance and step limits apply **per half**; exhaustion is explicitly recorded
as `chunk_length_limit`, not a successful endpoint. `page_points` is a target:
a single fiber is not fragmented, so a page can contain up to the larger of that
target and `2*maximum_steps+1` points. The shared packed writer rejects decoded
documents above 1 GiB. Route-cache bytes bound retained decoded geometry, not
active reader leases, per-worker route/page buffers, the prediction cache, or the
parallel inventory. Memory therefore also scales with workers and route limits.
Loads coalesce by key; no cache lock covers I/O or sampling. Only bounded trailing
history is kept for matching, while observers retain full emitted geometry.

By default routing coverage is the completed prediction's complete sparse core
inventory. `keys_xyz`, if supplied, restricts **seed-owner cores only** and does
not cut continuation at that subset. An explicit
`routes.inventory_keys_xyz` may define a smaller test coverage window. Stops at
its boundary are labeled `chunk_inventory_boundary`; missing files *inside* the
declared inventory remain errors. Do not supply only existing files to conceal
holes in an unfinished full-volume cache.

Output is Zstd level-3 compressed MessagePack using the shared chunk serializer:

- `routes.manifest.msgpack.zst`: immutable construction/route settings, source
  inventory with filenames/sizes/mtimes, seed-owner keys, and a generation ID.
- `components.msgpack.zst`: component count, largest sizes and graph timing.
- `path_reduction.msgpack.zst`: original path IDs mapped to their representative
  ID, `-1` for originally inactive paths or `-2` for ambiguous groups, conservative
  endpoint reasons, and per-chunk/global counts. Validated on resume.
- `continuations.msgpack.zst`: all canonical candidate endpoint connections,
  original degrees, accepted/rejected decisions and counts, validated on resume.
- `component_ID.routes.N.msgpack.zst`: bounded pages of full fibers. Each has source
  key/line/arc, `seed_base`, `seed_index`, `points_base`, `length_base`, two
  `stop_reasons`, oriented chunk/path `joins`, `owned_arcs` (including overlap
  aliases), and `cyclic_candidate`.
- `component_ID.routes.msgpack.zst`: final component index, written atomically
  **after** every page, with seed targets and page counts. Empty cores need no page.

Repeat the command to resume. Resume reads/validates pages and provenance, not
just filenames, and does not rewrite completed outputs. A partial component without
its final index is regenerated deterministically. Changed settings or source
inventory/metadata require a separate output directory; worker count and decoded
cache budget can change. Source fingerprints use size/mtime plus the explicit
immutable source identity, not a fresh cryptographic hash of every chunk.
Output must not be modified independently of its manifest/indexes.

The format is `vc-fiber-routes-v4-reduced`; use a separate output directory from
the old candidate-per-seed, v2 ownership and v3 prefilter exports. Existing chunk construction caches remain valid.
These are **piece-owned routes, not a physical fiber census**. A cached path interval
cannot be claimed twice, but distinct cached path IDs may still describe geometrically
overlapping traces outside the endpoint-relation/equivalence scope. No global
geometric union is claimed by the in-chunk reduction.
Cycle/ownership stops prevent repeatedly traversing already consumed cached pieces.
Summed length is therefore not guaranteed unique physical fiber coverage.
Width evidence participates in matching; the exporter
does not invent per-point tracing costs absent from the cached geometry.

Validation includes synthetic multi-owner/concurrent joins, exact seed selection,
flags, budgets, serial/parallel parity, corruption, partial publication, source
mismatch, interval ownership, corner-crossing paths and unchanged resume writes.
Cached ownership pilots/full export are under
`/ephemeral/hendrik/fiber3d_train2/evaluations/paris4_owned_routes_20261026_v1/`;
the historical candidate-per-seed timings remain under `full_route_export_20261025_v1/`.

Endpoint-compatible reduction results are under
`/ephemeral/hendrik/fiber3d_train2/evaluations/paris4_reduced_routes_20261026_v2/`.
On all 166258 cached Paris4 cores (GCC13 RelWithDebInfo, 64 CPU workers, one fresh
nonexclusive run): 35.95 s native, 41415 redundant paths fused, 152567 ambiguous paths
excluded from 310526 previously eligible paths. Output: 93487 routes, 71.4961 m summed
length, versus 279668 / 256.5674 m before reduction. This 49.1% exclusion rate is a
substantial conservative coverage tradeoff, not an accuracy improvement claim.
All 8854 exported handoffs were audited against the unique two-sided graph and
every exported source ID against reduction representatives. The 3407-core pilot
has exact 1/64-worker geometry/joins/stops/reduction/graph parity (18.9854 / 1.6079 s).
Full resume reused all 82486 components in 24.42 s without changing any output
size or mtime. The report includes exact invocation and source config.

Fusion-only comparison (`paris4_fusion_only_routes_20261026_v1/COMPARISON.md`):
same 166258 cached cores, 64 CPU workers and GCC13 RelWithDebInfo, no inference.
Disabling only whole-group rejection restored all 152567 unresolved paths while
preserving exactly the same 41415 duplicate fusions and original conflict flags.
Output: 237384 fibers / 216.9777 m, mean 0.914 mm, median 0.845 mm; native fresh
export 157.81 s. Strict output was 93487 / 71.4961 m; pre-fusion output was
279668 / 256.5674 m (duplicates retained). All 8819 exported handoffs and all
source-ID reduction mappings passed audit. This is a coverage comparison, not
a GT-accuracy evaluation. The experiment's config sets rejection false explicitly;
it does not change the historical default.

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

### Shared direct CPU tracer and sampling diagnostics

The VC3D native annotation tracer and direct reference benchmark share
`core/src/fiber_tracer/FiberTrace.cpp::traceOneWayCore`. VC3D calls
`traceFiberExtrapolation` or `traceFiberSegment` from
`apps/VC3D/LineAnnotationFiberSegments.cpp`. The benchmark calls
`traceFiberReplaySegment` from `FiberEvaluation.cpp`; reverse consistency calls
`traceFiberExtrapolation` from `FiberReverseConsistency.cpp`. Reference matching,
restart policy and reverse-check scheduling are wrappers, not a second beam
engine. GUI settings can differ, and external neural callbacks are a separate
option. This sharing claim does not mean the CUDA chunk engine is the same
implementation as the CPU engine.

`training_replay(..., profile=vc.fiber_trace.TraceProfile())` optionally exposes
existing native phase counters plus width sampling time. Use one profile per
concurrent call; it resets on entry, and reading/reusing an active profile raises
an error. Read it after completion with `profile.to_dict()`. The GIL is released
during native tracing. No tracing arithmetic or concurrency is changed by this
diagnostic API. Nested counters must not be summed with their parent timers.

The 2026-10-27 warm-cache profile used 32 distinct Paris4 directions, each clipped
to 1024 base voxels, and one native compute thread per trace. Same settings and
cases at 1 and 32 concurrent workers; GCC13 RelWithDebInfo, no inference/GPU.
The serial pass took 507.262 s (one repetition); parallel median was 31.560 s
(three repetitions, 31.317–32.025 s), about 16.1x throughput. Parallel process CPU
averaged 18.93 cores over the whole batch, including its uneven tail. No disk
read bytes or major faults were recorded during these timed batches.

| Phase | Serial case time | 32-worker summed case time |
|---|---:|---:|
| Normal sampling | 64.38% | 72.97% |
| Presence/direction sampling | 19.63% | 14.73% |
| Width sampling | 10.82% | 7.94% |
| Candidate scoring | 3.61% | 2.70% |

Normal request preparation/resolution alone consumed 43.85% of serial time.
This timer is named `normalPrefetchSeconds`, but includes in-memory preparation,
not just I/O. `LasagnaNormalSampler.cpp` builds a full
`vector<PreparedNormalPoint>` (three cube requests per candidate), resolves all
requests, then interpolates them in a second pass. The full frontier is already
batched: median candidate count was 648 at depth 1 and 52,488 at depth 2.

Priorities to investigate, not implemented by this profile:

1. Stream or cache-block normal request preparation/interpolation, reuse scratch,
   and retain local decoded-chunk references without materializing all requests.
2. Reuse coordinate/corner planning and validity samples across width,
   presence/direction and normal channel passes, preserving interpolation math.
3. If still useful, batch ready sampling requests across independent trace states
   without a global barrier. A verification reverse trace currently pauses its
   own forward trace; advancing that forward state simultaneously would require
   explicitly speculative work and rollback. Independent directions already run
   concurrently. Larger batches alone are not the first priority given the
   existing 52k-candidate frontier batches.

Hardware profiling was denied (`perf_event_paranoid=4`); native phase timings
establish the sampling bottleneck but do not isolate memory bandwidth, reference
count traffic or lock contention. A separate 4096-base case exercised nine reverse
checks with seven early convergences: unprofiled 76.935 s versus profiled 77.513 s,
with identical paths, costs and stop outcomes. This is a sampling microbenchmark,
not an extrapolation to full-volume runtime.

Command: `/home/hendrik/villa3/scripts/run_fiber_tests`, configured through
`local/fiber_tests.json`; suites `direct-sampling-profile`, bounded continuation
`direct-sampling-profile-finish`, and `direct-sampling-profile-report`.
Host-local results and full statistics:
`/ephemeral/hendrik/fiber3d_train2/evaluations/paris4_direct_separate_gates_20261027_v1/sampling_profile_v1/REPORT.md`.
Regression suite `direct-separate-gates-build` passed 112 native and 18 Python
tests (one optional skip), including profile parity, reset, concurrent-use guard
and exception cleanup.

### Sampling overhead: retained chunks and owned ROIs

The follow-up native benchmark `core/test/bench_fiber_sampling_overhead.cpp`
compares the current cached-request path with existing
`detail::FiberChunkTensor` scalar sampling. This is measurement only, not a switch
of the production direct tracer. Tensor normal/direction reconstruction and
width validity/units remain identical. The tensor's float-equivalent batch
override is deliberately not used for the double scalar-equivalent comparison.

On six Paris4 locations (three reference sites and three nearby storage seams),
five warm repeats per site, one CPU worker, GCC13 RelWithDebInfo:

| Operation | Cached ns/point | Direct-memory ns/point | Ratio |
|---|---:|---:|---:|
| Scalar trilinear, retained chunk bytes | 191.67 | 66.44 | 2.89x |
| Normal, owned ROI | 2697.82 | 200.50 | 13.46x |
| Presence + direction, owned ROI | 910.53 | 249.81 | 3.64x |
| Width, owned ROI | 589.11 | 58.76 | 10.03x |

These medians use 52,488-point deterministic local clouds, matching the observed
frontier size but not actual recorded beam positions. Three meaningful channel
operations together take about 4.20 microseconds versus 0.509 microseconds per
point (sum of medians, about 8.2x sampling-only difference). At 648 points the
normal advantage is 5.4x, while prediction/width are 3.8x/11.8x. Pregathered scalar
arithmetic alone takes 3.37 ns at 648 points and 7.71 ns at 52,488; it excludes
coordinate and corner gathering and cannot substitute for normal reconstruction.

`sizeof(LasagnaCubeRequest)` is 1048 bytes in this build. The normal batch's three
requests per point therefore create about 157 MiB of temporary request records
at 52,488 points. This is bookkeeping, not the underlying prediction data. Its
construction/destruction is included in the cached timings; no profiler timer
is enabled in these sampling calls.

Each owned ROI in this experiment occupies 285–293 KiB for all channels and its
interpolation apron. Warm ROI construction took 0.20–0.35 ms. Initial presence
chunk acquisition plus subsequent full ROI construction took 0.34–18.25 ms,
depending on already cached overlap; this is not a controlled cold-I/O benchmark.
Warm ROI setup amortizes after roughly 55–96 points if all three channel groups
are sampled. Including the observed first acquisition gives roughly 89–4908
points. Larger ROIs will have different preparation/storage costs.

All valid values and validity decisions matched exactly on every repeat,
including seam-crossing samples. 21 prediction and 18 normal sampler regression
cases passed. No production tracer/sampler/concurrency changes or full benchmark
rerun were made. These differences measure aggregate request, layout, allocation
and cache-access overhead—not exclusively locks, and not a full-trace speedup.

Run through `/home/hendrik/villa3/scripts/run_fiber_tests`, suite
`sampling-overhead`. Raw timings, per-site results, setup costs and complete
mean/median/min/max statistics:
`/ephemeral/hendrik/fiber3d_train2/evaluations/paris4_direct_separate_gates_20261027_v1/sampling_overhead_v1/REPORT.md`
and `results.json`/`summary.json` alongside it.

### Direct CPU tracing with retained decoded chunks

Set `TraceConfig.sampling_window_voxels = 64` to use the retained-cache backend
for direct CPU candidate-frontier sampling. `0` remains the default; valid nonzero
sizes are 8–256 stored voxels. This does **not** build copied ROI channel arrays.

The ownership/access flow is:

1. `FiberPredictionField::chunkRegionChannels` checks publication readiness.
2. `PinnedChunkRegion::pin` obtains ordinary decoded-cache `ChunkResult` references
   and retains their buffers. This is shared ownership, not OS memory locking.
3. `PinnedChunkGrid` builds an immutable table of raw pointers and strides for
   each channel. Construction resolves packed-channel byte offsets and checks
   payload extents. Missing/all-fill chunks use the source fill value; unpinned
   in-volume dependencies throw, rather than silently becoming fill.
4. `FiberChunkTensor` uses those tables for corner reads and the same interpolation,
   direction reconstruction, validity masking and width math as its owned backend.
   Interpolation performs no cache/map lookups, refcount updates, locks or I/O.

Each trace retains one guarded window and replaces it only when the next candidate
frontier leaves coverage. Old ownership is released before acquiring replacements.
Forward and reverse traces may each retain a window. Cache eviction can remove its
own reference without invalidating these views; buffers are freed when their last
owner releases them. Full decoded storage chunks remain alive, so retained bytes
can exceed the logical window size. This is not a second persistent cache.

Implementation: `core/include/vc/core/render/PinnedChunkRegion.hpp` and
`core/src/render/PinnedChunkRegion.cpp` provide the generic ownership/grid API;
`core/src/fiber_tracer/FiberChunkTensor.cpp` provides shared interpolation/storage
backends; `FiberTrace.cpp::scoreCandidateTasks` manages per-trace window reuse.

Current scope: explicit one-worker scalar tracing, single-option byte prediction
and normal channels on exactly matching lattices, byte/float width, no embeddings
or caller-supplied pinned region. Unsupported sources and oversized frontiers keep
the existing sampler. Read/readiness errors propagate. Initialization and committed
step sampling still use their existing APIs. Dense/GPU construction is unchanged;
`seedGrid` and `packGpu` deliberately require the owned backend.

`TraceProfile` reports window loads/fallbacks/time, maximum retained physical bytes
(deduplicated by payload within one window), pointer-table bytes, and ROI voxel
bytes allocated/copied (zero). Pointer-table bytes exclude region map nodes;
these counters are not total process memory or a global memory cap.

Validation includes exact values across seams, packed channel offsets, missing
chunks, masked normals, byte/float widths, independently chunked widths, cache
eviction and owner lifetime, and concurrent read-only sampling. The approved
runner suites `direct-sampling-window` and `direct-sampling-window-full` measure
cached Paris4 replay with unchanged tracing settings. Retained results are under
`paris4_direct_separate_gates_20261027_v1/sampling_retained_v1`; the earlier
`sampling_window_v1` measured the superseded **copied** ROI backend.

Measured 2026-10-27, GCC13 RelWithDebInfo, one native compute worker per trace,
up to 32 independent cases concurrently, cached inputs, no inference/GPU:

| Workload | Existing sampler | Retained cache chunks | Speedup |
|---|---:|---:|---:|
| 32 fixed prefixes, 1024 base voxels | 32.726 s | 6.338 / 6.307 s | 5.18× using mean |
| Case 136, 4096 base voxels | 77.346 s | 18.450 s | 4.19× |
| All 112 complete Paris4 directions | 459.18 s (saved baseline) | 87.524 s | 5.25× |

All replay output fields match exactly except reverse elapsed timings. Full-run
scores therefore remain 0 GT errors, 81.383% retained and 181 retained pieces
(minimum 512 base voxels), with 90 presence/width and 6 reverse filtered stops.
The full run made 8,132 window acquisitions, zero fallbacks and zero ROI voxel
copies. Maximum single-window retained physical storage was 47.25 MiB; pointer
tables peaked at 2.953 KiB. Acquisition time summed across cases was 4.949 s,
not additive wall time. These retention counters are not newly allocated arrays.

The copied-ROI experiment took 68.354 s on the same full benchmark. Retained
sampling is about 28% slower than that superseded experiment, but removes ROI
voxel copies and shares the regular decoded payloads instead. It remains about
5× faster than the original sampler. There is no claim of a new full-volume or
GPU throughput measurement. `REPORT.md` includes per-case mean/median/P95/min/max;
`results.json` includes profiles and output geometry. Full-run timing is one
measurement; pilot retained timing has two repetitions.

#### Initial retained implementation: diagnosed slowdown

The full112 saved comparison is 68.354s copied versus 87.524s retained: +19.170s
(28.0%). Same 1,427,719,122 candidate tasks and 8,132 window acquisitions. Summed
worker-wall phase timers localize the regression to sampling: width60.576→170.796s,
prediction392.014→529.285s, normals439.622→544.229s. Acquisition improved
17.933→4.949s; scoring234.282→233.192s and pruning31.757→31.337s stayed similar.
These sums are **not** additive process wall time.

Matched single-executable diagnostic,64stored-voxel windows,6real/seam sites,
5rotated repeats per site,52,488points per batch (median ns/sample):

| Operation | Copied ROI | Retained chunks | Paired median slowdown |
|---|---:|---:|---:|
| Normals | 209.88 | 287.40 | 39.5% |
| Presence/direction | 267.98 | 379.53 | 43.6% |
| Width | 61.00 | 131.63 | 116.0% |

The initial retained adapter had avoidable address work: `cube()` first built
z-contiguous dense indices; `retainedCorners()` decoded them back into XYZ using
integer divisions/remainders; `PinnedChunkGrid::corners()` then divided into chunk
coordinates/local offsets, checked bounds and resolved strides. This repeated
per channel rather than shared across channels on the same layout. Owned gathers
simply index their contiguous arrays with the already prepared offsets.

Controlled presence-scalar tests using the **same retained buffers and production
grid** measured63.01ns with the dense-index roundtrip versus47.92ns with direct XYZ;
paired median penalty15.04ns/gather. Prepared XYZ measured37.47ns, while interpolation
over pregathered corners measured3.83ns. Those last two are diagnostic lower bounds,
not proposed production implementations or exact cycle attribution. Warm retained
setup was0.047–0.055ms/site versus1.363–2.260ms for copied ROI setup. Thus copies are
not needed to eliminate this roundtrip overhead.

Memory layout/locality and generic grid checks also differ from owned arrays;
their individual contributions are not isolated. These sites retained14MiB of
physical payloads versusabout2MiB copied ROIs, but retained bytes are not all touched
per sample. Hardware-counter profiling was denied (`perf_event_paranoid=4`), so
there is no measured cache-miss/bandwidth attribution. Hot grid sampling itself
contains no locks or cache-service requests. No production optimization/default
change was made in this diagnostic task. The likely next change is direct XYZ
sampling with coordinate/chunk addressing reused per compatible channel layout.

Approved runner suite `retained-sampling-diagnosis`;43native sampler tests pass.
Exact parity asserted every timed repeat. Results, mean/median/min/max, small-batch
measurements and phase comparison: `sampling_retained_diagnosis_v1/REPORT.md` under
the same evaluation directory. Clouds are deterministic synthetic local positions,
not recorded beam frontiers; full replay timings above are from the saved runs.

#### Prepared corner addressing (first implementation)

Retained `FiberChunkTensor` sampling keeps absolute XYZ corners directly. It does
not create or decode dense ROI indices. A stack-local `Cube` lazily stores one
`PinnedChunkGrid::CornerAddress`; normal/prediction/width channels reuse it when
their layouts match. Independently chunked widths or other incompatible layouts
prepare their own address from the original XYZ. Owned ROI and owned seed-grid
paths continue using their existing flat array indices.

Generic API: `grid.prepareCorners(loXYZ, hiXYZ)` checks bounds and calculates
chunk entry indices and element offsets; `grid.corners<T>(address)` only validates
layout identity, gathers corner bytes/fills and converts dtype. Grid construction
accepts an optional `layoutPeer`. Identity is shared only after exact comparison
of volume/chunk geometry, window first/count and every entry's strides. Different
dtypes, fill values and packed-channel byte offsets are supported: offsets are
in elements and each grid retains its own data pointers and dtype. Incompatible
or default/uninitialized addresses throw before buffer access.

Prepared addresses borrow their layout identity; at least one participating grid
must remain alive while they are used. Grids retain decoded payload ownership;
the address does not. There are no per-sample shared ownership operations, mutable
shared scratch, cache lookups, locks or ROI voxel copies. Arithmetic/validity and
scalar-versus-batch precision are unchanged, as are scheduling and defaults.

Final reviewed build, full112Paris4 exact replay: **76.561s**, versus initial
retained87.524s (12.5% less wall time), copiedROI68.354s, and original459.18s
(6.00× speedup). A preceding prepared-address pass measured76.077s; the final
repeat includes defensive empty-plan/moved-from guards. Same32case concurrency,
one native worker per trace, GCC13 RelWithDebInfo, no inference/GPU. All outputs
except elapsed timings match exactly:0GTerrors,81.383%retained,181pieces. Zero
copied voxel bytes/fallbacks;8132window acquisitions; peak retained47.25MiB and
pointer tables2.953KiB per window. It remains about12% slower than copied ROIs.

Matched sampler large-batch medians (ns/sample), old retained → prepared:
normals287.40→251.28,presence/direction379.53→311.95,width131.63→91.98.
Reductions12.6%/17.8%/30.1%. The same-executable copied controls measured208.48,
268.65 and59.98ns respectively; remaining grid addressing/locality cost is not
eliminated. Plans are reused within channel groups, not across the separate
normal/prediction/width sampling passes. No new numerical approximation introduced.

156native cases and19Python tests passed (1skip), including generic mixed dtype/
fill sharing, packed offsets, independent width chunks, shifted-layout/default
rejection, lifetime and concurrent reads. Approved command:
`/home/hendrik/villa3/scripts/run_fiber_tests`, suites `direct-sampling-window`,
`direct-sampling-window-full`, `retained-sampling-diagnosis`,
`direct-separate-gates-build`, `direct-sampling-window-checks`. Results are in
`sampling_retained_prepared_v1/REPORT.md` and
`sampling_retained_prepared_diagnosis_v1/REPORT.md` beside the previous reports.
These contain full raw profiles/output geometry and latency distributions. The
opt-in control remains `sampling_window_voxels=64`; default0 is unchanged.

#### Unified scalar-equivalent channel sampling (current implementation)

`FiberChunkTensor::sampleCandidateChannels` now samples prediction, normal and
optional width in one frontier pass. Float tracer positions are widened to double
before a single coordinate preparation. Compatible grids share one corner address
across all channels; independently chunked width still re-prepares safely. The
existing scalar materialization helpers retain their exact arithmetic, validity,
width rounding and unit conversion. An invalid direction does not suppress valid
normal/width outputs. Disabled outputs are empty. Ordinary float batch, owned GPU
packing, generic/custom sources and multi-worker paths are unchanged. This path
also avoids temporary double-point/direction vectors and intermediate normal
derivative wrappers. No voxel copies, per-point cache calls or locks are added.

Full112Paris4 exact replay: **67.810s**, previous retained76.561s (**11.4% less
elapsed time**). This is approximately the historical copiedROI68.354s timing,
without its voxel copying. Same protocol and32case concurrency/one native worker
per case, no inference/GPU, GCC13 RelWithDebInfo. All output fields except timing
match the historical reference exactly:0GTerrors,81.383%retained,181pieces.
8132acquisitions,0fallbacks/voxel copies,47.25MiB maximum retained payload/window.
Pilot32prefixes:4.874/4.840s (mean4.857), previous5.527/5.521s. Case136long
diagnostic14.313s, previous15.896s. Full per-case mean10.930s,median7.848s,
P95=39.154s,min1.083s,max46.725s (concurrent latencies, not standalone).

Full replay measured **worker wall seconds** (not additive process elapsed time):

|Phase|Previous retained|Unified|
|---|---:|---:|
|All candidate channel sampling|1055.215|864.237|
|Candidate scoring|233.072|239.062|
|Pruning|31.670|30.491|
|Task construction|14.173|13.503|
|Chunk acquisition|4.999|5.043|

`TraceProfile.combinedChannelSampleSeconds` replaces the three frontier sampling
timers for this path; remaining `widthSampleSeconds` belongs to other calls.
There are1,427,719,122candidate tasks in both full runs. Current scoring is
167.4ns/task and combined sampling605.3ns/task when normalizing worker wall time;
do not equate these concurrent workload averages with isolated CPU cycle counts.

Stage microbenchmark medians, small648 / large52,488points, ns per point:

|Stage|648points|52,488points|
|---|---:|---:|
|Coordinates/floor/fractions|8.36|9.14|
|Chunk address preparation|33.00|37.24|
|Prepared 8-corner scalar gather|11.81|16.89|
|Address plus seven-channel gathers|137.81|198.05|
|Scalar interpolation, pre-gathered|3.33|4.18|
|One vector: table lookup + tensor accumulation|37.23|41.15|
|One vector: principal-axis solve + normalization|100.35|107.69|
|One vector: complete pre-gathered reconstruction|167.46|159.55|
|Prediction alignment/presence materialization|18.92|21.10|
|All channels, separate passes|659.65|671.27|
|All channels, unified pass|495.40|571.09|

Chunk addressing is a modest cost, not the dominant computation: six runtime
integer divisions, bounds checks and eight offsets. The expensive vector part is
the analytic principal-axis solve (sqrt/acos/cos/cross products/normalization),
not raw byte decoding, which already uses a256×256lookup table. Normal and fiber
direction each require a reconstruction when valid. Gathering/layout overhead
and candidate scoring remain significant; no lock or I/O attribution is implied.

Measurements use shared production helpers
`accumulateLasagnaCompactTensorCorners` and `finalizeLasagnaCompactTensor`; the
existing interpolation wrapper calls both unchanged. Six real/seam locations,
five rotated repeats each,32loops for648points and2loops for52,488points. They
are deterministic local clouds, not recorded frontiers. Stage tests write/read
prepared arrays (address plans are roughly144bytes), so these are throughput
measurements including buffer traffic, not pure arithmetic latency. They are not
additive production cost accounting. Hardware counters remain unavailable.

Validation:156native cases plus19Python tests pass,1skip; full historical replay
exact. Approved command remains `/home/hendrik/villa3/scripts/run_fiber_tests`.
Select `direct-sampling-window`, `direct-sampling-window-full`,
`unified-sampling-diagnosis`, `direct-separate-gates-build` in
`local/fiber_tests.json`. Reports and raw profiles:
`paris4_direct_separate_gates_20261027_v1/sampling_unified_v1/REPORT.md` and
`sampling_unified_diagnosis_v1/REPORT.md`. Defaults remain unchanged;
`sampling_window_voxels=64` enables the measured retained direct path.

#### Optional in-pass diagnostic accounting

`sampleCandidateChannels(..., FiberSamplingProfile*)` accepts an optional native
diagnostic profile. Its measured specialization partitions actual calls into
exclusive coordinate/address/gather, direction and normal tensor/axis, presence,
width and output/loop/control intervals. Integer-nanosecond stage sums equal
the timed root exactly; nested costs are not counted twice. Counters accumulate
across calls and remain valid on empty calls and exception unwinding. One profile
per thread; reset it explicitly between measurements. Ordinary calls select the
unmeasured specialization, with no per-point clocks or instrumentation branches.

This resolves the previous isolated-stage accounting gaps, but the clocks have
large observer overhead: large-batch mean1404.93ns instrumented versus545.13ns
plain. Do **not** use these instrumented stage times as uninstrumented costs or
subtract a guessed constant. The complete additive table and controls are saved
in `sampling_unified_inpass_v1/REPORT.md` beside the preceding reports. An important
measured workload fact is that normal tensor/axis reconstruction executes for
43.19%of these points, not for every point; a universal two-vector estimate is
incorrect here. Deeper profiling was stopped at the user's request after these
measurements; instrumentation remains disabled in normal tracing.

### Comparing exported routes with online reference replay

`FiberChunkRoutingRules` loads the completed export's `path_reduction.msgpack.zst`
and `continuations.msgpack.zst`. It checks their generation, matching/construction
settings and complete source-file inventory against `routes.manifest.msgpack.zst`.
The active cache identity, directory, scale and configuration are checked again
at replay time; rules from another cache cannot silently be applied.

Python usage with an existing `vc.fiber_trace.ChunkCache`:

```python
rules = cache.routing_rules("/path/to/completed-export")
legacy = cache.replay(reference_base, evaluation_normals, replay_config)
controlled = cache.replay(reference_base, evaluation_normals, replay_config,
                          rules=rules)
```

Both modes use `traceFiberChunkReferenceReplay` and the same reference protocol.
The optional rules reuse duplicate reduction, the complete two-sided uniqueness
graph, and the exporter's oriented ownership/overlap-alias callbacks. Legacy
defaults are unchanged. Ownership starts empty for **each independent replay
segment**: GT case order, parallel scheduling and forward/reverse evaluation must
not steal geometry from one another. This does not reproduce the global export's
seed-order ownership decisions.

For the actual export, `StoredRouteSet(lines_base, end_reasons).replay(...)` uses
the same initial matcher, prediction-based restart search and reference evaluator.
It follows the selected saved route's original vertices to its saved end, with no
new boundary join or geometric resampling. `fused_member_stop:` provenance wraps
the original endpoint reason; presence/invalid-width stops retain their filtering
semantics. Populate the immutable route set from the export's MessagePack pages.
Its segment R-tree narrows initial candidates without replacing the exact matcher.

Always report retained reference distance and pieces, not just GT errors. A cache
with no usable paths can report zero retained errors while covering no reference.
Distinguish no-initial-path, ambiguity, missing-overlap, ownership and presence
stops from GT-distance violations. An export for Paris4 covers only the Paris4
subset of a multi-scroll benchmark; historical runs using a different prediction
resolution are context, not a controlled comparison.

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

## Continuous CPU tracing with neural volume coverage rules

`scripts/run_fiber_stream_volume.py CONFIG.json OUTPUT` is an alternative to
constructing and stitching cached chunk paths. It uses the existing chunk seed
generator (`findFiberChunkSeeds`, Python `chunk_seeds`) but traces continuously
with the native beam implementation. A storage boundary does not reset direction
history or the cumulative arithmetic width mean. No neural model is run here;
the inputs are completed prediction Zarrs and the CT volume.

The neural coverage rules have a native spatial-index implementation in
`FiberVolumeCoverage.hpp/.cpp`; regression tests compare it against the shared
Python reference `vesuvius.neural_tracing.tracers.fibers.tracing.volume_policy`
used by the neural volume runner. With `policy_grid_base: 8`, the rules are:

- Collision radius 16 base voxels and absolute tangent dot product > 0.8;
  three consecutive contacts stop a direction. Sibling halves do not collide.
- Defer nearby starts within 128 base voxels; suppress covered main seeds within
  16 base voxels. Chunk NMS/tube candidates replace the neural plane/fill-in
  candidate generator; there is no additional fill-in pass.
- Preserve the neural CT-zero, duplicate-run trimming, minimum 48-sample piece,
  and endpoint-link rules. Paths are sampled at 8 base voxels for those rules.
- The sole extra collision condition is width compatibility: known widths must
  satisfy the tracer's `width_hard_relative_tolerance`, relative to the querying
  trace's accumulated width. Unknown widths cannot reject a collision.

This workflow does not enable the chunk conflict filter or FW/BW checks. Its
collision semantics are not the older normal-oriented ellipse policy. Neither
stopped directions nor discarded short pieces are GT error measurements.

Use the regular volume runner's prediction, cache, trace and chunk settings,
plus this section (distances here are base voxels):

```json
"stream": {
  "workers": 128,
  "active_fibers": 1280,
  "seed_workers": 8,
  "max_open": 32,
  "policy_grid_base": 8,
  "decision_base": 64,
  "maximum_base": 64000,
  "ct_locator": "/path/or/url/to/ct.zarr",
  "ct_level": 2
}
```

Set native `parallel_threads: 1` and leave `chunks.gpu_library` empty.
`FiberStreamVolume.cpp` owns one shared C++ worker pool. Priority queues first
serve ready collision/CT checks, then completion/admission, then beam advancement
and bounded seed preparation. `active_fibers` defaults to ten times `workers`
and counts seed pairs (two directional trace states each), **not OS threads**.
The legacy `rows` option is accepted as the fallback worker count.

`FiberTraceCursor` suspends the same beam loop used by synchronous VC3D tracing
at frontier boundaries, retaining its branches, scratch, direction history and
width mean. It can resume on any pool worker. CT cache misses register a
continuation with the regular native chunk-cache readiness listener and return
the worker to other work. Collision queries use native spatial indexes with
shared-reader locks; final ownership publication is ordered in a single queued
lane so concurrent finished paths cannot both evade duplicate suppression.
That lane does not park other workers. Python only prepares metadata, enters
the native runner with the GIL released, and forwards cancellation.

`progress.json` includes CPU seconds, busy workers, active fibers, CT suspensions
and per-priority work time/job counts (tiers 0: checks, 1: completion/admission,
2: trace/seed preparation; tier 3 reserved). `scheduler.json` records the native
backend and effective limits. This CPU-only workflow does not request GPUs or
run inference. The previous coordinator is retained only behind the explicit
`--python-reference` diagnostic flag for matched before/after measurements.

`busy_workers` counts assigned jobs, including jobs blocked in prediction-cache
reads; it is not CPU utilization. Prediction sampling-window loads currently
use blocking retained-chunk reads, unlike the asynchronous CT checks above.
The 2026-10-28 full-volume run exposed a shared-cache eviction/scheduler lock
cascade after a fast initial pilot: 224 tracing workers waited for chunks while
an eviction scan held a source mutex and a decode submitter held the shared
selection gate waiting for that mutex. Small-pilot throughput therefore does
not establish sustained full-volume throughput. The run was stopped for cache
integration and pressure testing; this limitation must be verified under a
filled cache before resuming production.

The cache pressure redesign removes that lock chain:

- Publishers acquire source prerequisites **before** the global scheduling gate;
  waiting for a busy source no longer freezes unrelated queue selection.
- Decoded payloads have a separate eviction LRU. Missing/all-fill chunks use a
  sparse, source-local two-bit bitmap (16-cubed chunk pages), not heavy LRU entries.
  Negative knowledge survives metadata eviction, but `invalidate()` clears it.
  Completed volumes need no refresh; live publishers still must invalidate when
  previously absent data becomes available. This does not change source mutability.
- A bounded two-thread housekeeping pool runs at most one eviction job per shared
  budget. It uses try-locks on busy sources. Headroom is two average decoded chunks
  or one outstanding-request wave, capped at one eighth of the budget. Average
  chunk bytes use an EMA; cleanup starts halfway into that reserve. Tiny caches
  holding fewer than sixteen average chunks use hard-limit enforcement only.
  Callers join eviction only after exceeding the hard decoded-byte ceiling.
- Byte accounting covers cache-owned payloads and recycled buffers, not caller-held
  sampling leases, in-flight decode buffers or total process RSS. Parked blocking
  readers retain their existing protection. Background eviction is serialized per
  shared budget to preserve victim selection; it is not one worker per source.

`progress.json.decoded_cache` reports decoded/recycled bytes, capacity, requested
reserve, pending requests, background passes/bytes evicted and hard-limit passes.
Use pressure tests, not just an initially empty-cache timing, to validate liveness.
The 512-MiB pilot also exposed a CT readiness race: a published chunk could be
evicted before its queued waiter observed it. CT wake checks now re-request missing
chunks instead of waiting indefinitely for an already-completed fetch.

On the matched 32-tile Paris4 pilot (224 workers, 2240 active pairs, 4096-base
maximum half-trace, GCC 13 RelWithDebInfo/LTO), three 64-GiB runs averaged
34.80 s versus the previous 34.54 s: no meaningful no-pressure speedup. A 4-GiB
run completed in 35.44 s while evicting 13.6 GiB in the background; a 512-MiB
stress run completed in 100.37 s with 1.5 TiB cumulative decoded eviction.
Both finished without stranded work. These are bounded pressure checks, not a
new full-volume ETA. Configs, individual timings and the initially stranded CT
pilot are retained in `evaluations/cache_pressure_20261028_v1/README.md`.

Output `tiles/X_Y_Z.msgpack.zst` pages belong to seed tiles, not clipping regions:
their paths can extend far beyond the tile. Each piece includes base coordinates,
seed provenance, endpoint reasons and links, plus the attributed output below.

### Extensible fiber attributes (native stream output v2)

The packed storage remains a MessagePack map compressed with Zstd. Native stream
pages now declare `version: 2`, `attribute_schema_version: 1`; `run.json` declares
`output_schema: "fiber-stream-attributes-v2"`. Old pages are rejected on resume;
there is no conversion, backfill or reuse of historical fiber output. Prediction
and CT caches are independent and reusable.

For N saved `points_base`, each piece has three named dictionaries:

- `point_attributes`: each attribute's `values` contains N entries.
- `segment_attributes`: N−1 entries; entry i belongs to points i and i+1.
- `fiber_attributes`: whole-piece values (scalars for the fields below).

Every attribute has `values`, `units` and a human-readable `meaning`. Generic
readers validate cardinalities and retain unknown optional attributes; a workflow
separately validates the fields/units it requires. Adding optional names does not
change the container version. Changing the meaning of an existing name requires
an explicit schema change. `FiberAttributes.hpp` provides shared creation,
validation, resampling and piece-summary helpers; `FiberPackedStorage.hpp` retains
unknown fields during MessagePack roundtrips. This is not the separate strict
VC3D annotation JSON schema.

| Association | Field | Units / semantics |
|---|---|---|
| Point | `fiber_normal` | Unit XYZ axis; null when unavailable |
| Point | `width` | Local predicted width, base voxels; **not** the running mean |
| Segment | `modulated_presence` | Actual chosen-step direction/width-modulated presence, before additive smoothness |
| Segment | `tracer_cost` | Actual incremental chosen-step objective, including smoothness and any enabled additional objective terms |
| Segment | `traced_length` | Source arc length represented by the segment, base voxels |
| Fiber | `width_mean` | Piece-local length-weighted mean of local point widths |
| Fiber | `modulated_presence_mean` | Length-weighted segment quality |
| Fiber | `modulated_presence_integral` | Sum of quality × represented arc length |
| Fiber | `tracer_cost_sum` | Sum of segment costs |
| Fiber | `tracer_cost_per_base_voxel` | Cost sum / represented arc length |
| Fiber | `traced_length` | Sum of represented arc lengths |
| Fiber | `coverage_width_reference` | Separate scalar resume state for the unchanged collision policy; **not** the reported piece-average width |

Scores, local widths and scoring normals travel with the chosen beam nodes and
are emitted when committed. If normal-aware scoring is disabled, only the
accepted point's normal is sampled for metadata; no scores are re-evaluated.
The seed normal is sampled once per half at startup. Reversing the backward half
reverses segment order without negating quality/cost. Normals are axes, not
signed traversal directions. At resampled points, widths interpolate linearly;
normals interpolate sign-aligned and normalize, with invalid values propagated.
Segment scores are re-binned from recorded values by source-arc overlap; clipping
apportions a step's cost rather than charging it twice. No new inference or path
re-scoring is performed. `traced_length` distinguishes source arc from a saved
chord across a bend. The old `width_history_base` and `cumulative_cost` fields
are absent. Running width means remain internal tracing state only, apart from
the single coverage reference needed to preserve resume behavior.

`run.json` records the complete settings identity. Repeating the command restores
completed pages and their ownership, then retries unfinished seed tiles. A lock
prevents simultaneous writers. Live traces are not checkpointed, and concurrent
ownership decisions are scheduling-dependent; resume is not a promise of a
bit-identical uninterrupted run. Output formats are versioned independently of
the existing cached-chunk route format.

Regression tests cover native seed ownership, exact cursor/synchronous beam
parity across worker migration, cold CT reads, completed-page resume, cancellation,
coordinate scaling, output roundtrip, and shared neural-policy parity. Local
production inputs and pilot timings remain under the ephemeral experiment
directory rather than becoming repository defaults.
