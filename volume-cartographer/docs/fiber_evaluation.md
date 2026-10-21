# Callable reference evaluation

`vc.fiber_evaluation.evaluate` runs explicit reference cases against a prediction
source using the native chunk or direct beam tracer, reference replay protocol and cooperative
case queue. It returns results directly: no subprocess, config reload, model
unload or GPU allocation. It is suitable for an existing application's evaluation
phase, including a caller that already owns a demand-inference producer.

This extracts the shared evaluation orchestrator and CP-bounded reference
selection. It does not include dataset splits, parameter sweeps, visualization,
fiberlet/skeleton evaluation or model construction. The
input schema is new and deliberately smaller than the experimental `replay-set`
CLI; old experimental configs are not silently accepted.

## Usage

```python
from vc.fiber_evaluation import evaluate
from vc.fiber_trace import EvaluationControl

control = EvaluationControl()
report = evaluate(config, base_directory="/data/evaluation",
                  control=control, progress=print)
```

The application can poll `control.status()` or call `control.cancel(reason)` from
another thread. A control is single-use. It reports total/completed cases and
started/finished/cancelled/error status. `finished` does not mean success if
`error` is nonempty. Errors propagate after all admitted work joins; a partial
case set is never returned as a successful evaluation. A successful cache chunk
remains reusable; failed builds are retryable with a fresh control and, for live
work, a fresh bridge session. Cancellation fails the old bridge closed. Progress
callbacks execute on the calling Python thread, never inside suspended native
contexts. Callback errors and Ctrl-C request cancellation and join the work.

Cancellation wakes publication waits and duplicate-cache waits. Arbitrary
blocking sampler I/O and active CPU kernels are not forcibly interrupted. A
configured live transport timeout bounds missing publication; it is not a
deadline for all I/O. Sources and attached readiness gates must not be mutated
while evaluating. The native scheduling API is supported on Linux/macOS;
unsupported platforms reject it explicitly. macOS has not been tested locally.

The equivalent thin CLI reads the config once and atomically replaces the output
only after success. Temporary output is created beside the destination, not in
the system temporary directory. Existing output is unchanged on failure.

```sh
python -m vc.fiber_evaluation evaluation.json results.json
```

Example configuration (replace paths, identity and points with your data):

```json
{
  "version": 1,
  "prediction_manifest": "predictions.lasagna.json",
  "normal_manifest": "normals.lasagna.json",
  "source_identity": "immutable-model-and-normal-dataset-identity",
  "chunk_cache": "chunk-paths",
  "scaledown_power": 0,
  "cache_bytes": 1073741824,
  "workers": 4,
  "contexts_per_worker": 8,
  "trace": {"replay_presence_cutoff": 0.1},
  "chunks": {},
  "replay": {"error_threshold_base": 20, "tangential_threshold_factor": 4},
  "reverse_interval_base": 128,
  "cases": [
    {"id": "forward", "points_base_xyz": [[100, 100, 100], [1000, 100, 100]]},
    {"id": "backward", "points_base_xyz": [[1000, 100, 100], [100, 100, 100]]}
  ]
}
```

All reference coordinates, intervals, returned points and lengths are **base
voxels**, not millimetres. Each case represents one direction; reverse the point
array to test the other. Optional `begin_arc_base` and `end_arc_base` select an
interval in that ordered polyline. No extrapolation or implicit reference
resampling is added. IDs must be unique nonempty strings.
Invalid geometry/bounds fail before case work starts. The caller must select
eligible reference spans or use the annotation selector below; no cohorts are bundled.

### Direct beam backend and annotation selection

Set `"backend": "beam"` (default is `"chunk"`) and omit `chunks` and
`chunk_cache`. This calls the existing beam engine directly; it does not build
chunk paths. Both backends use the same scheduler, metrics and result fields.
Beam width, lookahead, presence filtering, width and embedding costs use ordinary
`TraceConfig` settings. Direct starts/restarts use the reference point/tangent;
chunk starts choose an existing path in the acceptance ellipse. This adapter
does not import experimental prediction-based seed/trial search. Consequently
these backends need not produce identical paths or restart counts.

`reverse_interval_base` checks actual forward geometry, using the configured
acceptance radii, not the GT path. Direct reverse width gating uses the forward
committed width mean. Rejected windows use the shared replay rollback/filter
protocol. Presence eligibility remains the beam engine's candidate filter; it
does not introduce an additional GT-dependent decision.

Instead of explicit points a case can be `{"id": "fiber-a", "fiber_path":
"annotations/a.json", "direction": "both"}`. Files must already use base XYZ
coordinates. No coordinate metadata is interpreted as another transform.
The shared native `vc3dEvaluationReferencePlan` selects consecutive non-lasagna
CP spans and excludes tails before/after the first/last CP. Controls must match
dense line points exactly and in increasing order; no nearest-point guessing.
Version-1 annotations use their whole-fiber optimization mode. Version-3/4
annotations use each span's actual `interp_mode`. Excluded spans separate runs
and never become connecting edges. This is the historical
`cp-bounded-no-lasagna-v1` policy, not a new review/tag or dataset split policy.

Direction defaults to annotation `evaluation_direction`, otherwise both.
Output order is input file order, run order, then forward/reverse. Each result
has a unique expanded ID and original indices/arcs plus direction. Top-level
`reference_selection` records eligible/excluded spans and SHA-256 of each input
file; fully excluded fibers remain recorded with no cases. Files are read once
before worker startup. `reference_cases(annotation, id, direction=None)` exposes
the same selection without opening volumes or running traces.

Relative paths resolve against the config's directory (or `base_directory` in
the callable API); URLs are preserved. Optional `evaluation_normal_manifest`
separates acceptance normals from tracing normals; omitted uses the same normals.
`remote_cache_root` configures the existing native fetch cache. `cache_bytes` is
the shared decoded-volume budget, not a separate allocation per channel. Chunk
path cache identity must cover immutable predictions, normals and transforms.
Use a distinct directory/identity if any of those data change. No coordination
between separate processes writing a chunk-path directory is added.

`trace` uses `TraceConfig` field names. `chunks` and `replay` use `ChunkConfig`
and `ChunkReplayConfig` names described in [chunk tracing](fiber_chunk_tracing.md).
Defaults retain that API's numerical policy. Trace scale and a 16-base-voxel
step are derived from the prediction field; `parallel_threads` defaults to one
and must stay one because the queue owns CPU parallelism. Unknown settings are
errors. Width gating requires `enable_width: true` and the corresponding trace/
chunk width settings; it is not enabled implicitly. No precision changes.

Results retain case order regardless of completion order. Each case contains
segments, matches, failures, filtered stops, reverse checks, completed reference
arc and skipped seed-search distance. Top-level failure and filtered-stop counts
are separate. `retained_trace_length_base` sums the lengths of actual retained
trace polylines; it is neither reference coverage nor a scroll-weighted score.
Two directions intentionally count twice. There is no experiment-specific
combined score, automatic best selection or unit conversion.

## Live prediction

Optional `live_endpoint` contains `port`, `token`, `source`, and optionally
`timeout_seconds`/`heartbeat_wait`. The caller must already be serving the
[version-1 publication protocol](fiber_live_bridge.md). Evaluation opens and
closes only its own bridge connection, never starts or stops the producer.
Tokens are removed from returned/stored config. One source-level readiness
gate is shared across this batch; protocol case index/remaining-distance are
zero hints, not per-case dynamic priorities. Chunk readiness, not those hints,
controls correctness. More elaborate producer scheduling is not part of this
extraction. Offline evaluation requires already-published predictions; it does
not turn missing chunks into on-demand inference unless a bridge is configured.

### Connecting an existing demand producer

`lasagna.live_fiber_evaluation.evaluate_live` supplies the coordinator side of
that protocol around the same callable evaluator. It borrows `DemandStore`s,
validates each request against that source's product/channel paths and volume
bounds, and calls their existing `request`/`ready` methods. A ready response
means the coherent product has been published, not merely queued or inferred.
It does not duplicate tile scheduling, accumulation, interpolation or encoding.

```python
from contextlib import contextmanager
from lasagna.live_fiber_evaluation import evaluate_live

@contextmanager
def producer():
    # Application-owned context: existing DemandWorkerPool and GPU lease.
    # Construct deferred workers; do not block here waiting for a grant.
    with application_demand_session() as session:
        def pump():
            session.lease.tick()  # nonblocking allocation/pause/idle handling
            session.pool.pump()
        yield pump
    # The application session joins GPU workers BEFORE releasing its lease.

report = evaluate_live(config, {"scan": demand_store}, producer,
                       source="scan", base_directory="/data/evaluation")
```

`application_demand_session` above is the application's existing resource
owner, not a new factory supplied by this module. For CPU tests a factory can
directly enter `DemandWorkerPool(..., devices=["cpu"])` and yield `pool.pump`.
For GPU use it **must** retain the existing manager grant/lease discipline;
this adapter neither requests GPUs nor bypasses the manager. Automatic model,
store and adaptive-lease construction belongs to the later application
integration, not this small publication adapter.

The factory is entered lazily, at most once, only on a cache miss. Fully cached
evaluation needs no producer or GPU grant (`producer_factory=None` explicitly
rejects cache misses). One coordinator thread exclusively owns store mutation
and pumping; the caller must not concurrently pump those stores. No Python
callback runs on suspended native case stacks. Heartbeats keep publication
waits alive during a nonblocking resource wait; they never certify readiness.
Both factory entry and pump must return promptly, so resource waiting belongs
in the pump, not a blocking factory constructor.

The native consumers join before context teardown. The producer context then
closes/joins its workers and releases resources; borrowed stores remain owned
by the caller. Stop on cancellation/failure, without trying to complete an
unbounded inference backlog. Completed coherent chunks can be reused in a
new session; never reuse a failed producer's in-memory scheduling state.
No temporary config, evaluator subprocess, live config reload or model unload
is introduced. Tokens are still stripped from results. `DemandPublicationSession`
also exposes the scoped endpoint for another consumer of the same protocol.

Regression tests cover cold publication, warm no-producer reuse, validation,
failure and cancellation ordering. The opt-in real Paris4 CPU identity-model
test uses `LASAGNA_TEST_VOLUME` and `LASAGNA_TEST_ORIGIN_ZYX`; it runs the actual
demand worker/accumulator/publication pipeline and compares the published bytes
with the CT crop. This is correctness evidence, not a GPU speed measurement.

## Native and low-level Python APIs

`evaluateFiberChunkCases` in `vc/fiber_tracer/FiberEvaluation.hpp` takes a shared
chunk cache, sources, explicit cases and control. It uses the existing replay
core with cooperative cache waits and child construction jobs. Mutable shared
profiling sinks are rejected; any other native user callbacks must be thread-safe
and obey the scheduler's no-yield-under-lock rule. Configuration and reference
geometry are copied for the invocation; source/cache owners must outlive it.

Python callers with already-open handles can use
`cache.replay_many(cases, evaluation_normals, replay_config, control, workers=4,
contexts_per_worker=8, reverse_interval_base=0)` directly. Cases are
`EvaluationCase(float64_contiguous_Nx3, begin_arc_base=0, end_arc_base=None)`.
Native work releases the GIL; returned arrays own their memory. For a blocking
call's cancellation from Python, use a separate control thread or the high-level
`evaluate` wrapper, which also handles Ctrl-C. Ordinary single-case `replay`
remains available without the cooperative queue.

Direct equivalents are `beam_replay(field, normals, evaluation_normals, trace,
case, replay, reverse_interval_base=0)` and `beam_replay_many(field, normals,
evaluation_normals, trace, cases, replay, control, workers=4,
contexts_per_worker=8, reverse_interval_base=0)`. Native entrypoints are
`replayFiberBeamCase` and `evaluateFiberBeamCases` in the same header. Both batch
backends share the control/validation/scheduler implementation.

## Validation

With an existing native build and its Python environment:

```sh
cmake --build <build> --target vc_fiber_trace -j 8
python -m pytest volume-cartographer/python/tests/test_fiber_evaluation.py \
  volume-cartographer/python/tests/test_fiber_live_evaluation.py \
  volume-cartographer/python/tests/test_fiber_chunk_trace.py \
  --vc-build-python <build>/python
```

Tests compare serial versus batch replay, both directions, interval selection,
reverse checks, caller interruption, cancellation while publication/shared-cache
waiters are suspended, retry and CLI output. The documented cached Paris4
fixture variables in `fiber_chunk_tracing.md` also run the native batch and
callable orchestrator on a real 256-base-voxel reference span, comparing full
results with serial replay. No GPU inference or throughput claim is involved.
