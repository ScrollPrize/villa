# Plan: establish training measurements

## Two-stage per-sample pipeline benchmark

- Mechanically split the existing 3D `load_sample` operation at the current
  value-augmentation boundary. The synchronous public result and numerics must
  remain unchanged; raw load retains deterministic augmentation parameters.
- Add one shared reusable ordered two-stage pipeline implementation rather than
  another private trainer queue. Free stage-one workers claim monotonically
  generated sample indices; a bounded raw-ready queue feeds free augmentation
  workers; a bounded augmented-ready queue feeds deterministic reordering and
  batch assembly. Only bounded in-flight work is generated.
- Use subprocesses for both CPU stages. Carry tensors through torch
  multiprocessing shared-memory reductions and record queue/blocking costs;
  do not silently fall back to threads or unbounded futures.
- Bound all queued, executing and reordered samples with one global credit
  window released only at ordered delivery. The coordinator continuously
  drains completions into a bounded reorder map; it never waits for the next
  index while leaving later completions undrained. Queue size/empty probes are
  diagnostics only, never correctness conditions.
- The raw payload contains every identity field, raw volume/mask, deterministic
  augmentation parameters, CP/crop metadata and value-independent target spec.
  Synchronous `load_sample` composes the exact same stage-one and stage-two
  functions used by the benchmark.
- Make worker failure, early exit and lack of progress explicit errors. Cleanly
  stop/join workers and queues on success, error and interruption.
- Keep failure/control notification independent of capacity-limited data queues;
  poll process sentinels. Stop scheduling before teardown, grace-join, then
  terminate/reap only stuck owned children and close queue feeder resources.
- Add a standalone 3D benchmark CLI around the real S1a loader. Configure
  loading/augmentation worker counts and queue capacities; report end-to-end
  samples/s, batches/s, inter-batch stall distributions, per-stage service
  distributions, reorder high-water and queue wait/blocking. Save a JSON result
  with exact config, command and environment metadata.
- This first benchmark is one host-global CPU pipeline, not one pipeline/rank.
  Record workers, global credits, both queue capacities, native thread limits,
  approximate peak queued tensor bytes (~10 MiB/raw 128³ sample before metadata
  and augmented output), CPU utilization, cold/warm state, startup, sample
  range, IPC/stack cost and warmup exclusion. Set native threads/process to one.
- First validate with synthetic variable-delay stages: exact ordered values,
  no batch-level head-of-line worker assignment, bounded lookahead, error
  propagation, hard worker exit, early consumer stop, saturated result queue,
  slow earliest sample and repeated cleanup. Verify every split synchronous
  sample field exactly matches the original profile-disabled loader result.
  Then run the actual benchmark from `.venv_las` against the
  saved S1a configuration under `/ephemeral/hendrik/fiber3d_train2`.
- This phase does not alter the normal training path or claim the 221 samples/s
  target. Results select later augmentation kernels and worker topology.
- Strict deterministic batch order can still stall on a genuinely slow earliest
  sample. Per-sample scheduling removes batch-to-worker coupling; it does not
  permit later samples to change batch membership or overtake consumption.

### Spec update

Document the opt-in benchmark, deterministic per-sample scheduling, bounded
queues, process isolation, metrics and unchanged normal-training defaults.

### Docs updates

Document the queue ownership/lifecycle and exact benchmark command/results.
Keep `planning/task_log.md` to short implementation and result entries.

### Testing

Run focused synthetic pipeline and 3D loader tests, `git diff --check`, then a
real short S1a benchmark. Report warmup/sample count and mean/p50/p95/max stalls.

### Changelog update

Add one line when the reusable pipeline benchmark is complete.

## Loader-worker follow-up
- Add an opt-in profile_loader flag, requiring normal profile_enabled, to
  reuse existing per-batch worker timers without changing samples or scheduling.
- Carry worker ID/PID and index plus coordinator wait/receive timestamps in
  existing timing metadata; save per-rank JSONL. No extra tensor payloads.
- Compare stage distributions and interval overlap per worker/rank; distinguish
  batch service from post-service transport/order/prefetch waiting. Detect later
  batches already computed while an earlier batch blocks, without claiming
  worker-end timestamps prove delivery into the coordinator's queue.
- Run the same 120-step eager two-GPU experiment, trace disabled to avoid extra
  CUDA profiling overhead. Record detailed CPU timing overhead as a limitation.
- Test metadata propagation and unchanged batch values; update specs/docs and
  minimal task log with results. No BN or scheduling implementation changes yet.

## Single-threaded VC3D stage profile

- Add a per-call VC3D coordinate-sampling worker limit to sampler options and
  expose it through the Python binding, shared sampler factory, 3D loader
  config, and reusable benchmark CLI. Zero/omission preserves current automatic
  behavior; one forces serial coordinate interpolation without changing math.
- Keep source-read and decoder admission at one for the controlled run. Do not
  conflate these independent limits or resize process-global pools.
- Instrument blocking coordinate sampling around dependency discovery, chunk
  resolution, and interpolation. Add cumulative cache decoder counters around
  actual codec calls so per-request deltas exclude coordinate interpolation.
  Counters are monotonic and scoped to one cache state: attempts, successful
  chunks, input/output bytes, and summed codec duration are snapshotted around
  the blocking call and never reset.
- Carry native counters through existing sample profiling metadata. Summarize:
  decoder chunks/s and MiB/s, coordinate generation MP/s, coordinate sampling
  MP/s, and equivalent 128^3 samples/s for each stage. State explicitly when
  cache hits mean a sample performed no decode. Coordinate generation means
  Python `_build_geometry_maps` over `prod(patch_shape)` output points;
  dependency/resolve/pinning remains separately named; sampling means the
  native `samplePinnedPoint` interpolation loop over valid requested points.
  Aggregate native-unit rates use counts divided by summed service time;
  separately report host-wall rate/effective concurrency. Never present summed
  concurrent worker time as elapsed throughput.
- Run the stable argument-free benchmark launcher with all three VC3D limits at
  one, using the same warm S1a input and worker topology as prior comparisons.
  Compare pipeline throughput and raw-stage CPU/wall ratio against the recorded
  three-run read=1/decode=1 result. Also run a fresh-process/no-warmup profile
  against the existing persistent cache to observe codec work; do not delete
  cached data to manufacture misses. Zero codec work reports N/A.
- Add focused C++ tests for serial-option output equivalence/timing counters and
  Python tests for config/argument propagation and rate aggregation. Rebuild and
  reinstall only the explicitly requested VC binding, then run the benchmark.
  Validate negative rejection, omission/zero preserving automatic behavior,
  one forcing scalar dispatch, and positive limits being bounded without
  resizing the global pool. Verify every new stat reaches Python and preserve
  strict requested-level errors. Keep behavior portable without SIMD or
  platform-specific assumptions.

### Spec update

Document optional coordinate-sampler concurrency and opt-in stage metrics,
including their units and unchanged default behavior.

### Docs updates

Correct the earlier unsupported probe attribution and add the controlled
single-thread benchmark command, stage definitions, results, and limitations.

### Testing

Run targeted VC sampler/cache tests, Python config and benchmark aggregation
tests, `git diff --check`, binding smoke validation, and the real warm benchmark.

### Changelog update

Record the explicit sampler control and stage-resolved benchmark profiling.

## Requested-level dependency and pinned-sampling optimization

- Use the existing isolated `dependencySeconds` and `sampleSeconds` scopes on
  the real S1a 128^3 benchmark. Add bounded diagnostic counters where needed to
  distinguish visited coordinates, reused previous footprints, unique
  footprint changes, and actual hash probes. Report before/after service-time
  distributions and normalized rates; all runs remain CPU-only. Counters are
  worker-local plain integers aggregated after tile ranges, never atomics or
  event logs in the hot loop. Compare reference-disabled and optimized modes
  from the same instrumented binary so counter overhead is matched.
- First mechanically extract the corner visitor's existing current-chunk run,
  compact boundary layout, single-dependency offsets, immutable pinned-pointer
  access, and linear-before-hash dependency resolver into internal reusable
  primitives without changing visitor behavior; validate that refactor before
  the functional optimization. Then make dependency discovery and pinned scalar
  sampling consume those primitives. They compute the one, two, four, or eight
  unique chunk keys covering a voxel cell. Compare the current footprint with
  the immediately preceding coordinate and skip set work when identical.
  Preserve arbitrary coordinate order, invalid/out-of-bounds handling,
  non-power-of-two chunk shapes, and boundary behavior. A power-of-two shortcut
  may be used only behind an exact general fallback and measured separately.
- Preserve the coordinate contract exactly: sentinel and finite checks plus
  logical-level transform happen once, then both discovery and sampling consume
  the same level-coordinate voxel-cell footprint. The footprint distinguishes
  out-of-volume fill corners from required keys. Direct gathering is legal only
  when all eight in-volume corners occupy one valid payload and existing offset
  bounds/status checks succeed; direct and boundary paths share the existing
  ordered `std::fma`, clamp, and uint8-truncation interpolation helper.
- Preserve caller-specific policies explicitly: the corner visitor clamps its
  high corner at the final voxel and may fill missing corners independently;
  the scalar sampler currently treats an out-of-range +1 corner as fill and a
  missing required chunk as a missing/zero scalar sample. Shared layout code
  must parameterize these differences, not make either caller adopt the other.
- Do not reintroduce `LocalChunkCache` literally into discovery: discovery does
  not resolve payloads. Reuse its locality principle in the pinned sampling
  path with a worker-local last-key/result cache, and restore the existing
  single-chunk trilinear direct-gather fast path. Boundary cells retain exact
  multi-chunk lookup. Avoid mutable shared lookup state because the public
  sampler may use multiple workers.
- Reuse the current render implementation's exact mechanisms: single-chunk
  eight-corner gathering, current-chunk run reuse, compact boundary layouts,
  linear-before-hash unique-key resolution, tiled traversal, and pre-resolved
  immutable pinned payloads. A parallel second implementation is forbidden.
- Preserve byte-identical uint8 outputs and coverage for nearest/trilinear,
  serial/parallel, chunk-boundary, missing/fill, invalid, and non-power-of-two
  cases. No fast math, reduced precision, changed traversal results, or GPU use.
- Report dependency and interpolation target status independently. If either
  remains below 20 equivalent 128^3 samples/s, continue profiling/iteration or
  explicitly report the unmet target; total pipeline improvement alone cannot
  mark this optimization complete.

### Spec update

Document exact dependency-footprint reuse and worker-local pinned locality as
implementation guarantees, while keeping public sampler behavior unchanged.

### Docs updates

Record the isolated baseline, hotspot mechanism, before/after benchmark, build
type, dataset, worker limits, and portability/numerical-equivalence notes.

### Testing

Add focused C++ regression cases for repeated footprints, transitions across
each chunk axis, and a pre-change scalar reference or checked-in goldens.
Compare dependency sets, bytes, coverage, and requested/missing/error stats over
random transformed-level coordinates, negative/fractional edges, the last
voxel, sentinel/NaN/Inf, fill/missing/error, nearest/trilinear, non-power-of-two
chunks, and serial/parallel execution. Run sampler/cache tests, focused Python
tests, binding smoke checks, `git diff --check`, and the stable CPU-only S1a
benchmark.

### Changelog update

Record the exact requested-level dependency and pinned-gather acceleration once
validated.

## Two-GPU measurement implementation (authorized continuation)
- Add opt-in per-rank JSONL iteration/stage wall timers and bounded CPU/CUDA
  profiler traces to the real trainer. Record enqueue scopes accurately; do not
  add stage CUDA synchronizations or change training math for wall timing.
- Add explicit compile mode off/default/reduce-overhead, compiling the DDP
  wrapper while retaining raw_model for serialization/evaluation. No silent
  eager fallback on compiler errors. Defaults remain off.
- Use fixed opt-in model initialization seed for paired experiments.
- Run short eager and compile comparisons with batch 8 per rank on GPUs 0,1.
  Preserve BF16, LR, sampling, testing and visualization; record compilation
  startup separately from steady steps. Use an independent trace window so its
  overhead can be excluded from ordinary timings.
- Generate isolated configs/logs/traces with the specified venv under
  /ephemeral/hendrik/fiber3d_train2, reusing existing input cache.
- Test disabled instrumentation, trace lifecycle, compile selection and raw
  checkpoint keys, then inspect actual per-rank traces and compare timing
  distributions. Shared host contention remains a limitation.

1. Recover configuration/precision from reference events and checkpoint.
2. Audit the historical quality-versus-wall-time curve and timing distributions.
   Distinguish sparse logged samples from all-step statistics; explicitly report
   no crossing rather than extrapolating a threshold time.
3. Implement reusable read-only event analysis and synthetic regression tests.
4. Inspect current timers and compare coverage with the normal training path.
5. Once eight GPUs are free, run the reference configuration in a separate
   output directory for 300 optimizer steps, retaining tests and visualization.
   Record startup and warmup separately. Use a bounded opt-in torch CPU/CUDA
   profiler window after warmup, compare profiled/unprofiled throughput.
   Instrument normal training, never duplicate model/loss/optimizer code.
6. Separate loader wait, H2D, targets, forward, backward/DDP, optimizer,
   scalar collectives, test, visualization, snapshots and startup. Report
   patches/s, peak VRAM, cache state and mean/median/p95/min/max with commands,
   code revision, input, warmup and iteration counts.
7. Select experiments from task.md based on evidence; measure quality per hour.

Independent review clarifications:
- Preserve direction/presence loss and test/angle_mean_deg alongside total loss.
- Historical elapsed time begins at the first event; unrecorded process startup
  is unknown. Fresh runs must record actual launch time.
- Pin reference config/checkpoint identity and current code/environment; do not
  assume the historical and current test implementation are identical.
- The 300-step run measures throughput, not achievement of the one-hour goal.
- Optimization options remain future experiments for user selection.

The existing --benchmark is forward-only and single-process, not a DDP
training baseline. Do not disrupt occupied GPUs or call contended measurements
an isolated baseline. Profiling must not consume extra samples or change math.

## Spec update
Document opt-in profiler defaults/lifecycle, enqueue-versus-device timing,
compile modes and raw evaluation/serialization, optional initialization seed.

## Docs updates
Add baseline findings, exact analysis command and follow-up measurement protocol;
update status and current task log.

## Testing
Synthetic events: threshold crossing/no crossing, elapsed origin, timing warmup
filtering, missing series and config preservation. Audit the actual saved run.
Live instrumentation later needs trainer tests plus an actual eight-GPU run.
Current two-GPU validation adds trace lifecycle/disabled-path tests, compile
selection and unchanged raw checkpoint keys; compare eager/compiled losses
with identical seeds and batches and report numeric drift. Global batch 16
is not the reference global batch 64 convergence baseline.

## Changelog update
Record reusable reference audit and baseline findings.
Also record opt-in normal-training profiling and explicit compilation support.
