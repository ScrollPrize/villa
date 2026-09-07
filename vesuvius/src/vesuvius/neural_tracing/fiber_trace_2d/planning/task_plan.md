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

## Cache-local fused coordinate generation

Superseded for the current implementation by **Native fused coordinate
generation** below. The completed cache-local Torch implementation remains the
reference backend, but the former requirements for implicit sampler traversal,
bitwise equality, and no final coordinate map no longer govern this task.

- Profile `_build_geometry_maps` independently on deterministic S1a samples,
  separating grid creation, metadata conversion, centering, affine mapping,
  smooth-control interpolation, final translation, NumPy conversion/scaling,
  and validity generation. Report allocations/bytes where available and
  mean/p50/p95 service times with one native worker and no GPU.
- Add one reusable native implicit-grid descriptor containing output shape,
  float32 centers and inverse affine, selected-to-base scale/base shape, and
  smooth mode/control arrays/origin/spacing. Both requested-level dependency
  discovery and pinned sampling must consume one shared coordinate evaluator;
  do not copy Fiber augmentation math into independent implementations.
- Evaluate coordinates directly in traversal order. The initial implementation
  may recompute each coordinate in the dependency and sampling passes rather
  than storing a full 128^3x3 map. Smooth controls are small and remain
  cache-resident; only final sampled values and coverage are materialized.
- Preserve the current sequential inverse-smooth component order for 3D,
  float32 casts and operation order, affine convention, selected/base scaling,
  finite/bounds checks, missing behavior, deterministic partitions and target
  metadata. Do not algebraically combine selected-to-base scaling with the
  renderer transform. Keep the eager path as an explicit runtime
  reference/fallback.
- Treat Torch `einsum` reduction/contraction and division rounding as explicit
  exactness risks. Compare generated coordinate bits before sampling as well as
  final bytes. Keep Fiber-specific smooth math in its coordinate provider and
  parameterize the existing shared dependency/pinned traversal rather than
  creating another sampler implementation.
- Default-enable only after reference/optimized comparisons establish identical
  sampled bytes, coverage, dependency sets and metadata across smooth modes
  none/1d/2d/3d, nontrivial affine/scale, borders, missing chunks and
  non-power-of-two chunk shapes, singleton control axes, origin/last-voxel and
  NaN/Inf cases. Do not use fast math or GPU resources. Unverified compiler/
  architecture combinations retain the eager default rather than silently
  changing numerics.
- Measure cheaper stages independently before implicit recomputation: remove
  the redundant float32 CPU `astype` copy, then fuse selected-to-base scaling,
  validity and coordinate-layout conversion around the existing final Torch
  map. Compare that retained-map approach and bounded coordinate tiles against
  evaluating smooth coordinates twice.
- Benchmark one worker and the same 128-process raw-only workload. Attribute
  the change to coordinate work, queue/IPC and total raw throughput; record
  mean/p50/p95 and memory/residency limitations.

### Spec and documentation updates

Document the implicit-coordinate contract, reference fallback, exactness tests,
commands, artifacts and before/after results. Record any unmet 221-samples/s
global target explicitly.

## Native fused coordinate generation

- Add one portable scalar C++ primitive to the existing `vc` Python package.
  It accepts the output shape, float32 control point/affine metadata,
  low-resolution smooth-control arrays, selected-to-base scale and base shape,
  and returns contiguous final float32 base coordinates plus uint8 validity.
  It must not depend on a `Volume` or duplicate volume sampling.
- Traverse output coordinates once in contiguous order. Derive `(z,y,x)` from
  loop indices, apply the affine transform in registers, evaluate the current
  1D/2D/3D smooth controls directly from their cache-resident grids in the
  existing sequential component order, translate, scale and validate, then
  write only the final coordinate and mask. Do not allocate a meshgrid,
  full-resolution displacement field, index tensors or intermediate maps.
- Keep the C++ function reusable and independent of Fiber loader state. The
  Fiber3D loader owns conversion of `_Augment3DParams` into the native call and
  retains the eager implementation as an explicit reference backend.
- Define `geometry_backend` as `auto`, `native`, or `eager`. `auto` is the
  default and selects native when the new symbol is present, otherwise reports
  and uses eager for source-tree compatibility. `native` fails clearly if the
  binding is stale or absent. Every profile records the resolved backend.
- Coordinate equality is no longer required. Validate finite/bounds semantics
  exactly and compare native/eager coordinates using reported maximum and mean
  absolute error, plus sampled image/mask agreement on deterministic fixtures.
  Native acceptance requires maximum coordinate error <= 1e-3 base voxel,
  mean error <= 1e-5 base voxel, exact validity masks on finite deterministic
  fixtures, and sampled uint8 mismatch <= 1e-4 of valid voxels with maximum
  intensity difference <= 1. Boundary-focused fixtures must retain exact masks;
  any boundary disagreement fails rather than being averaged away. Do not use
  fast-math or reduced precision. Scalar C++ is the portable
  baseline on Ubuntu/macOS and amd64/arm64; SIMD is a later guarded option.
- First benchmark direct sequential low-resolution control lookups. Only add a
  precomposed low-resolution 3D vector field if isolated measurements show an
  end-to-end benefit. Such composition is approximate because interpolation
  and sequential coordinate displacement do not commute; record its error and
  expose it as a separate backend rather than silently replacing direct mode.
- Measure native versus eager generation in a single process and through the
  same 128-process raw S1a pipeline. Report command, dataset, build type,
  warmup/count, mean/p50/p95 generation time and end-to-end samples/s. Keep all
  benchmark runs CPU-only.

### Spec update

Document the native coordinate backend, its input/output contract, direct
smooth-control evaluation, explicit eager reference mode, numerical tolerance,
and clear missing-binding failure.

### Docs updates

Document build/runtime selection, implementation structure, validation error,
and before/after benchmark results. State whether low-resolution precomposition
was implemented or rejected after measurement.

### Testing

Add native binding tests for none/1d/2d/3d smooth modes, singleton control
axes, nontrivial affine/scale, negative and final-volume boundaries, invalid
and nonfinite arguments, allocation overflow, and deterministic repeatability.
The binding output contract is contiguous `(Z,Y,X,3)` float32 ZYX base
coordinates and `(Z,Y,X)` uint8 validity; affine rows map output-relative ZYX
to source-relative ZYX and backward 3D smooth controls are evaluated x, y, z.
Add loader tests for backend
selection and stale-binding errors. Run focused C++/Python tests,
`git diff --check`, isolated generation benchmarks and the stable CPU-only S1a
pipeline benchmark.

### Changelog update

Record the native fused coordinate generator and measured throughput gain.

## Native anisotropic value augmentation

- Extend the existing opt-in per-sample timing dictionary with separate
  normalization, contrast, brightness, gamma, noise, isotropic blur,
  anisotropic blur, and final-mask timings. Timers must not change random draws,
  values, or normal unprofiled behavior. Capture a CPU-only baseline with the
  stable full-pipeline launcher before replacing the implementation.
- Add one standalone native function to the existing VC Python extension. It
  accepts a contiguous float32 ZYX volume, the float32 ZYX principal axis, and
  along/across sigmas, and returns a new contiguous float32 ZYX volume. It owns
  no global pool, releases the GIL, and uses exactly one calling thread so the
  existing process pool remains the sole augmentation parallelism.
- Reproduce the current operation as three sequential passes: principal axis,
  first perpendicular, second perpendicular. Construct the same stable
  right-handed basis, Gaussian radius `ceil(3*sigma)`, normalized weights, and
  replicate-border behavior. Precompute each pass's constant trilinear stencil
  from its tap offsets, merge identical integer offsets, and traverse the
  volume once per pass using two reusable output buffers. Do not materialize
  coordinate grids or dispatch one operation per tap. Merge only within a pass;
  retain all three passes so repeated border clamping is unchanged. Preserve
  principal/directional normalization and the strict `abs(dot)>0.9` reference
  selection branch.
- Add `value_augmentation_backend = auto|native|torch`. `auto` uses native on
  eligible CPU tensors and falls back to the Torch reference when an older
  binding is installed. Explicit `native` fails clearly when unavailable or
  ineligible; `torch` remains the comparison/debug implementation. Auto
  fallback is restricted to a missing binding or explicitly unsupported tensor
  contract, never native validation/runtime errors. Define handling for
  noncontiguous, non-float32, and gradient-tracking tensors. Profiling records
  the resolved anisotropic backend and disabled-blur cases.
- Compare native and Torch outputs on deterministic interior, border, axial,
  oblique, singleton-axis, zero-sigma, degenerate-axis, and no-anisotropic-blur
  fixtures. Current Torch generation performs a float32 voxel-to-normalized-to-
  voxel coordinate round trip whose rounding can vary with position and shape;
  a constant stencil intentionally approximates that behavior in addition to
  reassociating weights. Test full 128-cubed and non-power-of-two volumes with
  impulses, ramps, high-contrast random values, realistic normalized values,
  and both sides of the strict basis-selection threshold. Report
  maximum/mean absolute error. The initial full-size random comparison measured
  max `6.18e-5` and mean `1.64e-6`, so require max <= `1e-4 * max(1,
  max_abs_input)` and mean <= `3e-6 * max(1, max_abs_input)` for the
  oriented-blur output. Verify identical shapes, dtype, contiguity,
  determinism, input immutability, and unchanged masks/other augmentation
  semantics. Reject invalid dimensions, sigmas, axes, and allocation overflow;
  never relax a failed tolerance silently.
- Rerun the same full S1a benchmark command and topology. Report total
  samples/s, augmentation mean/p50/p95, each new substage, summed CPU cores,
  queue waits, and before/after artifact paths. The user target is 1-2
  augmented samples/s per worker; derive it from augmentation service time or
  an isolated saturated augmentation run, not global throughput divided by
  workers. Include allocation/binding cost and matched sample identities. Keep
  GPUs hidden and native thread limits fixed throughout.

### Spec update

Document backend selection, the native three-pass stencil contract, one-thread
ownership, numerical tolerance, and retained Torch fallback.

### Docs updates

Add the measured substage baseline, implementation structure, exact benchmark
command, numerical comparison, before/after performance, and portability note
to the sample-pipeline benchmark document.

### Testing

Add focused binding validation and native-versus-Torch output tests plus loader
backend selection/fallback tests. Build the Release VC binding, reinstall only
that explicitly authorized editable package, run focused CPU Python tests, the
stable full-pipeline benchmark, and `git diff --check`.

### Changelog update

Record the native anisotropic augmentation and measured speedup once validated.

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
