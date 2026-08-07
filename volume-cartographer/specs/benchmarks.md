# Benchmark Requirements

## General

- Benchmarks must be reproducible and must identify the build configuration,
  compiler, architecture target, workload, repetition count, and reported
  metric.
- Performance gates must use deterministic or tightly controlled metrics.
  Host wall time may be reported for calibration, but must not be the sole CI
  regression gate.
- Benchmark fixtures must be synthetic or checked into the repository. They
  must not require network access or private datasets.
- Correctness checks must run with every performance measurement so a faster
  incorrect or empty workload cannot pass.
- Numeric behavior must remain unchanged unless a task explicitly permits a
  numerical change.
- Baselines and thresholds must be versioned with the benchmark and updated
  intentionally, with the reason recorded in the task log or changelog.

## Synthetic Rendering Benchmark

- The benchmark must exercise the production `ChunkedPlaneSampler`
  fine-to-coarse coordinate rendering path.
- It must use a deterministic synthetic chunked volume with four pyramid
  levels and pseudo-random decoded `uint8` chunk contents.
- Synthetic chunk materialization must occur before the measured region. The
  measured workload must represent rendering against controlled resident,
  missing, and fallback states, without storage, network, compression, or
  asynchronous cache scheduling costs.
- The coordinate fixture must include substantial contiguous runs of spatially
  correlated coordinates, matching the usual surface-rendering access pattern.
- The fixture must use trilinear sampling with every coordinate and all eight
  required voxels in bounds at every pyramid level. For a coordinate assigned
  to level `L`, at least one required chunk must be missing at each finer level
  and all required chunks must be resident data at level `L`.
- Both `tryGetChunk()` and `getChunkIfCached()` must return pre-materialized
  states without allocation or mutation. The benchmark must use
  `queueMisses=true` and `queuedFallbackLevels=0` so level 0 follows the normal
  request path and fallback levels use resident-only reads.
- The benchmark must provide these scenarios:
  - `full_res`: every coordinate is available at level 0.
  - `fallback_3`: every coordinate falls back through levels 0, 1, and 2 and
    is available at level 3.
  - `mixed_correlated`: equal quarters are available at levels 0, 1, 2, and 3.
  - `mixed_shuffled`: the same coordinates and fallback distribution as
    `mixed_correlated`, deterministically shuffled across output pixels.
- Every scenario must cover every output pixel and produce a deterministic
  checksum. An untimed validation pass must additionally compare every pixel's
  observed source level with its expected source level.
- The default native run must execute each scenario several times and report
  at least median elapsed time and throughput. It is a host calibration, not a
  portable CI gate.
- The deterministic amd64 run must use a Release build targeting the repository
  x86-64-v3 baseline, with `VC_MARCH_NATIVE=OFF`.
- The deterministic run must use Callgrind with explicitly configured I1, D1,
  and last-level cache geometry and branch simulation. It must report event
  counts per output pixel for all four scenarios.
- Callgrind must start with instrumentation disabled. Fixture construction,
  chunk materialization, warmup, allocation, and result validation must remain
  outside explicit client-controlled instrumentation boundaries. The
  denominator is `width * height * measured_repetitions`.
- Each instrumentation/timing interval must bracket the production sampler call
  directly. Checksum calculation, options construction, result-container
  bookkeeping, and validation are not rendering work and must remain outside.
- Callgrind does not emulate a complete CPU pipeline and therefore cannot
  produce literal processor cycles. The benchmark may report `modeled_cycles`
  using a versioned, documented fixed-cost formula over Callgrind events; this
  metric must always be labeled as modeled rather than hardware cycles.
- Estimated native throughput must use a versioned calibration independent of
  the regression baseline. The model must combine instructions, cache misses,
  and branch mispredictions with documented coefficients, normalize matched
  one/two/four-worker fixtures using measured effective parallelism, and apply
  one reference-host-specific nanoseconds-per-modeled-cycle conversion.
- Calibration observations must include at least five sequential fresh-process
  native samples per case, fixed CPU affinity, matched large fixtures for each
  worker count, and calibration-only locality workloads. Fit and held-out cases
  must be identified before fitting, with checked acceptance thresholds.
- Model fitting must be reproducible from checked-in observations without an
  optional numerical package. Coefficient bounds, objective, convergence,
  native min/median/max/MAD, fit residuals, and held-out residuals must be
  retained with the model.
- Estimated Mpx/s is valid only for the documented benchmark fixtures and
  reference CPU. Callgrind does not model hardware prefetching or simultaneous
  thread execution, so the estimate must retain its held-out error envelope and
  must not be described as a hardware cycle measurement.
- The deterministic benchmark must run on Linux amd64 only. Unsupported hosts
  must skip it clearly rather than silently substituting a different model.
- Individual deterministic scenarios must be independently runnable so CTest
  and CI can execute them concurrently.
- The benchmark must cover both sampler execution paths:
  - a small fixture below the sampler parallelization threshold for stable
    single-thread event counts;
  - a larger fixture above the threshold with an explicit, fixed render-sampler
    worker count.
- The production sampler must provide a documented worker-count override so
  tests and benchmarks can select the parallel path reproducibly without
  changing normal automatic worker selection.
- The override is immutable process configuration, validated before the first
  sampler call and used by both parallelization decisions and pool creation.
- Parallel Callgrind thresholds may be wider than serial thresholds to account
  for synchronization/runtime variation, but the image dimensions, tile size,
  worker count, repetitions, and scenario must remain fixed.
- Actual multi-worker execution must be proven by the native fixed-thread test.
  Callgrind serializes guest threads and is used to gate the aggregate cost of
  the production parallel task-submission and future-join path, not wall-clock
  parallel speedup.
- The deterministic matrix contains eight cases: every fallback scenario runs
  once with the serial fixture and once with the fixed-thread parallel fixture.
- The passive replay diagnostic matrix is separate: it includes `full_res`,
  `fallback_1`, `fallback_3`, `mixed_correlated`, and `mixed_shuffled` for
  workers 1--7. This five-scenario host-validation surface does not change the
  four-scenario/eight-case deterministic CI matrix.
- Native serial cases may execute concurrently. Native parallel cases must be
  scheduled so the aggregate configured worker count does not exceed the CI
  CPU allocation. Valgrind collection is the explicit exception: because each
  instrumented guest is serialized by Valgrind, CI uses the runner's available
  logical CPU count as Ninja concurrency and may run one independent Valgrind
  process per host CPU even when each guest configures four renderer workers.
- The CI regression graph must generate fresh separate-thread Callgrind
  profiles for all eight cases and fresh complete DRD dependency graphs for the
  four parallel cases. A collector publishes its completion manifest atomically
  only after all raw files and metadata validate; evaluators depend on those
  manifests and may not rerun collection.
- The CI relative modeled-runtime score uses the frozen synthetic-only
  data-read event model. Serial score is summed per-thread modeled work per
  render call. Parallel score is native FIFO replay makespan per render call
  with four workers plus one caller core, equal attribution,
  `residual_fraction=0.5`, zero wake latency, the frozen cross-thread release
  latency, and unit replay/dependency-excess scales. Process startup and native
  wall timing are excluded.
- The relative modeled-runtime score is not a validated absolute runtime claim.
  Each case must remain within the symmetric tolerance in its versioned
  reference, initially 10% in either direction. CI may neither refit the model
  nor rewrite references.
- The checked reference's top-level `tolerance` is the sole normal CI source
  for the symmetric acceptance width. Changing it is an explicit policy change
  independent of recalibrating the model or refreshing per-case scores.
- Before comparing scores, CI must require exact model hash, checksum,
  benchmark schema, compiler/version, architecture target, build type,
  Valgrind version, simulated cache geometry, fixture dimensions, repetitions,
  and worker count. Parallel traces must have no unresolved happens-before
  edge or unmatched blocking wait.
- The machine-readable summary must record the metric schema/model version,
  Callgrind events by name, compiler, architecture target, Valgrind version,
  cache geometry, dimensions, tile size, worker count, repetitions, checksum,
  and native timing statistics.
- A calibrated summary must additionally report modeled work and wall cycles,
  modeled cycles per pixel, the shared nanoseconds-per-modeled-cycle factor,
  estimated nanoseconds per pixel, estimated Mpx/s, and reference-host metadata.
- The complete deterministic benchmark should target approximately 10--30
  seconds on the four-core reference CI runner. Other runner sizes use their
  available logical CPU count and may have different wall time while preserving
  identical per-case artifacts and scores.
- CI must run the deterministic benchmark and reject regressions beyond the
  checked-in thresholds. Raw Callgrind outputs and a machine-readable summary
  must be retained as CI artifacts for diagnosis.

## Thread-Pool Dispatch Diagnostic

On the calibration host, pin all CPU policies to their common nominal frequency
before collecting native timings and restore their exact prior state afterward.
This avoids boost while respecting the minimum no-boost maximum enforced by
`amd-pstate-epp`:

```bash
sudo scripts/run_with_fixed_cpu_frequency.py set
sudo scripts/run_with_fixed_cpu_frequency.py restore
```

The `set` action saves a recovery snapshot under `/run` and refuses to replace
an outstanding snapshot. The `restore` action consumes that snapshot, including
the previous governor, frequency bounds, energy preference, and boost setting.

- A native diagnostic must be able to isolate render-style thread-pool dispatch
  from chunk lookup, shared ownership, output writes, and memory locality.
- It must use the production `utils::ThreadPool`, support serial execution,
  future-per-task submission, open- and closed-gate future controls,
  indexed-batch submission, inter-wave idle intervals, and repeated pool
  construction. It must expose worker count, task count, deterministic work per
  task, round count, requested and measured idle time, raw dispatch time, and
  paired-clock overhead.
- Every run must validate its deterministic result outside the measured region
  and report wall time, process CPU time, average occupied CPU cores, compiler,
  build type, architecture target, and checksum.
- Native dispatch timing is diagnostic and host-specific. It must not become a
  portable CI performance gate without a separately reviewed normalization and
  threshold protocol.
- Passive synchronization extraction must not require application markers or
  alter the benchmark binary. On Linux amd64 the dispatch fixture may use
  Valgrind's core scheduler and syscall debug streams. Renderer replay must use
  DRD vector clocks so nonblocking userspace synchronization is represented in
  addition to scheduler quanta and futex syscalls.
- A passive event stream must preserve thread IDs and global order, represent
  fixed-basic-block work slices, pair blocking futex waits with guest wakes or
  observed thread completion, and fail timing validation when dependencies are
  unresolved.
- Passive event playback must use the native persistent replay engine. Python
  remains responsible for Valgrind collection, dependency extraction, model
  fitting, and reports, but production callers must load each event graph once,
  cache named cost attributions, and submit ordered replay batches through the
  versioned protocol. Missing native replay is an error; there is no silent
  Python fallback. The Python replay is retained only as a temporary parity
  oracle and must not be used for production benchmark results.
- Valgrind elapsed timestamps must not be used as native duration or as work
  weights. State names must distinguish traced blocking from native scheduler
  state; simulated core idle is never an observed native metric.
- Callgrind work weights and native timing must cover the same explicit
  measured loop. The passive DRD dependency graph may span the process, but
  allocation, fixture construction, verification, and teardown must not be
  assigned measured work. Partial scheduler quanta and scheduler tie-breaking
  must be reported as sensitivity ranges.
- Native renderer replay must reserve one CPU for the submitting caller:
  `parallel_cores = workers + 1`. The one-CCD domain therefore ends at seven
  workers on CPUs 0--7. Comparing `workers` threads on only `workers` CPUs is a
  saturated diagnostic and must not be used as renderer validation.
- Renderer validation may not fit an effective-parallelism, contention, or
  worker-count coefficient. Valgrind supplies work events, scheduler slices,
  and DRD happens-before edges; native data validates the result but does not
  alter it. Every frozen case must remain within 20% median speedup error.
- Renderer diagnostics may motivate a generic synthetic hypothesis, but they
  may not choose its formula, bounds, coefficients, fit cases, holdouts, or
  acceptance thresholds. Existing renderer cases used to motivate a hypothesis
  are opened diagnostics rather than untouched holdouts. Freeze and hash the
  synthetic model before evaluating them, and do not refit after inspecting
  renderer results in the same experiment.
- Synchronization replay remains experimental. The reference-host timing model
  is restricted to the production futures path, at most seven physical-core
  workers, and no more futures than workers. Batch dispatch, deep queues, SMT,
  CPU saturation, and general scheduler contention are outside its domain.
- Every restricted native sample must use the same physical CPU affinity domain
  so worker-count changes do not also change CCD placement. On the 16-core
  reference host, the child caller and up to seven workers use CCD0 CPUs 0--7,
  while passive monitoring uses CPU 8 on CCD1; no SMT CPU is allowed. Runtime
  topology validation must reject a host where those CPU/cache relationships
  differ. Report passive `/proc` runnable-wait fraction as a scheduling-bubble
  diagnostic, not as a fitted scheduler-contention term or rejection gate.
- A scoped passive model may use synthetic-only Callgrind event costs, one
  bounded branch/cache stall-overlap fraction, one cross-thread release
  latency, and one bounded inferred-dependency-excess scale. Process startup
  must remain separate and may contain one per-additional-worker slope fitted
  from zero-work processes over workers 1-7. Renderer observations may not fit
  any work or scheduling parameter.
- The branch/cache overlap candidate is shared across synthetic work kinds and
  is constrained to `0 <= stall_overlap <= 1`:

  ```text
  mixed_stall = branch_stall + cache_stall
              - stall_overlap * min(branch_stall, cache_stall)
  event_cost = non_stall + mixed_stall
  ```

  Synthetic fit and unopened holdout cases must independently cross branch
  predictability, resident/L1/last-level working sets, and phase-separated
  versus interleaved operation order with matched operation totals. A
  pointer-chasing kernel may remain diagnostic but must not stand in for
  independent gather/interpolation work.
- The dependency-excess candidate is shared across workers, task fanout, work
  kind, and wave count and is constrained to
  `0 <= dependency_excess_scale <= 1`:

  ```text
  hard_cp = critical path excluding inferred DRD happens-before edges
  hard_lower = max(total_event_work / core_count, hard_cp)
  full_lower = max(hard_lower, full_dependency_critical_path)
  dependency_excess = full_lower - hard_lower
  fifo_excess = max(0, raw_fifo_replay - full_lower)
  adjusted_replay = hard_lower
                  + dependency_excess_scale * dependency_excess
                  + fifo_excess
  ```

  Hard critical-path edges include per-thread program order and thread
  lifecycle. The full path additionally includes inferred DRD vector-clock
  edges and the separately calibrated cross-thread release latency. Scale one
  must reproduce current FIFO replay exactly. Scale zero may remove only
  inferred dependency excess and must preserve hard task imbalance, program
  order, ideal work, and FIFO scheduling excess. Process startup is excluded.
  Fit and validation must use matched-total-work pairs that vary wave depth,
  balanced and deterministically skewed work, and `tasks <= workers <= 7`.
  Fit workers are 2/4/6; validation and sealed-holdout workers are 3/5 and 7.
  Handoff latency must be frozen before fitting the dependency scale.
- The older steady-state model contains exactly three host values:
  `work_ns_per_iteration`, `fixed_dispatch_ns`, and
  `per_future_dispatch_ns`. For ordinary futures with `tasks <= workers`, it is:

  ```text
  dispatch = fixed_dispatch_ns + tasks * per_future_dispatch_ns
  active_tasks = min(workers, tasks)
  total_work = tasks * work_iterations
  work = work_ns_per_iteration * total_work / active_tasks
  round_time = dispatch + work
  ```

  Within the accepted `tasks <= workers` domain, `active_tasks` equals `tasks`.
  Serial work calibration, zero-work dispatch calibration, and
  nonzero crossover validation must remain separate. Gate state, worker idle
  history, scheduler wait, and frequency readback are diagnostics and must not
  add accepted steady-state model parameters.
- Serial fit iterations are 0, 400k, 1.6M, and 6.4M; independent serial
  holdouts are 150k and 2.8M. Zero-work dispatch fit workers are 2, 4, and 6,
  with every task count from one through the worker count. Dispatch holdout
  workers are 1, 3, 5, and 7, also with every in-domain task count. Nonzero
  holdouts cross those holdout workers and task counts with 7.5k, 350k, and
  1.4M iterations per task. Five fresh sequential processes are required per
  case.
- The two-parameter dispatch fit must have full normalized-Jacobian rank,
  absolute parameter correlation below 0.98, no bound hits, and no fit residual
  above 15%. Serial holdout median and individual errors must be at most 5% and
  10%, respectively. The scoped passive threading holdout must have median
  absolute runtime error and RMS speedup error at most 20%, with maximum
  speedup error at most 30%. Per-case errors and all individual samples must
  still be reported.
- Open/closed gate controls, 0/100 us/1 ms idle probes, and repeated pool
  construction at workers 1/3/5/7 are reported separately. They neither affect
  calibration validity nor permit renderer timing claims.
- Although the current production `ChunkedPlaneSampler` can configure eight
  render workers, this host model must reject worker 8. Eight workers plus the
  submitting caller saturate the eight-core CCD and are outside the calibrated
  domain; the observed slowdown may be reported only as a diagnostic.
- Fixed-frequency evidence must be collected independently for fit and holdout
  blocks. Before and after each block, all CPU policies must retain a common
  fixed min/max target, performance governor, and disabled boost. Each monitored
  policy must have at least 20 `scaling_cur_freq` samples and a mean within 3%
  of target; instantaneous extrema are diagnostics because idle-policy values
  may be stale. Frequency restoration may be deferred only while the user is
  actively finalizing this benchmark and must use the saved recovery snapshot.
- Fine-to-coarse render estimates must sum one predicted wave per sampled
  pyramid level. End-to-end timing claims require separate one-, two-, and
  four-level renderer holdouts. Aggregate absolute-runtime median error and
  every per-case speedup error must be within 20% before timing claims are
  enabled. Imbalanced
  fallback coverage requires passive per-range work or maximum-range work as an
  input; it must not be absorbed by another fitted contention coefficient.
- Aggregate `/proc` scheduler accounting may distinguish execution, run-queue
  wait, and sleeping time, but it must not be presented as causal wake-to-run
  evidence. A direct causal claim requires native scheduler wake/switch tracing.
- Valgrind's user-space dependency stream does not reproduce native kernel
  wake placement, migration, or condition-variable scheduling. Any conversion
  from replay work to reference-host time must therefore use an explicit native
  calibration within the restricted host/model domain. Coefficients must remain
  shared across accepted worker and task counts rather than being refitted per
  configuration. Otherwise timing claims remain disabled.
- A renderer residual observed at one repetition count must not be fitted as a
  naked process intercept. Generic calibration must vary process repetition or
  synchronization-wave count to separate fixed startup/runtime cost from
  per-wave cross-thread handoff latency. An accepted latency term must be
  selected by passive critical-path synchronization or syscall-count features;
  Valgrind elapsed syscall time is not an accepted duration source. Renderer
  identity and worker count may not select the coefficient.
- Passive replay calibration must execute only standalone generic work and
  synchronization fixtures. Renderer binaries, rendering fixtures, production
  sampling, rendering coordinates, fallback scenarios, and renderer-derived
  observations may not participate in fitting, model selection, coefficient
  stability checks, or threshold selection. Renderer data may only validate a
  previously frozen synthetic model.
- The passive work-cost model must stay small and renderer-relevant. Its event
  basis may distinguish non-data instructions, data writes, L1 and last-level
  data misses, branch misses, and one branch/cache interaction. Random-gather
  read and random-scatter write diagnostics must cross the configured L1 and
  last-level cache boundaries. The synthetic matrix must have
  at least ten fit observations per parameter. Additional synthetic kernels
  overconstrain these shared coefficients; individual pathological-kernel
  residuals are diagnostics rather than reasons to add terms. Require full
  Jacobian rank, absolute parameter correlation below 0.98, no bound hits, and
  at most 20% median synthetic holdout error. The frozen model is accepted for
  rendering only when every renderer benchmark holdout is within 20% timing
  error; renderer observations may never alter its terms or coefficients. The
  one synthetic cross-thread release coefficient must have at most 15% fit
  error, 20% leave-one-case-out movement, and 20% median and individual
  holdout error.
- Fixed-host native collection and Valgrind replay remain Linux-amd64-only and
  outside portable CI. The synthetic fixture and model re-evaluation tests must
  compile or run on supported Ubuntu/macOS amd64/arm64 hosts without requiring
  host timing collection. Synthetic mixing must use explicitly defined state
  evolution rather than implementation-dependent standard-library shuffling.
- A timing model intended to survive arbitrary uninstrumented scheduler-code
  changes requires either passive native scheduler-event capture for a
  host-specific comparison or a deterministic full-system multicore simulator
  that executes the kernel scheduler. Callgrind/core replay alone is not such a
  simulator.
- Passive work-attribution sensitivity is diagnostic only. Every attribution
  policy must preserve each thread's complete synthetic event cost. `front`
  and `back` mean the first and last eligible passive events within a scheduler
  window; they do not imply known placement before or after synchronization.
  A sole trailing window receives the complete thread cost even when its
  relative weight is zero, and positive cost without an eligible event is an
  error. Renderer errors may compare frozen policies but may not select one.
- For the current renderer-model investigation, improvement decisions prioritize
  maximum absolute runtime error. Maximum speedup error remains reported, but
  it is monitor-only while every frozen renderer row remains below 20%. A
  candidate that improves speedup while worsening maximum runtime is not a
  runtime-model improvement.
- Alternative event-cost fitting objectives must use the unchanged predeclared
  feature basis and synthetic observations only. A minimax objective must
  minimize unweighted per-case relative error, measure every coefficient's
  feasible interval over the optimal face, and use a deterministic secondary
  objective. Selection requires a fresh synthetic holdout whose complete case
  tuples and working-set sizes are disjoint from all opened cases. A reduction
  in maximum error must exceed twice the largest per-case native-sample MAD;
  median/RMS and per-family regressions remain independently bounded. Freeze
  and hash code, benchmark, cases, and gates before collection. Renderer data
  remains a post-freeze diagnostic and cannot select the objective.
- Event-cost candidates containing nonlinear aggregate features must calculate
  those features independently per Callgrind thread and sum modeled thread
  costs; profiles with different event densities must never be merged before
  nonlinear feature extraction. Candidate evaluation freezes the current
  model, a matched legacy-basis refit, and the new-basis refit before renderer
  validation. Headline renderer reporting separates workers=1 from pooled
  workers=2--7 and retains supporting per-worker maxima and all rows.
- Passive feature schemas are identified by the complete ordered
  `feature_names` tuple, never by feature count. Existing six-feature legacy
  and seven-feature serialization schemas must retain exact predictions when
  other seven-, eight-, or nine-feature schemas are introduced. A base read
  feature is `Dr`; split cache features are `D1mr`, `D1mw`, `DLmr`, and
  `DLmw`; the branch/cache interaction remains
  `(D1mr + D1mw) * (Bcm + Bim) / max(Ir, 1)` per thread.
- A new event basis must pass opened-fit identification before consuming a
  sealed holdout: full normalized rank, maximum parameter correlation below
  0.98, no nonnegative-bound hit, stable multi-process native calibration, and
  at most 20% coefficient movement when each added calibration family is
  omitted independently. Failed bases are rejected by default, but an
  explicitly requested post-freeze renderer diagnostic may still evaluate
  them provided no coefficient or threshold is subsequently refitted. Generic
  density kernels may vary reads or writes per loop to
  identify event costs, but every fitted basis and its matched baseline must
  receive identical records, family weights, medians, objective, ridge,
  bounds, and training-derived scaling.
- Event-basis identification should include paired synthetic kernels that hold
  cache-line traversal fixed while varying accesses per line, plus crossed
  read/write subsets. Crossed cases must include sparse and dense reads and
  writes so total accesses, read/write misses, and instruction cost are not
  identified from single-feature kernels alone.
