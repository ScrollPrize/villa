# Synthetic Rendering Benchmark

`bench_render_synthetic` measures the production
`ChunkedPlaneSampler::sampleCoordsFineToCoarse` path without storage, network,
decompression, or asynchronous chunk scheduling. It uses a four-level synthetic
resident array whose deterministic pseudo-random chunk bytes are materialized
before measurement.

## Scenarios

Each case uses trilinear sampling and validates every output pixel against its
expected source pyramid level.

| Scenario | Residency pattern |
| --- | --- |
| `full_res` | Every coordinate is available at level 0. |
| `fallback_3` | Levels 0-2 are missing; every coordinate uses level 3. |
| `mixed_correlated` | Equal spatially coherent quarters use levels 0, 1, 2, and 3. |
| `mixed_shuffled` | The same coordinates and level mix, deterministically shuffled across output pixels. |

Calibration additionally uses `full_res_shuffled`, `fallback_3_shuffled`,
`full_res_cache_stress`, and `full_res_cache_stress_shuffled` to vary locality,
branch prediction, and simulated cache behavior without changing the four-case
CI reporting surface.

The `serial` fixture is `96x96`, below the sampler's `128x128`
parallelization threshold. The `parallel` fixture is `256x256` and uses
`VC_RENDER_SAMPLER_THREADS=4`. Native validation confirms that multiple worker
threads execute the latter path. The environment override accepts `1..8`, is
read once per process before the first sampler call, and leaves automatic worker
selection unchanged when unset.

## Native Calibration

Configure a generic Release build and build with the desired local parallelism:

```bash
cmake -S volume-cartographer -B volume-cartographer/build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DVC_MARCH_NATIVE=OFF \
  -DVC_TESTING=ON \
  -DVC_RUN_RENDER_BENCHMARKS=ON \
  -DVC_BUILD_APPS=OFF \
  -DVC_BUILD_UI_TRACER=OFF \
  -DVC_BUILD_FLATBOI=OFF
cmake --build volume-cartographer/build-release --parallel 32 \
  --target bench_render_synthetic
```

Run one native case:

```bash
VC_RENDER_SAMPLER_THREADS=4 \
  volume-cartographer/build-release/bin/bench_render_synthetic \
  --fixture parallel --scenario mixed_correlated --native-trials 9
```

The JSON output reports mean, median, and p95 wall time, median pixels/second,
checksum, observed worker count, compiler, build type, and architecture target.
Native timing is a hardware calibration and is not the portable CI gate.

The checked-in calibration uses five fresh processes per case with one long
trial in each process. The serial fixture uses 4096 repetitions pinned to CPU 2;
the large fixture uses 512 repetitions with one, two, and four workers pinned to
CPUs 2, 2-3, and 2-5 respectively. Calibration reports min, median, max, and
median absolute deviation rather than a p95 from only five samples.

## Thread-Pool Dispatch Diagnostic

`bench_thread_pool_dispatch` isolates the render sampler's pool submission and
completion pattern from chunk access and output memory. Its task body uses only
private deterministic integer state. It supports:

- `serial`: execute every task on the caller;
- `futures`: submit one future per task and retrieve results in order, matching
  the coordinate sampler's current synchronization shape;
- `batch`: use `utils::ThreadPool::run_indexed_batch()` and its shared latch.

Build and run a render-sized four-task case:

```bash
cmake --build volume-cartographer/build-release --parallel 32 \
  --target bench_thread_pool_dispatch
taskset -c 2-5 \
  volume-cartographer/build-release/bin/bench_thread_pool_dispatch \
  --mode futures --workers 4 --tasks 4 \
  --work-iterations 400000 --rounds 1024
```

The benchmark reports wall time per round, process CPU time, average occupied
CPU cores, and a checksum validated against an untimed serial oracle. Sweep the
task count while keeping `tasks * work-iterations` constant to measure task
granularity independently of total work.

On the Ryzen 9 5950X reference host, four tasks of 400,000 iterations achieve a
median 2.02x one-to-four-worker speedup and occupy 2.18 cores. Splitting the
same total work into 64 tasks of 25,000 iterations achieves 3.41x and occupies
3.31 cores. This diagnostic is native and host-specific; it is not a CI timing
gate.

### Passive Synchronization Replay

`run_thread_sync_replay.py` is a Linux amd64 feasibility diagnostic. It runs
the unmodified dispatch binary under Valgrind's `none` tool with core scheduler
and syscall debugging enabled. A fixed scheduler quantum supplies basic-block
work slices, and futex waits/wakes supply dependency edges, including the
atomic-futex path used by `std::future`. A separate per-thread Callgrind run
weights each thread's slices. Valgrind timestamps are never used as costs.

Run one case and retain the raw and normalized streams:

```bash
python3 volume-cartographer/scripts/run_thread_sync_replay.py \
  --benchmark volume-cartographer/build-release/bin/bench_thread_pool_dispatch \
  --output-dir volume-cartographer/build-release/render-benchmark/sync-replay \
  --mode futures --workers 4 --tasks 4 \
  --work-iterations 400000 --rounds 16 --quantum 10000 \
  --trace-trials 5 --native-trials 5 --cpu-affinity 2-5
```

Native validation pins the same four-worker process first to one CPU and then
to four CPUs and measures the whole process. This matches the passive trace
boundary, which includes startup, 32 warm-up rounds, measured rounds, and
teardown.

The initial equal-total-work feasibility results are:

| Tasks x iterations | Complete traces | Native 1-to-4 CPU speedup | Replay median (range) | Error |
| --- | ---: | ---: | ---: | ---: |
| `4 x 400000` | 5/5 | 1.86x | 2.68x (2.58-2.92x) | +43.7% |
| `64 x 25000` | 5/5 | 2.47x | 2.49x (2.37-2.63x) | +0.6% |

The stream correctly ranks the finer-grained case as more parallel, but it
does not predict coarse-batch scheduling delay within the predeclared 20%
bound. Repeated Valgrind traces also produce different worker task assignments,
so the replay distribution is mandatory. A per-worker-wake latency fitted to a
four-task trace underpredicts the 64-task speedup by about 25%, because
Valgrind's serialized task assignment produces a different set of worker
wait/wake edges than the native scheduler. Therefore the JSON output sets
`timing_claims_enabled=false`; the tool is currently useful for dependency
inspection and model experiments, not estimated benchmark time or a CI
performance gate.

#### Native mismatch diagnosis

Long measured-region runs show that the coarse mismatch repeats at every batch
boundary; it is not process startup. At equal total deterministic work, the
four-worker futures path scales from 2.01x with four tasks to 3.17x with 64
tasks, while occupied cores rise from 2.15 to 3.29. The indexed-batch
`notify_all` path has the same four-task behavior as futures `notify_one`, so
notification choice is not the principal term.

Passive `/proc/<pid>/task/*/schedstat` sampling over approximately 8.5-second
runs separated worker time as follows:

| Tasks | Worker running | Worker run-queue wait | Worker sleeping/unsampled |
| ---: | ---: | ---: | ---: |
| 4 | 18.03 s | 5.04 s | 10.89 s |
| 64 | 26.10 s | 6.14 s | 1.15 s |

All four workers had balanced CPU time. The dominant difference is therefore
an inferred batch-tail/barrier bubble: with one task per worker, early workers
sleep while the batch tail completes and must be made runnable again for the
next round. A deep queue keeps workers runnable and amortizes this bubble.
`schedstat` is aggregate evidence; without native scheduler wake/switch events,
it cannot assign the delay to an individual wake causally.

The worker CPU time per round was about 2.20 ms for the coarse case and 2.18 ms
for the fine case, versus about 2.05 ms of serial task service. The small
difference between coarse and fine CPU service does not explain the large
parallel-speedup gap; sleeping and queue-tail behavior do.

#### Reference-host calibration

For four-worker futures dispatch with `N >= 4`, the tested internal-time model
is:

```text
T4(N, W) = D4(N)
         + W / Psteady
         + alpha * W / N
         + beta * W / N, when N == workers
```

`D4(N)` is native zero-work dispatch/completion time and `W` is native serial
task-service time. Fitting the Ryzen 9 5950X observations produced
`Psteady=3.348`, `alpha=0.591 task-times`, and an additional exact-one-wave
penalty `beta=0.210 task-times`. A first model without `beta` was rejected after
missing the five-task holdout by 10.49%. The revised model used new untouched
task counts 6, 10, 20, and 40 at a third total-work size; errors were +7.87%,
-9.41%, -7.13%, and -3.62%.

The passive replay covers the whole process, while the benchmark timer excludes
pool startup, 32 warmup rounds, and teardown. A separate model

```text
whole_process_time = fixed_process_time + measured_rounds * round_time
```

was fitted at 7 and 127 rounds and tested at 31 and 511 rounds. The fitted
fixed terms were 78.6/45.1 ms for coarse one/four-CPU runs and 81.4/34.8 ms for
fine runs. Worst individual holdout error was 4.8%; median errors were generally
below 1%.

At the original 16-round replay boundary, the combined calibration predicted
1.814x speedup versus 1.855x observed for four tasks, and 2.531x versus 2.543x
for 64 tasks. This calibrates the current fixture on the reference host, but it
does not turn Valgrind into a native scheduler simulation. The required native
inputs are the process intercept, zero-work dispatch curve, serial service cost,
steady effective parallelism, and queue-tail class. Valgrind does not provide
Linux wake placement, run-queue migration, or a stable native worker assignment.

For arbitrary uninstrumented scheduling rewrites, use passive native
`sched_wakeup`/`sched_switch` capture for a host-specific comparison or a
deterministic full-system multicore simulator that runs the kernel. A
Callgrind/core replay by itself cannot remain calibrated across those changes.

A subsequent fixed-frequency test attempted one 13-parameter model shared
across 2, 4, 8, 16, and 24 workers, futures/batch submission, three queue
depths, and three work sizes. It was rejected: six of ten crossed worker/depth
holdouts exceeded 10%, with a maximum error of 67.3%; the SMT-capacity
parameter hit its bound and omission tests were unstable. Native occupied-core
counts showed why: effective runnable width changes with task duration and
queue depth even at a fixed configured worker count. Futures additionally
overlap caller submission with worker queue draining. A future passive model
must obtain both runnable-width and submission/drain-overlap structure from the
trace rather than treating capacity as a function of worker count alone.

#### Minimal single-CCD calibration

The accepted synthetic model fixes the entire benchmark child to CCD0 physical
CPUs `0-7`, including the submitting caller, and runs at most seven workers.
Frequency sampling runs on CPU 8 in the other CCD. The harness validates that
topology from sysfs and checks fit and holdout frequency windows independently
while the host remains pinned at 3.401 GHz.

The model has exactly three values and applies only to ordinary future-per-task
dispatch with `tasks <= workers`:

```text
dispatch = fixed_dispatch_ns + tasks * per_future_dispatch_ns
active_tasks = min(workers, tasks)
total_work = tasks * work_iterations
work = work_ns_per_iteration * total_work / active_tasks
round_time = dispatch + work
```

Serial work is fitted at 0, 400k, 1.6M, and 6.4M iterations and held out at
150k and 2.8M. Zero-work dispatch is fitted on workers 2/4/6 and all task counts
through each worker count. Holdout workers 1/3/5/7 use every in-domain task
count at zero work and at 7.5k, 350k, and 1.4M iterations per task. The 7.5k
case exercises the dispatch/work crossover. Every point contains five fresh,
sequential process runs.

The 2026-08-05 reference run produced:

| Value | Result |
| --- | ---: |
| Work | 1.788023 ns/iteration |
| Fixed dispatch | 8,333.21 ns/round |
| Dispatch per future | 101.412 ns/future |
| Maximum serial holdout median error | 0.034% |
| Maximum serial holdout individual error | 0.052% |
| Maximum threading holdout median error | 13.34% |
| Maximum threading holdout individual error | 14.71% |
| Maximum dispatch fit residual | 6.81% |

The dispatch Jacobian has rank two, coefficient correlation is 0.827, and
neither coefficient hit a bound. Fit and holdout frequency windows passed with
means of 3.381 and 3.380 GHz. The worst case is the seven-worker, seven-task,
zero-work holdout; nonzero work errors are generally much smaller, with the
seven-task crossover at 7.47% median error.

Open/closed gate controls changed medians by at most 0.6%. After subtracting
paired-clock overhead, a 1 ms inter-wave idle changed dispatch by at most 2.1%.
Repeated pool construction cost about 22 us at one worker and 235 us at seven.
These measurements are diagnostics, not extra model inputs. Lifecycle probing
is capped at 64 rounds because an earlier high-count diagnostic exposed a
long-running pool teardown/construction stall; the production pool was not
changed as part of this benchmark work.

Build and collect the calibration with:

```bash
cmake --build volume-cartographer/build-release --parallel 32 \
  --target bench_thread_pool_dispatch
python3 volume-cartographer/scripts/calibrate_thread_dispatch_shared.py \
  --benchmark volume-cartographer/build-release/bin/bench_thread_pool_dispatch \
  --output-dir /tmp/dispatch-calibration-minimal-w7-fixed-3401 \
  --trials 5 --target-seconds 0.5
```

`fit-observations.json`, `holdout-observations.json`, `diagnostics.json`, and
`model.json` retain the raw runs and result. `--reuse-calibration` and
`--reuse-diagnostics` regenerate derived output after an interrupted analysis
without recollecting completed blocks. Schema 3 replaces the rejected
six-coefficient result atomically; `model.json` contains only the three accepted
parameters. Synthetic calibration passes, but renderer timing claims remain
disabled until separate one-, two-, and four-level renderer holdouts each meet
the 20% median and individual-run bounds.

An exploratory eight-worker full-width run on the same CCD was about 44% slower
than this unsaturated model predicts, while its half-width wave remained within
1%. Eight workers plus the caller compete for eight cores, so worker 8 is an
out-of-domain saturation diagnostic, not accepted calibration data.

### Synthetic-only passive replay calibration

`calibrate_thread_sync_synthetic.py` calibrates passive replay without running
or reading any renderer. Its only executable workload is
`bench_thread_pool_dispatch`, using deterministic ALU work and the production
generic `utils::ThreadPool` futures path. Synthetic traces use DRD vector-clock
dependencies, Callgrind instruction counts, and one spare caller CPU.

The tested model is:

```text
runtime = fixed_process_ns
        + replay_makespan(
              instruction_work * ns_per_instruction,
              cross_thread_release_ns)
```

Warmup round count is explicit so synthetic work cannot silently enter the
process intercept. Even workers 2/4/6 are fit inputs; odd workers 1/3/5/7 are
held out. Task fanout, work duration, warmup, and measured rounds vary
independently. Cross-thread latency is applied to passive DRD/futex release
edges, and replay determines whether an edge extends the critical path.

The final synthetic-only experiment fitted:

| Value | Result |
| --- | ---: |
| Instruction work | 0.153434 ns/instruction |
| Fixed process | 1.823562 ms/process |
| Cross-thread release | 4.036710 us/release |
| Parameter correlation | 0.666 |
| Holdout median absolute error | 1.46% |
| Holdout maximum median error | 15.09% |
| Holdout maximum individual error | 15.45% |

The calibration is **not accepted**. Maximum fit error is 17.19% versus the
15% requirement, and leave-one-case-out coefficient movement is 10.53% versus
the 10% requirement. A per-thread lifecycle candidate was less stable and was
rejected. These limits are not relaxed based on holdout accuracy, and no
renderer timing claim is enabled.

Collect and refit with:

```bash
python3 volume-cartographer/scripts/calibrate_thread_sync_synthetic.py \
  --runner volume-cartographer/scripts/run_thread_sync_replay.py \
  --benchmark volume-cartographer/build-release/bin/bench_thread_pool_dispatch \
  --output-dir /tmp/thread-sync-synthetic-fixed-3401-v3 \
  --trace-trials 3 --native-trials 5
```

This runs five native whole-process trials for each of the one-core and
caller-plus-worker affinity configurations in every case. The final report
prints estimated and achieved holdout speedups together with each absolute
relative speedup error and its median, RMS, and maximum summary.

`observations.json` explicitly records `renderer_inputs_used: false`; model
fitting rejects any result whose workload is not `synthetic`.

### Passive renderer replay

`run_thread_sync_replay.py --workload renderer` combines two passive Valgrind
runs. Callgrind supplies per-thread instruction, cache-miss, and branch events.
DRD runs with vector-clock and scheduler tracing enabled, supplying
happens-before edges for userspace synchronization and thread lifecycle. The
application contains no replay markers or instrumentation added for this
analysis.

The native comparison keeps the configured worker topology fixed. Its serial
point pins that process to one CPU; its parallel point uses one CPU per worker
plus one for the submitting caller. All CPUs remain in CCD0, so seven workers
use CPUs 0--7. An earlier comparison incorrectly used only `workers` CPUs and
made the caller compete with a worker; that produced 25--30% apparent replay
errors and is not a valid renderer comparison.

At the fixed 3.401 GHz request, three DRD traces and five fresh native processes
were collected for every worker count 1--7 and each 16-repetition scenario. The
Release benchmark was rebuilt with GCC 16.1 and OpenCV 5.0 before this complete
sweep, so no result is mixed with the earlier OpenCV 4.13 binary.

| Workers | Full-resolution error | One-fallback error | Three-fallback error |
| ---: | ---: | ---: | ---: |
| 1 | -0.40% | -0.52% | -0.12% |
| 2 | +13.08% | +13.02% | +7.66% |
| 3 | +15.18% | +15.37% | +13.86% |
| 4 | +18.21% | +17.32% | +16.86% |
| 5 | +16.90% | +18.36% | +19.07% |
| 6 | +19.58% | +19.40% | +19.50% |
| 7 | +10.91% | +15.72% | +17.45% |

All blocking futex waits and all DRD vector-clock references resolved. Workers
2--7 have a systematic positive error, but a worker-count term does not explain
it: adding a worker slope changes RMS error only from 2.142% to 2.132% after an
additive correction.

The stronger diagnostic is in runtime space. Native parallel runtime equals
replay runtime plus 9.588 ms on average, and an affine fit has slope 1.00002. A
single 9.57 ms additive term reduces median absolute speedup error from 16.88%
to 1.28% and maximum error from 19.58% to 6.35%. A proportional runtime factor
is slightly worse. This term is not accepted calibration: all renderer rows are
now opened, and the fixed 16-repetition matrix cannot distinguish process-fixed
runtime from latency paid once per render wave.

The separately calibrated 101.4 ns per-future submission cost has no measurable
effect on these millisecond-scale residuals. The replay currently gives
synchronization dependencies zero native latency and Callgrind runs with
system-call timing disabled. The next generic calibration must vary repetition
count and fit `startup + waves * handoff`, then remove whichever term is not
identified. Any retained latency must be shared and driven by passive
cross-thread critical-path events, not renderer identity or worker count. New,
untouched renderer cases are required before timing claims can be enabled.

Run one case with:

```bash
python3 volume-cartographer/scripts/run_thread_sync_replay.py \
  --workload renderer \
  --benchmark volume-cartographer/build-release/bin/bench_render_synthetic \
  --output-dir /tmp/renderer-drd-replay-fixed-3401-v3 \
  --scenario fallback_3 --workers 7 --repetitions 16 \
  --trace-trials 3 --native-trials 5 \
  --cpu-affinity 0,1,2,3,4,5,6,7
```

The output retains raw trace paths, dependency counts, sensitivity simulations,
native distributions, and exact commands. DRD replay is a Linux-amd64 host
analysis and is not part of the approximately ten-second portable Callgrind CI
gate.

#### Worker-count observations

An additional five-process native sweep used approximately 1.6 million work
iterations per round. Physical-core affinities were balanced across CCDs; SMT
siblings were then added alternately across CCDs. The host used
`amd-pstate-epp`, the `powersave` governor, and enabled boost.

| Workers | One task/configured worker speedup | Queue-rich speedup | Queue-rich occupied cores |
| ---: | ---: | ---: | ---: |
| 2 | 1.70x | 1.68x | 1.68 |
| 8 | 2.78x | 3.36x | 3.52 |
| 16 | 5.06x | 6.16x | 6.74 |
| 17 | 6.26x | 5.56x | 7.54 |
| 23 | 8.04x | 3.80x | 7.73 |
| 24 | 7.24x | 2.98x | 7.32 |

These are same-worker-count one-CPU-to-many-CPU speedups. The
one-task-per-worker configuration changes task duration as worker count changes
and does not guarantee one actual execution wave. The queue-rich configuration
uses `16 * workers` tasks, so its dispatch cost also grows with worker count.

To isolate SMT, 16/17/23/24 workers additionally ran the identical
`384 tasks x 4,167 iterations` workload:

| Workers | Median ns/round | Giter/s | Relative to 16 workers |
| ---: | ---: | ---: | ---: |
| 16 | 551,707 | 2.900 | 1.000x |
| 17 | 591,732 | 2.704 | 0.932x |
| 23 | 618,385 | 2.588 | 0.892x |
| 24 | 764,237 | 2.094 | 0.722x |

SMT reduced throughput for this short-task synchronization-heavy fixture. The
17/23/24-worker samples were also substantially noisier than the 16-physical-
core result. A two-worker cross-CCD affinity was bimodal; the table uses the
compact physical-core `0,1` sensitivity result instead.

## Scoped Synthetic Event and Startup Calibration

The 2026-08-06 calibration aligns native timing and Callgrind to the same
client-controlled measured loop. Process startup is measured separately with a
zero-work fixture. Random-scatter stores are retained as a diagnostic but are
excluded from fitting because identical native runs showed allocation-dependent
timing modes. Stable random-gather reads cross the simulated 32 KiB D1 and
8 MiB last-level boundaries.

The six synthetic-only event terms are:

```text
non-data instructions
data writes
L1 data misses
last-level data misses
branch misses
L1 misses * branch misses / instructions
```

The expanded calibration adds two renderer-independent trilinear-gather
kernels. `mixed-grid-phase` selects four memory regions in long deterministic
phases; `mixed-grid-random` selects the same regions from deterministic state
bits on every iteration. The 36 mixed fit cases and 10 sealed mixed holdouts
cross both kernels, working sets from 16 KiB through 12 MiB, and three fit
iteration counts. Each iteration still performs exactly one trilinear gather.

The resulting fitted costs are `0.107890`, `0.212241`, `0.207016`, `1.536580`,
`5.491304`, and `16.279596` ns respectively. Synthetic mixed holdouts have
8.91% median, 16.67% RMS, and 31.49% maximum error. The single 31.49% case is
the phase-separated 8 MiB holdout and exceeds the predeclared 30% maximum.
Overall synthetic holdout median error is 7.09%.

The bounded branch/cache overlap candidate fitted
`stall_overlap = 1.29e-23`, its lower bound. Refitting with overlap fixed to
zero gives the same predictions within numerical precision. The overlap term
is therefore rejected; the expanded synthetic cases improve the event basis,
but they provide no evidence for an additional overlap coefficient. Serialized
pointer chasing and floating point dependency chains remain diagnostics
outside the model's claim.

Synchronization calibration compares replay with the benchmark's internal
`wall_seconds`; allocation, work-data construction, verification, and teardown
are not scheduling work. A separate 100-trial sweep at each worker count fits:

```text
process_startup(workers) = 2.101557 ms
                         + (workers - 1) * 23.947 us
```

The startup fit has 1.33% RMS and 2.03% maximum median error. Four additional
fit cases and three sealed holdouts combine the two mixed-grid kernels with
balanced and deterministically skewed per-task work. The two-parameter
candidate fits a cross-thread release latency of `3.385 us` and
`replay_idle_scale = 1.54e-13`. The idle scale hits its lower bound and is
unstable under leave-one-case-out fitting. It also worsens held-out speedup RMS
from 14.53% to 30.82% and maximum error from 27.69% to 67.82%. The candidate is
rejected and the generated model retains the `3.146 us` handoff-only baseline
instead of serializing the failed idle parameter.

The result identifies the modeling mismatch: the replay's apparent idle is
already dependency-critical-path time, not excess list-scheduler idle above
the dependency/work lower bound. Scaling only the latter cannot correct it.
All fitting inputs are standalone synthetic work and synchronization fixtures
on CPUs 0-7; renderer observations are validation only.

The frozen rejected model was evaluated, without refitting, against all five
opened renderer diagnostics and worker counts 1-7. Cases run strictly one
after another; neither scenarios nor worker counts overlap. The original three
scenarios use their scoped 23-repetition profiles and previous native medians.
The mixed rows use three passive traces and nine fresh native processes per
row. Native processes use 512 measured render calls for all correlated rows
and shuffled workers 1-2, and 128 calls for shuffled workers 3-7.

| Scenario | Runtime error median | Runtime error RMS | Runtime error maximum | Speedup error median | Speedup error maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full_res` | 9.72% | 14.40% | 27.76% | 6.02% | 9.73% |
| `fallback_1` | 16.84% | 19.82% | 29.63% | 6.24% | 10.76% |
| `fallback_3` | 23.40% | 23.84% | 29.51% | 4.95% | 9.51% |
| `mixed_correlated` | 19.54% | 20.50% | 26.12% | 5.51% | 10.03% |
| `mixed_shuffled` | 17.28% | 20.54% | 30.68% | 7.29% | 16.80% |

All 35 speedup predictions are within the 20% exploratory target: median error
is 5.67%, RMS is 7.22%, and maximum error is 16.80%. Absolute runtime has
19.36% median, 20.05% RMS, and 30.68% maximum error; only 19 of 35 rows are
within 20%. Re-evaluating with `replay_idle_scale=1` produces exactly the same
renderer values because raw replay makespan equals its dependency lower bound
for these graphs. The renderer improvement relative to the prior mixed result
comes from the expanded synthetic event calibration, not the rejected idle
term. `timing_claims_enabled` remains false.

### Dependency-excess experiment

The follow-up experiment replaces replay-idle scaling with a scale applied only
to critical-path excess introduced by inferred DRD happens-before edges:

```text
hard_lower = max(total_work / cores, critical_path_without_drd_edges)
full_lower = max(hard_lower, critical_path_with_all_edges)
prediction = hard_lower
           + dependency_excess_scale * (full_lower - hard_lower)
           + (raw_fifo_replay - full_lower)
```

This preserves long tasks, per-thread program order, lifecycle constraints,
ideal work, and explicit FIFO scheduling excess. Scale one reproduces existing
replay. The core futex trace and DRD trace do not share event identities, so
confirmed futex dependencies cannot currently be promoted into the hard DRD
graph; this limits the coefficient to an empirical measure of inferred DRD
excess.

Eight fit cases use four matched-work shallow/deep pairs at workers 2/4/6.
Four validation cases reserve matched pairs at workers 3/5, and a final
seven-worker pair remains sealed until selection. Every pair keeps task count,
work kind, working set, warmup, and total measured iterations fixed while
changing per-wave work and wave count. The handoff-only coefficient is fitted
from the prior synthetic matrix and frozen at `3.146 us` before fitting the
single new parameter.

The fitted `dependency_excess_scale` is `0.99999994`; it hits the upper bound,
and the 101-point profile minimum is exactly `1.0`. Validation runtime RMS is
`2.67548%` both with and without the parameter, while speedup RMS is `0.86863%`
in both models. The sealed seven-worker pair has 35.56% runtime RMS and 19.63%
speedup RMS, also unchanged. The candidate is rejected because it adds no
synthetic explanatory power.

The frozen rejected candidate was then evaluated without refitting. Across the
five renderer scenarios and workers 1-7, runtime error is 19.22% median,
19.99% RMS, and 30.49% maximum; speedup error is 5.79% median, 7.23% RMS, and
16.68% maximum. All 35 speedups remain within 20%, but only 19 runtime rows do.
The small change from the preceding table comes from selecting the
handoff-only latency, not from dependency scaling.

Replacing FIFO ready-event tie-breaking with deterministic round-robin was
also evaluated over the same frozen 35 rows. It changed no predicted runtime,
with a maximum difference of exactly zero nanoseconds. Under these graphs,
release times and per-thread program order determine the makespan before the
ready-event tie policy matters.

Work placement was then varied independently while preserving every thread's
complete frozen synthetic cost. The nine policies combine first-event
(`front`), uniform (`equal`), or last-event (`back`) placement within each
passive scheduler window with trailing-window weights 0, 0.5, or 1. Renderer
measurements were used only to calculate errors after prediction.

| Policy | Runtime median | Runtime RMS | Runtime max | Runtime <=20% | Speedup median | Speedup RMS | Speedup max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| front / 0 | 19.24% | 19.92% | 32.17% | 19/35 | 3.80% | 5.25% | 10.63% |
| front / 0.5 | 19.20% | 19.90% | 32.06% | 19/35 | 3.85% | 5.26% | 10.62% |
| front / 1 | 19.16% | 19.87% | 31.95% | 20/35 | 3.89% | 5.27% | 10.61% |
| equal / 0 | 19.29% | 20.02% | 30.54% | 19/35 | 5.73% | 7.22% | 16.71% |
| equal / 0.5 (baseline) | 19.22% | 19.99% | 30.49% | 19/35 | 5.79% | 7.23% | 16.68% |
| equal / 1 | 19.15% | 19.95% | 30.44% | 19/35 | 5.86% | 7.25% | 16.65% |
| back / 0 | 19.16% | 19.88% | 31.96% | 20/35 | 3.88% | 5.27% | 10.63% |
| back / 0.5 | 19.12% | 19.84% | 31.90% | 20/35 | 3.93% | 5.29% | 10.57% |
| back / 1 | 19.09% | 19.81% | 31.88% | 20/35 | 3.97% | 5.30% | 10.54% |

Front/back concentration reduces speedup RMS by about two percentage points,
but runtime RMS moves by at most 0.18 points and maximum runtime error becomes
slightly worse. Trailing-window weight has little effect. This identifies
within-window attribution as a speedup-sensitive assumption, but it does not
explain the absolute-runtime gap and no renderer-derived policy is selected.

The active improvement target is maximum absolute runtime error, not speedup
error. All nine attribution policies keep maximum speedup error below 20%; the
baseline is 16.68% and concentration reduces it to 10.54--10.63%. That
reduction is substantial but not currently actionable because maximum runtime
error worsens from 30.49% to at least 31.88%. The baseline's worst runtime row
is `mixed_shuffled` at seven workers (+30.49%), closely followed by
`fallback_1` at seven workers (+29.55%) and `fallback_3` at one worker
(+29.51%). The mismatch therefore is not isolated to mixed scheduling.

### Cache-serialization pressure experiment

The first runtime-focused follow-up added one passive feature:

```text
l1_miss_serial_pressure = l1_data_misses^2 / instructions
```

This is a miss-density proxy, not an observation of memory-level parallelism.
It is homogeneous under profile repetition but nonlinear across profiles, so
it is calculated per thread before thread costs are summed. Existing opened
pointer-chase diagnostics were fit inputs. Thirteen fresh sequential holdouts
used five pointer sizes plus two sizes each for cache-read, grid-sample,
mixed-grid-phase, and mixed-grid-random. Every case used five native trials.

The seven-feature fit is full rank, has 0.941 maximum parameter correlation,
no reported bound hit, and fits a 6.692 ns pressure coefficient. On fresh
pointer holdouts it improves maximum error from 93.66% for the matched
six-feature model to 63.81%, but fails the predeclared 30% gate. The largest
fresh non-pointer maximum regression is 3.08 percentage points, within its
five-point guard. The candidate is rejected from synthetic data alone.

Three synchronization pipelines were then refit from the same synthetic-only
observations and frozen before renderer evaluation:

| Pipeline | Max runtime, worker 1 | Max runtime, workers 2--7 | Max speedup |
| --- | ---: | ---: | ---: |
| Current model | 29.51% (`fallback_3`) | 30.49% (`mixed_shuffled`/7) | 16.68% |
| Matched six-feature refit | 29.08% (`fallback_3`) | 30.11% (`mixed_shuffled`/7) | 16.59% |
| Seven-feature candidate | 34.59% (`fallback_3`) | 33.13% (`mixed_shuffled`/7) | 16.12% |

The per-worker maximum absolute runtime errors are:

| Workers | Current | Matched six | Candidate seven |
| ---: | ---: | ---: | ---: |
| 1 | 29.51% | 29.08% | 34.59% |
| 2 | 27.39% | 26.93% | 32.19% |
| 3 | 23.36% | 22.85% | 27.74% |
| 4 | 22.54% | 21.99% | 26.59% |
| 5 | 21.23% | 20.91% | 23.82% |
| 6 | 29.46% | 29.10% | 32.16% |
| 7 | 30.49% | 30.11% | 33.13% |

All headline errors are positive overpredictions. The candidate worsens both
runtime targets and is rejected independently by synthetic holdouts; its small
speedup improvement is monitor-only.

### Minimax objective experiment

An unweighted Chebyshev fit was tested on the unchanged six-feature,
zero-overlap basis. The fit used only the original seven synthetic workload
kinds; pointer and rejected serialization-extension records were excluded. A
fresh 21-case holdout crossed all seven kinds with three unused working-set
sizes and five sequential native trials per case.

| Objective | Fit median | Fit RMS | Fit maximum | Fresh median | Fresh RMS | Fresh maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Matched soft-L1 | 6.24% | 20.07% | 55.70% | 11.51% | 20.23% | 37.66% |
| Minimax | 21.17% | 28.60% | 48.02% | 23.05% | 27.74% | 47.85% |

Minimax reduced the training maximum by 7.68 points but increased the fresh
maximum by 10.19 points. Its worst fresh case was an underprediction on the
intermediate-size branch workload. The maximum native-sample MAD was 0.83%, so
the required improvement margin was 1.65 points. The normalized design had
rank 6, condition number 3.39, and maximum optimal-face coefficient width
0.0151%; the rejection is therefore objective generalization, not numerical
non-identifiability.

The frozen renderer diagnostic was:

| Workers | Matched soft-L1 maximum | Minimax maximum |
| ---: | ---: | ---: |
| 1 | 29.51% | 57.45% |
| 2 | 27.39% | 54.39% |
| 3 | 23.36% | 49.38% |
| 4 | 22.54% | 48.03% |
| 5 | 21.23% | 42.46% |
| 6 | 29.46% | 44.75% |
| 7 | 30.49% | 54.31% |

The headline pooled workers 2--7 maximum rose from 30.49% to 54.39%; the
one-worker maximum rose from 29.51% to 57.45%. Maximum speedup error changed
from 16.68% to 16.00% and remains monitor-only. Minimax is rejected and does
not replace the soft-L1 objective.

### Read and split-miss feature experiment

Two requested basis changes and their combination were tested before consuming
a sealed holdout:

```text
data_reads = Dr
split L1 misses = D1mr, D1mw
split last-level misses = DLmr, DLmw
```

The interaction remains `(D1mr + D1mw) * (Bcm + Bim) / Ir`, calculated per
thread. New schemas are dispatched by their exact ordered feature names, so the
existing six-feature and serialization models retain identical predictions.

The opened matrix initially had 0.987 correlation between data reads and
non-data instructions. Generic four-read and eight-read loops plus an
eight-store loop were therefore added. Thirty-six opened calibration cases
cross four working-set sizes and three work counts. The stabilized collection
used five sequential native processes per case; maximum native range was 7.64%
and fixed-frequency validation passed.

| Basis | Parameters | Correlation | Condition | Bound hits | Fit median | Fit maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Matched legacy | 6 | 0.653 | 3.55 | 0 | 21.98% | 67.34% |
| Add `data_reads` | 7 | 0.699 | 5.07 | 1 | 20.85% | 67.80% |
| Split misses | 8 | 0.757 | 5.04 | 3 | 20.21% | 66.76% |
| Combined | 9 | 0.731 | 5.70 | 3 | 20.21% | 66.76% |

The read-only candidate fits `data_reads = 0.00435 ns`, while its existing
`data_writes` coefficient reaches zero. Removing the four-read family changes
the read coefficient by 347%; removing the eight-read family drives it to zero.
It therefore fails both the bound and 20% stability gates.

Both split candidates fit zero costs for L1 and last-level write misses. The
combined candidate additionally drives the base read cost to zero and becomes
numerically equivalent to the split-only fit. Omitting a read-density family
changes the last-level read coefficient by 45.5--140.6%, and omitting the dense
writer destabilizes the write-side boundary solution. Both fail the bound and
stability gates.

No candidate passed opened-fit selection, so the predeclared 40-case holdout
was not collected. At the user's direction, the fitted coefficients were then
frozen and all four synchronization models were refit identically before an
actual 35-case renderer diagnostic. No coefficient was changed afterward.

| Pipeline | Max worker 1 | Max workers 2--7 | Runtime median/RMS | Rows within 20% | Max speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous current | +29.51% | +30.49% | 19.22% / 19.99% | 19/35 | 16.68% |
| Density-matched legacy | -19.83% | -21.58% | 17.98% / 17.34% | 24/35 | 18.83% |
| Add `data_reads` | -19.22% | -21.27% | 17.45% / 16.82% | 27/35 | 18.74% |
| Split misses | -20.62% | -22.32% | 18.79% / 18.04% | 21/35 | 18.83% |
| Combined | -20.62% | -22.32% | 18.79% / 18.04% | 21/35 | 18.83% |

The signs show that the old model overpredicts its worst rows while every new
fit underpredicts its worst rows. Most of the improvement comes from adding the
density calibration families and refitting the original six-feature basis.
The explicit read term improves the matched refit by another 0.61 points for
one worker and 0.31 points for workers 2--7. Split misses are worse than the
matched refit, and the combined model is numerically identical to split-only.

### Crossed access-subset diagnostic

The initial read and split candidates were followed by a larger synthetic-only
identification experiment. Four cache-line traversal pairs issue one or eight
reads or writes while holding visited lines fixed. Four crossed families issue
`r1w1`, `r8w1`, `r1w8`, or `r8w8` per line, producing mixed access and miss
profiles instead of identifying coefficients from single-feature kernels. The
amd64 memory operations use guarded explicit assembly; other supported builds
use a C++ fallback and do not make fixed-host timing claims.

The 132-case extension removed the combined fit collapse. Its nine columns are
full rank, maximum correlation is 0.475, condition is 3.93, and every
coefficient is positive. In particular, `Dr=0.091087 ns`, `Dw=0.072237 ns`,
`D1mr=0.066331 ns`, `D1mw=0.152123 ns`, `DLmr=0.042316 ns`, and
`DLmw=0.047179 ns`.

| Frozen pipeline | Max worker 1 | Max workers 2--7 | Runtime median/RMS | Rows within 20% | Max speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous data reads | -19.22% | -21.27% | 17.45% / 16.82% | 27/35 | 18.74% |
| Crossed matched legacy | -13.55% | -18.12% | 11.11% / 11.68% | 35/35 | 17.97% |
| Crossed data reads | +7.03% | +16.80% | 4.42% / 6.77% | 35/35 | 16.44% |
| Crossed split misses | -11.76% | -16.92% | 9.04% / 10.17% | 35/35 | 17.80% |
| Crossed combined | +7.61% | +17.28% | 3.92% / 6.70% | 35/35 | 16.47% |

Combined now differs from split-only. Relative to read-only it improves median
absolute error by 0.49 points and RMS by 0.07, but worsens the prioritized
maxima by 0.58 points for worker 1 and 0.48 points for workers 2--7. Read-only
therefore has the best maximum-runtime result, combined has the best median/RMS,
and every expanded-model row remains below 20%. The maximum five-run native
range was 33.63% due to isolated slow samples, while the worst row's MAD was
2.52%. Medians are used throughout, but the experiment is not promoted as a
timing model.

Reproduce the opened calibration with:

```bash
python3 volume-cartographer/scripts/calibrate_synthetic_event_features.py \
  --phase fit \
  --benchmark volume-cartographer/build-release/bin/bench_thread_pool_dispatch \
  --base-observations /tmp/synthetic-event-costs-fixed-3401-v10-overlap/observations.json \
  --output-dir /tmp/synthetic-event-features-v2 --native-trials 5
```

Exit status 2 is expected because every candidate is rejected.

The crossed-subset diagnostic uses the same command with
`--output-dir /tmp/synthetic-event-features-v4-crossed`; its 132 density cases
are declared by `ALL_DENSITY_FIT_CASES`.

Reproduce the synthetic comparison with:

```bash
python3 volume-cartographer/scripts/calibrate_synthetic_event_costs_minimax.py \
  --benchmark volume-cartographer/build-release/bin/bench_thread_pool_dispatch \
  --base-observations /tmp/synthetic-event-costs-fixed-3401-v10-overlap/observations.json \
  --output-dir /tmp/synthetic-event-costs-minimax-v3 --native-trials 5
```

Reproduce the two synthetic fits with:

```bash
python3 volume-cartographer/scripts/calibrate_synthetic_event_costs.py \
  --benchmark volume-cartographer/build-release/bin/bench_thread_pool_dispatch \
  --output-dir /tmp/synthetic-event-costs --native-trials 5
python3 volume-cartographer/scripts/calibrate_thread_sync_synthetic.py \
  --runner volume-cartographer/scripts/run_thread_sync_replay.py \
  --benchmark volume-cartographer/build-release/bin/bench_thread_pool_dispatch \
  --output-dir /tmp/thread-sync-synthetic \
  --event-cost-model /tmp/synthetic-event-costs/model.json \
  --trace-trials 3 --native-trials 5
python3 volume-cartographer/scripts/evaluate_render_attribution_sensitivity.py \
  --model /tmp/thread-sync-synthetic/model.json \
  --event-model /tmp/synthetic-event-costs/model.json \
  --cases /tmp/renderer-attribution-cases.json \
  --output /tmp/renderer-attribution-sensitivity.json
```

## Deterministic Cost

Configure the non-default Release benchmark graph and run the complete gate:

```bash
cmake -S volume-cartographer -B volume-cartographer/build/ci-render-benchmark \
  -G Ninja -DCMAKE_BUILD_TYPE=Release -DVC_MARCH_NATIVE=OFF \
  -DVC_TESTING=ON -DVC_RUN_RENDER_BENCHMARKS=ON \
  -DVC_BUILD_APPS=OFF -DVC_BUILD_UI_TRACER=OFF -DVC_BUILD_FLATBOI=OFF
cmake --build volume-cartographer/build/ci-render-benchmark \
  --target bench_render_synthetic --parallel 32
ctest --test-dir volume-cartographer/build/ci-render-benchmark \
  --output-on-failure -R '^test_render_synthetic_fixture$'
jobs=$(nproc)
cmake --build volume-cartographer/build/ci-render-benchmark \
  --target render_valgrind_ci --parallel "$jobs"
```

Ninja owns the complete artifact graph. It schedules at most four independent
Valgrind collectors at once. This is valid even for four-worker guest fixtures
because Valgrind serializes each guest; native verification remains a separate
non-oversubscribed test.

Callgrind starts with instrumentation disabled. The executable materializes and
warms the fixture, allocates one output and coverage matrix per measured
repetition, then uses Callgrind client requests to measure only the production
sampling calls. Validation happens after instrumentation stops.

The fixed simulated cache is:

- I1: 32 KiB, 8-way, 64-byte lines
- D1: 32 KiB, 8-way, 64-byte lines
- LL: 8 MiB, 16-way, 64-byte lines
- Branch simulation: enabled

The older CTest entries remain available for aggregate-cost reports. Their
model version 2 calculates modeled work cycles as:

```text
Ir / modeled_ipc
+ l1_miss_cycles * (I1mr + D1mr + D1mw)
+ last_level_miss_cycles * (ILmr + DLmr + DLmw)
+ branch_mispredict_cycles * (Bcm + Bim)
```

Those legacy coefficients and effective parallelism values live in
`core/test/data/render_callgrind_model.json`. Aggregate parallel work is divided
by the calibrated effective parallelism for the fixed worker count, then one
shared `nanoseconds_per_modeled_cycle` factor produces estimated ns/px and
Mpx/s. The model is fitted reproducibly with:

```bash
python3 volume-cartographer/scripts/calibrate_render_model.py \
  volume-cartographer/core/test/data/render_calibration_observations.json \
  --output volume-cartographer/core/test/data/render_callgrind_model.json
```

The calibration uses six scenarios for fitting and reserves
`fallback_3_shuffled` and `full_res_cache_stress_shuffled` for held-out
validation. On the AMD Ryzen 9 5950X reference host, training maximum error is
7.78% and held-out maximum error is 12.84%. The eight CI report cases are within
about 5% of their native medians.

This remains a modeled cost, not literal hardware cycles. Callgrind does not
model pipeline overlap, hardware prefetching, DVFS, or true simultaneous thread
execution. In particular, ideal-cache miss counts do not fully express the
native difference between correlated and shuffled memory access.

### Fresh Valgrind replay gate

The CI gate uses `scripts/run_render_valgrind_ci.py` and writes under
`build/ci-render-benchmark/render-valgrind-ci/<fixture>/<scenario>/`. Every case
has a separate-thread Callgrind artifact. Parallel cases additionally have a
DRD trace/event artifact. Collection manifests are written atomically after
validation, and each raw file is hashed before evaluation.

The frozen model is `core/test/data/render_valgrind_ci_model.json`. It uses the
seven synthetic-only data-read event coefficients. Serial score is total
modeled thread work per render call. Parallel score is the native FIFO replay
makespan per call with these fixed controls:

```text
workers = 4
cores = workers + 1 = 5
split = equal
residual_fraction = 0.5
wake_latency = 0
cross_thread_release = 3157.1798928563853 ns
replay_idle_scale = 1
dependency_excess_scale = 1
```

Process startup and native timing are not part of this score. The calibration
did not pass its absolute-timing promotion gates, so the metric is a relative
modeled-runtime score only. It must not be described as estimated native wall
time. Each candidate score must fall in `[0.90, 1.10]` times the checked
reference.

The gate rejects model hashes, checksums, incomplete DRD dependencies, compiler
or Valgrind versions, cache geometry, architecture, fixture shape, repetition
count, and worker count that differ from the reference. CI retains the complete
`render-valgrind-ci/` tree even on failure.

The 2026-08-07 GCC 15.3.0/Valgrind 3.25.1 reference collection took 15.73 s on
the four-core calibration host. A second fresh collection and gate took 16.38
s. Its parallel score ratios ranged from 0.993 to 1.016; all serial scores were
identical. CI derives Ninja concurrency from `nproc`; four was a property of
that reference runner, not a workflow constant.

Each calibrated JSON summary reports `modeled_cycles_per_pixel` and
`estimated_mpx_per_second` together with the shared conversion and reference
host. The throughput estimate is scoped to these exact synthetic fixtures.

## Operating And Updating The Gate

See [VC3D Rendering Performance Gate](render_valgrind_ci.md) for local use,
artifact diagnosis, CI activation, synthetic-only model recalibration,
eight-case reference refresh, and intentional tolerance changes. Legacy
aggregate-cost baselines remain in `core/test/data/render_callgrind_baseline.json`
with coefficients in `core/test/data/render_callgrind_model.json`.
