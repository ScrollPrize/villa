# VC3D render attribution and lookup repair task log

## 2026-08-18 findings

- The frozen reference predates `81f6a8ffb` (`Vc3d renderorder`). Its parent
  passed the gate and the squash commit failed it. The later
  `QuadSurface::gen()` optimization cannot be the original trigger because the
  synthetic fixture calls `ChunkedPlaneSampler` directly.
- Callgrind and DRD are necessarily collected in separate Valgrind processes.
  Current evaluation passes per-thread Callgrind costs directly to a DRD graph
  under the same raw Valgrind IDs. Worker participation and executed work are
  not stable identities across those executions.
- `Graph::assignCosts()` iterates DRD threads. It fails when a DRD thread lacks
  a same-numbered Callgrind profile, but does not reject extra Callgrind thread
  costs. A Callgrind-only worker cost can therefore be silently omitted from the
  replay.
- Every CI case currently uses one Callgrind execution and one first-complete
  DRD execution. No median or repeated-trace aggregation protects the parallel
  score from attribution/scheduling variation.
- The event-cost model contains nonlinear interaction features. Each Callgrind
  thread must continue to be modeled independently before worker costs are
  summed; summing raw counters first changes the modeled total.
- Periodic Callgrind dumps at 10,000 basic blocks produce chronological,
  separate-thread delta profiles without changing the benchmark binary.
- Three repeated `parallel/full_res` DRD runs had the same measured render
  structure: seven main quanta and sixteen quanta on each of four workers. One
  whole-process trace differed by one unrelated main-thread quantum, confirming
  that the graph must be trimmed to the measured render.
- The existing benchmark timing calls appear passively in the DRD syscall log.
  The measured render is the unique pair of main-thread `clock_gettime` calls
  sharing a stack address. This provides boundaries without markers.
- Fair-scheduled `mixed_correlated` Callgrind runs execute one logical range per
  worker, but physical IDs vary. Their canonical worker instruction totals are
  stable at approximately 13.38M, 13.38M, 14.53M, and 16.17M. Independent DRD
  runs produce corresponding measured worker lengths of 36/37, 36, 40, and 44
  quanta. Rank matching is therefore viable; the two shortest ranges are an
  equivalent tie.
- `--fair-sched=no` is unsuitable: one physical worker may execute multiple
  logical ranges while another remains effectively idle.
- Reducing the fair-scheduled quantum from 10,000 to 1,000 basic blocks does
  not stabilize raw worker IDs. Physical IDs are therefore never a valid
  cross-run identity, regardless of quantum.
- A three-run, four-scenario passive matrix with the unmodified release binary
  validated canonical reconstruction:

  | Scenario | Canonical worker `Ir` stability | DRD measured quanta | Worst 32-bin cost-shape delta |
  | --- | ---: | --- | ---: |
  | `full_res` | 0.29% | `16,16,16,16` | 0.23% |
  | `fallback_3` | 0.74% | `56,57,62,71` | 0.36% |
  | `mixed_correlated` | 0.48% | `36,36,40,44` | 0.37% |
  | `mixed_shuffled` | 0.08% | `77,77,78,78` | 0.03% |

- Replaying every admissible mapping inside equal-quantum groups bounded the
  remaining identity ambiguity at 0.90% makespan for `full_res`, 0.11% for
  `mixed_shuffled`, and 0% for `fallback_3` and `mixed_correlated`. The gate can
  conservatively report the maximum mapping when a tie is equivalent instead
  of selecting an arbitrary worker.
- The passive task-start order is not sufficient to identify FIFO task order:
  multiple workers can dequeue inside one periodic Callgrind interval. Matching
  must use complete canonical workload signatures and enumerate equivalent
  ties, not first-observed worker order.

## Attribution options

### A. Main/worker role pooling (recommended first)

Keep thread 1 as the stable process-main role. Sum independently modeled
non-main Callgrind costs, then distribute that complete worker-pool cost over
all eligible non-main DRD work windows using the existing quantum/residual
weights. This removes invented worker identity, accepts different worker sets,
preserves total modeled work, and remains completely passive. DRD event thread
IDs are retained, so worker program order, blocking, wakeups, and cross-thread
dependencies continue to determine replay makespan; only cross-process cost
identity is pooled.

Tradeoff: per-worker Callgrind imbalance is deliberately discarded. DRD still
supplies worker participation, dependencies, blocking, and available work
windows. This is appropriate unless repeated tests show that worker-cost
imbalance itself carries necessary predictive information.

### B. Worker-cost permutation ensemble

Treat Callgrind worker costs and DRD worker schedules as unlabeled multisets.
Evaluate all worker assignments (small for the configured four workers) and use
the median while reporting min/max bounds. This preserves measured imbalance
without claiming IDs correspond.

Tradeoff: differing worker participation requires zero-padding or pooling, and
the selected statistic is less directly interpretable. The worst-case bound is
likely too pessimistic for a regression gate. Keep this as a diagnostic or a
fallback if role pooling loses important behavior.

### C. Repeated independent trace ensemble

Collect several complete Callgrind/DRD samples and gate a median role-pooled
score. This addresses residual variation in synchronization outcomes and cache
simulation.

Tradeoff: it multiplies the expensive Valgrind portion of CI and does not by
itself repair invalid raw-ID attribution. Consider it only after option A and
prefer repeating DRD alone if Callgrind total work proves stable.

### D. Deterministic benchmark scheduling or explicit task markers

Force task-to-worker assignment or add logical phase/task events so profiles
can be paired exactly.

Rejected as the primary solution: it changes or instruments the renderer being
measured, stops being passive, and can hide the scheduling regressions the gate
is meant to detect.

### E. Aggregate work divided by configured worker count

Discard DRD and derive runtime from total modeled work plus a fixed parallelism
factor.

Rejected: it cannot see startup, blocking, synchronization, or lost
parallelism, which are central requirements of this benchmark.

## Speed notes for phase 2

- `PendingRenderJob.chunkRequest` is created once from view ID plus render
  request ID. `renderFrame()` copies it into base/overlay sampling options, and
  all sampling calls in that job reuse it. The request is therefore constant
  per render job, but not across jobs or request-ID changes.
- `LevelAccess` already hoists shape, chunk shape, transform, dtype/fill, and
  level validity once per sampled level.
- `LocalChunkCache` already has a same-key `lastResult` fast path and pins up to
  eight chunk results. A repeated successful chunk does not invoke
  `IChunkedArray::tryGetChunk()` again within that local sampler cache. Separate
  tile/level sampling contexts do not share this local hit state.
- The caller still computes voxel-to-chunk integer divisions and assembles a
  `ChunkKey` before `LocalChunkCache` can detect that hit. A successful-chunk
  cursor with cached voxel bounds can bypass both for correlated samples.
- Production `ChunkCache` directly overrides contextual `tryGetChunk()`. The
  synthetic fixture overrides only the compatibility overload, so its measured
  path currently includes an extra virtual forwarding call that production
  rendering does not.
- `ChunkCache::tryGetChunk()` reloads shared state, appends source ID to a new
  full key, and acquires the cache mutex for each local-cache miss. A prepared
  source/request/level lookup handle may remove repeated setup, but should be
  attempted only if profiles still identify this path after the cursor/local
  cache improvements.

## Current decision

Prototype deterministic paired reconstruction instead of role pooling. Keep the
measured program unchanged and reject ambiguous pairings. Use role pooling only
as a diagnostic baseline. Do not update the reference or tolerance yet. Defer
speed implementation until attribution is correct enough to evaluate it, while
retaining the pre/post render-order commits as the A/B workload.

The no-code-change feasibility prototype passed. Production wiring may change
collector/parser/replay tooling, but must not modify or instrument the measured
renderer, benchmark, thread pool, or workload. Equivalent tie groups should be
enumerated and scored conservatively; non-equivalent ambiguous groups fail.

## Native implementation result

- The regular Ninja estimate now invokes Valgrind directly and runs raw
  Callgrind parsing, measured-window DRD parsing, canonical matching,
  chronological attribution, conservative replay, reference comparison, and
  JSON output in C++. Python is not in the evaluation path.
- The measured renderer, benchmark, thread pool, and workload were not changed.
- A fresh current-build parallel matrix produced these conservative scores:

  | Scenario | Score (ns/call) | Mappings | Best-to-worst mapping span |
  | --- | ---: | ---: | ---: |
  | `full_res` | 917,189 | 24 | 0.87% |
  | `fallback_3` | 2,515,155 | 2 | 0.12% |
  | `mixed_correlated` | 1,700,855 | 2 | 0.00% |
  | `mixed_shuffled` | 3,155,879 | 2 | 0.02% |

- Three additional sequential fresh `parallel/full_res` pairs scored 917,333,
  917,297, and 917,304 ns/call. The full range is 0.004%, confirming that the
  paired estimate is stable on the current binary.
- A fresh tied `full_res` group had 2.38% total modeled-cost spread but only
  0.22% normalized chronological-shape spread. The predeclared native gates are
  therefore separate: 5% total cost and 2% 32-bin shape. Every passing tie is
  still fully enumerated and scored by its maximum makespan.
