# Native Fiber Trace Locality And Scheduling Optimization Log

## Starting Point

- Retained baseline: 1.869s wall / 8.222s CPU, 6,910,839 candidates,
  4,318 generations, and 7 restarts over 87 segments.
- Dominant retained stages: fused pinned-corner gather/decode/score 0.845s,
  frontier construction 0.217s, pruning 0.167s, task construction 0.137s,
  and start sampling 0.132s.
- Prior active task details were intentionally discarded. Durable results remain
  in `planning/changelog.md` and git history.

## Planning

- The new task covers worker granularity, deterministic partial parent
  selection, compact frontiers, spatial chunk/cube ordering, unique-cube corner
  reuse, persistent two-depth pins, bounded envelope prefetch, rolling pins,
  nearby fixed caps, and adaptive escalation.
- The mandatory depth-one/depth-two decision barrier remains: second-depth
  coordinates cannot be generated until first-depth scoring selects parents.
- Representative benchmarks require explicit user approval immediately before
  every invocation. The exact existing command and cache path must be reused.
- No independent-agent tool is available in the current context. Direct review
  found the plan consistent with the current original-index determinism,
  shared-corner-visitor, cap-32, exact-lazy, exhaustive-mode, cache-budget, and
  portability requirements.
- The review confirmed that depth-two work cannot be batched before first-depth
  selection. Locality work must therefore use spatial ordering and a bounded
  session across the required decision barrier, not speculative expansion.

## Phase 1 Instrumentation

- Added profile-gated depth-one/depth-two candidate batch counts and sizes,
  submitted corner-worker task counts, unique integer voxel-cube counts and
  occupancy histogram, dependency overlap, parent-order time, frontier-storage
  time, and allocated/evaluated frontier slots.
- Dependency identities are collected from the existing sampler layout and are
  used only for aggregate overlap diagnostics. Candidate ordering, sampling,
  interpolation, and selection remain unchanged.
- Corrected evaluated-frontier accounting to count only parent batches actually
  processed when exact lazy evaluation terminates early.
- Extended the grouped-corner sampler test with two points sharing one voxel
  cube and one invalid point. It verifies callback semantics plus unique-cube,
  occupancy, worker-task, and dependency instrumentation.
- Built `vc_fiber_trace_metric`, `test_fiber_trace3d`,
  `test_chunked_plane_sampler_fallback`, and
  `test_lasagna_normal_sampler` successfully.
- Validation: 30 tracer tests, 16 chunk-sampler tests, and 11 Lasagna-normal
  tests passed.
- At Phase 1 implementation completion no representative benchmark had yet
  been run; the approved result is recorded below.

## Instrumented Representative Result

- Approved unchanged command completed with 1.921s wall / 8.197s CPU,
  6,910,839 candidates, 4,318 generations, and 7 restarts. This preserves the
  retained workload and quality; the 0.052s wall increase over the retained
  baseline includes profile-only locality maps and dependency collection.
- Depth one: 2,177 batches / 1,361,367 points, p50/p95 batch 648.
- Depth two: 2,141 batches / 5,549,472 points, p50/p95 batch 2,592.
- Corner scoring submitted 129,279 worker tasks. There were 30,707 unique
  integer voxel cubes for 6,910,839 valid points: 225.1 points/cube on average,
  with occupancy p50 60, p95 at least 64 due the bounded histogram, and max
  2,446.
- Dependency Jaccard overlap was 93.5% between lookahead depths and 93.8%
  across consecutive trace steps.
- Lazy lookahead allocated 108,381,159 frontier slots but evaluated only
  5,549,472 slots. Measured parent ordering was 0.070s and full frontier
  storage allocation/clearing was 0.183s.
- These results support all three main directions: coarser corner worker tasks,
  compact capped-frontier storage, and shared-cube/persistent-pin sampling.

## Worker Granularity Trial

- Fused corner scoring now requests at most one worker per 256 candidates.
  Frontier materialization and generic sampling retain their existing worker
  behavior.
- The existing contiguous static ranges and original callback indices are
  unchanged, so this is result-neutral.
- Added direct worker-count coverage for representative 648- and 2,592-point
  batches. All 31 tracer tests pass.
- Representative timing is pending explicit approval.

- Representative result: 2.119s wall / 6.553s CPU, with unchanged 7
  restarts, 6,910,839 candidates, and 4,318 generations. Worker submissions
  fell from 129,279 to 29,908, but corner gather rose from 0.537s to 0.936s.
  The trial reduced CPU by 20% but regressed wall time by 10%; it was rejected
  and reverted.

## Compact Capped Frontier

- Replaced lazy final-frontier arrays sized for every potential child with
  compact aligned task, score, and frontier arrays containing evaluated
  children only.
- Each compact frontier record carries its original global child index. Equal
  loss/depth selection and reached-target ties compare that index, preserving
  the former full-array generation-order behavior.
- Potential and evaluated child counts remain separate in exact-lookahead
  profiling. Exact lazy-versus-exhaustive test parity and all 30 tracer tests
  pass.
- The benchmark load gate reported 96% CPU idle in both live samples and a
  runnable queue of one. The representative result was 1.736s wall / 8.128s
  CPU, 7 restarts, 6,910,839 candidates, and 4,318 generations. Logical
  frontier slots fell from 108,381,159 to 5,549,472 and measured storage work
  from 0.183s to 0.027s. This 9.6% wall improvement over the instrumented
  baseline is retained.

## Benchmark Load Gate

- Per updated user direction, representative runs no longer require a separate
  approval when a short `vmstat` sample shows the host is quiet. If a compile,
  runnable queue, or material competing CPU load is present, do not run and
  wait for user clearance. The benchmark command and cache path remain fixed.

## Deterministic Partial Parent Selection Trial

- Capped lazy lookahead now uses deterministic partial selection followed by a
  full sort of the retained prefix by `(loss, original_parent_index)`.
  Uncapped exact mode still fully sorts all parents.
- Added direct coverage for capped ties, uncapped full ordering, and an empty
  prefix. All 31 tracer tests pass.
- The pre-benchmark load gate blocked measurement: live samples had runnable
  queues of 17 and 10 with only 25% and 55% CPU idle. No representative run was
  started; the trial remains pending measurement.
- A later 99%-idle run completed at 1.535s wall / 7.796s CPU with unchanged 7
  restarts, 6,910,839 candidates, and 4,318 generations. Parent-order time fell
  from 0.074s to 0.011s. The trial is retained.

## Unique Voxel-Cube Corner Reuse

- The shared chunked corner visitor now interns valid points by integer voxel
  cube, builds chunk dependency/offset metadata once per unique cube, gathers
  each physical volume's ordered eight corners once, and then invokes the
  existing callback for every original point with its unchanged fraction and
  index.
- Invalid points, clamped volume edges, mixed chunk layouts, boundary corners,
  missing-fill behavior, malformed extent checks, and callback indexing remain
  on the shared sampler path. All focused suites pass: 31 tracer, 16 chunked
  sampler, and 11 Lasagna-normal tests.
- On a fully idle host, the representative result was 1.352s wall / 6.831s CPU
  with unchanged 7 restarts and exact candidate/generation counts. Corner
  layout time fell from 0.141s to 0.004s, layout chunk runs from 272,607 to
  11,162, and total prediction-corner time from 0.849s to 0.675s. The trial is
  retained.

## New Measured Follow-Up

- After cube reuse, the separate final-frontier materialization pass still
  costs 0.195s, while pinning costs 0.046s. A result-neutral fused frontier
  construction trial was added to the plan ahead of persistent pin-session
  work because it has the larger measured ceiling.
