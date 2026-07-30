# Native Fiber Trace Lookahead And Pipeline Optimization Log

## Baseline

- Commit `2bf48dea0`: 21.155s wall / 619.366s CPU, 105,810,462
  candidates, 4,170 generations, and 8 restarts over 87 segments.
- Final baseline stages: task build 2.399s, corner batch 7.300s, fused candidate
  scoring 5.773s, frontier construction 3.239s, and pruning 1.257s.

## Process

- Previous task details were removed from the active task files as requested;
  durable results remain in `planning/changelog.md` and git history.
- The workflow requests independent plan review, but no independent-agent tool
  is available in the current context. A direct review found exact lazy
  expansion compatible with the existing post-lookahead pruning requirement.
  Approximate intermediate caps are isolated as opt-in experiments because
  they change that requirement.

## Exact Lazy-Lookahead Instrumentation

- Added result-neutral profile counters for exhaustive versus conservatively
  required intermediate parents and child candidates. The required count uses
  the selected reached loss or worst complete spatially accepted beam loss and
  includes every parent with a lower bound equal to the threshold.
- Added focused tests for equal-bound retention and conservative behavior when
  no finite threshold or complete final beam set exists.
- Validation: `test_fiber_trace3d` passes 28 cases.
- The representative benchmark is pending explicit user approval immediately
  before execution.

### Exhaustive Measurement

- User approved the representative benchmark immediately before execution.
- Result: 23.396s wall / 681.346s CPU, 8 restarts, 105,810,462 candidates,
  and 4,170 generations. The instrumentation and system variation made this
  slower than the committed 21.155s baseline; trace quality was unchanged.
- Across 2,067 final lookahead frontiers, exact reproduction requires 462,332
  of 1,290,087 parents (35.8%). Required parents are mean 223.673, p50 230,
  p95 320, and max 495.
- The exact lower bound predicts reducing second-step candidates from
  104,497,047 to 37,448,892 and total candidates from 105,810,462 to roughly
  38.8 million. This supports implementing exact lazy expansion before fused
  sampling/scoring.

## Exact Lazy-Lookahead Implementation

- Added default exact lazy final-lookahead evaluation plus an
  `--exhaustive-lookahead` control. Parents are ordered by cumulative lower
  bound and original index, the first 256 are evaluated together, and further
  batches contain 64 parents.
- Child tasks and scores are stored at their original exhaustive global indices
  so equal-loss ordering, reached-target selection, and spatial beam pruning
  see the same indices as the exhaustive implementation.
- Lazy evaluation stops only when the next unevaluated parent lower bound is
  strictly greater than the current best reached loss or complete spatially
  accepted beam threshold. Equal bounds are always evaluated.
- Added a synthetic lazy-versus-exhaustive trace parity test. Validation passes
  29 fiber-trace, 15 corner-sampler, and 11 Lasagna normal-sampler cases.

### Lazy Measurement

- User approved the representative lazy benchmark immediately before
  execution. The first tool result was lost during context compaction, so the
  user explicitly approved rerunning the same command with output captured to
  `/tmp`.
- Result: 12.868s wall / 395.718s CPU, 7 restarts, 47,001,222 candidates,
  and 4,171 generations. This improves the committed baseline by 1.64x in wall
  time and reduces candidates by 56%; quality remains within the at-most-eight
  restart acceptance criterion.
- Lazy evaluation processed 564,039 of 1,290,087 lookahead parents and
  45,687,159 of 104,497,047 second-step children. Batch granularity accounts
  for evaluation above the theoretical 462,059 required parents.
- Remaining measured stages are prediction corner sampling 4.916s, candidate
  scoring 3.310s, frontier construction 1.216s, task construction 1.049s,
  lookahead decisions 0.949s, and pruning 0.628s. Fused sampling/scoring remains
  worthwhile.

## Fused Sampling And Scoring

- Added a shared requested-level corner visitor to `ChunkedPlaneSampler` and
  routed the existing materializing corner API through it. Geometry,
  dependencies, prefetch, pinned chunk ownership, ordered corner gathering,
  fill handling, and errors therefore remain in one implementation.
- Added the corresponding Lasagna channel visitor and a persisted fiber-field
  visitor that combines prediction and required Lasagna-normal channels in the
  same pinned batch.
- Persisted candidate scoring now consumes one point's corner span directly,
  decodes all options and the normal, and writes the score at the original
  candidate index. It no longer materializes candidate-sized per-volume corner
  arrays or traverses temporary decoded options twice. Generic sources and
  nonparallel sampling keep the prior fallback path.
- Profiling attribution changes for the fused path: callback decode/scoring is
  included in `prediction_corner_gather_s`; it is no longer separately counted
  in `candidate_score_s`.
- Added visitor coverage for mixed chunk grids and invalid points. One initial
  fixture coordinate crossed into an intentionally unresolved mock chunk; it
  was corrected to test a resident valid point plus an explicitly invalid
  point. This was a test-fixture issue, not a retained implementation failure.
- Validation passes 29 fiber-trace, 16 corner-sampler, and 11 Lasagna-normal
  cases. `git diff --check` is clean. The representative fused benchmark is
  pending explicit user approval immediately before execution.

### Fused Measurement

- User approved the representative fused benchmark immediately before
  execution.
- Result: 7.624s wall / 51.303s CPU, 7 restarts, 47,001,222 candidates, and
  4,171 generations. Candidate count and trace quality exactly match the
  lazy-only measurement.
- The fused `prediction_corner_s` is 3.864s, including decode/scoring in its
  2.126s gather stage. The prior lazy-only run spent 4.916s in corner sampling
  plus 3.310s in separate scoring. Other measured stages are task construction
  0.792s, frontier construction 0.953s, lookahead decisions 0.859s, and pruning
  0.580s.
- This is a 1.69x wall speedup over lazy-only and 2.77x over the committed
  21.155s baseline. Because it remains short of the requested 5-10x range, the
  planned opt-in deterministic intermediate-parent caps remain worth testing.

## Intermediate Parent-Cap Experiments

### Cap 64

- Added a deterministic final-lookahead parent cap while retaining original
  parent lower-bound order and original global candidate indices. A zero cap
  remains the exact uncapped mode; the exact parity test sets it explicitly.
- Added focused coverage that verifies evaluated parents cannot exceed the cap
  per final lookahead frontier. Validation passes 30 fiber-trace, 16
  corner-sampler, and 11 Lasagna-normal cases.
- User approved running the byte-for-byte unchanged representative command.
- Result: 2.750s wall / 13.328s CPU, 6 restarts, 12,018,375 candidates, and
  4,168 generations. The run evaluated 132,160 final-lookahead parents and
  10,704,960 second-step children.
- This is 2.77x faster than the uncapped fused run and 7.69x faster than the
  committed baseline while improving the measured restart count. Cap 64 meets
  the requested performance range and is a viable retained default.

### Cap 32

- The first approved cap-32 run overlapped another compile. Its deterministic
  search result was 7 restarts and 6,910,839 candidates, but its 3.036s wall /
  9.437s CPU timing is explicitly discarded as contaminated.
- After the user confirmed resources were clear, the byte-for-byte unchanged
  command was approved and rerun.
- Clean result: 1.869s wall / 8.222s CPU, 7 restarts, 6,910,839 candidates,
  and 4,318 generations. It evaluated 68,512 final-lookahead parents and
  5,549,472 second-step children.
- Cap 32 is 1.47x faster than cap 64 and 11.3x faster than the committed
  baseline while remaining within the quality threshold.

### Cap 16

- User approved the byte-for-byte unchanged representative command.
- Result: 4.190s wall / 5.831s CPU, 10 restarts, 4,148,415 candidates, and
  4,331 generations. This exceeds the accepted maximum of 8 restarts and is
  rejected.
- The degraded trace also spent 2.726s pinning chunks reached by its changed
  path, so it was slower than cap 32 despite evaluating fewer candidates.

### Cap 8

- User approved the byte-for-byte unchanged representative command.
- Result: 1.327s wall / 4.154s CPU, 14 restarts, 2,675,511 candidates, and
  4,205 generations. This substantially exceeds the accepted restart limit and
  is rejected.
- Cap 32 is restored as the retained default. The CLI exposes
  `--lookahead-parent-cap`; `0` selects exact uncapped lazy evaluation.

## Final Validation

- Restored and rebuilt the retained cap-32 default after all isolated trials.
- Final focused results: 30 fiber-trace, 16 corner-sampler, and 11
  Lasagna-normal test cases pass.
- `vc_fiber_trace_metric --help` reports the cap-32 default,
  `--lookahead-parent-cap 0` exact mode, and full
  `--exhaustive-lookahead` mode.
- Updated the normative spec, code-structure documentation, status, and durable
  changelog. `git diff --check` is clean.
