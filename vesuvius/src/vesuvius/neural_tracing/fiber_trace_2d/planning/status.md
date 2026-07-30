# Native Fiber Trace Lookahead And Pipeline Optimization Status

- [x] Read repository and fiber-trace workflow instructions
- [x] Replace the prior active task with the focused continuation task
- [x] Write the implementation, testing, spec, and docs plan
- [x] Review the plan directly against current Trace2CP specifications
- [x] Instrument exact lazy-lookahead potential without changing results
- [x] Benchmark exhaustive parent requirements and candidate reduction
- [x] Implement and test exact lazy lookahead if supported by measurements
- [x] Benchmark exact lazy lookahead and verify at most 8 restarts
- [x] Implement and test fused sampling/scoring if still worthwhile
- [x] Benchmark fused sampling/scoring and verify at most 8 restarts
- [x] Test intermediate caps only after exact options proved insufficient
- [x] Update specs and durable docs for retained behavior
- [x] Run final focused tests, benchmark, and consistency review

Baseline: 21.155s wall / 619.366s CPU, 105,810,462 candidates, 4,170
generations, and 8 restarts over 87 segments.

The result-neutral instrumentation derives a conservative exact parent count
from each already-computed exhaustive frontier. Equal lower bounds are retained
and incomplete final beam sets require all parents. The focused fiber suite
passes all 28 cases.

The approved exhaustive benchmark retained 8 restarts and found that 462,332
of 1,290,087 intermediate parents are required (35.8%). Required parent count
is mean 223.7, p50 230, p95 320, and max 495. Exact lazy expansion predicts
37,448,892 second-step candidates instead of 104,497,047.

Exact lazy mode remains available with `--lookahead-parent-cap 0`; it expands
256 lowest-bound parents first, then batches of 64 until a strict exact
stopping condition is met. `--exhaustive-lookahead` evaluates the full
frontier. Original global child indices preserve exhaustive ties.

The approved lazy benchmark completed in 12.868s wall / 395.718s CPU with
47,001,222 candidates, 4,171 generations, and 7 restarts. It evaluated
564,039 of 1,290,087 lookahead parents and 45,687,159 of 104,497,047
second-step children. Relative to the committed baseline this is a 1.64x wall
speedup and a 56% candidate reduction without a quality regression.

Fused persisted sampling/scoring now uses a shared VC3D requested-level corner
visitor. It gathers each point's ordered corners from the already pinned chunks,
decodes prediction and normal tensors, and writes the candidate score before
moving to the next point. The existing materializing corner API is implemented
through the same visitor, while generic prediction/normal sources retain their
previous fallback. Focused tests pass: 29 fiber, 16 corner sampler, and 11
Lasagna normal sampler cases before the cap experiments.

The approved fused benchmark completed in 7.624s wall / 51.303s CPU with the
same 47,001,222 candidates, 4,171 generations, and 7 restarts as lazy-only.
Combined pinned-corner gathering, decode, and score time is 3.864s versus the
previous separate 4.916s corner plus 3.310s scoring stages. This is a 1.69x
wall improvement over lazy-only and 2.77x over the committed baseline.

The approved cap-64 benchmark completed in 2.750s wall / 13.328s CPU with
12,018,375 candidates, 4,168 generations, and 6 restarts. It is 7.69x faster
than the committed baseline and satisfies the quality threshold. Lower planned
caps remain to be tested independently.

The clean approved cap-32 benchmark completed in 1.869s wall / 8.222s CPU with
6,910,839 candidates, 4,318 generations, and 7 restarts. An earlier run that
overlapped a compile is excluded. Cap 32 is 11.3x faster than baseline and is
the current best accepted trial.

Cap 16 is rejected: its approved run took 4.190s wall and produced 10
restarts, exceeding the quality threshold. Cap 8 remains as the final planned
independent trial; cap 32 remains the retained candidate.

Cap 8 is also rejected: 1.327s wall but 14 restarts. All planned cap trials are
complete. Cap 32 is restored as the retained default, with `0` available for
exact uncapped lazy expansion.

Final validation on the restored cap-32 build passes 30 fiber-trace, 16 corner
sampler, and 11 Lasagna normal sampler cases. The metric CLI builds and reports
the retained default and exact/exhaustive controls; `git diff --check` is clean.
