# Status

## Independent fiber normal orientation

- [x] Plan and implementation independently reviewed.
- [x] Native second-pass implementation;142regression tests pass.
- [x] Pilot and full database validation; documentation/results complete.

- [x] Preserve prior experiments and create clean branch from updated origin/main.
- [x] Plan and independent review.
- [x] Download AFV and preserve source metadata.
- [x] Identify source scan and distinguish native coordinates from tracing scale.
- [x] Validate integrity and all geometry block structure.
- [x] Document findings, limitations and future global-SGD purpose.
- [ ] Future task: design and implement global assignment with PyTorch SGD.

## Shared BP merge

- [x] Commit AFV setup and merge fiber-exp5.
- [x] Preserve both codec packaging and fiber tracer dependencies at conflict.
- [x] Inspect shared constraint extraction and write bp_constraints.md.
- [x] Run available Python benchmark-support tests (12 passed).
- [x] Native rebuild/tests after user installed SQLite3 development dependency.

## Unsplit unsigned local-events benchmark (complete)

- [x] Plan and independent review in local_events_benchmark_plan.md.
- [x] Implement shared unsplit mode, CLI flag and strict endpoint behavior.
- [x] Add measurement-equivalence and endpoint-support regression tests.
- [x] Locate original Paris4 sheet-normal manifest used by prior BP benchmarks.
- [x] Build and native regression suite: 137 tests passed.
- [x] Native AFV benchmark adapter using shared collection/sampling/extraction code.
- [x] Remove unwanted normal alignment; unsigned extraction only.
- [x] Timed full-XY Z slabs: thickness256/1024/4096; three repetitions each.
- [x] Whole-volume estimate with boundary/density/cache caveats.
- [x] Persist results, commands and validation in unsigned_constraints_benchmark.md.

## Full unsigned extraction/database (complete)

- [x] Plan and independent review: full_extraction_plan.md.
- [x] Native catalog selection and transactional SQLite exporter.
- [x] Native rebuild and real-data pilot export validation.
- [x] Full512698fiber extraction/export:2,334,188events.
- [x] Full integrity/source completeness and per-fiber/type statistics.
- [x] Schema and reusable commands: constraint_database.md.

## Per-check rejection audit

- [x] Independent plan review; instrument counters without changing decisions.
- [x] Build/tests and exact slab measurement comparison against saved DB.
- [x] Full diagnostic rerun, all-measurement equivalence and rejection ledger.
- [x] Report per-check fractions and overlapping winding predicates.

## Parallel reciprocal removal / perpendicular iteration comparison

- [x] Plan reviewed; disable both parallel reciprocal checks by default.
- [x] Parameterize perpendicular iterations; instrument remaining losses/bounds.
- [x] Native139tests and exact legacy slab equivalence passed.
- [x] Full4/16/64iteration runs and comparison report; extra iterations do not help.

## Moving-window perpendicular refinement

- [x] Independent plan review and shared scorer implementation.
- [x] Regression tests and legacy pilot validation (140 tests;19,889exact events).
- [x] Full moving-window4/16/64 comparison and persistent results.
- [x] Default moving16; final140tests and exact legacy pilot passed.

## Continuous winding extraction (not downstream processing)

- [x] Plan reviewed; shared existing flags reused, historical configs pinned.
- [x] Raw local refinement avoids winding rounding; finite probe validation kept.
- [x]141tests and exact legacy pilot pass.
- [x] Full new database extraction/validation and result report (7,368,335events).

## Relation-specific rejection diagnostics

- [x] Plan reviewed; diagnostic counters and ledger tests implemented.
- [x]141native tests;all7,368,335raw measurements exact;per-type tables complete.
