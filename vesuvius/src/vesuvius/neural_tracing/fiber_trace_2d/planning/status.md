# Whole-Batch Loader Parallelization Status

- [x] Capture current user task in `planning/task.md`.
- [x] Create focused implementation plan.
- [x] Enable whole-batch queueing for load-only benchmarks.
- [x] Update pipeline defaults and tune the example config to the measured
  16 queued / 4 whole-batch / 4 CP-worker shape.
- [x] Update specs/docs for load-only whole-batch queue behavior.
- [x] Avoid fully independent loader clone startup by sharing parsed records and deterministic order cache.
- [x] Add optional VC3D sampler cache budget / I/O thread config.
- [x] Tune example config to measured best current loader shape.
- [x] Compile-check changed Python.
- [x] Run focused loader tests.
- [x] Run load-only profile benchmark and record before/after result.
- [x] Update current-task log and changelog.
