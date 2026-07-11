# Bidirectional Trace2CP Status

- [x] Capture current user task in `planning/task.md`.
- [x] Create focused task plan.
- [x] Review task plan against `planning/specs.md`, `planning/plan.md`, and
  `planning/task.md`.
- [x] Implement bidirectional Trace2CP tracing, scoring, and visualization.
- [x] Add focused regression tests.
- [x] Update specs, docs, changelog, and task log.
- [x] Compile-check changed Python.
- [x] Run focused fiber-trace tests.

Result: `--trace2cp-vis` now traces start-to-target and target-to-start on the
same segment strip, draws both traces, and reports `trace2cp_score` as the
average of the two directional normalized scores.
