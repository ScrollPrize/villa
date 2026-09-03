# Task Log

## Scope

- Benchmark the current Fiberlet tracer from both endpoints of every tagged
  reference run inside the fixed crop.
- Reuse `measureFiberReplayThreshold` and `traceFiberletGraphReplay`; do not
  introduce a parallel evaluation implementation.
- Store each endpoint and oracle-pruning run as Markdown and maintain a separate
  compact result index.

## Decisions

- Clip full reference polylines to the half-open benchmark crop and retain all
  contiguous in-crop runs.
- Reverse each run to create the opposite endpoint case.
- Stop accounting at the first failure. The aggregate denominator is twice the
  total in-crop reference length.
- Require an explicit positive base-voxel size for millimeter output.

## Independent Review

- Keep this benchmark's crop clipping separate from `direction-ablation`, whose
  reference diagnostics intentionally retain complete JSON polylines.
- Restrict the first seed to the initial replay seed window. Credit a valid,
  threshold-checked seed offset; return zero when no endpoint seed exists.
- Add shared opt-in stop-at-first-failure behavior while leaving ordinary replay
  restart semantics unchanged.
- Report both length-weighted success and binary completion, and distinguish
  all-direction mean credited length from failed-direction mean failure length.
- Emit versioned JSON and render Markdown from it; do not parse unstable console
  tables.
- Record dirty-tree identity, host/build/cache metadata, and one Markdown file
  per invocation. These are manual external-data evaluations, not CI gates.

## Implementation

- Added opt-in replay controls for an initial endpoint seed window and stopping
  at the first failure; ordinary replay defaults are unchanged.
- Added reusable crop-run preparation, bidirectional case generation,
  first-failure accounting, physical conversion, and versioned JSON output.
- Added `vc_fiber_trace_chunk reference-replay-benchmark`; reference directions
  run concurrently while each beam search uses one expansion worker.
- Added focused tests for crop re-entry, max-face clipping, forward/reverse
  failure accounting, non-distance failure aggregation, unit conversion,
  endpoint seed restriction, and first-failure termination.

## Validation

- Release build: `vc_fiber_trace_chunk`, `test_fiberlet_paths`, and
  `test_fiber_reference_replay_benchmark` build successfully.
- `test_fiber_reference_replay_benchmark`: 3 test cases passed.
- The full `test_fiberlet_paths` binary retains existing bit-exact prepared
  scoring failures at line 414 in this build. The newly added replay cases did
  not emit failures.
