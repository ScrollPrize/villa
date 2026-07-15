# 3D Trace2CP Metric Wiring Task Log

## Diagnosis

- The previous 3D follow-up did not fully satisfy the requested Trace2CP metric
  wiring.
- Implemented piece:
  - `fiber_trace_3d.trace2cp_bridge` can project dense 3D outputs onto a 2D
    Trace2CP strip and call the existing 2D scorer.
- Missing pieces:
  - `fiber_trace_3d/train.py` still uses `evaluate_dense_loss(...)` only.
  - 3D `best.pt` selection still uses dense loss, not `test/trace2cp_error`.
  - 3D configs do not expose explicit Trace2CP metric keys.
  - There is no 3D Trace2CP CLI visualization path.
  - There is no real train/CLI integration test for the bridge.

## Planning Result

- Replaced `planning/task.md` with the current missing-wiring task.
- Replaced `planning/task_plan.md` with a focused implementation plan for:
  - no-silent-simplification/no-silent-postponement reporting;
  - 3D Trace2CP config keys;
  - 2D geometry-loader reuse for metric coordinates;
  - 3D dense block inference and projection;
  - removal of the current 3D unit-sphere grid-search decoder from Trace2CP
    projection;
  - replacement with torch-vectorized analytic Lasagna 3x2 decoding using the
    established two-channel `atan2(...)/2` decode and Lasagna 3-plane
    reconstruction/sign-alignment logic;
  - fixed-set training metric evaluation;
  - TensorBoard/stdout/checkpoint selection wiring;
  - 3D CLI visualization;
  - focused tests and docs/spec updates.
- Updated `fiber_trace_2d/AGENTS.md` so future tasks must explicitly report
  simplifications, partial implementations, deferred items, unsupported cases,
  and intentionally skipped requirements in both `task_log.md` and the final
  response.
- Found existing analytic direction pieces that should have been used instead
  of grid search:
  - `fiber_trace_3d.direction.decode_lasagna_direction_2d`;
  - `fiber_trace_2d.direction.decode_lasagna_direction_xy`;
  - Lasagna `_decode_dir_angle` / `_estimate_normal` reconstruction logic in
    `lasagna/preprocess_cos_omezarr.py`.

## Validation

- Planning-only task; no code validation run yet.
