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

- Implemented `decode_lasagna_direction_3x2_analytic(...)` in
  `fiber_trace_3d.direction` and replaced `fiber_trace_3d.projection` with an
  analytic-only projection path. The previous candidate-grid decoder is no
  longer present in production 3D code.
- Updated `trace2cp_bridge` so 3D Trace2CP projection no longer accepts or
  forwards `candidate_count`.
- Added explicit 3D Trace2CP config parsing in `fiber_trace_3d.train`.
  Auto-built metric geometry now requires explicit 2D strip geometry keys when
  no separate `test_trace2cp_loader_config` is supplied.
- Added tiled 3D inference over 2D Trace2CP segment coordinates. It tiles in
  Trace2CP image space, reads a selected-level 3D bounding block for each tile,
  runs the 3D model, projects direction/presence back to 2D, and scores with
  the existing 2D Trace2CP scorer.
- Wired 3D test evaluation to log `test/trace2cp_error`,
  `test/trace2cp_raw_y_error_mean_px`, valid segment count, and skipped segment
  count. Checkpoint metadata now includes `metric_name`, and `best.pt` uses
  `test/trace2cp_error` when Trace2CP evaluation is enabled.
- Added `fiber_trace_3d.train --trace2cp-vis` for single-sample and
  `--fiber-json` whole-fiber inspection. It exports `trace2cp_3d_vis.jpg` and
  prints `trace2cp_error=...` or `trace2cp_error_mean=...`.
- Added explicit Trace2CP metric keys to `fiber_trace_3d/configs/loader_example.json`.
  Added disabled-by-default Trace2CP keys to `train_s1a_nml_all.json` because
  that config currently has no `test_datasets`.
- Added focused tests for analytic 3x2 round-trip decoding and synthetic
  fixed-set 3D Trace2CP evaluator wiring.

Validation command:

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`

Result:

`13 passed in 2.58s`

Regression command:

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py vesuvius/tests/neural_tracing/test_fiber_trace.py`

Result:

`332 passed in 9.85s`

## Reported Limitations / Deferred Items

- The 3D Trace2CP visualization is intentionally compact and does not recreate
  every panel from the mature 2D Trace2CP runner. It uses the same 3D metric
  projection/scoring path, but exports source+direction/traces and presence
  only.
- Whole-fiber 3D Trace2CP visualization stacks per-pair panels vertically
  rather than composing them into the 2D runner's shared arc-length canvas.
- Dense 3D Trace2CP inference currently tiles over 2D strip tiles and runs one
  3D block per tile. It is correct and bounded but not yet optimized for
  merging neighboring tile blocks.
