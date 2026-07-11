# Trace2CP Center-Biased Metric Task Log

## Plan Review

- `task.md`: the plan implements the requested center preference and `2x`
  considered distance at either CP.
- `specs.md`: the change is limited to Trace2CP metric selection/scoring and
  keeps the existing shared tracing/TTA code path.
- `plan.md`: pre-existing user edits remain untouched.

## Implementation Notes

- Added `_trace2cp_center_penalty`, using a linear multiplier from `1.0` at
  the midpoint between CP x coordinates to `2.0` at either CP x coordinate.
- Trace2CP closest-point selection now minimizes
  `actual_y_gap * center_penalty`.
- `trace2cp_score` now uses the considered center-penalized distance divided by
  the usable vertical strip span. The actual y separation remains available as
  `actual_y_error_px` / `raw_y_error_px` diagnostics.
- Trace2CP visualization/summary output now reports actual gap, considered
  gap, and center penalty.
- Added regression tests for the center penalty and for a case where the center
  candidate wins despite a larger actual y-gap.
- Added a reference-only comparison column to `trace2cp_vis.jpg` when
  `--med-tta` is active. The selected median-TTA result remains first; the
  second column is the base/reference inference without TTA. Non-TTA Trace2CP
  output stays single-column.
- Added a renderer regression test for the optional reference column.
- `planning/plan.md` and `planning/todo.md` were already modified in the
  worktree and were not edited for this task.

## Validation

- Passed:
  `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
- Passed:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  (`131 passed in 6.06s`)
- Passed:
  `git diff --check -- vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
