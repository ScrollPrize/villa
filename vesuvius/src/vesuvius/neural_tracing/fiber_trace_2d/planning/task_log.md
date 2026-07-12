# Trace2CP Last-Point Similarity Columns Task Log

## Notes

- Added `_trace_progress_similarity_map` for forward/reverse Trace2CP
  embedding-debug panels.
- The helper samples the previous accepted trace point's embedding and paints
  only the vertical column band around the newly placed point, using
  `ceil(step_px / 2)` as the radius.
- Unvisited columns remain `NaN`, which the existing fixed-scale visualization
  renders as black.
- Start CP, target CP, and same-fiber/global CP-bank similarity maps remain
  unchanged full-image fixed-scale cosine displays.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed: 160 tests in 7.14s.
- `git diff --check` on touched files passed.
