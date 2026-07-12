# Trace2CP Top-Model Z-Aware Monotone Path Task Log

## Implementation Notes

- Added `_trace2cp_top_monotone_direction_path_z`, whose DP state is
  `(top_offset_layer, y)`.
- The yellow top-model diagnostic path now uses raw per-layer top-model
  direction fields instead of the fused median direction field.
- The path can move between neighboring top-offset layers per horizontal step
  with a small z-transition penalty. Direction alignment cost is evaluated from
  the direction field of the path's selected/interpolated layer.
- The existing single-layer helper remains as a wrapper around the z-aware
  helper for tests and simple callers.
- The top-model debug label now reports `dp_layer_range` and
  `dp_layer_changes`.

## Deviations

- None.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py -k 'top_monotone_direction_path or top_direction_traces or top_model_direction_selection'`
  passed: 9 tests.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed: 221 tests.
