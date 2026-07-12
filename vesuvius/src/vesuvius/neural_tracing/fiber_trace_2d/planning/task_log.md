# Trace2CP Top-Model Monotone Path Task Log

## Implementation Notes

- Updated the NumPy monotone-x dynamic-programming helper for the top-model
  direction field to use fixed 8 px horizontal transitions.
- The DP path connects the two CP x-columns on the top-strip center row and
  integrates `1 - abs(dot(path_tangent, fused_direction))` across every pixel
  column crossed by each transition.
- Invalid fused-direction pixels get a fixed penalty rather than acting as hard
  barriers, so the diagnostic path still connects the CPs across missing field
  gaps while preferring valid pixels where possible.
- The top-model direction debug panel now draws the DP path in addition to the
  existing local forward/reverse traces.
- The single-pair summary now reports top local-trace stop reasons, point
  counts, DP path length, and DP invalid-pixel count.

## Deviations

- None.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py -k 'top_monotone_direction_path or top_direction_traces or top_model_direction_selection'`
  passed after fixed-step integrated-cost update: 7 tests.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed after the fixed-step integrated-cost update: 219 tests.
