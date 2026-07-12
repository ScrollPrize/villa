# Trace2CP Top-Model Direction Debug Task Log

## Implementation Notes

- Added `--trace2cp-top-model-dir-vis`.
- Added top-view model loading from `top_model_state_dict`; the flag fails
  clearly if the checkpoint does not contain top-model weights.
- The diagnostic chooses the traced fused top strip with z-search correction
  when available, otherwise the traced fused top strip at central z.
- The selected top strip is passed through the top-view model, its ambiguous
  direction channels are decoded, and sparse direction line indicators are
  overlaid on an additional top-strip panel/row.
- This is visualization-only and does not change Trace2CP scoring, z-search
  candidate selection, or traced line generation.

## Deviations

- None.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py -k 'top_strip_panel_accepts_rgb_direction_overlay or top_model_loader'`
  passed: 2 tests.
- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed: 211 tests.
