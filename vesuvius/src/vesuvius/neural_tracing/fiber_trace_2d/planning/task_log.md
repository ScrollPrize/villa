# Trace2CP Top-Model Offset Direction Debug Task Log

## Implementation Notes

- Changed `--trace2cp-top-model-dir-vis` from single-layer inference to
  fixed top-strip offset-stack inference.
- The stack uses offsets `-4..+4` selected-scale voxels in one-voxel steps
  around the z-corrected fused trace when available, otherwise around the
  central-z fused trace.
- The top model is run on every layer. For each pixel, the displayed direction
  is selected from the valid layer whose decoded direction has maximal
  horizontal alignment `abs(dx)`.
- The overlay is still diagnostic-only: Trace2CP scoring, tracing, and
  z-search layer selection are unchanged.

## Deviations

- None.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py -k 'top_model_direction_selection or top_strip_panel_accepts_rgb_direction_overlay or top_model_loader'`
  passed: 3 tests.
- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed: 212 tests.
