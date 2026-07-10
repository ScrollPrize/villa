# Task Log: Fused Forward/Backward Augmentation Coordinate Maps

## Implementation Notes

- Added `StripAugmentTransform` in `augmentation.py` as the shared paired
  geometric transform. Existing sampling-grid callers now route through its
  output-to-source map.
- Replaced smooth line/control-point nearest-grid inversion with vectorized
  source-to-output point mapping. For smooth offsets combined with affine
  transforms, the point inverse uses a small deterministic fixed-point solve
  over the requested points rather than an HxW distance search.
- Updated `FiberStrip2DLoader` line/CP mapping so augmented patches transform
  cached source-space line and CP coordinates instead of regenerating a
  shape-only centerline.
- Added regression tests for affine and smooth round trips, and a guard that
  smooth line/CP mapping does not call the dense nearest-grid search helper.
- Updated specs, code-structure docs, changelog, and status for the current
  task.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/augmentation.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
- Result: `78 passed in 3.80s`.
