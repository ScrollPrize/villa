# Trace2CP Side-Z Presence Blur Task Log

## Notes

- Task started from user request to blur side-view presence with Gaussian
  radii z=11 px and x=5 px before use/display.
- Scope is Trace2CP z-search/side-z presence only. Non-z presence has no z axis
  and remains unchanged.
- Implemented weighted Gaussian smoothing over presence stacks shaped
  `[z_layer, y, x]`. Invalid pixels are excluded from the weighted average.
- `_Trace2CpZPlaneCache.blurred_presence_for_layer()` lazily caches blurred
  per-layer presence maps; DP can request a whole layer list at once.
- Stepwise z-search, side/top-z experiment scoring, side-z DP presence stacks,
  z-corrected presence panels, z-pillar panels, and z-layer TIFF presence pages
  now use the blurred side-z presence view.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed: 253 tests.
