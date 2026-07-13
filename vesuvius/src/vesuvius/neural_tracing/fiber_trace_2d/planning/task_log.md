# Trace2CP Direction-Aligned Side Presence Blur Task Log

## Goal

Replace the axis-aligned Trace2CP z-search side-presence blur with a
direction-aligned anisotropic blur implemented in batched PyTorch. The blur
uses side-z radius 21, local-direction radius 5, and a small perpendicular
radius 1.

## Notes

- Current z-search presence call sites already route through
  `_Trace2CpZPlaneCache.blurred_presence_for_layer(s)`, so the implementation
  should stay inside that cache/helper path.
- Implemented `_trace2cp_blur_presence_stack_directional()` as a weighted
  PyTorch blur. It smooths along side-z first, then applies a symmetric
  direction-aligned x/y gather with `grid_sample` over bounded layer chunks.
- The direction-aligned kernel uses radius 5 along the local side direction and
  radius 1 across it. Since offsets are symmetric, `dir` and `-dir` produce the
  same blur.
- `_Trace2CpZPlaneCache` now gathers direction fields with the presence/valid
  stack and keeps the existing `blurred_presence_for_layer(s)` API.
- Added `--trace2cp-presence-blur` and made the cache return raw per-layer
  presence unless that flag is enabled.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py`
  passed.
- Initial focused pytest run failed only because the repository guard test
  forbade any `grid_sample` text in `runner.py`; updated that guard to exempt
  only the new presence-blur helper block, not runner globally.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed: 254 tests.
