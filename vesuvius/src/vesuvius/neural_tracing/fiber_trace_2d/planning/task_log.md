# Trace2CP Z-Search Layer TIFF Export Task Log

## Implementation Notes

- Added `--trace2cp-z-layers-tif` to the runner Trace2CP path.
- The flag is rejected unless `--trace2cp-z-search` is active.
- Added `_trace2cp_z_layer_tiff_stack`, which converts the existing inferred
  `_Trace2CpZPlaneCache` layers to uint8 pages without re-sampling:
  sorted slice pages first, then sorted presence pages.
- Single-pair mode writes `trace2cp_z_layers.tif`.
- Whole-fiber mode writes one pair-local TIFF per valid pair under
  `trace2cp_z_layers/`.
- Summary/stdout now report z-layer TIFF paths/page counts when exported.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed: 204 tests.
- `git diff --check -- vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/code_structure.md`
  passed.
