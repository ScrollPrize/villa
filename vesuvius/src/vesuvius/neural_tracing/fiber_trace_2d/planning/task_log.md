# Trace2CP Gaussian Refine-Smoothing Task Log

## Implementation Notes

- Replaced the Trace2CP refinement smoother's uniform moving-average kernel
  with a normalized finite Gaussian kernel.
- Kept existing refinement invariants: x columns are restored exactly, and the
  first and last CP endpoints are restored exactly after smoothing.
- Kept `--trace2cp-refine-smooth-window` as the kernel window control for
  compatibility; even values are still rounded up to the next odd window.
- Updated the focused smoothing regression to verify Gaussian values, endpoint
  preservation, and x-column preservation.

## Deviations

- None.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py -k smoothing`
  passed: 1 test.
- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed: 209 tests.
