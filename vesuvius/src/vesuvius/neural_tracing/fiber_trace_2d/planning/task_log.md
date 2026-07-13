# Trace2CP Side-DP Z Smoothness Re-Enable Task Log

## Implementation Notes

- Restored `_TRACE2CP_SIDE_DP_DZ_SMOOTH_PENALTY` to `0.5`.
- Kept `_TRACE2CP_SIDE_DP_Z_TRANSITION_PENALTY` at `0.0`.
- Updated the regression test that captures the side-DP penalty values.
- Updated specs, code-structure docs, changelog, todo, and task status.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - Result: `251 passed in 10.96s`

## Deviations

- None.
