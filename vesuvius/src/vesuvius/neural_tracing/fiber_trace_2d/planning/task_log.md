# Trace2CP Vertical Space Doubling Task Log

## Notes

- Current Trace2CP segment loading uses `2 * patch_shape_hw[0]`.
- Requested behavior is another doubling, so Trace2CP segment strips should use
  `4 * patch_shape_hw[0]`.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - Result: 157 passed in 6.05s.
- `git diff --check` on touched files returned clean.
