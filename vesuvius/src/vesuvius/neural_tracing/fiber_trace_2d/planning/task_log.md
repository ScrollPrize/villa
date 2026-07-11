# Trace2CP Target-Column Metric Task Log

## Notes

- Current code computes public `trace2cp_error` from closest trace-to-trace
  overlap, while the user wants target-column y error restored.
- Refinement/fusion still needs closest-point behavior, so the change should
  stay inside the public metric helper and summary/docs wording.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - Result: 157 passed in 6.09s.
- `git diff --check` on touched code/docs returned clean.
