# Python Native 3D Trace2CP Whole-Fiber Start CP Task Log

## Implementation Notes

- Added `--whole-fiber-start-cp-index` for Python native 3D Trace2CP.
- The existing `--start-cp-index` / `--target-cp-index` arguments remain
  explicit single-segment selectors.
- Whole-fiber tracing now initializes from the selected CP and traces through
  the final CP.
- Total and partial restart-rate denominators are measured along the original
  loaded fiber line from the selected CP, so suffix metrics are local to the
  tested suffix.
- Invalid start CP values that do not leave at least one target segment fail
  loudly.

## Deviations / Deferred Items

- No tracing, candidate scoring, inference, visualization layout, or metric
  formula changes beyond the selected suffix denominator.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py`
  - passed
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  - passed: 170 tests, 2 skipped
- `git diff --check`
  - passed
