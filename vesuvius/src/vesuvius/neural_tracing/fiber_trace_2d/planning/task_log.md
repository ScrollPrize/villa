# Full Test Trace2CP Evaluation Sentinel Task Log

## Notes

- Starting from user request to make `training.test_control_points: 0` evaluate
  all configured test control point samples.
- `training.test_control_points` now parses as a non-negative integer. Positive
  values preserve the existing deterministic random test window behavior.
- The `0` sentinel resolves to `test_loader.sample_count`, starts at sample
  index `0`, and uses flat CP order so the evaluated segment set matches
  whole-fiber Trace2CP visualization more directly.
- Test loss and Trace2CP metric both use the resolved effective test selection.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py -k 'test_control_points or test_interval'`
  - Passed: 5 passed, 140 deselected in 4.17s.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - Passed: 145 passed in 6.46s.
- `git diff --check -- <touched files>`
  - Passed.
