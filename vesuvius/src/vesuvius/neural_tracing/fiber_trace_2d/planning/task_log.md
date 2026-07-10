# Task Log: Augment-Vis Timing Cleanup

## Implementation Notes

- Added an unprofiled `descriptor_for_sample_index()` call before augment-vis
  source timing so first-use deterministic sample-order cache construction is
  not charged to the `descriptor` or first-row `total` timing.
- Left loader-side profiling unchanged for training/profile paths.

## Deviations

- None.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed: 69 tests.
