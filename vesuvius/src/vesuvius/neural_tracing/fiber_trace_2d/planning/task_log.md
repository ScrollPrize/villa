# Native 3D Trace Smoothness Weight Log

## Implementation Notes

- Raised `NativeTrace2CpConfig.smoothness_weight` default from `0.0` to `2.0`.
- Raised the CLI `--smoothness-weight` default from `0.0` to `2.0`.
- Kept the smoothing formula unchanged:
  `smoothness_weight * max(0, angle(previous_step_dir, step_dir) - free_angle)^2`
  in radians.
- Kept `--smoothness-free-angle-degrees` default at `10.0`.

## Deviations / Deferred Items

- No requested implementation item was deferred.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  passed: 79 tests.
