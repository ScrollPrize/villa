# Task Log: Report Direction Error In Degrees

## Implementation Notes

- Added `direction_angle_error_degrees(...)` in `direction.py`.
- The metric gathers the same CP-local supervised pixels as the MSE loss,
  decodes predicted two-channel Lasagna ambiguous direction vectors, folds the
  sign with `abs(dot)`, and reports errors in `0..90` degrees.
- Training loss remains encoded-channel MSE.
- Train/test TensorBoard scalar logs now include `angle_error_mean_deg`.
- Console progress and final training output now include mean angle error in
  degrees.

## Deviations

- None.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/direction.py` passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py` passed with 57 tests.
