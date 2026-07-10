# Task Plan: Report Direction Error In Degrees

## Scope

Add degree-based angular metrics for the existing V0 direction prediction path.

## Plan

- Add a helper in `direction.py` that computes folded unoriented angular error
  in degrees from model outputs and `DirectionSupervision`.
  - Gather predictions at the same CP-local supervised pixels used by the loss.
  - Decode the two-channel Lasagna ambiguous encoding to unoriented direction
    vectors using the existing decoder.
  - Compare to `supervision.tangent_xy` with `abs(dot)` and report angles in
    `0..90` degrees.
- Extend `_compute_batch_loss(...)` to return angle metrics while keeping MSE
  as the loss.
- Log mean angle error in degrees to TensorBoard for train and test.
- Print train/test angle error in degrees in console progress and final output.
- Keep checkpoint metric selection unchanged unless explicitly requested.

## Spec Update

Update `planning/specs.md` to state that direction MSE remains the training
loss and angle error in degrees is a reported metric.

## Docs Updates

Update `docs/code_structure.md` to document the degree metric.

## Testing

- Add focused direction tests that perfect predictions produce near-zero degree
  error and perpendicular predictions produce near-90 degree error.
- Run:
  - `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/direction.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`

## Changelog

Add a changelog entry because this changes public training/TensorBoard output.
