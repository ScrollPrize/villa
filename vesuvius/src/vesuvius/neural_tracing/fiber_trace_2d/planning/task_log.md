# Task Log: Disable 3D Fiber Model Normalization

## Implementation Notes

- Added `normalization: none` support to `Vesuvius3dUnetModel` while preserving
  its legacy default of `InstanceNorm3d` for other callers.
- Updated the `fiber_trace_3d` model wrapper to request `normalization: none`
  unconditionally.
- Added a regression test asserting that `FiberTrace3DNet` contains no
  `BatchNorm3d`, `GroupNorm`, or `InstanceNorm3d` modules.
- Updated specs, code-structure docs, and changelog to describe the 3D fiber
  no-normalization rule.

## Validation

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`

Result: `17 passed in 2.62s`.

## Deviations / Simplifications / Deferred Items

- None.
