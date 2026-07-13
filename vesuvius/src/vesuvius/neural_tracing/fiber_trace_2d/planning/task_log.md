# Trace2CP Sub-Voxel Z-Step Interpolation Task Log

## Implementation Notes

- Kept `z_step_voxels` as the Trace2CP z-search/DP state spacing.
- Changed `_Trace2CpZPlaneCache` so `z_step_voxels < 1.0` does not trigger
  model inference at sub-voxel offsets. Requested state layers now map to
  `z_voxels = layer * z_step_voxels`, then use the bracketing integer
  selected-scale side-z voxel layers.
- Added interpolation for sub-voxel state fields:
  - direction vectors are sign-aligned for the ambiguous representation,
    linearly interpolated, and normalized;
  - sheet/fiber presence is linearly interpolated;
  - embeddings are linearly interpolated and normalized when present.
- Kept debug slice image export on actually inferred integer side-z layers;
  sub-voxel state layers do not create extra sampled image pages.
- Updated specs, code-structure docs, changelog, and task status.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - Result: `251 passed in 10.11s`

## Deviations

- None.
