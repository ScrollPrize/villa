# Task Log: 3D CP Training Config And Segment Target Fix

## Implementation Notes

- Updated `fiber_trace_3d/configs/train_s1a_nml_all.json`:
  - `patch_shape_zyx` is now `[192, 192, 192]`;
  - `augment_shift_zyx` is now `[48.0, 48.0, 48.0]`;
  - `model_3d.unet_depth` is now `6`;
  - `training.sample_vis_interval` is explicitly set to `1000`.
- Added source-space AABB clipping for 3D fiber-line segments before mapping
  them into the output patch. This keeps supervision for long NML edges that
  overlap the crop even when an original endpoint lies outside the patch.
- Added `train_sample_3d/principal_slices` TensorBoard image logging:
  rows are `yx`, `zx`, and `zy` CP-centered slices; columns are image data,
  target presence, predicted presence, and direction angular error.
- Added focused tests for long-segment clipping and the training visualization
  sheet helper.

## Validation

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`

Result: `15 passed in 2.62s`.

## Deviations / Simplifications / Deferred Items

- None for this task.
