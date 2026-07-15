# 3D CP Training Config And Segment Target Fix Plan

## Implementation

- Update `fiber_trace_3d/configs/train_s1a_nml_all.json`:
  - set `patch_shape_zyx` to `[192, 192, 192]`;
  - set `augment_shift_zyx` to `[48.0, 48.0, 48.0]`;
  - set `model_3d.unet_depth` to `6`, producing stages
    `[16, 32, 64, 128, 256, 512]` and a `6^3` deepest map for `192^3`
    inputs.
- Add a small source-space segment clipping helper in
  `fiber_trace_3d/loader.py`.
- In `_build_targets`, clip each fiber-line segment to the source domain covered
  by `geometry.forward_map_zyx` before sampling the forward map.
- Use the clipped/mapped output segments for distance and direction target
  generation.
- Keep this as a target-generation fix. Do not skip samples or hide failures.
- Add a lightweight 3D training visualization helper in
  `fiber_trace_3d/train.py`:
  - render the CP-centered principal planes (ZY, ZX, YX);
  - show image data, target presence, predicted presence, and prediction-vs-
    target direction angular error;
  - log to TensorBoard at a configurable interval.

## Spec Update

- Add a 3D spec note that long fiber-line segments must be clipped to the
  sampled/forward-map source domain for target generation; endpoint-inside-only
  logic is invalid.
- Add a 3D config note for the S1A NML training config patch/shift/depth values.
- Add 3D training visualization semantics and interval key.

## Docs Updates

- Update `docs/code_structure.md` in the 3D loader/model section to mention
  clipped segment target generation and 3D training sample visualization.

## Testing

- Add or update a focused 3D loader test where the CP is inside the crop but
  both adjacent line vertices are outside the patch/domain. The target builder
  must still produce direction and presence supervision.
- Add a focused helper-level test for 3D training visualization output shape if
  practical without requiring TensorBoard.
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`

## Changelog

- Add a 2026-07-15 entry for the 3D S1A NML config and segment clipping fix.
