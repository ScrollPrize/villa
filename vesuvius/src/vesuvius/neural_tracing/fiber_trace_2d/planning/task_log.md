# Correct Fiber 3D Tiled Inference Output Task Log

## Planning Notes

- Started a corrective task because the previous fiber inference V0 output
  was incomplete for the intended Lasagna-compatible inference product.
- Verified current Lasagna `predict3d` behavior in
  `lasagna/preprocess_cos_omezarr.py`:
  - writes `grad_mag`, `nx`, and `ny` OME-Zarr groups;
  - records those groups in `.lasagna.json`;
  - encodes compact hemisphere vectors by flipping to `z >= 0`, then writing
    `round(component * 127 + 128)` for `nx/ny`;
  - builds scalar pyramids for scalar channels and
    `build_normal_omezarr_pyramid(...)` for `nx/ny`.
- Verified current fiber adapter persists raw internal seven-channel option
  bundles, which must be changed to persisted `presence/nx/ny`.
- Replaced the active task and task plan with a plan to make fiber inference
  produce Lasagna-style `.lasagna.json` plus OME-Zarr scale-space pyramids.
- Tightened the plan after review: fiber inference must reuse the existing
  shared tiled runner resume path unchanged. The only fiber-specific resume
  input is the adapter product-completeness hook for `presence/nx/ny`.

## Deviations / Deferred Items

- No implementation has been done yet for this corrective task.
- The plan preserves multi-option model outputs as separate
  `presence/nx/ny` products; it does not collapse branches or recurrent
  options.

## Validation

- Planning-only update; no tests run yet.
