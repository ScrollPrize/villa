# 3D Prefetch Coordinate Shape And 64-Patch Augmentation Config Plan

## Scope

Fix the 3D prefetch crash caused by passing `[Z,Y,X,3]` coordinates to the
VC3D dependency API, and make the 64-patch S1A NML config useful for testing
implemented augmentations.

## Implementation

1. Update `fiber_trace_2d.sampling.Vc3dCoordinateSampler.chunk_requests_for_coords`.
   - Accept any coordinate tensor shaped `[...,3]`.
   - If the coordinate grid is 2D, call VC3D once as before.
   - If it is higher-rank, flatten it exactly like training sampling:
     `[Z,Y,X,3] -> [Z*Y,X,3]` or generally `[...,H,W,3] -> [prod(...),W,3]`,
     then call `collect_coords_dependencies` once.
   - De-duplicate returned chunk requests by `(store_identity, key)`.
   - Preserve exact VC3D metadata returned by the binding; do not reconstruct
     cache paths in Python.

2. Leave 3D training sampling unchanged.
   - The normal 3D sample path still materializes the 3D volume via the existing
     loader/sampler path.
   - This task only changes dependency discovery for prefetch.

3. Update `train_s1a_nml_all_64_sd2.json`.
   - Keep `patch_shape_zyx: [64,64,64]`, `base_volume_scale: 2`, and shift 16.
   - Keep affine and value augmentations on.
   - Enable implemented smooth displacement, isotropic blur, and anisotropic
     blur with moderate values appropriate for fast 64-voxel tests.
   - Leave shear and ringing absent/unsupported.

4. Tests.
   - Add a test using a fake VC3D volume whose dependency API rejects non-2D
     coordinate shapes.
   - Verify 3D coordinate grids are flattened into one VC3D-compatible 2D
     dependency call and de-duplicated.
   - Run focused 3D tests.

## Spec Update

- Document that 3D prefetch flattens regular 3D coordinate volumes into one
  VC3D-compatible 2D dependency surface using the same convention as training
  sampling.
- Document that the 64/sd2 config is an experimental fast 64-voxel config with
  implemented smooth/blur augmentations enabled.

## Docs Updates

- Update `docs/code_structure.md` prefetch notes if necessary to mention the
  3D dependency flattening.
- Update `planning/local_development.md` with the 64/sd2 prefetch/training
  intent if the command documentation is touched.

## Changelog

- Add a short 2026-07-15 entry for the 3D prefetch rank fix and 64/sd2
  augmentation config correction.

## Non-Goals

- Do not add shear/skew or ringing support.
- Do not change 3D training sample order.
- Do not change VC3D cache metadata handling.
- Do not change 3D target materialization or model architecture.
