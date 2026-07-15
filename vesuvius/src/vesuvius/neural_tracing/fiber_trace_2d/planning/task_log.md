# 3D Prefetch Coordinate Shape And 64-Patch Augmentation Config Task Log

## Notes

- Started from user-reported crash:
  `ValueError: coords_xyz must have shape [H, W, 3]` during
  `fiber_trace_3d --prefetch` with `train_s1a_nml_all_64_sd2.json`.
- Root cause: the 3D loader sends a regular `[Z,Y,X,3]` coordinate volume to
  `Vc3dCoordinateSampler.chunk_requests_for_coords`, while VC3D dependency
  collection currently accepts only 2D coordinate surfaces.
- Corrected implementation: prefetch dependency collection now follows the
  regular training sampling adapter and flattens `[Z,Y,X,3]` to `[Z*Y,X,3]`
  for one VC3D dependency call, rather than issuing one dependency call per
  Z surface.
- The 64/sd2 config already had affine/value augmentations enabled, but smooth
  displacement, isotropic blur, and anisotropic blur were set to no-op values.

## Deviations Or Deferrals

- Shear/skew and ringing remain unsupported because the 3D loader explicitly
  rejects those keys and the current spec keeps them out of scope.

## Validation

- Focused tests:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  passed with `19 passed`.
- `git diff --check` passed.
- Full remote prefetch smoke was not completed in this task; the regression
  test covers the reported VC3D shape failure with a fake VC3D dependency
  binding that rejects non-2D coordinate inputs.
