# 3D Prefetch Coordinate Shape And 64-Patch Augmentation Config

The 3D prefetch path crashes for
`fiber_trace_3d/configs/train_s1a_nml_all_64_sd2.json` because it passes a
full `[Z,Y,X,3]` coordinate volume into the VC3D dependency API, whose current
Python binding expects a 2D coordinate surface shaped `[H,W,3]`.

Required behavior:

- Fix 3D prefetch so it can generate chunk requests for 3D CP-centered patches
  without changing the training coordinate sampling path.
- Keep using VC3D dependency metadata and Python prefetch downloads.
- Preserve deterministic sample order and augmentation parameter generation.
- Update tests to cover 3D chunk dependency request generation.
- Update the 64/sd2 config so the implemented important augmentations are
  enabled for fast 64-voxel patch experiments:
  - keep affine/value augmentations enabled;
  - enable implemented smooth displacement;
  - enable implemented isotropic blur;
  - enable implemented anisotropic blur;
  - do not add shear or ringing because the 3D loader intentionally rejects
    those unsupported keys.
- Document the prefetch shape behavior and the 64/sd2 augmentation intent.
