# 3D CP Training Config And Segment Target Fix

Fix the 3D CP-centered training path after the first S1A NML training run.

Required behavior:

- The S1A NML 3D training config must use the intended larger 3D patch size:
  `patch_shape_zyx: [192, 192, 192]`.
- The configured CP shift must match that patch scale: `augment_shift_zyx:
  [48, 48, 48]`.
- The 3D U-Net depth must be fixed to a depth appropriate for 192-voxel
  patches rather than the shallow 4-stage setup.
- 3D label generation must not discard a fiber segment only because one of the
  original line vertices is outside the sampled patch/source-map domain. It
  must keep segment portions that overlap the crop by clipping to the
  forward-map/source domain before mapping to output coordinates.
- This is a training target-generation fix, not a data-skip workaround.
- 3D training TensorBoard visualization must show the three principal slices
  through a sampled CP, including image data, presence, and direction/angle
  information.
