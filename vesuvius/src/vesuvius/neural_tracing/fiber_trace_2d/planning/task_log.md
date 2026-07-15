# 3D Fiber Follow-Up Task Log

## Planning Notes

- Current 3D affine augmentation is present:
  - `augment_shift_zyx` controls CP placement inside the output patch;
  - `augment_rotation_degrees` samples a 3D axis-angle rotation;
  - isotropic scale and axis flips are also wired.
- Current 3D target generation is matrix-based and therefore only general
  enough for affine transforms.
- Corrected smooth displacement plan after review: the 3D path must use the
  same paired-map contract as 2D. It should build both `backward_map_zyx`
  (output -> source) and `forward_map_zyx` (source -> output) as concrete maps
  at augmentation construction time. Labels should transform source line/CP
  coordinates through `forward_map_zyx`; they should not be derived only from
  the output-to-source map and a Jacobian.
- Full 3D smooth displacement should be implemented only through an explicitly
  invertible paired construction, such as smooth coupling/triangular stages,
  not arbitrary dense inverse estimation.
- Anisotropic blur should be a torch value augmentation after volume sampling,
  not a geometric/image-space resampling step.
- Full Trace2CP wiring still needs a bridge that samples 3D checkpoint outputs
  at 2D Trace2CP strip coordinates, projects 3D Lasagna directions into the
  strip frame, and reuses the existing public Trace2CP metric.
- Shear/skew and ringing stay out of scope and should continue to be rejected
  when configured non-zero.

## Validation

- Planning-only task so far; no tests run yet.
