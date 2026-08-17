# VC3D short line-segment handling

Use one 32-base-voxel sampling distance for generated line strips and control-
point click replacement.

Requirements:

- Keep every annotation control point and both line endpoints as exact strip
  supports. Internal optimized line points define the path but are not supports.
- Resample each control-point span by optimized-polyline arclength using the
  interval count whose physical spacing is closest to 32.
- Keep control-point spans at or below 32 base voxels as one strip interval.
- Declare the generated strip's along-line scale as exactly `1/32`, so short
  physical spans are expanded to one nominal display interval.
- Use exactly seven cross-strip samples at 32 base voxels for both the line
  surface and side slice, giving a fixed 192-base-voxel extent close to the
  previous typical width.
- Remove the unused configurable cross extents and cross-sample count.
- On generated-view clicks, compare existing controls by optimized-polyline
  arclength rather than line-index distance.
- Collapse every existing control within an inclusive 32-base-voxel arclength
  radius into one control at the clicked location.
- Preserve seed, interpolation-span, branch-link, and endpoint semantics while
  collapsing controls.
