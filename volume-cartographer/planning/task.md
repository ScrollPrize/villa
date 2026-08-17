# VC3D short line-segment handling

Use one 32-base-voxel sampling distance for generated line strips and control-
point click replacement.

Requirements:

- Keep every original line/control-point endpoint as an exact strip support.
- Subdivide spans longer than 32 base voxels using the existing nearest-spacing
  interval selection.
- Keep spans at or below 32 base voxels as one strip interval.
- Declare the generated strip's along-line scale as exactly `1/32`, so short
  physical spans are expanded to one nominal display interval.
- On generated-view clicks, compare existing controls by optimized-polyline
  arclength rather than line-index distance.
- Collapse every existing control within an inclusive 32-base-voxel arclength
  radius into one control at the clicked location.
- Preserve seed, interpolation-span, branch-link, and endpoint semantics while
  collapsing controls.
