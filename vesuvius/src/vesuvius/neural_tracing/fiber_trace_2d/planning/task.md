# Trace2CP Short Z-Search Plan

Plan a short out-of-plane z-search extension for Trace2CP combined tracing.

Requested behavior:

- At Trace2CP start, infer not only the regular center-strip embedding field
  but also fields offset by two selected-scale voxels in `-z` and `+z`.
- During tracing, search over the current z layer and neighboring `+-1` z
  steps. Choose the next step using the existing regular 2D angle evidence
  combined with the best embedding evidence, while ignoring out-of-plane angle
  effects for now.
- At every trace step, ensure the two neighboring z layers around the current
  layer exist as inferred image/direction/embedding fields; extend the inferred
  stack as needed.
- For the Trace2CP connection/closest-approach logic, use Manhattan-style
  distance `abs(y distance) + abs(z voxel distance)`, where z is measured in
  actual selected-scale voxels, not layer steps. Apply the same center-distance
  magnification currently used for y-only closest-point selection.
- Correct z linearly during fused connection construction in the same way y is
  corrected, producing fractional z positions in the fused/refined line.
- For visualization, show z-corrected images reconstructed column by column
  from the z-layer stack. Since the selected z layer can differ by trace
  direction, show separate forward and backward z-corrected views. This
  reconstruction must not re-sample the volume; for each image column
  independently, it rounds that column's z value to the nearest already
  inferred z layer and copies that layer's image content for the column.
