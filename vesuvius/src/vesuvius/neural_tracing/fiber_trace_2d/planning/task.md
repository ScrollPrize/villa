# Task: H/V constraints from stored crop traces

Extract the first discrete-labeling inputs directly from a durable Fiberlet crop
trace artifact.

- Uniformly resample every stored trace at a common base-voxel spacing.
- Split each trace into evenly sized overlapping arc-length pieces, targeting
  `512` base voxels with `128` base-voxel overlap.
- Find the closest resampled points for every distinct piece pair within a
  configurable spatial threshold.
- Measure a normalized parallel-versus-perpendicular score. Parallel evidence
  comes from a synchronized tangent walk around the closest points, including
  bounded phase refinement; perpendicular evidence comes from the closest-point
  tangent pair.
- Measure the connector's Lasagna winding integral after modulating every
  integration sample by `abs(dot(connector_direction, normal))`.
- Do not spatially search pieces split from the same original trace. Connect
  consecutive pieces with an exact parallel-score-one, winding-zero constraint.
- For this first stage, keep constraints in memory and report extraction timing,
  rejection counts, and decile statistics rather than publishing a new artifact.

The implementation consumes the stored float64 crop polylines. It must not
reopen or reconstruct the source Fiberlet graph.
