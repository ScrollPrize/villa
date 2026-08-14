# Task: straightened fiberlet DP and scored graph joins

Replace the experimental world-axis half-voxel fiberlet DP with a directed
candidate-local search in the straightened anchor-to-anchor domain.

- Use deterministic planes normal to a curved cubic-Hermite centerline fitted
  from both anchor positions and directions. Space ordinary planes by 2 stored
  prediction voxels of centerline arclength and allow transverse movement on a
  0.5-prediction-voxel grid.
- Evaluate prediction presence, prediction direction, and Lasagna normals at
  the resulting floating-point XYZ positions through native-volume
  interpolation. Do not quantize the positions back to world voxel axes.
- Preserve exact floating-point anchor endpoints and deterministic acyclic DP.
- Keep base-volume coordinates at the CLI and artifact boundaries.
- Apply the same local fiberlet alignment and Lasagna-normal tangent/normal
  smoothness metric to graph joins. The existing strict 45-degree join bound
  remains a feasibility constraint, not the complete join objective.
- Benchmark the supplied Paris4 `fiberlet-replay --along 512` workload against
  the current world-axis half-grid result.
