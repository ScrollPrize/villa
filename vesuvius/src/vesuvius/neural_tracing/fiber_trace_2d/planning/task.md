# Task: fiberlet graph replay

Extend the experimental C++ fiberlet pipeline so it can trace through a graph
of short fiberlets and compare that route with a dense reference fiber in the
existing failure-replay workflow.

- Generate candidate anchor pairs at every shorter cell distance through the
  configured outer radius, rather than only on the outer shell.
- During integer DP, accept a step only when its unoriented angle to the sampled
  dense fiber-prediction direction is less than 25 degrees. Invalid fiber
  predictions cannot admit a nonzero step under this hard constraint. Lasagna
  normals continue to affect only the existing curvature split.
- Build a graph from every accepted fiberlet. A traversal may join two
  fiberlets at an anchor only when the directed entering/leaving angle is
  strictly below 45 degrees.
- Add a separate replay command that uses deterministic beam search with
  anchor-level lookahead, records when its exact forward reference distance
  first exceeds the replay threshold, continues through anchor-bounded
  postroll, and publishes the resulting fiberlet route for napari visualization
  with the reference and existing greedy trace.
- Keep all CLI spatial values and artifacts in base-volume coordinates. The
  fiberlet formats are experimental; update them directly without repair or
  compatibility paths.
- Default replay failure-tube extraction to 128 base voxels along the reference
  on each side and a 64-base-voxel radius.
- Use that single `--along` extent for all three comparison trajectories:
  reference geometry, greedy trace before/after failure, and the fiberlet graph
  search interval. Derive greedy postroll from it and remove the independent
  postroll-length setting.
- Fiberlet graph replay must not terminate at its first reference-distance
  failure. Record that first failure, then continue routing for the same
  effective `--along` distance, matching greedy replay's post-failure behavior.
  Report complete postroll separately from truncation caused by graph
  exhaustion. Preserve graph semantics by stopping only at an anchor: finish
  the edge containing the failure and stop after the first later complete edge
  whose anchor reaches or exceeds the postroll distance.
