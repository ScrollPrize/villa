# Native 3D Trace Fusion Pairwise Meeting

Change native 3D Trace2CP forward/reverse fusion so it no longer depends on
straight CP-axis progress as the primary meeting-coordinate.

Required behavior:

- Build the fused line from the best pair of points on the forward and reverse
  traces.
- Score each candidate point pair by:
  - Euclidean distance between the two points, times a factor of `1.0` for now;
  - plus the traced arc length from the forward trace start to the forward
    candidate;
  - plus the traced arc length from the reverse trace start to the reverse
    candidate.
- Pick the candidate point pair with the lowest score.
- Use the midpoint of that pair as the fusion meeting point.
- Lerp/warp the forward partial trace from start to the forward meeting point
  and the reverse partial trace from target to the reverse meeting point toward
  the shared midpoint, then concatenate and arc-length resample.
- Preserve existing single-pair and whole-fiber native 3D Trace2CP interfaces.
- Add tests that catch cases where straight-axis progress fusion picks the
  wrong meeting point.
