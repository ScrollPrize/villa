# Trace2CP Traced Top Strip Visualization

Correct Trace2CP top-strip visualization so the comparison output includes both
the original/init strip and strips reconstructed from the traced fused line.

- Single-pair Trace2CP should show the VC3D-style top strip sampled from the
  same segment source as an original/init comparison.
- Single-pair and whole-fiber Trace2CP should also show a VC3D-style top strip
  constructed from the fused traced line projected onto the central z slice.
- When z-search is active, single-pair and whole-fiber Trace2CP should also
  show the fused z-corrected top strip constructed from the fused traced line.
- Do not change Trace2CP scoring, tracing, metrics, or training behavior.
