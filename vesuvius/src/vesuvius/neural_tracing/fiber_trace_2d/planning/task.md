# Trace2CP Top-Model Direction Debug

- Keep `--trace2cp-top-model-dir-vis` diagnostic-only.
- Keep the top-strip layer stack at offsets `-4..+4` selected-scale voxels in
  one-voxel steps around the z-corrected fused trace when available, otherwise
  around the central-z fused trace.
- Change the per-pixel top-direction fusion from selecting the single layer
  closest to horizontal to taking the median of all valid offset-layer
  directions within 45 degrees of horizontal.
- Treat the top direction as Lasagna-ambiguous/unoriented during fusion:
  align each contributing direction sign before taking the median so opposite
  signs cannot cancel.
- Make both visualization-only top traces equally visible in the debug panel.
