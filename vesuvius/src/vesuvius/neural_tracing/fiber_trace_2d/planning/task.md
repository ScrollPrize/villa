# Trace2CP Top-Model Direction Debug

- Add an opt-in Trace2CP runner argument that loads the jointly trained
  top-view model from the checkpoint.
- For now, use it as a diagnostic visualization rather than changing scoring.
- If z-search produced a z-corrected fused top strip, use that trace as the
  center layer; otherwise use the fused traced top strip at the central z layer.
- Also sample top-strip layers at offsets `-4..+4` selected-scale voxels in
  one-voxel steps around that center layer.
- Run the top model on every offset layer and, per pixel, display the direction
  from the layer whose decoded direction is most aligned with image-horizontal
  (`abs(dx)` is maximal).
- Append an additional top-strip panel/row with sparse top-model direction
  line indicators over the chosen center rendering.
