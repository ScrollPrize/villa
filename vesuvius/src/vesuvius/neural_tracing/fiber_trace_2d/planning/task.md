# Trace2CP Top-Model Direction Debug

- Add an opt-in Trace2CP runner argument that loads the jointly trained
  top-view model from the checkpoint.
- For now, use it as a diagnostic visualization rather than changing scoring:
  run the top model on the traced top-strip rendering.
- If z-search produced a z-corrected fused top strip, use that image;
  otherwise use the fused traced top strip at the central z layer.
- Append an additional top-strip panel/row with sparse top-model direction
  line indicators over the chosen top rendering.
