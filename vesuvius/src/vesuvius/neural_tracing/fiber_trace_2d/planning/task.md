# Trace2CP Top-Slice Presence Visualization

- Add additional Trace2CP top-slice visualization rows for the side-strip
  sheet/fiber-presence output.
- These rows must use the existing side model presence head, not the optional
  top-view model.
- Show the projected presence below the regular top-strip slices for:
  - original/init top strip;
  - traced fused central-z top strip;
  - traced fused z-corrected top strip when z-search is active.
- Keep this visualization-only. It must not affect Trace2CP scoring, z-search,
  training, or top-model inference.
- Replace the previous top-presence projection with a z-pillar debug view:
  each image column is the presence sampled from the inferred side-slice stack
  across z layers. With `--trace2cp-z-max-layer 40`, the image is 81 px high.
- For the z-search fused trace z-pillar, shift each column by that column's
  selected z value so the center row is relative z=0 around the used layer.
