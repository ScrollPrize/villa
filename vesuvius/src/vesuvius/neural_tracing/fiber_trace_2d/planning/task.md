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
- The presence top rows must not repeat one sampled CP/trace value over the
  top-strip row direction. They should sample the available inferred side
  presence slices at the corresponding side-strip coordinates.
