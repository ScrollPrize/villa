# Merge Main Line-Annotation GUI Changes

- Merge `main` into `fiber-3d-ext3` and retain the fiber annotation features.
- Adopt the new annotation toolbar/menu, tag controls, schematic full-width
  overview map, in-place generated-view refresh, and pane lifecycle fixes.
- Keep both existing rendered strip viewers:
  - the top-view `lineSurface` strip;
  - the `lineSideSlice` strip.
- Both rendered strips remain ordinary interactive viewers with scrolling,
  panning, zooming, control overlays, and per-span mode/metric/message labels.
- Do not restore the older fixed-height, fit-to-width, non-interactive rendered
  top strip. The schematic overview map is additional UI, not a replacement for
  either rendered strip.
