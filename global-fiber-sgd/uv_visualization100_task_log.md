# Saved 100-sweep fiber-map export

- Plan uv_visualization100_plan.md independently reviewed. CPU-only saved-state
  rendering: no optimization, inference or GPU allocation.
- Both conventions, 2000px full-map TIFF height, equal U/V scales, LZW tiles.
  No external decorations on raster; histogram TIFF/SVG separate. Fitted winding
  boundary lines remain inside map. SVG/PDF retain annotations and histograms.
- Installed matplotlib creates actual vector PDF paths/text; portable page size
  uniformly capped at180in, no tooltips or rasterized map. Same selected pieces.
- Added fresh layout_only route with piece identity checks and source hashes.
- Seven tests passed, independent implementation review completed. Fixed review's
  PDF font-size mismatch and restarted export; final run completed both maps.
  Checked TIFF detail/histogram visually, PDF has vector paths and no image objects.
  CCW395549 / CW395118 pieces; raster108962x2000 /105507x2000.
  Artifacts and approved reproduction command: uv_visualization100_results.md.
  No task requirements deferred; prior20-sweep maps untouched.
