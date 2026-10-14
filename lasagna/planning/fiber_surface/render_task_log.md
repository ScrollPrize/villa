# Matched surface rendering — 2026-10-14

- Follow-up: regenerate fibers/composite with8px lines (4x previous2px), reuse
  existing CT/mask and validated attachment hashes. Shared overlay writer avoids
  duplicating drawing/compositing. Config render_deformed_thick_fibers.json;
  overwrite only fiber outputs, preserve original CT and render provenance.
- Refresh completed in3.57s for790pieces;3tests passed including exact2→8pixel
  thickness. Output metadata overlay_refresh.json records the new width.

- Follow-up preference: future CT-only output is ct.jpg (quality95), not TIFF.
  Existing outputs remain untouched; overlay/composite/validity stay TIFF.

- Requested CT render and same-shape fiber overlay of one deformed surface.
- Select untangle_central_deformed_v1/continued.tifxyz and saved final attachments.
- Reuse shared bilinear interpolation and native vc chunk-aware CT sampler;
  Python only orchestrates bands and draws overlays. No GPU reservation needed.
- Previous repair log archived as repair_task_log.md; repair work paused.
- Independent review: use continued attachments, preserve piece gaps, record
  missing/error coverage and clarify UV parameter scale vs physical resolution.
- Tests: affine surface and exact overlay mapping; same-source piece gap remains
  unconnected. New test plus adapter suite:6passed. Real crop visually inspected.
- v1 TIFF generation succeeded but summary serialization failed on NumPy int64;
  fixed Python-int dimensions and reran v2. Installed older native open_url
  defaulted cache to cwd; now explicitly passes shared cache root when supported.
  v1's self-created cache moved into its output directory (ct-cache), not deleted.
- Mapping uses vertex-aligned pixels rather than shared renderer canvas-centre
  convention; shared bilinear field interpolation is reused directly with exact
  inverse overlay mapping, tested. This avoids edge extrapolation.
- Final v2 completed in56.61seconds,14701x6051pixels,790H pieces/788sources,
  100%native sampling coverage. CT scan Paris4 20260411134726 atlevel2,
  64native sampler threads,16GiB decoded cache,sharedcache explicitly selected.
  Outputs lasagna/data/render_deformed_central_v2/{ct,fibers,ct_fibers,valid}.tif.
  Exact command documented in lasagna/docs/fiber_surface.md with frozen config
  lasagna/configs/render_deformed_surface.json. No GPU reserved; surface untouched.
- Visual inspection confirms CT tissue detail with distinct colored H polylines.
  Limitations: existing surface folds remain; lines interpolate fitted attachment
  samples, not every AFV vertex; this is a single zero-offset CT slice, not a slab.
- Final independent implementation review found no blocker. Added anisotropic
  grid/invalid-mask test:7tests passed. No singleton pieces in this output.
  Missing/error chunks both0; band latency p50=.491s,p95=.708s. Diff check clean.
