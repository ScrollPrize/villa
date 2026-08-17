# Plan: Fiberlet-centered replay strips and indexed overview JPEGs

## Geometry and rendering

1. Extract the existing batched normal sampling plus default
   `buildLineViewSurfaces()` construction into one file-local helper used by
   both failure-local strip generation and the full replay overview. Do not add
   an alternate strip geometry or renderer.
2. Keep the current reference-centered top and side strips and their yellow
   reference, red greedy, cyan fiberlet, and restart-marker overlays.
3. Build an additional top and side surface for every fiberlet replay segment
   with at least two route points. Preserve reset boundaries as disconnected
   components, render each component through the same native-group coordinate
   conversion and fine-to-coarse CT renderer at overview scale eight, and join
   rendered components only in the 2D inspection raster with explicit black
   separators. Persist a deterministic component placement table containing
   source segment index, matched reference-arc interval, and top/side raster
   column ranges, including separator semantics.
4. In the fiberlet-centered pair, draw the fiberlet centerline in cyan and the
   stored matched reference points in yellow. Map greedy trace points and both
   tracers' failure arcs through the fiberlet segment's stored monotonic match
   arcs so the comparison and restart diagnostics remain visible without new
   nearest-point matching. Greedy samples outside a component's covered arc are
   omitted; samples in overlapping component intervals are shown in each such
   component. Fiberlet failures are assigned to their recorded source segment.

## Indexed JPEG parts

1. Extend the current wrapped compositor so every horizontal interval is one
   vertically stacked four-strip block ordered reference top, reference side,
   fiberlet top, fiberlet side. Preserve the existing 32,000-column proportional
   wrap-range construction and validate that each complete block independently
   fits within 65,000 rows and columns.
2. Split every strip into proportional, exact half-open column ranges for each
   block. Copy every raster column exactly once without resizing. Stack as many
   complete four-strip blocks as fit below 65,000 rows in one image; continue
   with another indexed image only when the next complete block would exceed
   that limit.
3. Publish immutable `replay/full_strip.NNNNNN.jpg` artifacts and stable
   `fiber_replay.NNNNNN.jpg` aliases for every part. Record ordered part paths,
   hashes, image shapes, progress fractions, page index, and the four source-
   column/row ranges for every ordered block in the strict version-2 replay
   root. Remove the unpublished singular JPEG path and clean stale indexed
   aliases after shorter or non-visual runs.
4. Print every stable overview part path after publication.

## Tests and validation

1. Extend overview rendering coverage to prove both fiberlet-centered CT strips
   are rendered, use the fiberlet route as their centerline, retain overlays,
   and preserve disconnected replay components.
2. Replace the panel-wrapping compositor test with exact 65,000-boundary and
   multi-part tests that verify all four rasters are copied once without
   resampling, no four-strip block is split across pages, and every output
   dimension remains within the limit.
3. Update replay publication tests for indexed immutable/stable JPEGs,
   deterministic rewrites, manifest metadata, and stale-alias cleanup.
4. Build `test_fiber_replay` and `vc_fiberlets` with `-j32`, run the focused
   suite, run `git diff --check`, and, if available, run a bounded Paris4 visual
   replay to inspect the generated part count and dimensions.

## Spec update

- Replace each reference-only wrapped block with four reference-/fiberlet-
  centered strips, retain vertical wrapping within an image, and spill complete
  blocks across indexed <=65,000-pixel JPEG parts only when required. Specify
  default line-view geometry, segment separation, stored match-based overlays,
  indexed artifact naming, metadata, and stale cleanup.

## Documentation update

- Update `volume-cartographer/docs/fiberlets.md` with the four-strip layout,
  fiberlet-centered interpretation, indexed paths, size bound, manifest fields,
  and command-line output.

## Changelog update

- Record the fiberlet-centered overview pair and indexed JPEG publication.
