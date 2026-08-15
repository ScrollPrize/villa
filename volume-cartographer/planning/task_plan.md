# Task plan

## Problem

Generated ribbons currently sample a globally uniform arclength grid and then
connect adjacent samples with quads. A control-point bend between two samples
is replaced by a straight chord, so the rendered centerline and its inverse
point-position mapping depart from the stored polyline.

## Implementation

1. Compute original control-point arclengths as before.
2. For every nonzero control-point segment, choose the positive interval count
   whose segment-local spacing is closest to the target spacing.
3. Append segment-local supports, including the segment endpoint exactly and
   avoiding duplicated boundary columns. This makes every non-duplicate
   control point a support.
4. Store explicit support arclengths in `LineStripPositionMap`.
5. Convert original positions to strip columns and back by piecewise-linear
   interpolation through explicit support and original arclength arrays.
6. Keep the scalar `QuadSurface` along scale as the mean support density; it is
   nominal view metadata, while exact geometry and interaction use the support
   map.
7. Preserve duplicate-point canonicalization without emitting zero-width
   geometry.

## Tests

- Verify a short right-angle line retains its bend even when the target spacing
  exceeds the full line length.
- Verify uneven straight segments are subdivided independently at the closest
  available spacing and retain their shared control point.
- Verify original-position/strip-column round trips with nonuniform supports.
- Retain reversed-line and duplicate-point coverage.
- Build and run `test_lasagna_line_view_surfaces` and run `git diff --check`.

## Spec update

Replace the globally uniform ribbon-grid requirement with control-point-
preserving, segment-local target-spacing subdivision and explicit support
arclength mapping.

## Documentation updates

Update `docs/line_annotation_fibers.md` to distinguish exact support geometry
from nominal mean strip density.

## Changelog

Record the bend-preserving ribbon correction under 2026-08-15.

## Independent review

- Original segment endpoints are the only geometric knots of the stored
  piecewise-linear line, so retaining each non-duplicate endpoint prevents
  chord shortcuts across bends.
- Comparing the floor and ceiling interval counts minimizes absolute spacing
  error for each segment.
- Explicit support arclengths remove the invalid uniform-column assumption from
  both mapping directions.
- Duplicate points cannot define a finite interval and intentionally share the
  preceding support column, matching existing canonical behavior.
- No volume sampling, interpolation kernel, pyramid selection, or stored line
  geometry changes.
