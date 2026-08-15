# Task log

## Findings

- Uniform arclength support points lie on the stored polyline individually,
  but `QuadSurface` joins adjacent columns directly. A control-point bend that
  is not itself sampled is therefore replaced by a chord.
- The existing bidirectional map derives arclength from one scalar column
  spacing, so it must change together with the support geometry.
- `QuadSurface` exposes one scalar density per axis. With segment-local support
  spacing, the along-axis value can only be nominal; exact geometry remains in
  the point grid and exact interaction mapping in the support arclength array.

## Deviations

- None.

## Implementation

- Added explicit strip support arclengths and segment-local interval selection.
- Preserved every non-duplicate control point as a generated ribbon column.
- Changed both mapping directions to interpolate through the explicit support
  arclength array.
- Retained the mean interval as nominal `QuadSurface` density metadata.
- Added focused bend, segment-spacing, reverse-orientation, round-trip, and
  duplicate-point checks.

## Validation

- `cmake --build volume-cartographer/build --target test_lasagna_line_view_surfaces -j 8`
- `volume-cartographer/build/bin/test_lasagna_line_view_surfaces` (21 passed)
- `cmake --build volume-cartographer/build --target VC3D -j 8`
- `git diff --check`
