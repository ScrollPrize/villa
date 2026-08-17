# VC3D short line-segment task log

## Findings

- Current strip subdivision already retains every original endpoint and keeps a
  short span as one interval.
- Current strip scale is derived from total arclength divided by total interval
  count, so mixed short/long spans distort one another's displayed pitch.
- Current generated-click replacement uses `abs(linePosition difference) <=
  0.5`, which depends on producer point density rather than physical arclength.
- The local control update API supports one inserted/moved control, not several
  simultaneous removals.
- Branch links are indexed by the live control vector and therefore require an
  explicit old-to-new index remap during a multi-control collapse.
- The initial implementation incorrectly made every optimized line point a
  strip support. Production now supplies the annotation control positions, and
  only those controls plus line endpoints are fixed supports; each intervening
  optimized polyline is resampled by arclength.
- The generated ribbons have no physical fiber-width input. Main's default
  cross width was typically about 200 base voxels.
- `surfaceHalfWidth`, `sideSliceHalfDepth`, and `crossSamples` have no production
  overrides; only `test_lasagna_line_view_surfaces` changes them.
- The unused cross configuration fields and median-step helper are removed.
  Seven rows at 32 voxels give a fixed 192-vx extent close to main's typical
  physical width. The generated-pane camera fitting passed scale-adjusted
  `QuadSurface::size()` values back as grid indices; this task corrects that
  related pre-existing extent calculation with `gridSize()`.

## Deviations

- The private Qt controller has no isolated interaction-test target. The pure
  collapse operation and its old-to-new mapping are unit tested, reciprocal
  branch synchronization was reviewed and compiled in the full VC3D target,
  but modal confirmation and asynchronous UI orchestration are not directly
  exercised by an automated test.
- The related generated-pane camera-fit correction is compiled in the full
  VC3D target but its private dialog helper has no isolated unit-test seam.

## Validation

- Built with all 32 cores:
  `cmake --build volume-cartographer/build --parallel 32 --target test_lasagna_line_view_surfaces test_fiber_slice_geometry test_line_annotation_generated_views VC3D`
- `test_lasagna_line_view_surfaces`: 24 test cases passed.
- `test_fiber_slice_geometry`: 10 test cases passed.
- `test_line_annotation_generated_views`: 76 test cases passed.
- `git diff --check`: passed.

## Strip sampling follow-up

- Removed `surfaceHalfWidth`, `sideSliceHalfDepth`, `crossSamples`, and the
  median optimized-point cross spacing. Both ribbons use seven rows at 32
  voxels for a fixed 192-vx extent.
- Generated strips preserve exact annotation controls and line endpoints, and
  resample the optimized polyline by arclength between those supports.
- Generated-pane initial fitting now converts actual `gridSize()` endpoints to
  surface coordinates instead of reusing scale-adjusted `size()` as grid
  indices.
- Rebuilt `test_lasagna_line_view_surfaces` and `VC3D` with 32 cores.
- `test_lasagna_line_view_surfaces`: 24 test cases passed.
- `test_fiber_slice_geometry`: 10 test cases passed.
- `test_line_annotation_generated_views`: 76 test cases passed.
- Final `git diff --check`: passed.
