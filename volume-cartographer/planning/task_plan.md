# VC3D short line-segment plan

## Implementation

1. Define one public 32-base-voxel generated-line sampling constant and make it
   the default `LineViewConfig` target.
2. Retain per-consecutive-optimized-point nearest-spacing subdivision and every
   exact optimized-point support, but declare only the along-strip grid density
   from the configured target rather than the mean physical source interval.
   Remove cross-direction configuration and the median-point-spacing heuristic;
   build both ribbons with exactly 21 rows at the shared 32-base-voxel pitch.
3. Add one reusable cumulative-arclength mapping for fractional line positions,
   inverse mapping, and inclusive radius matching. Use it for both deduplication
   and the existing maximum-control-distance gate, whose setting is also in
   base voxels.
4. Implement control collapse as a pure operation returning ordered controls,
   every old-to-new index mapping, the replacement index, and adjacent dirty
   spans. The replacement receives the clicked position, clicked geometry, and
   an invalid optimizer index. It inherits seed state from any collapsed seed
   and outgoing interpolation policy from the rightmost collapsed control;
   unaffected span policies remain unchanged.
5. Before mutation, confirm the whole operation if any matched control is
   linked. Remap every branch through the explicit many-to-one mapping and
   remove only duplicate links to the same remote endpoint.
6. Continue the existing local update path for insertion or one replacement.
   Use fiber-mode optimization for a multi-control collapse, marking only the
   replacement's adjacent spans dirty because the local single-control updater
   does not accept simultaneous removals. Preserve rollback snapshots across
   the asynchronous update.
7. Correct generated-pane initial fitting to use the QuadSurface grid dimensions
   when converting the first and last grid samples to surface coordinates.

## Specification updates

- Change the generated-line strip invariant from a 50-vx mean-density model to
  a fixed 32-vx display pitch with exact supports and segment-local subdivision.
- Require a fixed 21-row, 640-base-voxel cross extent for both generated
  ribbons, independent of optimized-line sampling.
- Add the 32-vx arclength replacement/collapse rule.

## Documentation updates

- Update the line-annotation fiber documentation to describe fixed 32-vx strip
  display pitch in both directions, fixed cross extent, and arclength-based
  control replacement.

## Testing and validation

- Extend `test_lasagna_line_view_surfaces` for fixed `1/32` scale in both
  directions, fixed 21-row/640-vx cross extents, exact short endpoints, and
  unchanged long-span subdivision. Remove tests for obsolete cross overrides.
- Extend `test_fiber_slice_geometry` for forward/inverse fractional arclength
  conversion, inclusive multi-match selection, and maximum-distance behavior
  on nonuniform/curved polylines.
- Add focused control-collapse tests for one/multiple/all matches, seed inside
  and outside the set, endpoint metadata ownership, old-to-new index mapping,
  adjacent dirty spans, and unchanged unrelated metadata.
- Cover many-to-one branch remapping and duplicate-link handling at the
  controller helper boundary where private branch state is available.
- Build with all 32 available cores and run the focused tests, then run diff
  checks.

## Changelog update

- Add one dated line for fixed-pitch generated strips and arclength control
  replacement.
