# Plan: detailed replay overview and restart markers

## Implementation

1. Keep native selected-group coordinate sizing as the overview's base grid,
   then call the unchanged shared fine-to-coarse coordinate renderer with
   render scale eight for both reference top and side surfaces. Do not change
   the per-failure OBJ/MTL/TIFF rendering contract.
2. Validate failure arrays before any CT rendering. Map each greedy and
   fiberlet failure's stored absolute reference arc through
   the same selected-reference fractional-column mapping used by trace points.
   Draw a JPEG-safe three-pixel vertical marker band over the complete top and
   side strips at the failure location: red for greedy and cyan for fiberlet.
   Overlapping marker pixels use magenta so coincident/adjacent failures from
   both tracers remain explicit. These mark the pre-reset error's
   `failure_reference_arc`, rather than the later reset seed position; no
   separate reset-seed marker is drawn.
3. Keep the yellow reference centerline visible over the two evaluator traces.
   Preserve disconnected evaluator segments and strict match validation.
4. Avoid JPEG's 65,500-pixel dimension limit by wrapping long 8x strips into
   deterministic longitudinal panels of at most 32,000 columns in the same
   JPG. Choose panel count from the larger unwrapped strip width, divide panels
   by equal selected-reference fractions, and independently map those fraction
   boundaries to contiguous `[begin,end)` top and side source-column ranges.
   Copy the already rasterized ranges exactly once, so traces and marker bands
   split without dropped columns or cross-panel bridges. Each panel has fixed
   top/side label rows and fixed inter-panel spacing. Reject before allocation
   if the computed JPEG width or height would exceed 65,500. Short or
   `--length`-bounded runs remain one panel. Never downsample the requested 8x
   pixels to fit.
5. Extend the typed overview payload and strict publisher validation with
   render scale, marker width/semantics, full unwrapped top/side dimensions,
   composed JPEG dimensions, and per-panel reference fractions, top/side
   source ranges, and composed row ranges. Persist the same fields and colors
   in root metadata.

## Tests

1. Extend the synthetic overview test to assert exact 8x top/side dimensions
   from a non-unit OME transform, near-full-height JPEG-surviving red/cyan
   marker bands at expected projected failure columns, a magenta equal-arc
   overlap marker, disconnected traces, and unchanged strict match behavior.
2. Test the compositor directly with synthetic rendered strips whose unwrapped
   width crosses the panel limit. Prove deterministic proportional top/side
   ranges, an exact-boundary trace/marker column, complete exactly-once source
   coverage, no rescaling, and no bridge between panel rows.
3. Retain regression coverage that the per-failure renderer still produces
   native-size TIFFs and scale-one manifest metadata.
4. Build `test_fiber_replay` and `vc_fiberlets` with `-j32`, run the focused
   C++ suite, run `git diff --check`, and regenerate a bounded Paris4 overview
   to inspect the increased detail and metadata.

## Spec Update

- Change the root replay overview from native group pitch to 8x sampling.
- Add failure/restart marker semantics and deterministic JPEG-safe panel
  wrapping while retaining selected-interval and shared-renderer contracts.

## Documentation Updates

- Document the 8x overview, red/cyan vertical failure markers, wrapped panel
  layout, and unchanged per-failure artifact resolution in
  `volume-cartographer/docs/fiberlets.md`.
- Update status, current-task log, and changelog with validation results.
