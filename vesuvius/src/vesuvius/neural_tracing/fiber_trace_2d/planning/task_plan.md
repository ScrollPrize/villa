# Trace2CP Top-Slice Presence Visualization Plan

## Interpretation

- "Fiber presence in additional top views" refers to the side-strip presence
  head already predicted by the Trace2CP side model.
- The top-view model must not be used for this feature. The presence map has no
  top-offset axis, so the visualization will project side presence onto
  top-strip-sized panels along the corresponding init/direct/z-corrected trace
  columns.

## Implementation

- Add optional projected-presence fields to `_Trace2CpPairEvaluation` for the
  original/init, traced central-z, and z-corrected top-strip rows.
- Add a helper that samples the inferred side-strip presence at each top-strip
  pixel's corresponding side-strip coordinate: `x` from the top column and
  `trace_y(x) + top_row_offset` from the row. For z-corrected output, feed it
  the z-selected side-presence image produced from the inferred side slices.
- Generate the projected presence rows during `_evaluate_trace2cp_pair` when
  top-strip debug is enabled and the checkpoint exposes a presence head.
- Extend single-pair and whole-fiber Trace2CP overlay rendering to append the
  projected presence rows immediately after the regular top-strip rows.
- Pass the new fields through existing trace2cp export calls.

## Spec Update

- Document that Trace2CP top-strip visualizations may include projected
  side-model presence rows.
- State explicitly that these rows are fixed-scale visualization-only output
  and do not use the top-view model or change scoring.

## Docs Updates

- Update `docs/code_structure.md` Trace2CP visualization notes with the
  projected top-presence rows.

## Tests

- Extend single-pair overlay tests to verify projected top-presence rows append
  to the existing top-strip column.
- Extend whole-fiber overlay tests to verify projected top-presence rows append
  to the stitched fiber visualization.
- Run the focused loader/runner tests.

## Changelog

- Add a dated note for the projected Trace2CP top-slice presence visualization.
