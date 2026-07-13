# Trace2CP Timing Rows Plan

## Implementation

- Add a small timing-row dataclass and helper functions for recording,
  aggregating, and printing Trace2CP stage timings.
- Record timings inside `_evaluate_trace2cp_pair` for:
  - segment source construction and sampling,
  - center model inference,
  - reference/base tracing,
  - optional TTA field build/inference and median-TTA tracing,
  - combined DP or z-DP tracing,
  - z-corrected debug image/presence reconstruction and layer TIFF assembly,
  - similarity debug and top-strip/top-model debug stages.
- Add export-level rows for overlay/write/summary generation where useful.
- Print one compact table per Trace2CP command. For whole-fiber mode, aggregate
  all pair timings into stage rows with count, total, mean, and max.

## Spec Update

- Document that Trace2CP commands print timing rows by stage.

## Docs Updates

- Update `docs/code_structure.md` Trace2CP runner section with the timing table
  behavior.

## Tests

- Add lightweight unit coverage for the timing aggregation helper.
- Run the full fiber_trace_2d loader test file.

## Changelog

- Add a dated changelog entry for Trace2CP timing rows.
