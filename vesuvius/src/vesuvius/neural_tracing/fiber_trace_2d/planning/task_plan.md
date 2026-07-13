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
- Update side-strip joint DP tracing so it no longer inherits the short
  `--line-trace-step` as its DP transition length. Use a fixed longer
  transition for finer effective angular resolution while preserving
  `--line-trace-step` for output resampling.
- Pass the existing candidate-angle limit into the side DP and penalize
  direction alignments outside that cone so combined tracing is closer to the
  old direct candidate tracer.
- Add opt-in progress output inside the shared monotone DP helper. CLI
  Trace2CP side/z/top DP call sites should pass labels; tests and internal
  direct helper calls remain quiet by default.
- Include elapsed time and ETA in throttled progress rows.

## Spec Update

- Document that Trace2CP commands print timing rows by stage.
- Document the side-strip DP transition length and candidate-angle penalty.
- Document Trace2CP DP progress rows with ETA.

## Docs Updates

- Update `docs/code_structure.md` Trace2CP runner section with the timing table
  behavior.
- Update the Trace2CP combined tracing notes with the side-DP angle/transition
  behavior.
- Update the Trace2CP docs with DP progress behavior.

## Tests

- Add lightweight unit coverage for the timing aggregation helper.
- Add lightweight unit coverage for DP progress ETA output.
- Run focused Trace2CP tests and the full fiber_trace_2d loader test file.

## Changelog

- Add a dated changelog entry for Trace2CP timing rows.
