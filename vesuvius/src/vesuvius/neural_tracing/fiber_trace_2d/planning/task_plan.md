# Trace2CP Target-Column Metric Plan

## Scope

- Change only the public metric calculation used by Trace2CP CLI output and
  training test evaluation.
- Keep closest-point selection, center penalty, fused-line generation, and
  optimized refinement unchanged.
- Apply the metric semantics uniformly to direction-only, median-TTA, and
  combined direction/embedding Trace2CP paths because they all share the same
  bidirectional result construction.

## Implementation

1. Update `_trace2cp_metric_from_traces`.
   - Interpolate the forward trace at the target CP x-column.
   - Interpolate the reverse trace at the start CP x-column.
   - Compare those y values to their corresponding CP y values.
   - Use the mean raw endpoint y error divided by the horizontal CP span as
     `trace2cp_error`.
   - If either endpoint column is not reached, use the existing default
     maximum segment y error for that direction.
   - Keep capping by the default maximum y error so edge exits do not dominate.

2. Leave refinement untouched.
   - `_trace2cp_refinement_from_traces` continues to use
     `_closest_trace2cp_approach`.
   - Summary/debug text should identify closest-point values as refinement
     diagnostics, not as metric semantics.

3. Update summary labels where helpful.
   - Replace metric "closest" wording with endpoint/target-column wording in
     summaries and docs.
   - Keep dataclass fields stable unless a larger API cleanup is needed.

## Spec Update

- Replace the current public `trace2cp_error` closest-gap spec with endpoint
  target-column y-error semantics.
- Keep the closest-point spec only for refinement/fusion diagnostic behavior.

## Docs Updates

- Update `docs/code_structure.md` Trace2CP section to describe target-column
  metric semantics.
- Update `planning/changelog.md`, `planning/status.md`, and
  `planning/task_log.md` for this task.

## Tests

- Update unit tests for `_trace2cp_metric_from_traces` and bidirectional
  Trace2CP scoring to expect target-column metric values.
- Add/adjust a regression where closest gap is good but endpoint target-column
  error is worse, proving the public metric no longer follows the closest
  intersection.
- Run the focused fiber_trace_2d test module.
