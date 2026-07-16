# Trace2CP Compact Geometry CP-Span Coverage Plan

## Implementation

- Extend `_required_line_ranges_for_record(...)` so compact geometry samples
  consecutive CP-to-CP line spans in addition to CP source-window ranges.
- Keep strict invalid-normal behavior: sampled bad line points remain invalid
  and Trace2CP fails if the actual segment crosses them.
- Preserve line-level invalid reasons in `_FiberLineGeometry`.
- Improve the Trace2CP interval-crossing error to report invalid runs,
  overlapping valid intervals, stored Lasagna failure reasons, and direct-probe
  values for any unexpected unsampled invalid point.

## Spec Update

- Update compact-geometry preload specs to include consecutive CP-to-CP spans,
  not only CP source windows.
- Document that unsampled invalid compact-geometry points are a diagnostic bug
  path and should not occur after preload for requested Trace2CP spans.

## Docs Updates

- Update `docs/code_structure.md` compact-geometry notes to describe CP-span
  coverage and detailed invalid-interval diagnostics.

## Testing Plan

- Add a unit test proving disjoint CP source windows still produce one required
  CP-to-CP preload span.
- Keep tests for trimming invalid margin and rejecting actually invalid
  CP-to-CP compact geometry.
- Run focused compact-geometry tests.

## Changelog

- Add a changelog line for compact-geometry CP-span preload coverage and
  improved invalid-interval diagnostics.

## Deviations Or Deferrals

- No normal interpolation/fill is added. Invalid sampled Lasagna data still
  invalidates the segment.
