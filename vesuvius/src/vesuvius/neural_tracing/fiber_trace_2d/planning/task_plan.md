# Plan: Python Native 3D Trace2CP CP Label Placement

## Implementation

1. Update `_draw_trace_panel(...)` so control point labels are collected during
   marker drawing and then rendered at the bottom edge of the strip, clamped to
   panel bounds.
2. Keep the point marker itself unchanged and visible.
3. Update `_whole_fiber_span_control_point_labels(...)` so labels include CP
   indices: `cp=<idx> d=<distance>`, `cp=<idx> d=inf`, or `cp=<idx> miss`.
4. Keep all tracing and metric behavior unchanged.

## Spec Update

- Document bottom-strip CP label placement and index-bearing label strings.

## Docs Update

- Update `docs/code_structure.md` with the label format and debug usage.

## Validation

- Add/update tests for label strings and bottom placement.
- Run the 3D neural tracing test file and `git diff --check`.
