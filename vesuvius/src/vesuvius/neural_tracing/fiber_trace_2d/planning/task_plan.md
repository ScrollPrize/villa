# Native 3D Whole-Fiber Trace2CP Eight-Row Visualization Plan

## Implementation

- Change the whole-fiber panel block composer to support eight rows per visual
  span instead of the current four.
- For each restart-delimited whole-fiber span, keep the existing initial
  side/top volume and presence panels.
- Build a regenerated Trace2CP segment source from the span's stitched traced
  line using the same refined-source helper used by single-pair rendering.
- Render regenerated side/top volume panels and regenerated side/top 3D
  presence panels, with only the regenerated centerline drawn thinly.
- Project all control points belonging to the displayed span into each side/top
  strip source and draw them on the initial and regenerated rows.
- Label each displayed control point with the native whole-fiber segment
  in-plane distance measured at that CP plane. The span start CP is labeled as
  zero distance; target CPs use the segment's `in_plane_error_voxels`, or a
  miss label if the plane was not reached.
- Preserve the existing partial-output overwrite behavior after each segment.

## Spec Update

- Update `planning/specs.md` so native 3D whole-fiber visualization explicitly
  requires eight rows, including regenerated/fused side/top strips and their
  presence panels, and requires span control-point markers plus CP-plane trace
  distance labels on those panels.

## Docs Updates

- Update durable planning docs only; no separate user-facing docs are needed
  for this narrow visualization layout fix.

## Tests

- Add or update a focused test that the whole-fiber span renderer returns eight
  panels, calls the refined-source path, and projects span control points into
  initial and regenerated strips with distance labels.
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
- Run `git diff --check` over touched files.

## Non-Goals

- Do not change native 3D tracing, fusion, scoring, candidate search, or model
  inference behavior.
