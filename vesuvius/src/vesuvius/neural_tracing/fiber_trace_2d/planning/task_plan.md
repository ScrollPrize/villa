# Trace2CP Top-Model Monotone Path Plan

## Implementation

- Add a NumPy dynamic-programming helper for a monotone-x top-strip path with
  state `(top_offset_layer, y)`.
- Use the top-strip center row for both CPs, matching the existing top-strip
  visualization convention.
- Round CP x columns to image columns, step in fixed 8 px horizontal increments
  plus the exact target column, and allow a bounded vertical transition band
  proportional to the horizontal step.
- Allow bounded z/top-offset layer transitions per horizontal step, with a
  modest penalty for `abs(delta_layer)` so the path can use another layer when
  direction evidence is better but does not hop layers for free.
- Use ambiguous direction alignment cost for each transition:
  integrate `1 - abs(dot(normalized([dx, dy]), normalized_direction_at_pixel))`
  over every pixel column crossed by the transition, using the direction field
  at the path's interpolated z layer.
- Add a fixed penalty for invalid direction pixels in the selected layer so
  the path can still connect across gaps, while preferring valid field pixels
  where possible.
- Backtrack the minimal-cost path ending at the target CP row and return it as
  polyline coordinates plus layer indices for debug reporting.
- Draw the DP path in the `--trace2cp-top-model-dir-vis` debug panel along
  with the existing forward/reverse local traces.
- Include top-trace stop reasons, point counts, DP path length, and DP invalid
  pixel count plus layer min/max/transition count in the single-pair trace
  summary/debug label.

## Spec Update

- Document the z-aware monotone-x DP path, its direction-alignment cost, and
  z-transition penalty.
- Document that it is visualization-only for now and does not change Trace2CP
  scoring or z-search.

## Docs Updates

- Update `docs/code_structure.md` Trace2CP/top-view runner notes.
- Add a changelog line and current-task log entry.

## Tests

- Add focused tests that a horizontal field produces a CP-to-CP center path,
  that an 8 px step path prefers gradual corrections over one large jump, that
  invalid pixels are avoided where possible, that a full invalid barrier is
  still crossed, and that the z-aware path uses an alternate layer when that
  layer has better direction evidence.
- Run:
  - `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
