# Trace2CP Top-Model Monotone Path Plan

## Implementation

- Add a NumPy dynamic-programming helper for a monotone-x top-strip path.
- Use the top-strip center row for both CPs, matching the existing top-strip
  visualization convention.
- Round CP x columns to image columns, step in fixed 8 px horizontal increments
  plus the exact target column, and allow a bounded vertical transition band
  proportional to the horizontal step.
- Use ambiguous direction alignment cost for each transition:
  integrate `1 - abs(dot(normalized([dx, dy]), normalized_direction_at_pixel))`
  over every pixel column crossed by the transition.
- Add a fixed penalty for invalid fused direction pixels so the path can still
  connect across gaps, while preferring valid field pixels where possible.
- Backtrack the minimal-cost path ending at the target CP row and return it as
  polyline coordinates.
- Draw the DP path in the `--trace2cp-top-model-dir-vis` debug panel along
  with the existing forward/reverse local traces.
- Include top-trace stop reasons, point counts, DP path length, and DP invalid
  pixel count in the single-pair trace summary/debug label.

## Spec Update

- Document the monotone-x DP path and its direction-alignment cost.
- Document that it is visualization-only for now and does not change Trace2CP
  scoring or z-search.

## Docs Updates

- Update `docs/code_structure.md` Trace2CP/top-view runner notes.
- Add a changelog line and current-task log entry.

## Tests

- Add focused tests that a horizontal field produces a CP-to-CP center path,
  that an 8 px step path prefers gradual corrections over one large jump, that
  invalid pixels are avoided where possible, and that a full invalid barrier is
  still crossed.
- Run:
  - `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
