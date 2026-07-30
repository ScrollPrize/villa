# Task Log: Native 3D Trace2CP Refined Presence Visualization

## Implementation Notes

- Added strip-coordinate-grid plane-normal reconstruction for native 3D
  Trace2CP presence visualization.
- Extended `_sample_presence_on_strip` with opt-in
  `scale_by_strip_tangent_plane`.
- The scale factor is
  `sqrt(1 - dot(predicted_direction, strip_plane_normal)^2)`, so it is
  invariant to Lasagna/fiber direction sign ambiguity and measures alignment to
  the displayed strip plane rather than a single tangent vector.
- Single-pair fused presence panels and whole-fiber regenerated presence panels
  use the modulated display. Original/input presence panels remain raw
  presence.

## Deviations / Deferred Items

- None.

## Validation

- `PYTHONPATH=vesuvius/src:. python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py vesuvius/tests/neural_tracing/test_fiber_trace_3d.py` passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k 'strip_presence'` passed: 2 passed, 178 deselected.
- `git diff --check` passed.
