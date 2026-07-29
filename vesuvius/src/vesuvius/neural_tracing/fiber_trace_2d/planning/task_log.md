# Python Native 3D Trace2CP CP Label Task Log

## Implementation Notes

- `_draw_trace_panel(...)` now draws CP marker ellipses without nearby text and
  renders CP labels at the bottom edge of the strip.
- Bottom labels are clamped horizontally into the panel bounds and keep the
  existing translucent text background.
- Whole-fiber labels now include CP indices:
  - `cp=<idx> d=<distance>`
  - `cp=<idx> d=inf`
  - `cp=<idx> miss`

## Deviations / Deferred Items

- No tracing, metric, inference, output selection, or filename behavior was
  changed.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py`
  - passed
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  - passed: 167 tests, 2 skipped
- `git diff --check`
  - passed
