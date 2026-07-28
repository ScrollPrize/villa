# Task Log: Native 3D Trace2CP Metric-Only Default

## Implementation Notes

- Added `--vis` to `fiber_trace_3d.trace2cp_tool`; it defaults to false.
- Added `render_visualization` to `run_native_trace2cp`.
- Whole-fiber mode now skips the render callback, initial status canvas, and
  partial JPG updates unless `--vis` is enabled.
- Single-pair mode now skips strip-source construction, panel rendering, and
  JPG writing unless `--vis` is enabled.
- Summary JSON is still always written and includes `visualization_enabled`.
  The `export` field is only populated when a visualization was actually
  written.
- No trace search, scoring, fusion, or metric semantics were changed.

## Deviations / Deferred Items

- None.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "native_3d_trace2cp_cli_defaults"`
  passed: 1 passed, 143 deselected.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  passed: 142 passed, 2 skipped.
- `git diff --check` passed.
