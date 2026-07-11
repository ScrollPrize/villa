# Direction Visualization Cell Spacing Log

Current task: adjust the `--dir-vis` direction overlay from 4x4 display-pixel
cells to 8x8 display-pixel cells, with 6-display-pixel anti-aliased direction
segments.

Implementation notes:

- Updated `_direction_field_overlay_rgb` to default to stride 4, while
  `--dir-vis` still uses a 2x nearest-neighbor image scale.
- The existing proportional segment-length formula now yields a 6-display-pixel
  segment inside each 8x8 display-pixel cell.
- Direction segments are drawn onto a 4x supersampled RGBA overlay and
  downsampled before compositing, leaving the underlying image scaling
  unchanged.
- Updated dir-vis summary metadata, specs, and code-structure docs.

Validation:

- Compile check:
  `AGENTS_AGENT_MODE=1 PYTHONPATH=vesuvius/src:. python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py`
  passed.
- Focused overlay test:
  `AGENTS_AGENT_MODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py -k dir_vis_overlay_scales_and_strides`
  passed: `1 passed, 105 deselected in 1.90s`.
- Full fiber 2D loader/runner helper test module:
  `AGENTS_AGENT_MODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed: `106 passed in 5.12s`.
- Diff whitespace check:
  `git diff --check -- vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed.
