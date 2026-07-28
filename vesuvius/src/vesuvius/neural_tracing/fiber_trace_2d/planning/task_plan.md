# Plan: Native 3D Trace2CP Metric-Only Default

## Goals

- Keep native 3D Trace2CP tracing, stdout metrics, and JSON summary output
  unchanged.
- Make JPG visualization and partial panel updates opt-in with `--vis`.
- Preserve current single-pair and whole-fiber selection behavior.

## Non-Goals

- Do not change trace search, beam scoring, smoothing, fusion, or metric
  semantics.
- Do not change output filenames when visualization is explicitly enabled.

## Implementation Steps

1. Add a `render_visualization` flag to the native 3D Trace2CP runner and a
   CLI `--vis` option defaulting to false.
2. In whole-fiber mode, only create/update `trace2cp_native_3d_vis.jpg` and
   install the segment render callback when `--vis` is set.
3. In single-pair mode, only build strip visualization sources, render panels,
   and save `trace2cp_native_3d_vis.jpg` when `--vis` is set.
4. Always create `trace2cp_native_3d_summary.json`, always print the existing
   metric lines, and mark whether visualization was enabled in the summary.
5. Add a CLI default regression test for metric-only mode.

## Spec Update

- Add that native 3D Trace2CP is metric-only by default.
- Add that `--vis` explicitly enables JPG rendering and partial image updates.

## Docs Updates

- Update `docs/code_structure.md` native 3D Trace2CP command notes to show
  metric-only default behavior and `--vis` for visualization.

## Validation Commands

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "native_3d_trace2cp_cli_defaults"`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
- `git diff --check`

## Changelog Update

- Add one changelog line for metric-only native 3D Trace2CP default behavior.
