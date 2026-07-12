# Trace2CP Top-Model Direction Debug Plan

## Implementation

- Add `--trace2cp-top-model-dir-vis` to the runner.
- Extend checkpoint loading with a top-view model loader that requires
  `top_model_state_dict` and instantiates the same V0 architecture with one
  scalar output channel for the top distance-transform head.
- Thread the optional top model through single-pair and whole-fiber Trace2CP
  evaluation chains.
- During top-strip debug construction, choose the z-corrected traced fused top
  strip when available, otherwise the traced fused top strip at z=0.
- Run the top model on that selected top strip, decode its two direction
  channels, and render sparse unoriented direction line segments over the same
  top image.
- Append the rendered panel to the existing top-strip column/rows.

## Spec Update

- Document that `--trace2cp-top-model-dir-vis` is visualization-only and does
  not alter Trace2CP scoring or z-search selection yet.

## Docs Updates

- Update `docs/code_structure.md` Trace2CP/top-view runner notes.
- Add a changelog line and current-task log entry.

## Tests

- Add focused tests for top-model checkpoint loading failure/success where
  practical and for RGB top-strip panel handling.
- Run:
  - `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
