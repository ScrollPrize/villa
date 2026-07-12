# Trace2CP Top-Model Direction Debug Plan

## Implementation

- Add `--trace2cp-top-model-dir-vis` to the runner.
- Extend checkpoint loading with a top-view model loader that requires
  `top_model_state_dict` and instantiates the same V0 architecture with one
  scalar output channel for the top distance-transform head.
- Thread the optional top model through single-pair and whole-fiber Trace2CP
  evaluation chains.
- During top-strip debug construction, choose the z-corrected fused trace when
  available, otherwise the fused trace at z=0.
- Build a top-strip offset stack from that trace using offsets `-4..+4`
  selected-scale voxels in one-voxel steps.
- Run the top model on each offset layer, decode its two direction channels,
  and choose the per-pixel direction from the valid layer with the largest
  horizontal alignment `abs(dx)`.
- Render sparse unoriented direction line segments from that selected direction
  field over the center top-strip image.
- Append the rendered panel to the existing top-strip column/rows.

## Spec Update

- Document that `--trace2cp-top-model-dir-vis` searches fixed top-strip normal
  offsets for the most-horizontal top-model direction and is visualization-only;
  it does not alter Trace2CP scoring or z-search selection yet.

## Docs Updates

- Update `docs/code_structure.md` Trace2CP/top-view runner notes.
- Add a changelog line and current-task log entry.

## Tests

- Add focused tests for top-model checkpoint loading failure/success where
  practical, RGB top-strip panel handling, and horizontal-best direction
  selection across offset layers.
- Run:
  - `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
