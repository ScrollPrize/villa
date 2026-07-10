# Task Plan: Simplify Training TensorBoard Direction Overlay

## Scope

Only change the TensorBoard training batch visualization. Do not change
augment-vis exports, supervision, loss computation, sampling, or model output.

## Plan

- Keep the existing transformed centerline overlay as the background layer.
- Remove CP-neighborhood box drawing and CP marker tick drawing from the
  training TensorBoard visualization.
- Draw exactly one short predicted direction segment at the transformed CP
  coordinate, on top of the centerline.
- Add a focused regression for the drawing helper that detects the green
  prediction segment and confirms yellow/cyan CP visualization colors are gone.

## Spec Update

Update `planning/specs.md` to state that TensorBoard batch overlays show the
centerline plus one predicted CP direction segment only.

## Docs Updates

Update `docs/code_structure.md` training logging notes.

## Testing

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`

## Changelog

No changelog entry needed; this is a small visualization-only tweak.
