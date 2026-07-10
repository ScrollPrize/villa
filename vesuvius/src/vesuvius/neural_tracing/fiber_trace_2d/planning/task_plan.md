# Task Plan: Batch Visualization Samples Different CPs

## Scope

Change TensorBoard batch direction overlays so they choose representative
patches across loaded control-point samples instead of taking the first
flattened patches, which are usually multiple strip-z offsets from one CP.

## Plan

- Add a small visualization-selection helper in `train.py`.
- Group flattened batch patches by loaded control-point sample order.
- Prefer the strip-z offset closest to zero for each control-point group.
- Fill the contact sheet with one representative patch per CP first, then any
  remaining patches only if the sheet has unused capacity.
- Use the selected patch indices in `_make_training_visualization` for both
  train and test overlays.

## Spec Update

Update `planning/specs.md` so TensorBoard batch overlays explicitly select
examples across different loaded CP samples, preferring the center strip-z
offset.

## Docs Updates

Update `docs/code_structure.md` TensorBoard documentation with the same
selection behavior.

## Testing

- Add a focused unit test that creates a synthetic multi-CP, multi-offset batch
  and asserts visualization selection uses different CP groups before repeated
  offsets from the first CP.
- Run:
  - `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`

## Changelog

No durable changelog entry is needed; this is a debug/TensorBoard visualization
selection fix without config or data semantics changes.
