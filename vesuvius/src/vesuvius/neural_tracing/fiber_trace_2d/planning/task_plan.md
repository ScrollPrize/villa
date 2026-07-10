# Task Plan: Augment-Vis CP Marker And Label Padding

## Scope

Only change augment-vis debug rendering. Do not change sampled images,
coordinates, masks, augmentation parameters, or training behavior.

## Plan

- Draw a final visualization-only CP crosshair at each patch's
  `FiberStripSample.control_point_xy`.
- Leave the center pixel open so the exact sampled CP pixel remains visible.
- Render contact-sheet labels in a dedicated top band for each cell instead of
  drawing text over the image.
- Include the per-cell transformed CP coordinate in `augment_summary.txt`.
- Update the existing augment contact-sheet export test for the larger labeled
  cell height and visible CP marker.

## Spec Update

Update `planning/specs.md` to state that augment-vis contact sheets draw a CP
crosshair from transformed control-point coordinates and reserve a label band.

## Docs Updates

Update `docs/code_structure.md` under runner/augment-vis documentation.

## Testing

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`

## Changelog

Add a concise changelog entry for the augment-vis rendering change.
