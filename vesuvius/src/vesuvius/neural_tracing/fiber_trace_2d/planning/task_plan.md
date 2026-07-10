# Task Plan: Shift/Scale Augmentation Order

## Scope

Implement the first item from `planning/todo.md`: geometric shift must be
applied in scaled output space rather than original source space.

## Plan

- Update the inverse output-pixel-to-source coordinate map so output-space
  shift is subtracted before undoing flip/scale/shear/rotation.
- Update the affine source-point and centerline coordinate maps so shift is
  added only after scale/flip and before the output center offset.
- Keep smooth-offset fallback behavior on the same source-coordinate grid path
  so image sampling and line/control-point coordinates stay aligned.
- Add a regression test for combined shift+scale verifying that a transformed
  source point lands at the shifted scaled output coordinate and that the
  sampling grid at that output pixel points back to the same source point.

## Spec Update

Update `planning/specs.md` to state that geometric shift is an output-space
translation applied after scale.

## Docs Updates

Update `docs/code_structure.md` to mention the affine geometric order used by
`augmentation.py`.

## Testing

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/augmentation.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`

## Changelog

Add a concise changelog entry for the augmentation composition fix.
