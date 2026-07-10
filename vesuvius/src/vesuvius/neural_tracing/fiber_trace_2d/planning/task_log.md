# Task Log: Batch Visualization Samples Different CPs

## Implementation Notes

- Started from the first item in `planning/todo.md`.
- Plan is local to TensorBoard/debug visualization selection and should not
  affect sampling, training loss, augmentation, or cache behavior.
- Added `_select_visualization_patch_indices` so training/test batch overlays
  prefer one center strip-z patch from each loaded CP sample before filling
  remaining cells.
- Updated the TensorBoard overlay path to use the selected patch indices.
- Added a focused unit test covering multi-CP, multi-offset selection.
- Updated `planning/specs.md` and `docs/code_structure.md`.

## Deviations

- None.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed with `43 passed in 3.23s`.
- `git diff --check` over touched files passed.
