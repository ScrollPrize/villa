# 3D Test Visibility And Direction Projection Plan

## Implementation

1. Update `_draw_projected_cp_direction(...)` in `fiber_trace_3d/train.py`.
2. Compute the in-slice projection magnitude from the 3D direction vector.
3. Use the normalized in-slice direction only for orientation, and multiply the
   display radius by the projection magnitude for line length.
4. Preserve the existing thin anti-aliased line renderer.
5. Add a shared helper for writing a 3D principal-slice sample sheet to
   TensorBoard.
6. Use that helper for both train and test sample visualization.
7. In `run_configured_tests(...)`, write `test_sample_3d/principal_slices` when
   dense test loss is configured and flush the writer after test logging.

## Testing

- Add/update focused tests that compare in-plane versus out-of-plane projected
  line extent.
- Add/update focused tests for the shared sample-sheet writer if practical.
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
- Run `git diff --check`.

## Spec Update

- Document that the 3D train/test sample projected direction overlay scales
  line length by the in-slice projection magnitude.
- Document that configured dense 3D tests log a test principal-slice sheet.

## Docs Updates

- Update `docs/code_structure.md` for the train/test visualization behavior.

## Changelog

- Add a short 2026-07-15 changelog entry for projection-length-correct
  direction overlays and test sample visualization.
