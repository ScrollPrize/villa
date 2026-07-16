# 3D Multi-Dir TensorBoard Presence And Oblique Slice Visualization Task Log

## Findings

- The TensorBoard branch-presence helper returned close-to-normal raw
  presence, other raw presence, and a normal-weighted close-presence view.
- The two oblique rows did not draw the GT line overlay in the oblique image
  panel.
- Oblique target/context panels sampled the already rasterized 3D target
  volume instead of projecting transformed line segments into the oblique
  frame, which could make the cross slice look like it had sideways line
  motion.
- Dense-line/NML samples returned a zero `target_tangent_zyx`, so oblique row
  construction fell back to a fixed default tangent instead of the local GT CP
  tangent.

## Implementation Notes

- Branch-presence visualization now returns raw close/other branch presence
  plus raw max/min/average branch presence aggregates.
- Train/test sample sheets now use seven columns: image, target/context,
  close, other, max, min, average.
- Dense-line/NML target specs now compute the transformed CP tangent using the
  same source-to-output tangent mapping as CP-only specs.
- Oblique rows now project transformed line segments into their row/column
  axes and gate by distance to the oblique plane normal. The same projection is
  used for image overlays and target/context panels.
- Oblique target/context panels fall back to sampled target volume only when no
  transformed segment metadata produces a visible projected line.

## Deviations Or Deferrals

- None.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/loader.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/train.py vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "branch_presence_view or train_sample_3d_sheet or oblique_line_presence or dense_line_batch_keeps_cp_tangent"`
  passed: 6 passed, 60 deselected.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  passed: 66 passed.
- `git diff --check` passed.
