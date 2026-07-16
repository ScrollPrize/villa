# 3D Multi-Dir TensorBoard Presence And Oblique Slice Visualization Plan

## Implementation

- Change branch-presence visualization helpers to return close-to-normal raw
  presence, other-branch raw presence, max branch presence, min branch
  presence, and average branch presence.
- Update train/test TensorBoard layout text and sample-sheet assembly to use
  seven columns: image, target/context, close, other, max, min, average.
- Compute and store transformed CP tangents for dense-line/NML samples, not
  only CP-only samples.
- Project transformed line segments into oblique row/column coordinates for
  the GT-tangent and perpendicular/cross rows.
- Use the oblique-frame line raster for oblique target/context panels, with
  sampled target volume only as a fallback when no segment metadata exists.

## Spec Update

- Document the seven-column 3D TensorBoard sheet layout.
- Document that oblique GT overlays and target/context panels are projected in
  the oblique slice frame.
- Document that dense-line/NML samples carry transformed CP tangents for
  oblique visualization construction.

## Docs Updates

- Update `docs/code_structure.md` 3D training visualization notes.
- Add a changelog entry for the TensorBoard layout and oblique slice fixes.

## Testing Plan

- Update existing sheet-width and branch-presence helper tests for the new
  aggregate columns.
- Add a regression test that NML dense-line samples keep a transformed CP
  tangent for oblique visuals.
- Add a regression test that oblique line presence uses row/column/normal
  axes rather than an axis-aligned slice assumption.
- Run:
  - `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/loader.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/train.py vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "branch_presence_view or train_sample_3d_sheet or oblique_line_presence or dense_line_batch_keeps_cp_tangent"`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`

## Changelog

- Record the 3D TensorBoard presence-column update and oblique slice GT
  projection/tangent fix.

## Deviations Or Deferrals

- None planned.
