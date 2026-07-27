# 3D Single-Output Presence Visualization Simplification Plan

## Scope

- `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/train.py`
- `vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
- `vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/specs.md`
- `vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/code_structure.md`
- active planning/status/changelog files

## Plan

1. Add a single-output sample-sheet helper for exactly seven-channel outputs.
   It should decode the predicted Lasagna 3x2 orientation, return raw presence,
   presence weighted by `abs(dot(pred_axis, slice_normal))`, and presence
   weighted by `abs(dot(pred_axis, GT_tangent))`.
2. Keep the existing multi-output helper and branch summary columns unchanged
   for conditioned or multi-branch outputs.
3. Route principal and oblique sample-sheet rows through a variable panel list.
   Single-output rows should contain image, target/context, and the three
   simplified prediction-presence panels.
4. Pass the transformed/target CP tangent to the oblique row builder so tangent
   modulation uses the same local target tangent as the row geometry.
5. Update TensorBoard layout text, specs, docs, changelog, and tests.

## Spec Update

- Document that single-output 3D train/test sample-sheet rows use three
  prediction-presence panels: raw, slice-normal-weighted, and GT-tangent-weighted.
- Keep the existing multi-output/conditioned branch summary layout in the spec.

## Docs Updates

- Update `docs/code_structure.md` with the same single-output versus multi-output
  sample-sheet distinction.

## Testing

- Add/adjust unit tests for:
  - single-output sheet width,
  - single-output normal/tangent modulation values,
  - existing multi-branch normal-selection helper behavior.
- Run:
  - `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/train.py vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k 'sample_3d_sheet or branch_presence or single_output_presence'`
  - `git diff --check`

## Changelog Update

- Add a 2026-07-27 entry noting the simplified single-output 3D presence
  visualization.

## Non-Goals

- Do not change training losses, target materialization, model outputs, or
  multi-output/conditioned branch visualization behavior.
