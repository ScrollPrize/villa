# Trace2CP Top-Slice Presence Visualization Log

- Started from the user clarification that the requested top-slice presence
  visualization must use the existing side-strip presence output, not the
  optional top model.
- Planned a visualization-only projection of side presence into top-strip-sized
  rows for init, traced central-z, and z-corrected traced top strips.
- Added projected-presence fields to `_Trace2CpPairEvaluation` and pass-through
  arguments for the single-pair Trace2CP overlay.
- Replaced the earlier projected top-presence helper with
  `_side_presence_z_pillar_image`. It samples the inferred side-presence stack
  directly: each row is one z-search layer and each column samples that layer at
  `(x, trace_y(x))`.
- Updated the z-search fused trace panel to pass the full `(x, y, z)` trace
  into `_side_presence_z_pillar_image`, shifting each column by the selected
  z layer so the center row is relative to the used z value.
- Whole-fiber rendering now stitches the same projected top-presence rows as
  the single-pair renderer.
- Updated specs, code-structure docs, changelog, and focused renderer tests.
- Validation:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py -k "side_presence_z_pillar_image or trace2cp_overlay_can_add_top_strip_column or trace2cp_fiber_overlay_adds_top_strip_rows"`
    passed: 4 passed, 242 deselected.
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
    passed: 246 passed.
