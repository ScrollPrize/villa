# Trace2CP Top-Slice Presence Visualization Log

- Started from the user clarification that the requested top-slice presence
  visualization must use the existing side-strip presence output, not the
  optional top model.
- Planned a visualization-only projection of side presence into top-strip-sized
  rows for init, traced central-z, and z-corrected traced top strips.
- Added projected-presence fields to `_Trace2CpPairEvaluation` and pass-through
  arguments for the single-pair Trace2CP overlay.
- Added `_project_side_presence_to_top_strip`, which samples the side-strip
  presence map at the side coordinate corresponding to each top-strip pixel:
  `x` from the top column and `trace_y(x) + top_row_offset` from the row.
  This avoids repeating one presence value down an entire top-strip column.
- Whole-fiber rendering now stitches the same projected top-presence rows as
  the single-pair renderer.
- Updated specs, code-structure docs, changelog, and focused renderer tests.
- Validation:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py -k "trace2cp_overlay_can_add_top_strip_column or trace2cp_fiber_overlay_adds_top_strip_rows"`
    passed: 2 passed, 242 deselected.
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
    passed: 244 passed.
