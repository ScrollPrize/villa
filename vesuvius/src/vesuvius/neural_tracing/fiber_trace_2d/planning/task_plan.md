# Trace2CP Traced Top Strip Visualization Plan

## Scope

- Add top-strip panels to single-pair `trace2cp_vis.jpg`.
- Keep the original/init top strip for comparison.
- Add traced-line top-strip sampling for both single-pair and whole-fiber
  Trace2CP visualization.
- Do not change scoring, tracing, z-search candidate selection, metrics, or
  training.

## Implementation

- Request top-strip debug sampling from `_evaluate_trace2cp_pair` in
  single-pair export.
- Add traced top-strip sampling in the loader:
  - interpolate the fused trace at each output column;
  - sample the center 3D point and Lasagna normal/offset axis from the segment
    coordinate grid at that traced side-strip point;
  - derive the top-strip side axis from traced tangent and normal;
  - sample rows along that side axis;
  - for the non-z version, force the trace z offset to zero;
  - for z-search, use the fused trace's selected z offsets.
- Extend `_draw_trace2cp_overlay` to accept optional original, traced z=0, and
  traced z-corrected top-strip images/valid masks.
- Render the top strips as an additional debug column:
  - original/init `original top strip VC3D lineSurface`;
  - traced `traced fused top strip z=0`;
  - optional `traced fused top strip z-corrected`;
  - draw the fiber centerline and start/target CP markers at the top-strip
    center row.
- Update whole-fiber overlay to stitch the original, traced z=0, and optional
  traced z-corrected top-strip rows.

## Spec Update

- Clarify that both single-pair and whole-fiber Trace2CP visualizations include
  original/init top-strip comparison and traced fused top-strip output.

## Docs Updates

- Update `docs/code_structure.md`.
- Update `planning/status.md`, `planning/task_log.md`, and changelog if useful.

## Tests

- Add focused unit coverage that `_draw_trace2cp_overlay` appends a top-strip
  column, and includes the z-corrected panel when provided.
- Run:
  - `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
