# Trace2CP Traced Top Strip Visualization Log

## Implementation Notes

- `_evaluate_trace2cp_pair` now builds three visualization-only top-strip
  variants when requested: original/init comparison, traced fused central-z,
  and traced fused z-corrected when z-search is active.
- `FiberStrip2DLoader.sample_trace2cp_traced_top_strip_source` samples a
  VC3D-style top strip from the traced fused line by interpolating the trace at
  each output column, sampling the segment coordinate grid and Lasagna
  normal/offset axis at that point, deriving the side axis from traced tangent
  and normal, and sampling rows through the volume.
- Single-pair `trace2cp_vis.jpg` renders the original, traced central-z, and
  optional z-corrected top strips as a comparison debug column.
- Whole-fiber `trace2cp_fiber_vis.jpg` stitches the same variants as separate
  rows in global CP x coordinates.
- Trace2CP scoring, tracing, metrics, and training behavior are unchanged.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py`
  - Result: passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - Result: `188 passed in 7.34s`.
- `git diff --check`
  - Result: passed.
