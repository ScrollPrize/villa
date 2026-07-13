# Trace2CP Top-Direction DP Optimized-Line Diagnostics Log

## Implementation Notes

- Extended traced top-strip sampling with optional per-column top-row offsets.
- `_trace2cp_top_model_direction_overlay` now returns the yellow top-DP path
  and selected top-offset layers as debug data.
- Trace2CP pair evaluation reconstructs optimized top strip, optimized
  side-z slice, top z-pillar presence, and side-column presence panels from
  the top-DP optimized line.
- The optimized line now folds both top-DP components into side displacement:
  selected top-offset layer plus the DP row offset from the old top-strip
  center. The optimized top-strip panel draws the new centerline as straight
  rather than re-drawing the pre-reslice curved DP path.
- Single-pair overlays and whole-fiber rows append these panels below the
  existing top-model direction visualization.
- Specs, code docs, and changelog were updated.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - Result: `256 passed in 10.71s`
