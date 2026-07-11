# Whole-Fiber Trace2CP Visualization Task Log

## Notes

- Started from the existing single-pair Trace2CP runner path.
- Whole-fiber mode should reuse configured loader records and reject a
  `--fiber-json` path that is not part of the active config.
- Added `FiberStrip2DLoader.flat_sample_indices_for_fiber_json(...)` so runner
  code can resolve a configured fiber JSON to deterministic flat CP indices.
- Extracted `_evaluate_trace2cp_pair(...)` so single-pair and whole-fiber
  Trace2CP share the same segment loading, model prediction, optional
  median-TTA, tracing, and scoring behavior.
- Added `--trace2cp-vis --fiber-json <path>` to evaluate all in-range CP pairs
  for `--trace2cp-target-offset` and write `trace2cp_fiber_vis.jpg` plus
  `trace2cp_fiber_summary.txt`.
- Added long-strip composition by translating pair-local images and trace
  points into the selected fiber's shared arc-length x coordinate system.
- `--fiber-json` rejects `--trace2cp-target-cp-index` because whole-fiber mode
  already derives all targets from `--trace2cp-target-offset`.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - passed: `134 passed in 6.04s`.
- `git diff --check -- vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - passed.
