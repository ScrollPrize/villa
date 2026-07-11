# Trace2CP Metric Task Log

## Implementation Notes

- Added `_Trace2CpMetricResult` and `_trace2cp_metric_from_traces` in
  `runner.py`.
- Public Trace2CP metric is now closest actual vertical trace gap divided by
  horizontal CP span.
- The no-overlap fallback uses centerline-to-valid-edge y distance divided by
  horizontal CP span.
- Existing center-biased closest approach remains in the refinement result as
  a visualization/refinement diagnostic.
- Added `_trace2cp_metric_bidirectional` for training/test evaluation so the
  metric path does not build fused/refined visualization traces.
- Single-pair and whole-fiber Trace2CP summaries/stdout now report
  `metric_error` and keep `refine_score` separate.
- Training test evaluation now computes averaged held-out CP-to-next-CP
  `test/trace2cp_error`, logs it to TensorBoard, and selects `best.pt` by that
  metric when `test_datasets` is configured.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - Passed.
- `git diff --check -- vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/task.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/task_plan.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/status.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/specs.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/code_structure.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/changelog.md`
  - Passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py -k 'trace2cp_metric or bidirectional_trace_scores_closest_approach or training_with_test_dataset_uses_test_interval_for_snapshots'`
  - Passed: 4 passed, 136 deselected.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - Passed: 140 passed.
