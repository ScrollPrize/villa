# Reachable-Area Contrastive Similarity-Mean Sparsity Loss Task Log

## Notes

- The reachable area should be exactly the same mask already used for
  contrastive pixel negatives.
- The default no-mask path should stay valid-only for direct unit calls that do
  not provide `negative_candidate_mask`.
- Implemented by passing `valid & reachable` to the similarity-mean sparsity
  term whenever `negative_candidate_mask` is supplied.
- Updated the existing edge-handling contrastive regression so the masked
  similarity mean ignores unreachable 5x5 edges and averages over the reachable
  3x3 center area.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py -k contrastive_embedding_loss`
  - Result: 3 passed, 157 deselected.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - Result: 160 passed in 6.73s.
- `git diff --check -- vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/embedding.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/specs.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/code_structure.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/changelog.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/task.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/task_plan.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/status.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/task_log.md`
  - Result: clean.
