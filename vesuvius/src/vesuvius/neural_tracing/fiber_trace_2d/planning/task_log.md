# Contrastive Similarity-Mean Sparsity Loss Task Log

## Notes

- The normalized similarity target uses the same `0.5 + 0.5 * cosine` space as
  existing embedding-similarity TensorBoard images.
- No config key is added; the requested target is fixed at `0.1`.
- Implemented the term in `contrastive_embedding_loss` as valid-pixel
  per-CP MSE between the normalized similarity-image mean and `0.1`.
- The new sparsity term is added to the existing balanced positive/negative
  pair loss before applying `training.contrastive_weight`.
- Added metrics and TensorBoard scalars for similarity mean loss, observed
  mean value, target, and sample count.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py -k contrastive_embedding_loss`
  - Result: 3 passed, 157 deselected.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - Result: 160 passed in 6.61s.
- `git diff --check -- vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/embedding.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/specs.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/code_structure.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/changelog.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/task.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/task_plan.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/status.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/task_log.md`
  - Result: clean.
