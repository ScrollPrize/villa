# Cross-Fiber Contrastive CP Negatives Task Log

## Notes

- Current contrastive loss already samples normalized CP embeddings and derives
  per-sample fiber IDs, so the new cross-fiber CP negative term can be added
  locally in `embedding.py`.
- The loss should keep single-fiber batches unchanged. When no other fiber is
  present in the batch, there is no cross-fiber component to average.
- Implemented the cross-fiber term by masking normalized CP embedding pairs
  whose fiber IDs differ.
- Updated `load_fiber_group_batch` so contrastive batches concatenate
  consecutive same-fiber CP groups. This is required for normal contrastive
  training batches to contain other-fiber CPs for the new negative term.
- Contrastive training now advances by groups-per-step, so adjacent steps do
  not overlap groups after the batch construction change.
- `negative_loss` remains the aggregate negative branch used in the objective.
  New component metrics expose pixel-negative and cross-fiber CP-negative
  losses and sample counts.
- TensorBoard keeps the existing aggregate contrastive scalar names and adds
  component scalar names for the two negative branches.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py -k "contrastive_embedding_loss or fiber_group_batch"`
  - Result: 5 passed, 155 deselected.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - Result: 160 passed in 5.81s.
- `git diff --check -- vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/embedding.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/specs.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/code_structure.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/changelog.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/task.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/task_plan.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/status.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/task_log.md`
  - Result: clean.
