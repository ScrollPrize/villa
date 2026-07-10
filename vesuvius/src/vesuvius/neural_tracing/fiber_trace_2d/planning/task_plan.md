# Parallel CUDA Training Pipeline Plan

## Implementation

- Add `training.pipeline_workers` as the concurrent whole-batch loader worker
  count. Default it to `training.pipeline_depth` so existing CUDA pipeline
  configs immediately allow more than one in-flight `load_batch` call.
- Keep deterministic sample semantics by submitting steps concurrently but
  consuming completed whole batches strictly by step number.
- Keep `training.pipeline_depth` as the bounded queue depth for in-flight
  whole-batch loads/prepared batches.
- Move CUDA batch preparation submission into a small background preparation
  executor. The main training thread should wait only for the current prepared
  batch event, then run forward/backward/step.
- Preserve the side CUDA stream and prepared CUDA tensors introduced by the
  prior task.

## Spec Update

- Replace the single whole-batch producer wording with deterministic concurrent
  whole-batch loading.
- Document `training.pipeline_workers` and the ordered-consumption guarantee.
- Clarify that `prep_submit_ms` measures main-thread queue refill overhead, not
  the full preparation work.

## Docs Updates

- Update `docs/code_structure.md` training/config sections with
  `pipeline_workers` and the background preparation executor.

## Testing

- Run:
  `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py`
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`

## Changelog

- Add a short changelog entry for concurrent whole-batch loading and background
  CUDA-preparation submission.
