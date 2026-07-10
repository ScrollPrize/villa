# Parallel CUDA Training Pipeline Task Log

## Implementation Notes

- Added `training.pipeline_workers`; `0` means use `pipeline_depth`.
- `_TrainingBatchPipeline` now uses `pipeline_workers` concurrent whole-batch
  loader workers while consuming steps in deterministic order.
- `_CudaPreparedBatchPipeline` now submits load+prepare work through a
  background preparation executor. The main thread waits for the current
  prepared batch and only refills the queue, so `prep_submit_ms` should now be
  small queue overhead rather than full preparation work.
- Updated `loader_example.json` with explicit `pipeline_depth: 4` and
  `pipeline_workers: 4`.
- Updated specs, code-structure docs, and changelog.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed: 104 tests.

## Follow-Up Signals

- If `prep_wait_ms` rises, preparation is no longer hidden and the next
  bottleneck is CUDA prep/value augmentation or supervision building.
- If `wait_ms` rises, whole-batch loading is the bottleneck; increase
  `training.pipeline_workers`, `training.pipeline_depth`, or `loader_workers`
  carefully while watching CPU utilization.
