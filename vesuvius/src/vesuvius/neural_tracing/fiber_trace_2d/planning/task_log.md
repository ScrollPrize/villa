# Task Log: Pipeline Training Batch Loading And GPU Work

## Planning Notes

- The current `run_training` loop loads one batch synchronously, then runs
  forward/backward/optimizer synchronously.
- `FiberStrip2DLoader.load_batch` already owns deterministic CP ordering,
  invalid-sample skips, VC3D coordinate sampling, and optional image/value
  augmentation.
- The plan keeps the shared loader path and only pipelines whole future batch
  preparation around the existing deterministic step indices.
- The main non-obvious constraint is image/value augmentation: today it happens
  inside the loader. For useful overlap, CPU-loaded batches should be prepared
  without value augmentation, then value augmentation should run on the training
  device just before model forward using the existing augmentation code and
  stored augmentation parameters.
- Clarification: coordinate generation and geometric coordinate augmentation
  must not be moved to CPU. The pipeline should preserve the configured
  `augment_device` path for that work and only overlap it carefully with the
  rest of training.

## Implementation Notes

- Added `training.pipeline_enabled` and `training.pipeline_depth`.
- CUDA training and non-load-only CUDA benchmarks now use a bounded
  `_TrainingBatchPipeline` that submits future whole-batch loads while the
  current batch trains.
- Whole-batch loading stays single-producer because `load_batch` already owns
  CP-level worker parallelism and one cache-trace scope.
- `FiberStrip2DBatch` now carries deterministic per-patch augmentation
  parameters. `load_batch(..., apply_image_augmentation=False)` can therefore
  defer torch image/value augmentation without changing the geometric
  coordinate path or sample order.
- Added `FiberStrip2DLoader.apply_batch_image_augmentation(...)` so the main
  training thread can apply the same batched value augmentation immediately
  before normalization/forward.
- CPU training keeps the synchronous path by default; load-only benchmark keeps
  measuring only loader work.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed with 95 tests.
- Approved local load-only benchmark command completed from
  `/home/hendrik/business/aiconsulting/vesuviuschallenge/data/fiber_train`:
  100 batches, 6400 patches, `17388.6 ms`, `368.06 patches/s`.
