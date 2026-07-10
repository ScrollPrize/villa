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
