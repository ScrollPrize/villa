# CUDA Training Preparation Pipeline Plan

## Implementation

- Add a prepared training-batch object containing the original loader batch,
  CUDA image tensor, supervision tensors, valid mask shape metadata, preparation
  timing, and an optional CUDA event/stream.
- Add a training-only helper that flattens a `FiberStrip2DBatch`, applies
  deferred value augmentation as torch tensors on the training device, normalizes
  images, and builds direction supervision.
- For CUDA training with `pipeline_enabled`, add a bounded preparation pipeline:
  consume loaded CPU batches in deterministic step order, submit CUDA
  augmentation/preparation for future batches on a side stream, and wait on the
  side-stream event immediately before model forward.
- Keep CPU training and load-only benchmark paths synchronous.
- Keep existing `FiberStrip2DLoader.apply_batch_image_augmentation()` behavior
  for runner/tests.

## Spec Update

- Document that CUDA training pipelines have two stages: CPU/VC3D batch loading
  and CUDA image-preparation on a side stream.
- Document that normal training avoids the NumPy round trip for deferred value
  augmentation, while runner/debug APIs may still return NumPy.
- Document new timing fields for preparation work and wait.

## Docs Updates

- Update `docs/code_structure.md` training/loader sections with the prepared
  CUDA batch path.

## Testing

- Run the focused loader/training tests:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
- Run a benchmark/profile smoke command if local VC3D/cache inputs are available.

## Changelog

- Add a short changelog entry for CUDA-stream training preparation overlap.
