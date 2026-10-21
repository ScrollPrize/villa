# Asynchronous Fiber3D batch preparation

`fiber_trace_3d.cuda_batch_pipeline.FiberTrace3DCudaBatchPipeline` is the
existing asynchronous preparation stage extracted from the development trainer.
It wraps a CPU batch source and prepares a bounded number of future batches on
a dedicated CUDA stream while the caller trains on the preceding batch.

This extraction does not switch the training CLI to a new loading path, change
augmentation parameters, or alter model/loss behavior. Optional
[GPU value augmentation](gpu_value_augmentation.md) can run on the preparation
stream. Effective-batch accumulation and the later trainer integration remain
separate extraction stages. The current mask-aware CPU augmentation is unchanged.

## API and ownership

```python
from vesuvius.neural_tracing.fiber_trace_3d.cuda_batch_pipeline import (
    FiberTrace3DCudaBatchPipeline,
)

# The application must already hold a GPU-manager grant for this device.
# source can be FiberTrace3DTrainingBatchPipeline or another single-consumer
# batch iterator with release_current() and close().
prepared = FiberTrace3DCudaBatchPipeline(
    source, config=loader_config, device=device, output_stride=1, depth=2,
)
try:
    for batch in prepared:
        train_one_batch(batch)
finally:
    prepared.close()
```

One background thread obtains a complete CPU batch, pins the CUDA-bound tensor
fields, and releases its source lease only after the pinned copy completes.
Volume/mask slots may then be reused without changing the transfer source.
Nonblocking transfer and target creation run on the private CUDA stream.
The pinned batch remains owned until its completion event has finished; device
tensors are recorded on the consumer's current stream before returning them.

`depth` bounds preparation credits. In addition to the batch held by the caller,
at most `depth` future batches may be prepared or under preparation. Depth is
not a worker count or a global training batch-size setting.

Sources may defer `blur` or `all` value augmentation. Pass the same mode as
`value_augmentation_suffix` to the CUDA wrapper; it calls the shared deferred
dispatcher between H2D and target creation, with autocast explicitly disabled.
Batch timing separates transfer, value augmentation and target creation.

`FiberTrace3DBatch.to(..., non_blocking=True)`, `pin_memory()` and
`record_stream(stream)` share the existing batch field definitions. CPU target
indexing metadata deliberately remains on CPU. Pinning copies CUDA-bound fields
without mutating the CPU batch or transferring its ownership implicitly.

Call `pause()` to stop further CUDA submission and synchronize already-submitted
work before evaluation on the same GPU; call `resume()` afterwards. This is a
local preparation pause, **not** a manager allocation release: prefetched device
batches still occupy memory. A cooperative allocation change must close/drain
the GPU cohort before acknowledging resource release.

`close()` closes the underlying source to wake a blocked queue reader, wakes
paused/semaphore-blocked preparation, joins the thread and drops prepared
batches. It is idempotent. Exceptions in source loading, pinning or target
creation are propagated to the consumer with their original cause.

## Output lattices

The source target builder supports `output_stride=1` (default) and integer
coarse strides. Input tensors, coordinates and sample identities remain in the
original input lattice. Only target tensors/indices are produced on the output
lattice. Coarse validity is conservative: a cell containing any invalid input
voxel is excluded. Partial edge cells use ceiling-sized output shapes.

The builder does not resize the model. A caller requesting stride greater than
one must use a model producing the corresponding output lattice. The existing
training CLI still uses its unchanged default stride-one path.

## Provenance and validation

Batch transfer helpers, the asynchronous stage and stride-aware target builder
come from `3f0aeecaa` (following `66f3134b7`'s output-lattice support). The
index-less CUDA device resolution fix and shared deferred dispatcher come from
the later frozen development source `2192cc5ac`. The GPU suffix extends the
stage with the existing `29a7289c5` implementation. The stage now has a separate importable module; no second
preparation implementation is added to the trainer.

CPU tests cover unchanged target defaults, coarse CP placement, conservative
validity, bounded admission, failure propagation, pause/resume and shutdown:

```bash
cd vesuvius
python -m pytest -q tests/neural_tracing/test_cuda_batch_pipeline.py -k 'not cuda_real'
```

The optional CUDA tests use the same real Paris4 annotation and CT fixture as
`test_sample_pipeline.py`. Set `VC_QUEUE_TEST_VOLUME` to the concrete cached
level-one CT array and `VC_QUEUE_TEST_FIBER` to an annotated Paris4 fiber JSON.
Select the intended native binding build before import. No source data is
modified and no chunks are copied. In a GPU-manager-reserved child process:

```bash
FIBER_ASYNC_CUDA_TEST=1 python -m pytest -q -s \
  tests/neural_tracing/test_cuda_batch_pipeline.py -k cuda_real
```

The reserving test launcher must set `FIBER_COOPERATIVE_TEST_GPUS` and
`CUDA_VISIBLE_DEVICES` from its granted UUIDs before launching the child. Tests
compare all inputs, masks, metadata and target tensors exactly against serial
preparation at strides 1, 2 and 4, including resumed rank-local sample mapping.
They also run train→test→train through the same shared CPU workers, including
partial batches and ring wraps, without recreating those workers.
CPU-only installations can test coordination without initializing CUDA; actual
CUDA execution requires a CUDA-capable PyTorch build. No throughput improvement
is claimed by this extraction alone.
