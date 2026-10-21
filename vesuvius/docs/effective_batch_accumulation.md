# Fixed effective-batch accumulation

`fiber_trace_3d.gradient_accumulation.NormalizedGradientAccumulator` is the
existing one-backward utility extracted from `3a4a7b2c1`. It accumulates a fixed
logical optimizer batch while allowing different microbatch sizes and rank
counts, without accumulating a separate model-sized gradient bank per loss.

This is a library extraction, not a switch of the current training CLI. The
production loss-statistics API and geometry-count staging are included; wiring
the training loop and cooperative worker lifecycle remain subsequent stages.
Shared [snapshot and in-RAM handoff state](training_state.md) is also available
for that later lifecycle integration.
No model defaults, sampling policy or loss objective changes.
The old component-by-component multiple-backward implementation is not included.

## Target-only normalization

Callers prepare all local normalization counts for one complete optimizer
update **before any model forward**. Counts may depend on input/GT and target
geometry, but not on other samples' predictions or how work was partitioned.

The three components are `direction`, `positive_presence` and
`negative_presence`. `LossStatistics` carries differentiable numerators and
detached counts, plus reporting-only angle/branch/score statistics. Its optional
embedding fields default to zero for source compatibility; this accumulator
rejects differentiable embedding numerators.

A presence numerator is a sum of per-nonempty-patch/group means, so its count
is the number of such groups, not necessarily its voxel count. Direction uses
the supervised channel-element count for its weighted squared-error sum. Counts
must use the same validity/mask and empty-group conventions as their numerators.
Do not substitute patch size, dense-volume size, nominal batch size or an
approximate count for the actual target support.

`targets.target_normalization_counts(batch, conditioned=False)` provides those
counts from materialized targets. It respects validity and presence masks. With
conditioned queries, both positive queries contribute direction/positive
groups and both dense queries contribute background groups, including positives
as in the existing conditioned objective. Counts do not inspect predictions.

`train.compute_losses(..., _return_statistics=True)` and
`train.compute_conditioned_losses(..., _return_statistics=True)` expose the
actual differentiable numerators from the shared production calculation. They
do not reconstruct sums by re-running masked loss functions. Without that
option they return the existing seven metric keys, using the shared
`_finalize_loss_statistics` helper. Ordinary multi-branch loss remains available
through the existing default API, but its statistics option raises because
routing can change group counts based on predictions. No embedding loss is
introduced by this stage.

Sums and direction counts use FP32 even when conditioned model outputs are
BF16. Otherwise the BF16 count representation can round a valid integer (for
example, 2,952 to 2,944) and disagree with the target prepass. Model forward and
elementwise error processing retain their configured precision. FP32 reduction
can change last-bit rounding relative to the former mean reductions; it does
not change the objective or weaken the exact count check.

## Geometry-only effective-batch staging

`sample_pipeline.MacroBatchTargetSource` wraps a leased CPU batch source before
`FiberTrace3DCudaBatchPipeline`. Supply the local microbatch sizes for one
complete optimizer update. It takes owned CPU copies of all tensor fields
before releasing source credits, reuses `materialize_targets` on the training
device with a storage-free final-shaped image, and attaches the aggregate local
`normalization_counts` to every microbatch. The prepass transfers validity and
geometry, never the actual CT image, and never calls a model. Deferred image
augmentation cannot change these counts. Target construction in the real CUDA
preparation path uses the same output stride and masks.

```python
staged = MacroBatchTargetSource(
    cpu_batches, config=loader_config, device=device, output_stride=output_stride,
    batch_sizes=local_microbatch_sizes, conditioned=False,
)
prepared = FiberTrace3DCudaBatchPipeline(
    staged, config=loader_config, device=device, output_stride=output_stride,
    value_augmentation_suffix="all",
)
```

CPU storage is bounded by one local optimizer batch, in addition to the
underlying queue and CUDA preparation window. Staged batches own their storage;
`release_current()` is therefore a no-op. `close()` closes the source and clears
staging. The source's `prepare_geometry(operation)` hook lets a future
cooperative caller delimit the geometry work; it defaults to calling the
operation directly. This hook is not a new manager integration.

This wrapper requires at least one local microbatch and a finite source must
end at a complete optimizer-update boundary. A rank with no samples bypasses
the wrapper and supplies zero counts while participating in accumulation
collectives. Do not change sample membership, refinement geometry or target
settings between the count prepass and forward. Counts are preserved through
batch slicing, pinning and transfer.

For each component `c`, the update gradient is:

```text
weight[c] / global_count[c] * sum_over_all_samples(gradient(numerator[c]))
```

Empty counts are clamped to one in the denominator; their numerators must be
zero. Constructor count reduction uses SUM across ranks. `add(statistics)`
combines the weighted normalized numerators and performs exactly **one**
backward per microbatch, with no collective communication. `reduce_and_assign()`
SUM-reduces gradients at the optimizer boundary, never averages them again.

The caller must zero gradients before constructing an accumulator. It also owns
autocast, any GradScaler initialization/scaling state, unscaling, clipping and
the single optimizer step after reduction. Master parameters must be float32;
forward/backward processing can still use BF16/FP16 autocast. The optional
`loss_scale` stays applied through reduction for the caller's usual AMP boundary.

Do not also enable normal DDP backward gradient synchronization: that would
communicate per microbatch and double-reduce or mis-scale these gradients.
All ranks construct/finalize once per update, including ranks with zero work.
An entirely unused parameter remains `grad=None`; one used on another rank
receives the globally reduced gradient even if locally unused.

The exact local count check and all-rank mismatch consensus come from the later
frozen `2192cc5ac` implementation. Each rank compares actual accumulated counts
with its own prepared counts in the same order. The mismatch flag shares the
reporting reduction so all ranks fail together before gradient assignment if
any rank's counts changed. No numerical tolerance is used for that check.

## Canonical sample membership

`effective_batch_partition(step, effective_batch_size, rank, world_size,
microbatch_size)` returns canonical sample-ID tuples for one update. Membership
is always `[step * effective_batch_size, (step + 1) * effective_batch_size)`;
physical ranks divide that interval, then split their slice into microbatches.
Nondivisible batch sizes and ranks with zero assigned samples are supported.

`sample_pipeline.VirtualBatchSampleIndex` preserves the existing logical-rank
batch stream when fewer physical ranks execute it. For example, six physical
ranks can execute eight logical batches of 16 samples, still consuming exactly
128 samples per update. Some physical ranks execute two logical batches while
others execute one. Virtual world size must be at least physical world size.
This mapping is a separate source API, not a new dataset shuffle policy.

## Model and mode restrictions

The existing guard rejects standard BatchNorm/SyncBatchNorm, dropout and RNN
modules because their state/randomness can depend on sample/rank placement.
This is not an exhaustive proof for arbitrary custom modules: callers must
also avoid their own cross-sample communication/state and random operations.
GroupNorm and other genuinely per-sample normalization can be used.

`validate_target_only_loss_config` is an opt-in validation helper for this
extraction stage. It rejects embedding and ordinary multi-branch training modes
pending their later compatible loss extraction. It does not change existing
inference support or silently disable an already-supported training CLI mode.
Conditioned counts/statistics are available through the production API above;
the current trainer loop is not yet wired to bounded accumulation.

## Validation

The source one-backward regression compares gradients and AdamW parameters with
a full effective batch of 128 samples for microbatches 1, 3, 16 and 128. Additional
tests cover sample partitions, zero-work ranks, unused parameters, FP16/BF16
autocast, packing/reporting and all-rank count mismatch failure:

```bash
cd vesuvius
python -m pytest -q tests/neural_tracing/test_effective_batch_accumulation.py \
  tests/neural_tracing/test_macro_batch_targets.py -k 'not cuda'
```

Real-data cases use the actual Paris4 annotation/CT fixture described in
[asynchronous preparation](async_training_preparation.md). A small seven-channel
pointwise head isolates accumulation behavior from architecture/convergence.
The tests consume the actual production statistics, rather than reconstructing
them through a test-only loss adapter. An independent per-patch formula verifies
the ordinary loss and gradients; conditioned cases exercise the existing query
semantics. Prepass tests cover masks, empty groups, strides 1/2/4, owned queue
slots, errors, repeated updates, exact FP32/BF16 counts and transfer metadata.
Two AdamW updates are compared with the full-batch gradient and parameters.

CUDA cases require an externally acquired GPU-manager reservation and UUID mask
for **two** GPUs, with `FIBER_COOPERATIVE_TEST_GPUS` set before child launch:

```bash
FIBER_ASYNC_CUDA_TEST=1 python -m pytest -q -s \
  tests/neural_tracing/test_effective_batch_accumulation.py \
  tests/neural_tracing/test_macro_batch_targets.py -k cuda
```

They check real Paris4 microbatch accumulation and NCCL reductions with unequal
and zero-work ranks, ordinary/conditioned prepass gradients and two AdamW updates
at strides 1/2/4, and count transfer through asynchronous GPU augmentation.
CPU distributed checks use Gloo. No external data are
modified or downloaded. Gradient/optimizer comparisons retain the source FP32
rounding tolerances; cross-device accumulation order is not bit-identical.
No full-trainer convergence or throughput result is claimed by this utility
extraction alone. Loss statistics/finalization are extracted from `dca6d0f2b`
and target staging from `3a4a7b2c1`, adapted to retain this branch's masks. The
shared count helper uses those same masked group conventions. Tests expose and
fix the source's inherited BF16 conditioned count rounding; no unrelated newer
losses, dataset changes or manager code are imported.
