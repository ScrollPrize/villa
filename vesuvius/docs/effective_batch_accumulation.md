# Fixed effective-batch accumulation

`fiber_trace_3d.gradient_accumulation.NormalizedGradientAccumulator` is the
existing one-backward utility extracted from `3a4a7b2c1`. It accumulates a fixed
logical optimizer batch while allowing different microbatch sizes and rank
counts, without accumulating a separate model-sized gradient bank per loss.

This is a library extraction, not a switch of the current training CLI. Full
geometry-count staging, model/loss integration and cooperative worker lifecycle
remain subsequent stages. No existing model defaults or loss behavior change.
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
Conditioned models are not rejected by configuration; the caller must supply
their corresponding target-only query counts/statistics. This stage does not
wire a conditioned loss adapter into the trainer either.

## Validation

The source one-backward regression compares gradients and AdamW parameters with
a full effective batch of 128 samples for microbatches 1, 3, 16 and 128. Additional
tests cover sample partitions, zero-work ranks, unused parameters, FP16/BF16
autocast, packing/reporting and all-rank count mismatch failure:

```bash
cd vesuvius
python -m pytest -q tests/neural_tracing/test_effective_batch_accumulation.py -k 'not cuda'
```

Real-data cases use the actual Paris4 annotation/CT fixture described in
[asynchronous preparation](async_training_preparation.md). A small seven-channel
pointwise head isolates accumulation behavior from architecture/convergence.
The test adapter obtains direction, positive and background terms from the
unchanged production loss, rather than adding another production loss function.
Two AdamW updates are compared with the full-batch gradient and parameters.

CUDA cases require an externally acquired GPU-manager reservation and UUID mask
for **two** GPUs, with `FIBER_COOPERATIVE_TEST_GPUS` set before child launch:

```bash
FIBER_ASYNC_CUDA_TEST=1 python -m pytest -q -s \
  tests/neural_tracing/test_effective_batch_accumulation.py -k cuda
```

They check real Paris4 microbatch accumulation and NCCL reductions with unequal
and zero-work ranks. CPU distributed checks use Gloo. No external data are
modified or downloaded. Gradient/optimizer comparisons retain the source FP32
rounding tolerances; cross-device accumulation order is not bit-identical.
No full-trainer convergence or throughput result is claimed by this utility
extraction alone.
