# Deferred GPU value augmentation

This is an extraction of the direct Gaussian and batched image-value chain from
the existing Fiber3D development implementation (`29a7289c5`), not a new blur
algorithm. The common deferred dispatcher and shared-source collation come from
`2192cc5ac`. It is opt-in and leaves the current training CLI unchanged.

## Preparation flow

1. Existing CPU workers sample the CT and construct compact supervision metadata.
2. The same stage-two worker either finishes the legacy mask-aware CPU recipe,
   or returns the existing deferred augmentation parameters and blur kernels.
3. Existing shared tensor slots carry volume and validity arrays. IPC contains
   descriptors, not a second large-tensor transport path.
4. The asynchronous preparation thread pins those tensors, releases CPU leases,
   transfers them to its CUDA stream, completes the selected value suffix in
   float32 with autocast disabled, and materializes targets.

Both rank-local `FiberTrace3DTrainingBatchPipeline` and source-tagged
`FiberTrace3DSampleQueue` use the same factories and collation. Set
`defer_cuda_value_augmentation` to `"blur"` or `"all"` on the pipeline or
`SampleQueueSource`. Set the matching `value_augmentation_suffix` on
`FiberTrace3DCudaBatchPipeline`:

```python
source = SampleQueueSource(config, defer_cuda_value_augmentation="all")
# Construct the shared owner and its QueuedBatchView as usual.
prepared = FiberTrace3DCudaBatchPipeline(
    view, config=config, device=device, output_stride=1, depth=2,
    value_augmentation_suffix="all",
)
```

The application must already have a GPU-manager allocation. This utility does
not acquire resources or change manager protocol/capabilities. Local preparation
pause does not release GPU allocations; follow the cohort's full drain/unload
protocol for that.

`"blur"` computes complete-patch normalization, contrast, brightness, gamma and
seeded noise on CPU, then defers only blur. `"all"` defers that whole chain,
using the source implementation's compact uint8 CT staging; CUDA converts the
batch to float32 once. This mode is for the existing 8-bit CT sampling pipeline,
not arbitrary floating-point volumes. Shared validity arrays remain boolean and
unchanged. No implicit mixed-precision changes are introduced.

After completion, `deferred_augmentation_params` and `deferred_blur_kernels`
are cleared. CPU target-index metadata deliberately stays on CPU. Batch timing
adds `cuda_pipeline_value_augmentation_ms`, separately from transfer/targets.

## Existing kernel and image semantics

- Complete-patch normalization uses every sampled value, including zero-filled
  missing/out-of-bounds CT. It does not mask image statistics or zero the result
  after augmentation. Normalization, contrast, brightness, gamma, seeded noise,
  isotropic blur, then directional blur run in that order.
- Isotropic Gaussian kernels are normalized discrete 1D arrays, truncated at
  `ceil(3*sigma)`. Three grouped 1D convolution passes apply them independently
  per sample, centering shorter kernels in the batch's largest support.
- Each directional kernel is built directly from its oriented Gaussian formula
  in float64 and stored as normalized float32 weights. Its rectangular support
  follows the projected principal-axis radii. No runtime interpolation or
  sample-along-direction loop is used for this recipe.
- Directional grouping is the existing deterministic greedy policy: process
  descending kernel volume, break ties by sample index, and merge only when
  padded work is at most 1.25 times the sum of individual kernel volumes.
  Each sample retains its own weights; grouping never shares one sample's kernel
  with another. Groups use rectangular grouped `conv3d` and replicate borders.
- Noise is seeded by canonical sample identity, not transport order or batch
  partition. CPU and CUDA RNG implementations need not generate identical
  noise; replay and reordering on the same device preserve sample noise.

## Compatibility boundary

The default CPU `apply_value_augmentation_3d` retains its mask-aware statistics,
final masking and original oriented three-pass blur. Those are not mathematically
identical to the complete-patch direct-convolution recipe. The latter is exposed
as `apply_complete_value_augmentation_3d`; both deferred modes use that recipe.
Their native CPU prefix shares the original native transform helper with an
all-valid compatibility mask instead of adding another binding implementation.

Validity-aware targets remain unchanged even on the deferred image path. Source
development later removed validity from the whole trainer; that broader semantic
change is deliberately not made here. Likewise, this stage consumes existing
parameter descriptors without replacing the current loader's fixed blur-sigma
sampling or changing its RNG draw order. The later trainer/config integration
must carry any updated parameter-selection recipe explicitly.

This PR does not yet enable GPU augmentation in the production training CLI or
change model, loss, effective batch, annotation or refinement behavior.

## Validation

CPU formula/grouping and descriptor-transport checks:

```bash
cd vesuvius
python -m pytest -q tests/neural_tracing/test_gpu_value_augmentation.py -k 'not cuda'
```

CUDA tests run only inside a GPU-manager-reserved child with the granted UUID
mask and `FIBER_COOPERATIVE_TEST_GPUS` set before CUDA initialization:

```bash
FIBER_ASYNC_CUDA_TEST=1 python -m pytest -q -s \
  tests/neural_tracing/test_gpu_value_augmentation.py -k cuda
```

For the real-data integration cases, use the `VC_QUEUE_TEST_VOLUME` and
`VC_QUEUE_TEST_FIBER` fixture variables documented in
[async preparation](async_training_preparation.md). They compare direct and
asynchronous augmentation/targets on the same actual Paris4 annotated crops in
train→test→train phases, with partial ring wraps and persistent worker PIDs.
Direct and queued same-device results must be exact. CPU versus CUDA formula
checks retain the source regression's float32 rounding tolerances; they do not
assert identical RNG streams across devices. No throughput claim is added by
this extraction alone.
