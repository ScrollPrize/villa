# Local Fiber3D metric embeddings

## Normal validity

Invalid normals (including missing chunks and stored zero grad_mag) cause bounded
whole-sample replacement within the same scroll/sampling group. No per-voxel
mask is added. Retries keep canonical queue IDs and deterministic seeds; after
128 replacements the loader fails loudly. Network/configuration errors propagate.

## Fresh combined-data training

Config: `fiber_trace_3d/configs/train_combined_5scroll_s1a_embedding_ema_16x8.json`.
Five scrolls have equal probability. Paris4's JSON collection (both directories
together) and S1a NML have equal conditional weight:10% each of all training
samples; the other four scrolls each20%. Existing arclength sampling, dense
supervision and held-out test sets are retained.

Fresh initialization, LR0.01 with100-update warmup, global batch128, microbatch
cap16/GPU,128³ input, GroupNorm, stride4, BF16,2M-update cap. EMA0.99/0.999,
tests/checkpoints every100updates and historical observational metrics remain.
The LR is inherited from nonembedding tuning, not a retuned embedding optimum.

Run through the default cooperative supervisor (not torchrun):

```bash
/home/hendrik/.venv_las/bin/python -m vesuvius.neural_tracing.fiber_trace_3d.train /home/hendrik/villa/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/train_combined_5scroll_s1a_embedding_ema_16x8.json
```

The manager auto-starts if necessary. Missing config-matched memory measurements
trigger exclusive calibration using real test/training work, followed by an
adaptive registration with RAM state retained. No separate manual calibration,
manager configuration or unmanaged fallback is needed. The embedding config
cannot reuse nonembedding calibration. Default socket selection and pause/resume
follow [cooperative training](cooperative_training.md).
Outputs are under `/ephemeral/hendrik/fiber3d_train2/runs`.

## Objective

```json
"embedding": {"enabled": true, "channels": 16, "pairs_at_128": 65536, "weight": 1}
```

- Each128³patch reserves32768positive and32768negative pairs. The total scales
  linearly with maximum input dimension, not voxel count.
- Positive endpoints are independently sampled by arclength on the selected
  trace, with replacement, regardless of predicted presence.
- Negative output-grid endpoints are sampled with replacement proportional to
  detached presence after exclusion. Weight1 is10times weight0.1; zero is never
  sampled. No eligible positive weight produces zero negative numerator, **but
  retains the configured negative normalization budget**. No uniform fallback.
- Loss: `weight * (sum(1-cos_positive)/positive_budget + sum(relu(cos_negative))/negative_budget)/2`.
  Budgets sum over the effective optimizer batch and depend only on supplied
  geometry/configuration. Other patches' predictions cannot alter loss weights.
  Empty GT support is an error; pairs never cross patches.
- The existing target prepass appends the two budgets, globally sums counts
  before forward, and performs one combined backward per microbatch for all
  five loss components. No extra component gradient banks/forward replay.
  At update end, observed counts must equal prepared counts.

This intentionally changes the old all-zero-negative case: remaining patches
are no longer upweighted when another patch has zero eligible presence.

## Geometry and normal data

Every training/test dataset needs a matching local `lasagna_manifest_path`
with base shape and grad_mag/nx/ny channels. Relative paths resolve against the
training config. A channel may name an explicit remote Zarr URL; the shared
native LasagnaDataset resolves it using the existing `volume_cache_dir` root.
`configs/embedding_normals/` binds four published normal predictions;
Paris4 uses its existing complete local normal inference. No new cache format
or manifest download mechanism is introduced.

Output coordinates are mapped through inverse augmentation to base coordinates.
Find the nearest continuous annotated segment point and sample its surface normal.
Exclude the inclusive ellipse with20base-voxel normal radius and80base-voxel
tangential radius. Transforms, source scale, rotation/scale and nonlinear warp
are included. Invalid required normals replace the sample, not the exclusion
with a spherical fallback. `vc.volume.NormalSampler` calls the exact existing
C++ LasagnaNormalSampler: native chunk cache, decoded corner tensors and
principal-axis interpolation (safe across encoding sign discontinuities).
Sampling releases the GIL and uses one native sampling thread per queued loader
worker. Each worker retains one sampler per manifest, not per fiber. Normal
validity is checked before fetching the CT patch, avoiding wasted rejected reads.

Python API: `vc.volume.NormalSampler(manifest, cache_root).sample(points_xyz,
threads=1)` takes contiguous float32 `[N,3]` base coordinates and returns float32
XYZ unit axes plus boolean validity. It uses the native manifest contract: normal
channels reference 3D ZYX arrays with explicit level paths. Unsupported channel
layouts are errors, never a switch to the older Python reader.

Loader workers build eligibility in bounded blocks. It travels through existing
shared slots/pinned transfer. Macro-batch staging owns its nested eligibility
tensors before releasing shared slots. Pair lookup, FP32 cosine arithmetic and
gradients run on the GPU; model BF16 execution is unchanged. Seeds are per stream,
reduced modulo `2**63-1`; partitioning does not change them. Low-order changes in
predicted presence can still alter random choices near a sampling boundary.

## Scalars and exports

Console/TensorBoard expose embedding total/positive/negative and
`embedding_negative_pairs`. The latter now means **normalization budget**, not
actually drawn pairs. Component scalars include configured embedding weight.
Raw/EMA testing and snapshots include the head. Historical observation retains
its original direction/presence-only metric through the full-grid wrapper.

```python
result = model(volume, return_embeddings=True)
prediction = result["prediction"]
embedding = torch.nn.functional.normalize(result["embeddings"].float(), dim=1)
```

Default forward and volume inference retain direction/presence products.
No embedding Zarr or Atlas export. Conditioned embeddings and ordinary
multi-branch training remain unsupported.

## Validation

Approved mutable launcher: `vesuvius/scripts/run_fiber3d_cooperative_tests`;
select `embedding`, `embedding-data`, `cuda-embedding` in its JSON control.
The data suite explicitly opts into real host data; ordinary CI does not fetch it.
CUDA suite holds a manager-exclusive lease and tests two GPUs (FP32 and BF16),
with per-worker8GiB allocator caps. CPU tests include three Gloo ranks with one
zero-work rank, mixed excluded/supported patches, full-versus-split gradients,
one backward per microbatch, deterministic/detached sampling, queue mask ownership,
normal geometry and checkpoint compatibility. FP32 oracle disables TF32 only in
the test. BF16 gradient comparison allows expected low-order batch/reduction
differences; measured relative L2 was0.0021734. No production precision change.

Embedding throughput/quality must be measured on the actual new run; these
correctness tests do not establish a negligible training overhead.

2026-09-13 installed-binding checks:49 CPU tests, all configured real-data sources
and a two-worker shared queue passed; both manager-exclusive GPU tests passed.
Warm synthetic32768-point sampling (one CPU core,1warmup+3measured) took
4.974/5.015/5.225ms min/median/max natively versus118.130/118.343/118.674ms with the
former Python/Zarr sampler. This isolates sampling, not full training throughput.
Logs: `/ephemeral/hendrik/fiber3d_train2/cooperative-tests/20260913_094306_012315`.

Binding rebuild/install used in this workspace:

```bash
CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 AGENTS_AGENT_MODE=1 AGENTS_ALLOW_INSTALL=1 uv pip install --python /home/hendrik/.venv_las/bin/python --no-deps --reinstall -e ./volume-cartographer
```
# Local contrastive competition (2026-09-13)

New fresh config: `configs/train_combined_5scroll_s1a_contrastive_ema_16x8.json`.
Same combined five-scroll data, Paris4 JSON/NML10% each, LR0.01,128 effective
batch, EMA99/999 and existing tests. Legacy configs retain pairwise loss.

At128³,2048 GT anchors each have one GT positive and32 presence-sampled
off-GT negatives:65536 negatives plus2048 positives. The normal ellipse and
native loading stay unchanged. Local competition is
`logsumexp([0, (negative_cosine-positive_cosine+0.2)/0.1])`, averaged by the
fixed number of anchors. Temperature/margin/negatives_per_anchor are explicit
embedding config settings. There is no additional positive margin loss.

Counts are known from geometry before forward; a patch with no eligible positive
presence contributes zero loss with its budget intact. One backward/microbatch,
no inter-sample softmax or other prediction-dependent normalization.

Test and test_ema_* TensorBoard groups include `embedding_similarity/` means,
positive_patch_p05, negative_patch_p95/p99, negative_above_0p9,
hardest_negative_mean and positive_hardest_gap. Quantiles are averaged per patch;
they are not pooled percentiles. `embedding_similarity_count/` gives the patch
coverage per metric; undefined negative metrics are excluded, not treated as0.
No extra forward is needed. Training does not calculate these test diagnostics.

```bash
/home/hendrik/villa/vesuvius/scripts/run_fiber3d_experiment production-start
```

The approved runner's production config now selects the contrastive config;
it starts through the manager and does not resume old weights. GPU validation
uses `run_fiber3d_cooperative_tests` with the cuda-embedding suite.

For a read-only audit of the latest production run's raw/EMA similarity tags:

```bash
/home/hendrik/villa/vesuvius/scripts/run_fiber3d_experiment production-embedding-metrics
```

Validation: 55 CPU tests passed (7 gated skips); five manager-exclusive CUDA
tests passed, including FP32/BF16 accumulation equivalence. Isolated one-GPU,
one-patch FP32 embedding forward/backward (65536 negatives, 16 channels,
32³ output; five warm measurements) took 2.363ms median versus 2.486ms for
the old pairwise loss. This is not a full training-throughput measurement.
