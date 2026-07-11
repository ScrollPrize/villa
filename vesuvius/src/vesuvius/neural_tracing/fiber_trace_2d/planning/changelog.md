# Changelog

## 2026-07-10

- Switched 2D `--dir-vis` direction overlays to 8x8 display-pixel cells with
  6-pixel anti-aliased direction segments.
- Added real process CPU timing to the 2D fiber-trace benchmark/profile table
  so loader summed-worker timing can be compared against actual CPU usage.
- Parallelized the 2D CUDA training load+prepare path by submitting exact training steps to concurrent workers, added one-offset strip-cache and CP-tangent/no-coordinate-retention fast paths, and documented measured throughput limits.
- Removed the 2D training default patch-count warning, added a real
  `batch_size`/`control_points_per_step` validation error, and switched
  variable-sigma value-augmentation blur to the measured unfold-based batched
  implementation.
- Set the measured default CUDA training pipeline queue to 8 batches, default
  whole-batch loader workers to 4, added a startup print for effective
  pipeline settings, and removed invalid-sample skip reason spam from hot
  training loads.
- Added configurable concurrent whole-batch CUDA training loaders and moved
  CUDA preparation submission into a background preparation executor.
- Added CUDA side-stream training preparation for deferred image/value
  augmentation, normalization, and supervision tensors, plus prep/outside
  timing diagnostics.
- Added `--trace2cp-vis` runner inspection for CP-pair segment tracing,
  optional median-TTA scoring, normalized trace-to-target score output, and
  single-panel JPG/summary export.
- Added CUDA training batch pipelining with bounded deterministic whole-batch
  futures, deferred torch value augmentation, and profile wait timing.
- Removed avoidable warm-path loader threading overhead by prewarming cached
  deterministic random pass orders and reusing a persistent CP worker executor
  across `load_batch` calls.
- Slimmed strip-coordinate cache reads/writes by using zyx source coordinates
  and deriving xyz tensors on load while preserving the existing cache key
  identity for supported entries.
- Batched 2D strip-z image loading through `CoordinateSampler.sample_coord_batch`
  and added `loader_workers` CP-level parallel `load_batch` construction with
  deterministic accepted-sample ordering.
- Added 2D loader profile wall/work/threading-factor reporting so parallel
  loader timings are not confused with summed worker timings.
- Batched 2D strip training sample augmentation across strip-z offsets:
  sparse line/CP lookup now uses direct batched bilinear gather, coordinate
  augmentation stacks fused maps, and post-load value augmentation runs on the
  loaded image stack.
- Replaced formula-based 2D strip point mapping with concrete cached
  `backward_map_xy`/`forward_map_xy` tensors so coordinate augmentation and
  line/CP mapping use the same fused geometric augmentation maps.
- Cached fused 2D strip augmentation transform constants/smooth controls,
  batched line+CP mapping, and reused transform/line mappings across matching
  strip-z offset patches.
- Added paired forward/backward strip augmentation transforms and routed
  smooth line/CP mapping through direct source-coordinate forward/backward
  point transforms instead of iterative or dense nearest-grid inversion.
- Cached source-space line/CP coordinates in strip-coordinate cache v2 and split coordinate profiling into descriptor/cache/source/line columns.
- Added configurable CP-local strip-coordinate caching shared by training, visualization, and prefetch source-grid construction.
- Kept 2D fiber-strip source grids, strip-z offsets, coordinate augmentation, and line/control-point transforms torch-native until explicit VC3D/export NumPy boundaries.
- Added opt-in `--med-tta` runner line tracing that uses per-step median directions across reference and fixed TTA direction fields.
- Added 100-batch `train.py --benchmark`, `--profile`, and `--load-only` modes for patch-throughput, per-stage timing, and loader-only diagnostics.
- Switched the default 2D fiber-strip direction model to a 10-block, 64-channel residual CNN.
- Switched the default 2D direction ResNet normalization to 8-group GroupNorm.
- Added `--dir-vis` runner direction-field inspection for checkpointed per-pixel direction predictions.
- Added the V0.1 runner line-tracing inspection mode for tracing checkpointed direction predictions on one deterministic side-strip patch.
- Added fixed test-time augmentation flock visualization to the V0.1 line-trace runner output.
- Added folded unoriented direction angle-error reporting in degrees for 2D fiber-strip train/test output.
- Fixed prefetch `idx` progress semantics so it reports the cache-complete safe sample prefix rather than dependency-generation progress.
- Added `prefetch_sampler_workers` to tune dependency producer concurrency separately from download worker concurrency.
- Added `training.max_sample_index` for bounded deterministic-prefix reuse and prefetch `idx` progress reporting.
- Restored 2D fiber-strip training/prefetch to deterministic pseudo-random full-dataset CP passes instead of flat sequential CP order.
- Switched 2D fiber-strip training/prefetch to flat CP order and made explicit `--prefetch-steps` override configured `training.max_steps`; `--prefetch-steps 0` now covers every configured training/test CP once.
- Added 2D fiber-strip `test_datasets` evaluation with deterministic held-out batches and test-loss current/best snapshot cadence.
- Added `--augment-vis` contact-sheet CP crosshairs and separate label bands so labels no longer cover image pixels.
- Fixed 2D fiber-strip affine augmentation composition so shift is applied in output/scaled space and image sampling stays aligned with line/control-point coordinates.

## 2026-07-09

- Tightened 2D fiber-strip prefetch so VC3D only reports dependency metadata while Python handles direct-source uncompressed chunk downloads, atomic cache writes, zero-byte `.empty` markers, and retry/progress behavior.
- Replaced 2D fiber-strip prefetch sampling/discarding with dependency-only chunk discovery, Python-side VC3D cache classification, `.empty` missing markers, bounded parallel fetching, and compact progress output.
- Unified 2D fiber-strip training, augment-vis, and prefetch around the shared torch-vectorized source-strip path; prefetch covers the configured augmentation envelope through dependency-only chunk discovery.
- Added training-oriented 2D fiber-strip prefetch mode with `--prefetch`, `--prefetch-steps N`, and `--prefetch-steps 0` for all configured steps.
- Added the V0 2D fiber-strip direction training path with Lasagna ambiguous two-cos-channel targets, CP-local supervision, TensorBoard logging, and current/best snapshots.
