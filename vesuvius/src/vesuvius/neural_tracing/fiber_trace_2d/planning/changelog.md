# Changelog

## 2026-07-10

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
