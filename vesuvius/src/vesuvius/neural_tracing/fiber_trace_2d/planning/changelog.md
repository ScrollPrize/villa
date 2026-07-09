# Changelog

## 2026-07-09

- Unified 2D fiber-strip training, augment-vis, and prefetch around the shared torch-vectorized source-strip path; prefetch now covers the configured augmentation envelope through sampler-level cache-aware coordinate sampling.
- Added training-oriented 2D fiber-strip prefetch mode with `--prefetch`, `--prefetch-steps N`, and `--prefetch-steps 0` for all configured steps.
- Added the V0 2D fiber-strip direction training path with Lasagna ambiguous two-cos-channel targets, CP-local supervision, TensorBoard logging, and current/best snapshots.
