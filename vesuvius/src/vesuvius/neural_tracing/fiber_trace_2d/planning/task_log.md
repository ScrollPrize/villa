# Training Batch Config Validation And Prep Slowdown Task Log

## Implementation Notes

- Removed the hard-coded `expected 64` training warning. Non-default flattened
  patch counts now run silently.
- Added `_validate_training_batch_config()` and call it from training and
  benchmark paths. Top-level `batch_size` is the CP-sample batch size and must
  match `training.control_points_per_step`.
- Replaced per-patch batched-value Gaussian blur loops with
  `_gaussian_blur_2d_batch()`, using grouped separable convolutions. This keeps
  different blur sigmas per patch without launching one vertical and one
  horizontal convolution per patch.
- Added regression tests for batch config mismatch and batched blur equivalence
  against the single-patch path.
- Updated specs, docs, and changelog.

## Benchmark Notes

- Before the grouped blur change, warm benchmark/profile rows commonly showed
  `prep`/`prep_gpu` around `8-15 ms/patch`, which corresponds to roughly
  one second or more of CUDA preparation per 128-patch step.
- After grouped blur, common warm rows dropped to about `0.25-2.1 ms/patch`.
  Heavy augmentation rows still appear around `7 ms/patch`.
- The benchmark command was interrupted after enough rows for comparison rather
  than waiting for all 100 batches.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/augmentation.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed: 106 tests.
