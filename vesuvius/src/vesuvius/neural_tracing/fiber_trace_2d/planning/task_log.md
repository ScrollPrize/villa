# Task Log: Training Benchmark And Profiling Mode

## Implementation Notes

- Added optional loader profile plumbing through `load_batch` and `build_sample`
  while keeping the default path unchanged when no profile dict is passed.
- Added `train.py --benchmark` for a 100-batch training-work benchmark that
  skips test evaluation, TensorBoard, run-directory creation, and snapshots.
- Added `train.py --profile` for per-batch stage timing rows plus a final
  milliseconds-per-CNN-patch summary.
- Added `train.py --load-only` for the benchmark path. It still runs
  deterministic sample selection, coordinate generation/augmentation, and
  volume sampling, but skips value augmentation and model training work.
- Added tests for loader profile collection and benchmark throughput reporting
  without training run-directory side effects.
- Added a test that load-only benchmark/profile reports zero image-augmentation,
  forward, and backward/step timing.

## Deviations

- None.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed: 67 tests.
- `git diff --check -- <touched files>` passed.
