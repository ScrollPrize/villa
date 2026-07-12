# Default Training Without Embedding Loss Task Log

## Implementation Notes

- Added `_embedding_channel_count(training)` so training/benchmark model
  construction creates embedding output channels only when
  `training.contrastive_enabled` is true.
- Removed contrastive keys from the standard `loader_example.json`; the example
  now represents the intended direction+presence training run.
- Kept contrastive support unchanged for explicit opt-in configs: enabling it
  still requires positive `contrastive_embedding_channels`.
- Updated specs and code-structure docs to describe contrastive embedding as
  explicit experimental opt-in.
- Added tests for standard example defaults, stale disabled embedding channel
  counts, and explicit enabled contrastive channel counts.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed: 191 tests.
- `git diff --check` passed.
