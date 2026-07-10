# Task Log: Test Dataset Evaluation And Snapshots

## Result

- Added `training.test_interval`, `training.test_control_points`, and
  `training.test_start_sample_index`.
- Added top-level `test_datasets`; it reuses the normal loader config with only
  `datasets` replaced.
- Added deterministic no-grad test evaluation at step 1, every
  `training.test_interval`, and the final step when `test_datasets` is present.
- Changed snapshot behavior with `test_datasets`: `current.pt` is written on
  test evaluation steps, and `best.pt` is selected only by test loss. Train-only
  runs keep existing `checkpoint_interval` and train-loss best behavior.
- Added test TensorBoard scalars/cache diagnostics and test batch visualization
  on test evaluation steps.
- Updated `loader_example.json`, specs, code docs, and changelog.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`

Both passed; focused pytest result was `42 passed in 3.03s`.
