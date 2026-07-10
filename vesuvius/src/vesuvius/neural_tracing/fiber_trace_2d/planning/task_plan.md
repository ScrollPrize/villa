# Task Plan: Test Dataset Evaluation And Snapshots

## Scope

Add a validation/test dataset path to the existing V0 2D trainer. Keep the
existing training batch construction, model, supervision, and augmentation
semantics unchanged.

## Plan

- Extend `FiberStripTrainingConfig` with:
  - `test_interval`: positive step interval for test evaluation and snapshots.
  - `test_control_points`: deterministic CP samples per test evaluation.
  - `test_start_sample_index`: deterministic starting sample index for the
    fixed test batch.
- Extend raw config handling with top-level `test_datasets`. When present,
  instantiate a second `FiberStrip2DLoader` using the same loader settings but
  with `datasets` replaced by `test_datasets`.
- Add a no-grad evaluation helper that runs one deterministic test batch and
  returns loss/supervision count/visualization outputs.
- Run test evaluation at step 1, every `test_interval`, and the final step when
  test data is configured.
- Write `snapshots/current.pt` at the same test cadence when test data is
  configured. Continue to write current at step 1/final for train-only runs.
- Write `snapshots/best.pt` from lowest test loss when test data is configured;
  keep training-loss best only when no test dataset is configured.
- Log test scalars and test visualization to TensorBoard when enabled.
- Update the example config with `test_datasets` and test interval keys.

## Spec Update

Update `planning/specs.md` to describe `test_datasets`, deterministic test
evaluation, test-triggered current snapshots, and validation-loss best
snapshots.

## Docs Updates

Update `docs/code_structure.md` training/config documentation with the new test
dataset keys and snapshot behavior.

## Testing

- Add focused tests for:
  - config parsing of test settings;
  - `test_datasets` replacing training datasets for the test loader;
  - test-interval snapshots and best snapshot metadata based on test loss.
- Run:
  - `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`

## Changelog

Add a one-line changelog entry because this changes training configuration and
snapshot semantics.
