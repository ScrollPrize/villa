# Full Test Trace2CP Evaluation Sentinel Plan

## Implementation

- Parse `training.test_control_points` as a non-negative integer instead of
  clamping it to at least one.
- Add a helper that resolves the effective test sample count:
  - positive values keep the existing subset size;
  - `0` resolves to `test_loader.sample_count`.
- Use the resolved count for both fixed test-batch direction loss and
  Trace2CP metric evaluation.
- Keep Trace2CP metric sample ordering as the deterministic training test
  order; when the count covers the whole test dataset, the averaged set matches
  all CP-to-next-CP pairs, independent of ordering.

## Spec Update

- Document `training.test_control_points: 0` as the full held-out test set
  sentinel.
- State that `best.pt` still uses averaged `test/trace2cp_error` when
  `test_datasets` is configured, now over the effective test sample count.

## Docs Updates

- Update `docs/code_structure.md` test config and snapshot notes.
- Update `planning/changelog.md`.
- Replace `planning/task_log.md` with current-task notes and validation only.

## Tests

- Add parser coverage for `test_control_points: 0` and negative rejection.
- Add a regression test that a config with `test_control_points: 0` evaluates
  all held-out test samples and records all valid CP-to-next-CP segments.
- Run the focused test file:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
