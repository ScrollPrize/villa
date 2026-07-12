# Trace2CP Metric Stdout And Presence Visualization Task Log

## Result

- Changed single-pair Trace2CP CLI output so the selected metric prints as a
  standalone `trace2cp_error=<value>` line before the diagnostics line.
- Added optional fixed-scale presence probability visualization when
  `--trace2cp-use-presence` activates presence scoring:
  single-pair `trace2cp_vis.jpg` gets a presence column and whole-fiber
  `trace2cp_fiber_vis.jpg` gets a presence row.
- Updated Trace2CP docs/specs for the stdout format and presence debug panel.
- Added focused overlay regression tests for the single-pair presence column
  and whole-fiber presence row.

## Deviations

- None.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - Result: `183 passed in 7.78s`.
- `git diff --check`
  - Result: clean.
