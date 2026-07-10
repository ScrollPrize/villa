# Task Log: Opt-In Augment-Vis Profiling

## Implementation Notes

- Added `--augment-profile` for `runner.py --augment-vis`.
- Default augment-vis export no longer passes profile dictionaries, uses no-op
  timers, and does not print augment timing tables or output timing lines.
- Profile mode repeats the same augment entries twice using the same CP-local
  source geometry:
  - pass 1 reports cold/first-use costs;
  - pass 2 reports warmed costs.
- Both profile tables retain `total/no-first` and `avg/no-first`.
- Augmentations are still processed one patch at a time; batching was not
  changed in this task.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed: 71 tests.
