# Task Log: Median TTA Line Tracing

## Implementation Notes

- Added `runner.py --med-tta` for `--line-trace-vis`.
- Switched line-trace TTA from fixed flip/rotation/shift variants to a shared
  random geometric TTA set sampled from regular training augmentation ranges.
  The flock and median trace use the same generated TTA fields.
- Added `--line-trace-tta-count` with default `100`; `--med-tta-count` remains
  an alias for the same count.
- Added reference-space median-TTA tracing. Each step samples the reference and
  random TTA direction fields, transforms TTA orientations back to reference
  space, resolves ambiguous sign against the previous step, and steps along the
  normalized median direction.
- The existing two-column line trace visualization remains unchanged unless
  `--med-tta` is supplied; median TTA adds a third column.
- Removed hardcoded CPU coordinate generation from non-prefetch center-patch and
  direct chunk-request paths. Prefetch dependency generation remains CPU-only.
- Added focused tests for median-TTA reference-space tracing, ambiguous
  direction sign handling, and source-grid point/direction mapping used by
  random geometric TTA.

## Deviations

- None.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed: 69 tests.
- `git diff --check -- <touched files>` passed.
