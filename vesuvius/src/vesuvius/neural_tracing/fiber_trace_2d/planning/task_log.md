# Task Log: Native 3D Whole-Fiber Restart Metric Units

## Implementation Notes

- Replaced the primary native whole-fiber metric with
  `native_trace2cp_fiber_restarts_per_kvx`.
- Reference length is measured along the original loaded fiber line from the
  first to last control point, in selected-level voxels.
- Added optional physical-length reporting from explicit metadata:
  `native_trace2cp_fiber_restarts_per_meter` and `reference_length_meters`.
- Whole-fiber live progress and partial visualization status now include the
  per-meter metric and `physical_unit=m` when explicit physical voxel-size
  metadata is available.
- Physical voxel size is read from explicit dataset/record voxel-size keys or
  OME multiscales with spatial units. The implementation does not parse voxel
  size from filenames.
- The old segment fraction is retained only in the JSON summary as
  `restart_fraction_per_segment`.

## Deviations / Deferred Items

- None.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "whole_fiber_trace"`
  passed: 5 passed, 141 deselected.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  passed: 144 passed, 2 skipped.
- `git diff --check` passed.
