# Task Log: Native 3D Trace2CP Metric Output Formatting

## Implemented

- Changed human `err/kvx` output in native 3D whole-fiber progress, partial
  visualization status, and final metric lines to one decimal place.
- Changed human `err/m` output to one decimal place and added the
  parenthesized mean traced run length in millimeters when VC3D voxel-size
  metadata is available.
- Kept JSON summary metrics full precision.
- Updated the specs and targeted tests for the new stdout format.

## Deviations / Deferred Items

- None.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "whole_fiber_progress_reports_compact_error_units_when_known or whole_fiber_error_format_helpers_use_one_decimal"`
  passed: `2 passed, 148 deselected`.
- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "whole_fiber_trace or whole_fiber_progress or whole_fiber_error_format or native_3d_whole_fiber"`
  passed: `12 passed, 138 deselected`.
