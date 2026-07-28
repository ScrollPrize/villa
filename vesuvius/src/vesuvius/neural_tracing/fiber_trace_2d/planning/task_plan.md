# Plan: Native 3D Whole-Fiber Restart Metric Units

## Goals

- Replace the primary whole-fiber restart metric
  `restart_count / segment_count` with `restart_count / reference_length_kvx`.
- Measure reference length along the original loaded fiber line, in the same
  selected-level voxel coordinate system used by native 3D tracing.
- If explicit physical voxel-size metadata is available, additionally report
  `restart_count / reference_length_meters`.

## Non-Goals

- Do not change tracing, restart detection, error thresholds, fusion, or
  visualization behavior.
- Do not infer physical size from volume filenames.

## Implementation Steps

1. Extend `NativeWholeFiberResult` with reference length and normalized restart
   metric fields.
2. Compute the reference length from the original fiber-line arc between CP0
   and the final CP.
3. Add physical-length support from explicit dataset/record/OME metadata:
   base voxel-size keys, selected-level voxel-size keys, or OME multiscales
   with spatial units.
4. Update whole-fiber stdout and JSON summary to use
   `native_trace2cp_fiber_restarts_per_kvx` as the primary metric and emit
   `native_trace2cp_fiber_restarts_per_meter` only when available.
5. Include the same per-meter physical unit in live whole-fiber progress and
   partial visualization status when explicit voxel-size metadata is available.
6. Keep the old segment fraction only as explicitly named
   `restart_fraction_per_segment` in the JSON summary.
7. Update tests and docs.

## Spec Update

- Native 3D whole-fiber Trace2CP metric is restarts per kvx, not restarts per
  segment.
- Optional restarts per meter is reported only with explicit physical voxel
  size metadata.
- Whole-fiber progress output includes the per-meter metric and meter length
  while running, plus `physical_unit=m`, when the same metadata is available.

## Docs Updates

- Update `docs/code_structure.md` native 3D Trace2CP metric description.

## Validation Commands

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "whole_fiber_trace"`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
- `git diff --check`

## Changelog Update

- Add one changelog line for the length-normalized native 3D whole-fiber
  restart metric.
