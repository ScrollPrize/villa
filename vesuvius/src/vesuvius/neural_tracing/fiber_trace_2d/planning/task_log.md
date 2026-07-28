# Task Log: Fail-Fast Native 3D Trace2CP Acceleration Comparison

## Implementation Notes

- Added `--debug-compare-normal-sampler[=MODE]` to the native 3D Trace2CP CLI.
  Supported modes are `all`, `sparse-direct`, and `sparse-corner-principal`.
- The comparison mode wraps the restored production geometry-loader normal
  sampler. The tracer still receives and scores with production normals.
- Added `_DebugSparseLasagnaNormalSampler` for debug-only sparse Lasagna
  sampling:
  - `sparse-direct` samples `grad_mag/nx/ny` at candidate points and reads
    `FitData3D.normal_3d`, matching the kind of direct sparse path that caused
    problems.
  - `sparse-corner-principal` samples `nx/ny` at the eight channel-grid corners,
    reconstructs the tensor/hint on those values, and decodes with the existing
    `_principal_tensor_axes` helper. This isolates sparse read/coordinate
    differences from direct compact-normal interpolation differences.
- Added `_FailFastNormalComparisonSampler`, which raises immediately on
  valid-mask mismatch or on angular difference above
  `--debug-normal-angle-threshold-degrees`.
- The failure message includes sampler label, call number, point index,
  selected-level ZYX coordinate, valid flags, normals, and angle/threshold where
  applicable.
- Fixed the debug-only Lasagna streaming import path so `fit_data.py` can resolve
  its script-style `lasagna_volume` import without changing normal tracer runs.

## Debug Run Results

- `--debug-compare-normal-sampler=sparse-direct --debug-normal-angle-threshold-degrees 1.0`:
  failed fast on the first candidate-normal call. First mismatch:
  `point_index=2`, `angle_degrees=1.050298`,
  `point_zyx_selected=[4545.16796875, 5062.00439453125, 3406.501708984375]`,
  baseline normal `[0.30029154, 0.87094468, 0.38894793]`, accelerated normal
  `[0.31770453, 0.86614174, 0.38582677]`.
- `--debug-compare-normal-sampler=sparse-corner-principal --debug-normal-angle-threshold-degrees 1.0`:
  completed the whole 87-segment fiber without any normal mismatch above
  1 degree. This isolates the first confirmed divergence to the direct sparse
  `normal_3d` style decode, not to sparse chunk reads in general at this
  threshold.

## Deviations / Deferred Items

- The accelerated samplers are debug-only and are not used for production
  scoring. This is intentional so the restored baseline metric stays unchanged
  while we localize acceleration differences.
- No long traces or JSON diff reports are written; the user requested fail-fast
  behavior instead.

## Validation

- `python -m py_compile lasagna/normal_encoding.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`: passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "normal_comparison or lasagna_normal_sampler or normal_aware_smoothness or cumulative_tangent_smoothness or trace_paths_sample_candidate_normals"`: 11 passed, 149 deselected.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`: 158 passed, 2 skipped.
- `PYTHONPATH=vesuvius/src:lasagna:. python -m vesuvius.neural_tracing.fiber_trace_3d.trace2cp_tool --help | rg -n "debug-compare-normal-sampler|debug-normal-angle"`: passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "normal_comparison or lasagna_normal_sampler"` after the import fix: 4 passed, 156 deselected.
- `git diff --check`: passed.
