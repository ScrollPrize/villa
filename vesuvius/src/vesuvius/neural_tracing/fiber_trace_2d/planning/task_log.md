# Task Log: Fail-Fast Native 3D Trace2CP Acceleration Comparison

## Implementation Notes

- Added `--debug-compare-normal-sampler[=sparse-corner-principal]` to the
  native 3D Trace2CP CLI.
- The comparison mode wraps the restored production geometry-loader normal
  sampler. The tracer still receives and scores with production normals.
- Added `_DebugSparseLasagnaNormalSampler` for debug-only sparse Lasagna
  sampling. It samples compact `nx/ny` only at the eight channel-grid corners,
  reconstructs the tensor/hint on those values, and decodes with the existing
  `_principal_tensor_axes` helper.
- Removed the invalid raw compact-normal path after confirming it reproduced
  the mismatch. Native 3D Trace2CP now has no debug or production path that
  interpolates raw compact `nx/ny` and then reads `FitData3D.normal_3d`.
- Added `_FailFastNormalComparisonSampler`, which raises immediately on
  valid-mask mismatch or on angular difference above
  `--debug-normal-angle-threshold-degrees`.
- Inverted the comparison wrapper return path so normal-aware smoothness now
  receives sparse corner/tensor normals after they pass comparison against the
  baseline sampler.
- The failure message includes sampler label, call number, point index,
  selected-level ZYX coordinate, valid flags, normals, and angle/threshold where
  applicable.
- Fixed the debug-only Lasagna streaming import path so `fit_data.py` can resolve
  its script-style `lasagna_volume` import without changing normal tracer runs.

## Debug Run Results

- Removed historical raw `normal_3d` debug path result: it failed fast on the
  first candidate-normal call, confirming why it must not exist. First
  mismatch:
  `point_index=2`, `angle_degrees=1.050298`,
  `point_zyx_selected=[4545.16796875, 5062.00439453125, 3406.501708984375]`,
  baseline normal `[0.30029154, 0.87094468, 0.38894793]`, accelerated normal
  `[0.31770453, 0.86614174, 0.38582677]`.
- `--debug-compare-normal-sampler=sparse-corner-principal --debug-normal-angle-threshold-degrees 1.0`:
  completed the whole 87-segment fiber without any normal mismatch above
  1 degree. This shows sparse chunk reads with baseline-style corner/tensor
  reconstruction can match the geometry-loader sampler at this threshold.
- Re-ran after removing the bad raw mode from the CLI/code:
  `err/kvx=0.7`, `restarts=11`, `segments=87`, `err/m=72.0 (12.7mm)`,
  `trace_wall_s=129.556`, no debug comparison failure.
- Next run target: same command, but now the trace is driven by the sparse
  corner/tensor normals instead of only comparing them.
- Accelerated-primary run with `--beam-lookahead-steps 1` completed without
  comparison failure but produced 11 restarts. This was the wrong comparison
  command for the 3-restart reference.
- Accelerated-primary run with the reference command's
  `--beam-lookahead-steps 2` completed without comparison failure and recovered
  the expected quality: `err/kvx=0.2`, `restarts=3`, `segments=87`,
  `err/m=19.6 (38.2mm)`, `trace_wall_s=679.960`.

## Deviations / Deferred Items

- The accelerated sparse corner/tensor sampler is still only selected by the
  debug comparison flag, but when selected it now drives tracing after passing
  fail-fast comparison.
- No long traces or JSON diff reports are written; the user requested fail-fast
  behavior instead.

## Validation

- `python -m py_compile lasagna/normal_encoding.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`: passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "normal_comparison or lasagna_normal_sampler or normal_aware_smoothness or cumulative_tangent_smoothness or trace_paths_sample_candidate_normals"`: 11 passed, 149 deselected.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py` after final cleanup: 158 passed, 2 skipped.
- `PYTHONPATH=vesuvius/src:lasagna:. python -m vesuvius.neural_tracing.fiber_trace_3d.trace2cp_tool --help | rg -n "debug-compare-normal-sampler|debug-normal-angle"`: passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "normal_comparison or lasagna_normal_sampler"` after the import fix: 4 passed, 156 deselected.
- Same focused test after removing the bad raw mode: 4 passed, 156 deselected.
- Same focused test after inverting the comparison wrapper to return the
  accelerated normals: 4 passed, 156 deselected.
- `git diff --check`: passed.
