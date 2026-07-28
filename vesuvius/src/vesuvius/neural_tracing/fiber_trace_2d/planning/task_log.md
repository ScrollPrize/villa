# Task Log: Fail-Fast Native 3D Trace2CP Acceleration Comparison

## Implementation Notes

- Added `--debug-compare-normal-sampler[=sparse-corner-principal]` to the
  native 3D Trace2CP CLI.
- The comparison mode wraps the restored production geometry-loader normal
  sampler. The tracer still receives and scores with production normals.
- Added `_SparseCornerLasagnaNormalSampler` for sparse Lasagna
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
- Made `sparse-corner-principal` the default native 3D Trace2CP normal sampler
  via `--normal-sampler`; `baseline` remains an explicit fallback.
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
- User-provided no-debug command showed no acceleration because it still used
  `debug_compare_normal_sampler=off` and the old baseline normal sampler:
  `trace_wall_s=472.279`, `trace_candidate_normals=291.478`,
  `lasagna_normal_sample=290.463`.
- Updated the no-debug path so the same command now uses
  `normal_sampler=sparse-corner-principal`.
- No-debug sparse run before moving tensor decode to torch:
  `err/kvx=0.2`, `restarts=3`, `segments=87`, `err/m=19.6 (38.2mm)`,
  `trace_wall_s=374.490`. The remaining normal cost was outside zarr sampling:
  `trace_candidate_normals=191.216` while
  `sparse_normal_prefetch+sparse_normal_sample=5.716`.
- Ported sparse corner/tensor reconstruction and principal-axis power
  iteration to torch to remove the per-call NumPy conversion/decode path.
- No-debug sparse run after torch normal decode:
  `err/kvx=0.2`, `restarts=3`, `segments=87`, `err/m=19.6 (38.2mm)`,
  `trace_wall_s=200.708`. This is 2.35x faster than the user's
  baseline-normal no-debug run (`472.279s`) with the same restart count.
  Remaining top stages: `trace_candidate_score=146.309`,
  `inference_forward=38.637`, `field_sample_lookup=35.018`,
  `trace_candidate_normals=19.312`.
- Removed a progress-only GPU-to-CPU sync from point lookup and one root-mask
  sync from beam tracing. Rerun was effectively unchanged:
  `trace_wall_s=202.279`, `restarts=3`, so this was not the main remaining
  bottleneck.

## Deviations / Deferred Items

- The accelerated sparse corner/tensor sampler is now the default native 3D
  Trace2CP normal sampler. Baseline remains selectable with
  `--normal-sampler baseline`.
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
- Focused tests after torch principal-axis decode:
  6 passed, 155 deselected.
- `git diff --check`: passed.
