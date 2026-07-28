# Plan: Fail-Fast Native 3D Trace2CP Acceleration Comparison

## Implementation

1. Reintroduce the accelerated Lasagna-normal sampler as debug-only code.
   - Load Lasagna streaming data only when the comparison flag is set.
   - Use sparse `FitData3D.grid_sample_fullres(...)` for `grad_mag`, `nx`, and
     `ny` at the exact candidate points.
   - Provide at least a direct sparse-normal variant so we can reproduce and
     localize the suspected divergent path.

2. Add a fail-fast comparison wrapper.
   - Wrap the restored `_NativeLasagnaNormalSampler` as the primary sampler.
   - Call debug alternate samplers on the same points.
   - Return primary normals to the tracer so the traced metric stays baseline.
   - Raise `ValueError` immediately on valid-mask mismatch or angle difference
     above the configured threshold.

3. Add CLI controls.
   - `--debug-compare-normal-sampler` enables the parallel comparison.
   - `--debug-normal-angle-threshold-degrees` controls angular fail threshold.

4. Keep diagnostics compact.
   - Error messages include path label, call index, local point index, selected
     coordinate, baseline/alternate valid flags, baseline/alternate normals, and
     angle in degrees.
   - No long trace or large report is written.

## Spec Update

- Add a debug-only native 3D Trace2CP comparison mode that is explicitly
  non-production and fail-fast.
- Reaffirm that default scoring uses the restored geometry-loader sampler.

## Docs Updates

- Update planning specs and task log/status.
- Add a changelog note for the debug comparison tool.

## Tests

- Add focused unit tests for the fail-fast wrapper:
  - valid mismatch raises.
  - angular mismatch above threshold raises.
  - below-threshold comparisons return baseline normals unchanged.
- Run focused `test_fiber_trace_3d.py` tests and `git diff --check`.
