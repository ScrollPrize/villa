# Plan: Native 3D Trace2CP Accelerated Normal Debug Run

## Implementation

1. Reintroduce the accelerated Lasagna-normal sampler as debug-only code.
   - Load Lasagna streaming data only when the comparison flag is set.
   - Use sparse `FitData3D.grid_sample_fullres(...)` only to read `grad_mag` at
     candidate points and compact `nx`/`ny` at the eight channel-grid corners.
   - Decode each compact corner, blend the sign-invariant tensor/hint, and
     recover the axis with the same principal-axis helper as the baseline path.
   - Do not provide or keep any path that interpolates raw compact `nx`/`ny`
     and then reads `FitData3D.normal_3d`.

2. Add a fail-fast comparison wrapper.
   - Wrap the restored `_NativeLasagnaNormalSampler` as the primary sampler.
   - Call debug alternate samplers on the same points.
   - Return the sparse corner/tensor normals to the tracer after comparison
     succeeds, so the traced metric exercises the accelerated path.
   - Raise `ValueError` immediately on valid-mask mismatch or angle difference
     above the configured threshold.

3. Add CLI controls.
   - `--debug-compare-normal-sampler[=sparse-corner-principal]` enables the
     parallel comparison.
   - `--debug-normal-angle-threshold-degrees` controls angular fail threshold.

4. Keep diagnostics compact.
   - Error messages include path label, call index, local point index, selected
     coordinate, baseline/alternate valid flags, baseline/alternate normals, and
     angle in degrees.
   - No long trace or large report is written.

## Spec Update

- Add a debug-only native 3D Trace2CP comparison mode that is explicitly
  fail-fast but can drive tracing with the accelerated sampler.
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
