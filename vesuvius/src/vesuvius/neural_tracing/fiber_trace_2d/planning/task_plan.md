# Plan: Native 3D Trace2CP Point-Lookup Optimization

## Implementation

1. Add a direct cached-block trilinear sampler.
   - Sample only the requested point corners from each cached inference block.
   - Preserve current selected-level coordinates, trusted-core checks,
     trilinear interpolation semantics, branch decoding, and validity rules.
   - Do not interpolate compact Lasagna `nx`/`ny` raw values for normal
     reconstruction; this only changes model-output field lookup.

2. Replace the Trace2CP field point lookup hot path.
   - Keep the existing block routing and lazy block inference.
   - Use direct corner gathers instead of stacking whole blocks into a
     per-call `grid_sample` batch.
   - Keep the previous branch selection and scoring math unchanged.

3. Test larger missing-block inference batches.
   - Increase the default only if the same benchmark keeps the same metric
     result and improves timing.
   - If memory or quality regresses, record that and leave the prior default.

4. Keep the analytic principal-axis experiment explicit.
   - Leave `eigh` as the default.
   - Record that the analytic tensor method was slower/quality-worse in the
     previous experiment and is not part of this default optimization.

## Spec Update

- Add that native 3D Trace2CP model-output field sampling uses direct
  cached-block trilinear corner gathers for point lookup.
- Reiterate that normal reconstruction still uses the sign-invariant tensor
  path; raw compact `nx`/`ny` must not be directly interpolated as vector
  directions.

## Docs Updates

- Update `task_log.md` with implementation notes, benchmark commands, and
  before/after timing.
- Add a changelog note if the optimization is retained.

## Tests

- Run focused Trace2CP unit tests.
- Run `python -m py_compile` for the modified tool.
- Run `git diff --check`.
- Run the approved whole-fiber benchmark command after each retained change.
