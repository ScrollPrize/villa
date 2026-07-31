# Task Log: Trace Native Extrapolation By Length

## 2026-07-31 - Findings

- `traceFiberExtrapolation()` currently creates a synthetic
  `extrapolation_distance` plane normal to the initial outward direction.
- `traceOneWayCore()` gives that request
  `ceil(distance * max_step_factor / step)` generations.
- A curved fiber can trace many times the requested extrapolation length without
  crossing the initial-direction plane. The observed 1,876-point fallback is a
  real runaway to that inflated budget, not only a misleading warning.
- `BeamState` already records accumulated `tracedLength`, so exact length
  completion can share the existing beam search without a parallel tracer.

## Review

- The length limit will be a private `traceOneWayCore()` mode used only by
  `traceFiberExtrapolation()`; public CP-to-CP requests continue to use their
  existing target planes and `max_step_factor` budget.
- No separate review agent is used because higher-priority collaboration
  instructions prohibit delegation unless explicitly requested. The plan was
  reviewed directly against the shared tracer and current extrapolation specs.

## Deviations

- The first test run showed that `makeTraceTargetPlaneSet()` rejected the empty
  set before the private length mode could run. It now permits empty planes only
  when `traceOneWayCore()` was explicitly given a trace-length limit; public
  target-directed calls still reject missing planes.

## Implementation

- Added `reachedTraceLength` to distinguish exact trace-budget completion from
  `reachedTargetPlane`.
- Added a private optional length limit to `traceOneWayCore()`. In that mode the
  generation budget is `ceil(distance / step)` and does not use
  `maxStepFactor`.
- Reused `BeamState::tracedLength` to recognize completion and clipped the last
  returned polyline segment to the requested double-precision length.
- Removed the synthetic `extrapolation_distance` plane from
  `traceFiberExtrapolation()` and made VC3D accept `reachedTraceLength`.
- Kept partial `no_valid_candidates` paths as native volume-edge truncations;
  no-progress failures still retain Lasagna and emit the diagnostic warning.

## Validation

- `test_fiber_trace3d`: all 44 cases passed.
- `test_line_annotation_generated_views`: all 51 cases passed.
- The slanted extrapolation regression sets `maxStepFactor=100`, does not cross
  the old initial-direction plane within the request, and still returns exactly
  three steps with 10 trace voxels of arc length.
- A boundary regression verifies a 6-voxel request with a 4-voxel standard step
  samples its second candidate at 6 rather than 8, so data invalid beyond the
  requested endpoint cannot cause a false edge truncation.
- Built the production `VC3D` target with `-j32`; only the existing Qt
  incomplete-type/SFINAE warnings were emitted.
- `git diff --check` passed. Whole-file `clang-format --dry-run` is not a useful
  gate in this checkout because the existing touched files produce thousands of
  format violations unrelated to this change; changed lines were reviewed
  directly against their surrounding style.
