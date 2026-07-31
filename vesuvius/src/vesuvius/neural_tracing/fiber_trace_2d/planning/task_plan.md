# Plan: Trace Native Extrapolation By Length

## Scope

- Change the shared C++ one-way tracer so extrapolation has a dedicated
  trace-length completion condition.
- Keep CP-to-CP target planes, intersection acceptance, and fusion unchanged.
- Update VC3D to recognize trace-length completion as successful native
  extrapolation.

## Implementation

1. Add a result flag that distinguishes trace-length completion from
   target-plane completion.
2. Pass an optional private trace-length limit through `traceOneWayCore()` and
   candidate frontier construction.
3. For trace-length mode, derive the maximum generation count directly from
   `ceil(distance / step)` and ignore `max_step_factor`.
4. Mark candidates complete once accumulated traced arc length reaches the
   limit, and clip the final returned segment to the exact requested length.
5. Make `traceFiberExtrapolation()` use the length limit with no target planes.
6. Accept `reachedTraceLength` in the VC3D open-tail replacement path while
   retaining the existing invalid-direction edge truncation rule.

## Tests

- Update the straight extrapolation regression to require trace-length rather
  than target-plane completion and exact arc length.
- Add a slanted trace regression whose initial-tangent plane is not reached at
  the requested length; verify it still stops exactly at that length and that a
  large `max_step_factor` does not extend the trace.
- Retain the invalid-direction partial-tail regression.
- Run `test_fiber_trace3d` and `test_line_annotation_generated_views`.
- Build production `VC3D` with `-j32`.

## Spec Update

- Replace the synthetic extrapolation distance-plane contract with exact traced
  arc-length termination and state that `max_step_factor` applies only to
  target-directed tracing.

## Docs Updates

- Update `docs/code_structure.md` and
  `volume-cartographer/docs/line_annotation_fibers.md` to describe length-based
  extrapolation and edge truncation.

## Changelog

- Record the correction from synthetic target-plane extrapolation to a hard
  trace-length budget.

## Review

- Verify the new completion mode cannot alter CP-to-CP target-plane paths.
- Verify all completion and failure result fields remain unambiguous.
