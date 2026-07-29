# Plan: Python Native 3D Trace2CP Continuous Whole-Fiber CP Handling

## Context

The Python whole-fiber 3D Trace2CP path now uses target-local planes instead of
the CP-to-CP chord, but it currently terminates as soon as every target plane
has been crossed once. For skewed/far crossings this can stop too early and can
snap the endpoint back to an old crossing. Separately, successful CP crossings
were still treated like segment boundaries: whole-fiber tracing restarted the
next one-way tracer from the selected crossing and reselected start direction /
smoothing state. The tracer should continue through accepted CP planes as one
continuous trace; CP planes are only metric/checkpoint events unless a segment
fails.

## Implementation

1. Crossing tracking
   - Update the target-plane crossing helper so it checks every target plane on
     every step, not only not-yet-crossed planes.
   - Store a new crossing if the plane was not crossed before or if the new
     crossing has a smaller in-plane CP error than the stored one.
   - Keep crossing tests local to the current segment target CP only.

2. Threshold-aware one-way trace acceptance
   - Add an optional one-way `target_plane_accept_threshold_voxels` argument.
   - With no threshold, keep existing behavior for single-pair and synthetic
     callers: all planes crossed is sufficient.
   - With a threshold, accept only when all planes are crossed and the best
     crossed-plane error is `<= threshold`.
   - If all planes are crossed but the best error is still above threshold,
     continue stepping from the actual candidate point, not from the selected
     crossing.
   - If the budget/failure condition ends after all planes were crossed but
     above threshold, return the best selected crossing/error so whole-fiber
     restart reporting can show the actual in-plane error.

3. Continuous whole-fiber state
   - Extend one-way trace results with terminal live state: actual stepped
     point, previous step direction, smoothing-history direction, and cached
     sampled-current direction when available.
   - Add one-way continuation arguments so a trace can start from an existing
     live state without CP-start direction resampling.
   - In whole-fiber mode, accepted CP crossings must keep the actual live trace
     endpoint and terminal state. The selected crossing remains stored for
     metric/error reporting only.
   - Keep snapping to the selected crossing as the default for single-pair
     visualization/diagnostic callers.
   - On failure/restart, reset the live state and initialize the next run from
     the failed CP's local fiber tangent as before.

4. Whole-fiber wiring
   - Pass the existing whole-fiber `error_threshold_voxels` into the Python
     one-way tracer when using the real tracer.
   - Keep fake `trace_segment_fn` tests compatible by not requiring test helpers
     to accept the new production-only argument.

5. Regression tests
   - Add a Python one-way test where all planes are crossed early with an error
     above threshold, then later crossed closer to the CP and accepted.
   - Add a whole-fiber test where a CP plane is crossed between trace steps and
     the next CP segment starts from the live stepped endpoint rather than the
     selected plane crossing.
   - Keep existing all-plane and selected-best-error tests passing.

## Spec Update

- Update the native 3D Trace2CP spec to state that all target-local planes being
  crossed is necessary but, in whole-fiber thresholded tracing, not sufficient:
  the selected best in-plane CP error must also be within the configured
  threshold.
- State that later closer crossings of the same target plane replace earlier
  farther crossings.
- State that whole-fiber success does not restart or snap live tracing state to
  the selected crossing; selected crossings are metric events, while only
  failures reset from a CP.

## Docs Updates

- Update `docs/code_structure.md` in the native 3D Trace2CP section with the
  threshold-aware crossing and continuous whole-fiber state behavior.

## Tests

- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
- Run `git diff --check`.

## Changelog

- Add a short 2026-07-29 entry for threshold-aware Python target-plane
  acceptance.

## Explicit Scope

- C++ tracer parity is intentionally deferred until the Python behavior has been
  validated on the failing fiber.
