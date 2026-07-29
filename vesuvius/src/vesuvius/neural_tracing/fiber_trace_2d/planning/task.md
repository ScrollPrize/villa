# Python Native 3D Trace2CP Continuous Whole-Fiber CP Handling

Fix the Python native 3D Trace2CP whole-fiber path so reaching a CP plane does
not reinitialize tracing or smoothing state when the segment succeeds.

Requirements:

- Scope this task to the Python tracer only. Do not adapt the C++ tracer until
  the Python behavior has been validated.
- Target-plane crossings are only for the current segment's target CP. The
  tracer must not consider previous segment planes or any CP-to-CP chord plane.
- Keep tracing while budget remains when all target-local planes have been
  crossed but the best crossed-plane in-plane CP error is still above the
  configured whole-fiber threshold.
- Continue to update target-plane crossing candidates so later/closer crossings
  can replace earlier far crossings of the same plane.
- Accept a one-way target-plane result only when all configured target-local
  planes have been crossed and the best in-plane crossing error is at or below
  the threshold.
- In whole-fiber mode, an accepted CP crossing is a checkpoint/metric event,
  not a trace restart. The live trace must continue from its actual stepped
  point with previous direction, sampled-current direction, and smoothing
  history preserved.
- Only failed segments count as restarts and reset tracing from the failed CP.
