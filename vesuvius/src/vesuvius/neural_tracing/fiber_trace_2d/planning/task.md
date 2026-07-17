# Native 3D Whole-Fiber Trace2CP

Extend the native 3D Trace2CP tool so `--fiber-json` defaults to tracing the
entire fiber when no explicit `--start-cp-index/--target-cp-index` is given.

Required behavior:

- Start tracing at the first control point.
- Trace continuously from CP plane to CP plane through consecutive fiber
  segments.
- For each segment, require the trace to reach the target CP plane within that
  segment's step budget.
- When the target CP plane is reached, compute the in-plane error from the
  traced crossing to the target CP.
- Continue tracing if the in-plane error is within a threshold, defaulting to
  100 selected-scale voxels.
- On failure, record the traced reference-line distance up to the last
  successful CP plane, count one restart, and resume tracing from the failed
  target CP.
- Report the whole-fiber error as
  `restart_count / segment_count`.
- Keep explicit single-segment mode for debugging when both CP indices are
  supplied.
- For whole-fiber rendering, keep the four useful panel types: side volume,
  side 3D presence, top volume, and top 3D presence.
- Update the output image segment by segment while tracing/rendering so partial
  fiber progress is visible during long whole-fiber runs.
- When a restart happens, cut off the failed trace before it overlaps the next
  CP region, then stitch the displayed trace again from the restart CP.
