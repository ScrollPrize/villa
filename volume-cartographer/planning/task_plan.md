# Task plan

## Implementation

1. Add task-ID cancellation to the keyed scheduler. Cancellation succeeds only
   while a task remains pending; running tasks are not interrupted.
2. Add explicit background ownership to unresolved cache entries while
   retaining per-view generation slots for GUI ownership.
3. During atomic view-snapshot replacement, remove the old view slots, install
   the new slots, and cancel entries with no remaining owner.
4. Pass the latest allocated render generation when clearing a view. Retain a
   generation watermark so late asynchronous work cannot recreate a closed
   view slot; a later generation may reopen the same stable view ID.
5. Guard probe-to-source and source-to-decode handoffs with the same scheduler
   selection gate used for view publication. Stale running work may finish its
   current stage but cannot queue another one.

## Testing

- Verify targeted cancellation does not affect running scheduler tasks.
- Verify render replacement and view closure remove stale pending work.
- Verify another view and explicit background demand preserve shared work.
- Verify a running stale download does not enter decode.
- Verify closed-generation work is rejected and a newer generation reopens the
  same view ID.
- Make decode-priority coverage wait until both candidates are genuinely
  pending before asserting their selection order.
- Build VC3D and repeatedly run the complete chunk-cache suite.

## Documentation

- Update the render/fetch specification and changelog with ownership and
  cancellation semantics.
