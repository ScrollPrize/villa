# Task: retire stale interactive chunk work

Prevent repeated interactive renders from growing the probe, source-read, and
decode queues with chunks that no current view still needs.

- Track GUI ownership by `(view ID, view version)` on each unresolved chunk.
- Track explicit non-GUI/background ownership independently.
- Atomically remove superseded GUI ownership when a render publishes its new
  dependency snapshot.
- Remove a view's ownership from every source when that view closes or changes
  source, rejecting late work from the cleared generation.
- Cancel only pending work that has no remaining owner.
- Do not interrupt running work, but prevent stale work from entering another
  queue stage.
- Preserve shared chunks requested by another view or background caller.

Decoded values, rendering, and cache residency semantics must remain unchanged.
