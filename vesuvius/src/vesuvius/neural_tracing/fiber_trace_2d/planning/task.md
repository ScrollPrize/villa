# Task: parallel Fiberlet crop seed tracing

Parallelize independent anchor-seeded Fiberlet crop traces. Trace candidates
concurrently, but serialize result integration and anchor coverage. If an
earlier accepted trace covers the seed of an already-computed concurrent
candidate, discard that candidate during serial integration.

Keep the existing deterministic strongest-first semantics, limits, geometry,
and numerical behavior. Use the host CPU count by default and measure the
optimized Release implementation before and after the change.
