# Task: remove serial full-corridor replay setup

`vc_fiberlets fiberlet-replay` must not materialize every anchor cell in the
full reference corridor before starting the cache-backed tracers. A full-fiber
radius-768 replay currently spends tens of seconds on one core at zero progress
because dense segment-to-cell expansion is performed serially and then repeated
for scheduling.

Make cached replay select cache chunks directly, enumerate exact anchor cells
only inside a requested chunk, and preserve the existing radius, NMS, path
containment, cache identity, and eager/cached numerical contracts.
