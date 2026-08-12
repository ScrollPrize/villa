# Task: prioritize interactive chunk download and decode work

Refactor VC3D regular-chunk request scheduling so remote downloads and cached
chunk decode work follow current GUI demand instead of submission order alone.

Each interactive viewer has a stable numeric ID, a last interaction/focus
position in viewport coordinates, and a versioned set of missing chunk
occurrences. Before each accepted GUI render starts, run a reduced-resolution,
stratified pre-pass which maps viewport samples to required regular chunks.
Retain multiple distant viewport occurrences for folded surfaces, deduplicate
nearby occurrences per chunk, and publish the complete view snapshot only after
the pre-pass has finished without holding the shared scheduler lock.

Interactive work is selected lexicographically by active view, coarse pyramid
level, and distance to that view's focus point. GUI misses not found by the
pre-pass remain GUI work with no location and sort last only within their view
and level. Existing queued work must adopt current view demand. Non-GUI work
uses a separate admission queue, with bounded work-conserving fairness between
GUI and non-GUI work.

The pre-pass must reuse surface geometry needed by the following render:

- when direct rendering needs full framebuffer coordinates, generate them once,
  use a strided view for the pre-pass, and reuse them for the full render;
- when `SurfaceCache` serves the frame, probe the shared
  `SurfaceGeometryTileCache`, so generated geometry tiles are immediately
  reused by SurfaceCache fills rather than running a separate surface
  generation path.

All GUI-originated regular chunk requests, including direct sampler misses,
fallback/composite/overlay requests and asynchronous SurfaceCache fills, must
carry GUI request context. Rendering values, interpolation, transforms and
cache contents must remain unchanged.
