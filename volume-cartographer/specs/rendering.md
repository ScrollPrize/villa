# Rendering Requirements

This document will contain the durable requirements for interactive and batch
rendering. The requirements will be drafted after the synthetic rendering
benchmark described in `benchmarks.md` is established.

## Derived Cache Lifetime

- Clearing a surface's derived caches while switching active surfaces or
  projects must not invalidate data used by in-flight renders.
- Cache construction, reset and snapshot acquisition use the same mutex.
  A renderer retains immutable, reference-counted matrix snapshots after
  releasing that mutex; it does not hold it throughout pixel sampling.
- A validity snapshot includes its matching all-valid flag.
- Eviction changes neither rendered coordinates/normals nor interpolation,
  and does not cancel in-flight rendering.
- This contract covers gen()/validMask() versus unloadCaches(), with points
  already loaded and geometry unchanged. Geometry edits, point unloading and
  channel access continue to require external synchronization.
