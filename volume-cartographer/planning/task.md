# Task: keep SurfaceCache fill dependencies view-independent

Restore SurfaceCache tile-fill chunk prefetches to background ownership.

- Exact tile dependencies must not be cancelled when a viewer replaces its
  sparse per-frame demand.
- SurfaceCache fills are non-GUI/background work and must use the context-free
  chunk API.
- Remove the now-unused viewer request context from SurfaceCache scheduling and
  its callers.
- Preserve stale-tile publication checks and the bounded incomplete-tile retry
  guard.
- Add regression coverage for background prefetch surviving view replacement.
