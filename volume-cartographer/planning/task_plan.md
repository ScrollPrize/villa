# Task plan

## Problem

SurfaceCache computes exact chunk dependencies for a 128x128 surface tile and
its normal band, then blocks a fill worker until those chunks resolve. Commit
`e9416cc21a` attached the current viewer request to that prefetch. A later
`replaceViewDemand()` can therefore cancel exact tile dependencies absent from
the sparse frame prepass, wake the blocking fill, and cause a transient
cancellation to be stored as an incomplete tile.

## Implementation

1. Change SurfaceCache tile-fill prefetch back to the context-free overload so
   dependencies receive explicit background ownership and survive replacement
   of any viewer snapshot.
2. Remove `ChunkRequestContext` from `SurfaceCache::requestView()`, queued fill
   closures, and `State::runFill()` because SurfaceCache no longer participates
   in per-view chunk ownership.
3. Update base and overlay VC3D SurfaceCache call sites to the simplified API.
4. Keep `viewTiles`/epoch checks unchanged: they still prevent stale computed
   tiles from being published, while already-admitted background dependencies
   are allowed to resolve.
5. Keep the incomplete-tile retry guard unchanged. Genuine missing/error data
   can still produce an incomplete tile; render replacement cancellation can
   no longer do so.

## Tests

- Add a focused ChunkCache regression proving a context-free prefetch remains
  pending and resolves after an interactive snapshot replacement removes all
  GUI ownership for that key.
- Build and run `test_chunk_cache`.
- Build `VC3D` to validate both SurfaceCache call sites and the public signature.
- Run `git diff --check`.

## Spec update

Change `planning/spec.md` so asynchronous SurfaceCache fills are explicitly
background-owned and independent of replaceable per-view demand. Clarify that
viewer contexts apply to direct render misses and sparse frame demand, not the
exact dependency prefetch performed by a derived tile fill.

## Documentation updates

Update `docs/remote_file_cache.md` to document the ownership boundary and why
SurfaceCache dependencies may continue after a view replacement.

## Changelog

Record the SurfaceCache ownership regression fix under 2026-08-15.

## Independent review

- Context-free prefetch already maps to `Entry::backgroundDemand`; no new
  ownership or queue mechanism is needed.
- Background ownership is intentionally not cleared by `replaceViewDemand()`.
- The fill remains asynchronous and does not block UI/render threads.
- Existing epoch and visible-tile checks continue to reject stale publication.
- Continuing an already-admitted stale dependency download is an explicit
  tradeoff of view-independent background ownership and restores behavior from
  before `e9416cc21a`.
- No rendering values, interpolation, cache keys, or pyramid selection change.
