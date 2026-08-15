# Task log

## Findings

- `SurfaceCache::runFill()` requests exact dependencies beyond the sparse frame
  prepass, especially for the default 32-sample normal band.
- Passing the viewer's `ChunkRequestContext` makes those dependencies part of
  the replaceable view snapshot.
- `replaceViewDemand()` erases absent per-view slots, cancels undemanded pending
  work, erases unresolved entries, and wakes `prefetchChunks(wait=true)`.
- The fill then treats unavailable decoded chunks as sample failures and can
  publish an incomplete tile, consuming the three-attempt retry guard.
- Before `e9416cc21a`, the context-free prefetch set background demand and was
  not affected by view replacement.

## Deviations

- None.

## Validation

- `cmake --build volume-cartographer/build --target test_chunk_cache VC3D -j8`
  passed. Existing Qt SFINAE incomplete-type warnings were unchanged.
- `volume-cartographer/build/bin/test_chunk_cache` passed 78 test cases,
  including context-free prefetch survival across view replacement.
- `git diff --check` passed.
