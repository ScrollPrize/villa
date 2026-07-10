# Task Plan: Strip Coordinate Cache

## Implementation

1. Add a top-level config key `strip_coord_cache_dir`.
   - `null`/missing disables the cache.
   - Paths are resolved relative to the config file like other local paths.
2. Add a small disk cache around `build_strip_source`.
   - Build the descriptor and requested source size first.
   - Compute a stable cache-family key from volume path, selected scale,
     pixel spacing, strip-z center offset, strip-z step, CP coordinate, and
     fiber-line identity.
   - Store one `.npz` entry per family with source shape metadata, frame,
     source coords, valid mask, and strip offset axes.
3. Cache lookup behavior:
   - A cache entry is usable when cached height and width are both at least the
     requested source height and width.
   - On usable larger hits, center-crop the stored tensors to the requested
     source size.
   - On misses or too-small hits, generate the source grid through the existing
     Lasagna/VC3D-equivalent path and atomically replace the cache entry if the
     new source is larger or no entry exists.
4. Keep the cache at the shared source-build level so training, augment-vis,
   center-patch runner loads, line/dir visualizations, and prefetch use the
   same behavior without separate code paths.
5. Do not cache final random augmented patches, sampled image values, or
   strip-z-offset-specific final grids.

## Spec Update

Update `planning/specs.md` with the new `strip_coord_cache_dir` key, cache key
semantics, larger-entry center-crop hit behavior, atomic writes, and shared
caller coverage.

## Docs Updates

Update `docs/code_structure.md` and `planning/local_development.md` to describe
the coordinate cache and where to configure it.

## Tests

- Add config parsing coverage for `strip_coord_cache_dir`.
- Add a loader test that builds a source once, confirms a cache file exists,
  then builds the same source through a fresh loader with Lasagna normal reads
  disabled to prove the shared source path is using the cache.
- Add a loader test that first writes a larger cached source and then requests a
  smaller source, asserting the smaller source is served from the cache and has
  the requested shape.
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`

## Changelog

Add one 2026-07-10 changelog line for the configurable strip-coordinate cache.
