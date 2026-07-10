# Task Plan: Cache Source Line Coordinates And Split Coord Profiling

## Implementation

1. Bump strip-coordinate cache version.
   - Use a new cache family prefix/version so existing `.npz` files are ignored.
   - Keep corrupt or incompatible cache entries as misses.
2. Extend `_StripSource`.
   - Store source-space `line_xy` and source-space `control_point_xy` tensors.
   - Persist those tensors in the strip-coordinate cache.
   - Center-crop larger cached source line coordinates for smaller source-size
     requests.
3. Use cached source line/CP coordinates.
   - For unaugmented patches, return the cached source line/CP tensors.
   - For augmented patches, continue computing transformed output line/CP
     coordinates per patch because those depend on the random augmentation.
4. Split profile accounting.
   - Keep aggregate `coord_gen` for compatibility.
   - Add separate profile totals/columns for `descriptor`, `coord_cache`,
     `source_geom`, and `line`.
   - `coord_cache` maps to `strip_coord_cache`; `source_geom` maps to
     `line_window + lasagna_normals + strip_coords`; `line` maps to
     `line_coords`.

## Spec Update

Update `planning/specs.md` so strip-coordinate cache entries include source
line/CP pixel coordinates and old cache versions are ignored.

## Docs Updates

Update `docs/code_structure.md` to document cached source line/CP coordinates
and the split profiling columns.

## Tests

- Extend strip-coordinate cache tests to assert source line/CP coordinates are
  restored from cache.
- Add a profile summary/header test for the new split profile keys.
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`

## Changelog

Add one 2026-07-10 changelog line for cached source line coordinates and split
coord profiling.
