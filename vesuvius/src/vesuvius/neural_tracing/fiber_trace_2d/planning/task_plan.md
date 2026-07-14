# Compact In-RAM Fiber-Line Geometry Cache Plan

## Findings

- The current `strip_coord_cache_dir` cache stores dense per-CP source grids:
  `coords_zyx[H,W,3]`, `valid_mask[H,W]`, and axis fields such as
  `offset_axis_zyx[H,W,3]` / `side_axis_zyx[H,W,3]`.
- With the example config, a `64x64` patch expands to about `374x358` source
  pixels because the augmentation envelope covers shift, rotation, shear,
  scale, and smooth offset.
- Current cache measurements:
  - typical current entry: about `3.35 MB`;
  - legacy entries with duplicate xyz/zyx arrays: about `6.5 MB`;
  - current cache directory: about `818 GiB` for `215,646` files.
- The side-strip builder already derives dense coordinates from a compact
  column representation:
  - cubic Hermite centerline interpolation over arc length;
  - interpolated/renormalized mesh-normal axis per column;
  - interpolated/renormalized side axis per column;
  - dense side grid as `center[x] + normal[x] * row_offset[y]`.
- Estimated full-fiber compact side geometry sizes:
  - regular JSON train: about `331 MiB` for center + normal + side + validity;
  - S1A NML train: about `73 MiB`;
  - combined, this is easily RAM-resident in one process.
- The NML case is especially wasteful under per-CP dense caching because median
  transformed selected-scale CP spacing is about `11 px` while each source
  window is about `358 px` wide.

## Design

### Compact Geometry Object

Add a compact per-record geometry representation, for example:

- `CompactFiberLineGeometry`
  - `record_index`;
  - `line_points_xyz` / original line metadata only where still needed for
    diagnostics;
  - `arc_start`, `arc_end`, `pixel_spacing_base`;
  - `columns_center_zyx: float32[num_columns, 3]`;
  - `columns_normal_axis_zyx: float32[num_columns, 3]`;
  - `columns_side_axis_zyx: float32[num_columns, 3]`;
  - `columns_valid: bool[num_columns]`;
  - `cp_column: float32[num_control_points]`;
  - `cp_valid: bool[num_control_points]`;
  - `frame_at_cp` or compact frame fields when needed by supervision/metadata.

The compact cache should store already-interpolated per-column data. It must
not store only sparse NML/JSON line vertices if that would require redoing the
expensive line-window/normal interpolation path at sample time.

### Startup Preprocessing

- Build records exactly as today: parse JSON/NML, apply optional affine
  transform, validate bounds, and preserve deterministic record/control-point
  identity.
- After records are loaded, build the shared compact geometry store once:
  - sample Lasagna normals for the line points required to build the full
    component geometry;
  - build frames with the existing frame construction semantics;
  - generate per-column center/normal/side arrays over the whole fiber/component
    at selected-scale pixel spacing;
  - compute direct CP column positions from each control point's exact line
    index/cumulative arc value;
  - mark CPs invalid if their required source window would cross invalid
    geometry or invalid Lasagna normal data.
- Show a startup progress line/bar while compact geometry is built:
  - records/components processed;
  - CPs processed/valid/skipped;
  - elapsed time and estimated remaining time;
  - final resident byte estimate for compact geometry.
- Do not fabricate normals. If a line point required for a component cannot be
  sampled from Lasagna channels, handle it explicitly:
  - either mark the whole component invalid, or split into valid contiguous
    intervals if that can preserve current CP-local skip semantics without
    inventing normals;
  - in either case, training/prefetch should skip invalid CPs deterministically
    as it does for current CP-local invalid samples.

### Sharing And Ownership

- Introduce a `FiberLineGeometryStore` owned by the top-level loader/provider.
- All batch loading, prefetch, augment-vis, line/dir/Trace2CP tooling, and
  top-view sampling should receive references to this store rather than building
  geometry independently.
- For threaded loader workers in one process, share the same Python object
  read-only.
- For `pipeline_isolated_loaders`, stop each isolated loader from rebuilding
  compact geometry:
  - construct records + geometry store once in the provider;
  - pass the shared records/store reference into worker loaders;
  - keep the compact arrays immutable after construction.
- Current `fiber_trace_2d` training does not use DDP or `torch.distributed`.
  The implementation target is therefore exactly one compact geometry store per
  training process, shared by all worker threads and by any cloned loaders.
  Multiprocess/DDP support is out of scope for this task unless it is added to
  `fiber_trace_2d` later; this task must not introduce extra per-worker copies.

### Hot-Path Sampling

- Replace `build_strip_source()` dense-cache lookup/store with direct compact
  geometry lookup:
  - descriptor lookup gives `(record_index, cp_index)`;
  - `geometry = store.by_record_index[record_index]`;
  - `anchor_col = geometry.cp_column[cp_index]`;
  - crop the source column range needed for the configured augmentation
    envelope without searching.
- Reconstruct source side grids by broadcast math:
  - `coords_zyx[y,x] = center_zyx[x] + normal_axis_zyx[x] * row_offset[y]`;
  - `offset_axis_zyx[y,x]` and `side_axis_zyx[y,x]` are broadcast views or
    cheaply expanded tensors from per-column axes.
- Reconstruct top-view grids from the same compact columns:
  - `coords_zyx[y,x] = center_zyx[x] + side_axis_zyx[x] * side_offset[y]`
    plus any requested normal/z offsets by column.
- Preserve current line/control-point pixel mapping:
  - source-space line/cp coordinates should derive from column coordinates and
    CP anchor positions;
  - geometric augmentation still transforms those coordinates through the
    existing fused map.
- Avoid dense axis materialization where a downstream consumer can use
  broadcasted per-column axes directly. If a consumer requires a dense tensor,
  materialize only for the current batch/source window.

### Config/API

- Remove `strip_coord_cache_dir` from example configs and documented config
  keys.
- Remove or ignore the disk strip-coordinate cache implementation. The loader
  should not read, write, or require `.npz` coordinate cache files.
- Keep the existing Zarr chunk cache and prefetcher unchanged except that
  prefetch dependency generation must use compact geometry instead of the old
  dense disk cache path.
- Add optional diagnostic config keys only if needed, for example:
  - `fiber_line_cache_progress_interval`;
  - `fiber_line_cache_dtype` later, if 16-bit storage is explored.
  Keep the first implementation in float32 for behavioral comparison.

## Spec Update

- Replace the `strip_coord_cache_dir` disk-cache section in `planning/specs.md`
  with compact in-RAM full-fiber geometry requirements.
- State that dense source coordinates are no longer cached per CP.
- State that compact geometry is built once at loader startup from transformed
  fiber coordinates and Lasagna normals.
- State that CP lookup is direct through precomputed record/control-point
  column metadata, not searched in the hot path.
- State that the compact cache is shared read-only across all loader workers in
  one process.
- Keep VC3D/Lasagna strip semantics unchanged: compact reconstruction must
  produce equivalent coordinates to the current dense grid builder.

## Docs Updates

- Update `docs/code_structure.md`:
  - describe `FiberLineGeometryStore` and compact full-fiber representation;
  - remove the current `strip_coord_cache_dir` disk-cache description;
  - document startup preprocessing/progress output;
  - document memory estimates and where to inspect profile fields.
- Update `planning/local_development.md`:
  - remove guidance that recommends clearing/reusing
    `vesuvius_fiber_trace_strip_coord_cache`;
  - document the benchmark commands for compact-cache vs old/no-cache
    comparison.
- Update configs:
  - remove `strip_coord_cache_dir` from `loader_example.json`,
    `loader_example_s1a_nml.json`, and temporary/example configs tracked in
    the subproject.

## Testing

- Unit tests:
  - compact side geometry reconstruction matches
    `build_side_strip_patch_grid_tensor_from_line_window()` for representative
    line windows within tight float32 tolerance;
  - compact top geometry reconstruction matches
    `build_top_strip_patch_grid_tensor_from_line_window()`;
  - CP-to-column lookup returns the exact anchor for all control points without
    searching;
  - invalid Lasagna samples mark CPs/components invalid and deterministic batch
    loading skips them.
- Loader regression tests:
  - JSON and NML dataset sample counts and deterministic ordering remain stable;
  - `load_batch`, `load_top_batch_for_batch`, prefetch dependency generation,
    augment-vis source generation, and Trace2CP source generation all use the
    same compact store path;
  - `pipeline_isolated_loaders` does not duplicate the compact store inside one
    process.
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`

## Benchmark / Validation

- Measure startup preprocessing time and memory estimate for:
  - `loader_example.json`;
  - `loader_example_s1a_nml.json`.
- Run old benchmark path after implementation:
  `PYTHONPATH=$SRC/volume-cartographer/build/python-bindings/python:$SRC/vesuvius/src:$SRC python -m vesuvius.neural_tracing.fiber_trace_2d.train $SRC/vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/configs/loader_example.json --benchmark --load-only --profile`
- Compare against the measured baselines:
  - warm dense disk coord cache: about `94.86 patches/s`,
    `coord_gen=24.17 ms/patch`;
  - no coord cache: about `12.33 patches/s`,
    `coord_gen=271.0 ms/patch`.
- Success target:
  - compact in-RAM geometry should be close to or faster than warm dense disk
    cache for the hot path;
  - startup preprocessing should be bounded and reported;
  - disk use for strip-coordinate geometry should drop to zero.

## Changelog

- After implementation, add a 2026-07-14 changelog entry: dense per-CP disk
  strip-coordinate cache removed; compact full-fiber in-RAM geometry store
  added for training, prefetch, and visualization paths.

## Open Implementation Notes

- Full-fiber preprocessing touches more line points at startup than CP-local
  loading. The implementation must be strict about invalid Lasagna data and
  must not silently propagate nearest normals.
- `FiberStrip2DLoader.clone()` currently constructs clone records with local
  samplers. The compact geometry store must be passed through cloning by
  reference and must not be rebuilt for cloned loaders.
