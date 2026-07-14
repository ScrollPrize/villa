# Compact In-RAM Fiber-Line Geometry Store Task Log

## Implementation Notes

- Removed the persistent dense `strip_coord_cache_dir` source-coordinate cache
  from config parsing, example configs, and source construction.
- Added a compact loader-owned geometry store that samples Lasagna normals at
  loader startup, builds valid contiguous frame intervals, and shares that store
  by reference with cloned/threaded loaders.
- Source-grid construction now looks up record/control-point compact geometry,
  evaluates only the requested source columns, and broadcasts rows from the
  interpolated frame axes.
- Preserved the existing training visualization contract: source-space
  centerlines are still one point per source column rather than sparse fiber
  vertices.
- Kept the existing Zarr chunk cache/prefetcher. A transient bug from removing
  the `uuid` import was fixed; prefetch still uses UUID temp names for atomic
  chunk downloads.

## Plan Deviations / Rationale

- Restored eager dataset-load construction after an intermediate lazy variant;
  this matches the requirement that compact geometry is built during loader
  construction with progress output.
- Instead of storing one fixed full-fiber per-column coordinate grid, the store
  keeps exact frame/interpolation arrays and evaluates the requested source
  columns at each CP's exact arc/pixel phase. This avoids a subtle alignment
  approximation for CPs whose arc length is not integer-aligned to the selected
  voxel spacing.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed: `264 passed in 7.99s`.
- `PYTHONPATH=/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/volume-cartographer/build/python-bindings/python:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3 python -m vesuvius.neural_tracing.fiber_trace_2d.train /home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/configs/loader_example.json --benchmark --load-only --profile`
  completed:
  - startup compact geometry build: `464` records, `14773` valid CPs,
    `184` skipped CPs, `63.5 MiB`, `8m53s`;
  - benchmark: `100` batches, `12800` patches, `76.76 patches/s`;
  - hot path profile: `compact_geometry=0.012 ms/patch`,
    `coord_gen=25.866 ms/patch`, `source_geom=22.010 ms/patch`,
    `loading=7.100 ms/patch`.
