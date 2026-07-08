# 2D Fiber Trace Initial Loader Task Plan

## Scope

Implement initial batch loading of fiber-strip patches around random control points from the fiber dataset.

The deliverable is a loader/iterator plus a small tester/runner. It loads VC3D-equivalent fiber side-strip patches from explicit surface/segment coordinates and does not use the neural-tracing crop loader.

## Reuse Targets

- Copy/adapt the VC3D fiber JSON parsing behavior from `vesuvius.neural_tracing.fiber_trace.fiber_json` into the 2D package.
- Use VC3D side-strip/surface/segment sampling semantics for patch coordinates.
- Prefer exporting/reusing the VC3D side-strip coordinate code directly. If direct reuse is impractical, port the same algorithm closely and allow only small rounding/interpolation differences.
- Use Lasagna normals where needed to construct aligned strip frames.
- Reuse only the existing Vesuvius/neural-tracer Zarr chunk cache/fetching path for prefetch and cache population.
- Do not use the existing neural-tracing crop-loading path for image loading.

## Configuration

- Add a Vesuvius-style JSON config for the loader/runner.
- Dataset entries identify fiber JSON files or globs, base volume Zarr path, cache settings, volume scale/group, and any Lasagna normal/manifest input needed for strip frames.
- Loader settings include `batch_size`, strip patch size, strip-z offsets, seed, deterministic tester sample index, and prefetch worker count.
- The default strip-z offsets are `-7..8`, giving 16 patches per selected control point.
- Prefetch worker count is capped at 16.

## Data Model

- Define a dataset record for one fiber source and its corresponding base-volume metadata.
- Define a sample descriptor with record index, fiber path, control-point index, control-point coordinates, local tangent, strip frame, strip-z offset, and explicit surface/segment coordinates.
- Define a batch object with image tensor, coordinate tensor, strip offsets, and fiber/control metadata.
- Image tensor layout should keep the per-control-point strip offsets explicit, for example batch, offset, channel, height, width.

## Deterministic Sampling

- Normal batch loading samples random control points from the parsed fiber dataset.
- Sampling must be deterministic from the configured seed.
- The tester/runner accepts a specified control-point sample index.
- The specified tester index maps deterministically into the full fiber/control-point dataset.
- Changing batch size or sample count must not change previously addressed samples.

## Strip Patch Loading

- For each selected control point, construct the VC3D-style strip frame from the fiber tangent and Lasagna normal information.
- Build one explicit VC3D-equivalent side-strip surface/segment coordinate grid for each strip-z offset.
- Sample each strip-z patch independently.
- Load voxel values from the base Zarr volume at the explicit coordinates.
- Do not call crop readers for image loading.

## Prefetch

- Add a prefetch mode for the loader/runner.
- Prefetch constructs the same explicit patch coordinate grids as loading.
- Convert coordinates to base-volume Zarr chunk keys.
- Deduplicate chunk requests.
- Skip chunks already present in the configured cache.
- Fetch missing chunks into the cache through the existing chunk fetching path.
- Print progress with count, MiB/s, and ETA.
- Prefetch only remote base-volume chunks.

## Tester/Runner

- Add a small command-line runner under the 2D fiber trace package.
- Runner inputs are config JSON path, deterministic control-point sample index, optional batch size override, and optional prefetch flag.
- Runner outputs loaded batch tensor shape/dtype, selected fiber/control metadata, strip offsets, and chunk/cache summary.

# fullfillment

## plan.md reduced scope

- 2D slice-based fiber refinement/interpolation: fulfilled by loading 2D fiber-strip patches as the first data milestone.
- Work on 2D slices only: fulfilled by 2D sampled VC3D side-strip coordinate grids.
- 2.5D by moving in z on 2D slices: fulfilled by independent strip-z offset patches.
- Use Lasagna normals to extract aligned slices: fulfilled by using Lasagna normal information in strip-frame construction.
- Fiber side strips as defined in VC3D: fulfilled by reusing/exporting the VC3D side-strip coordinate code or porting it closely with parity tests.
- Data streamed and cached from S3: fulfilled by Zarr chunk-key prefetch and cache population.
- Use VC3D strip extraction to get side-strip views of CPs: fulfilled by loading CP-centered strip views through explicit VC3D-style coordinates.
- Slices plus/minus voxels along strip z: fulfilled by the default `-7..8` strip-z offsets.

## task.md

- Copy fiber parsing from existing fiber code: copy/adapt `fiber_trace.fiber_json` parsing semantics into the 2D package.
- Write initial data-loader/iterator: add the deterministic loader and batch iterator for fiber-strip patches.
- Use fiber side-strip code from VC3D/Lasagna/fiber-trace: use VC3D-style side-strip coordinates and Lasagna normal inputs.
- Initial batch loading around random CPs: normal batches sample random control points from the fiber dataset.
- Load `+-7/8` patches around the CP strip: load 16 independently sampled strip-z offset patches per selected CP.
- Write a data-loader tester/runner: add a command that loads a batch from a specified deterministic CP sample index.
- Specified CP is deterministic random index: map the requested index deterministically into the full fiber/control-point dataset.
- Each patch sampled independently: build and sample a separate coordinate grid per strip-z offset.
- Link/use VC3D sampling methods: use VC3D-equivalent side-strip surface/segment coordinate sampling, not crop loading.
- Initially load a 2D strip image: the sampled image is 2D, but its coordinates must follow the VC3D side-strip surface/segment construction rather than a flat planar simplification.
- Use prefetch approach: compute needed chunks from coordinates before loading image data.
- Download chunks using chunk fetching into cache: fetch missing base-volume chunks through the existing cache/fetch path.
- Vesuvius-style JSON config: define dataset and loader settings in JSON.
- Inspect VC3D fibers/jsons/side-strip creation: implementation follows the VC3D fiber JSON and side-strip coordinate algorithm, with only rounding/interpolation differences accepted.
- Inspect Vesuvius chunk fetching: implementation reuses chunk cache/fetch behavior only.
- Prefer plain Vesuvius over neural-tracer dependencies: use shared Vesuvius utilities where possible and keep neural-tracer reuse limited to chunk cache/fetching.
- Write docs explaining code structure: document config, loader, coordinate sampler, prefetch, and runner layout.
- Update specs.md: keep implemented initial-loader specs as bullets.

## Tests

- Add tests for config parsing, deterministic CP indexing, fiber parser parity, strip offset generation, fake-volume batch shape, coordinate-grid generation, and prefetch chunk dedup/cache skipping.
- Tests must not require network access.

## Documentation

- Document the package structure in `docs/`.
- Keep `docs/specs.md` limited to the initial-loader behavior.
