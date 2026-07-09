# 2D Fiber Trace Loader Code Structure

This package currently implements a loader/debug runner for 2D fiber side-strip
patches around VC3D fiber control points. It is not yet a training pipeline.

The important behavior is:

- read VC3D fiber JSONs and Lasagna manifests;
- select deterministic control-point samples;
- build VC3D-equivalent side-strip coordinate grids;
- sample image values through the VC3D blocking coordinate sampler;
- optionally apply coordinate-space geometric augmentation and torch value
  augmentation;
- export JPG batches and augmentation contact sheets for inspection;
- prefetch addressed base-volume chunks into the configured cache.

## Module Map

`fiber_json.py`

- Re-exports the existing VC3D fiber JSON parser from
  `vesuvius.neural_tracing.fiber_trace.fiber_json`.
- Keeps control-point and line-point parsing semantics shared with the existing
  3D fiber trace code.

`strip_geometry.py`

- Defines `FiberStripFrame`, `FiberStripGrid`, and `FiberStripLineWindow`.
- Locates the CP-local line window that can affect a requested strip width.
- Requires each selected control point to be an exact member of `line_points`;
  mismatch is treated as corrupt fiber data.
- Ports the VC3D/Lasagna side-strip frame construction:
  - tangent from neighboring line points;
  - mesh normal from Lasagna normals projected into the tangent plane;
  - side direction from normal x tangent;
  - frame transport and roll smoothing along the local line window.
- Builds explicit coordinate grids for side strips using cubic Hermite
  interpolation over line arc length.
- Provides a torch-vectorized dense grid builder,
  `build_side_strip_patch_grid_from_line_window_torch`, which keeps the same
  frame semantics but vectorizes per-pixel interpolation work.

`loader_support.py`

- Contains small Zarr-array helpers used by the fallback/manual sampling path.
- Computes chunk requests for explicit trilinear coordinate samples.
- Provides `sample_array_trilinear` for fake/local-array tests.

`sampling.py`

- Defines the `CoordinateSampler` interface used by the loader.
- `Vc3dCoordinateSampler` is the production sampler:
  - opens local paths with `vc.volume.Volume.open`;
  - converts `s3://bucket/key` to the matching public HTTPS URL;
  - opens remote paths with `Volume.open_url`;
  - calls `Volume.sample_coords(..., blocking=True)` so missing chunks are
    fetched/decoded before sampling returns;
  - calls `Volume.collect_coords_dependencies` for prefetch chunk discovery.
- `NumpyZarrCoordinateSampler` remains useful for tests and local fake arrays.
- `make_coordinate_sampler` currently returns the VC3D sampler for normal
  runtime.

`augmentation.py`

- Defines `FiberStripAugmentConfig` and `FiberStripAugmentParams`.
- Builds geometric augmentation maps in strip pixel coordinates. The image is
  never geometrically warped after loading; instead, output pixels map into an
  oversized source coordinate grid, and final 3D coordinates are sampled once.
- Implements affine transforms, flips, smooth row offsets, value augmentation,
  line-coordinate mapping, and debug line overlays.
- Value augmentation runs as torch tensor operations on the configured device:
  brightness, contrast, gamma, noise, and separable Gaussian blur.
- Debug line overlays are drawn only as the final visualization step. The line
  coordinates themselves are transformed geometrically, not raster-warped.

`loader.py`

- Parses Vesuvius-style JSON configs into `FiberStrip2DConfig`.
- Opens the base volume and Lasagna manifest channels.
- Validates selected base-volume shape against the Lasagna manifest shape.
- Skips any fiber whose control points are outside the manifest/base-volume
  bounds.
- Loads only CP-local Lasagna normals needed for the requested strip window.
- Builds one sample as all configured strip-z offsets around one control point.
- Builds a batch by stacking deterministic samples.
- Computes prefetch chunk requests from explicit final coordinates.
- Implements `build_augmented_center_strip_source` to build CP-local source
  geometry once for augmentation contact sheets, then reuses it for each
  augmentation variant.

`runner.py`

- Provides the command-line entry point:

  ```bash
  python -m vesuvius.neural_tracing.fiber_trace_2d.runner
  ```

- Loads a config, optionally prefetches chunks, loads a batch, exports JPGs, and
  exports augmentation contact sheets.
- Prints augment-visualization timing rows and raw image stats.

`configs/loader_example.json`

- Example local Staticsheep config using:
  - PHercParis4 78keV masked base volume through the public Vesuvius S3 path;
  - local Lasagna manifest;
  - 128x128 strips;
  - 16 strip-z offsets;
  - current augmentation extrema.

## Config Shape

Top-level keys used by `load_config`:

- `datasets`: non-empty list of dataset entries.
- `batch_size`: number of control-point samples per loaded batch.
- `patch_shape_hw`: `[height, width]` output patch size.
- `strip_z_offset_count`: number of parallel strip-z offsets per sample.
- `strip_z_offset_step`: offset step in selected-scale voxels.
- `seed`: deterministic control-point sample seed.
- `prefetch_workers`: capped to 16.
- `volume_cache_dir`: optional cache directory for remote volume chunks.
- `volume_cache_offline`: passed to the Vesuvius Zarr cache opener.
- `volume_cache_retry_seconds`: passed to the Vesuvius Zarr cache opener.
- `augment_*`: parsed into `FiberStripAugmentConfig`.

Dataset entries must contain:

- `fiber_paths` or `fiber_glob`;
- `base_volume_path`;
- `base_volume_scale`;
- `lasagna_manifest_path`.

Optional dataset keys:

- `base_volume_auth_json` / `volume_auth_json`;
- `lasagna_auth_json`;
- legacy `volume_path` / `volume_scale` aliases.

`strip_z_offsets` is intentionally rejected; use count and step.

## Dataset Construction

`FiberStrip2DLoader.__init__` validates `batch_size`, derives strip-z offsets,
opens all configured records, and counts available control points.

For each dataset entry:

1. Open the selected base-volume level through `open_zarr`.
2. Load the Lasagna manifest with the existing 3D fiber trace helpers.
3. Open level 0 of the base volume and validate it against
   `lasagna_volume.base_shape_zyx`.
4. Validate the selected base-volume level shape against the manifest-derived
   level shape.
5. Open required Lasagna channels: `grad_mag`, `nx`, and `ny`.
6. Resolve fiber paths/globs and parse each VC3D fiber JSON.
7. Skip the whole fiber if any control point is outside the base-volume bounds.
8. Store a `_Record` containing fiber, volume, sampler, manifest channels, and
   scale metadata.

Only control points are checked during construction. Non-control line points are
not globally sampled up front. When a CP-local strip needs normals for a local
line window, missing or invalid Lasagna samples in that local window raise a
detailed error instead of being replaced.

## Sample Selection

The loader uses deterministic stateless sampling by sample index:

- `_random_flat_index(sample_index)` hashes `(seed, "cp", sample_index)` into a
  NumPy RNG seed;
- that RNG chooses one flat control-point index across all loaded records;
- `_locate_flat_index` maps it back to `(record_index, control_point_index)`.

Changing batch size changes which sample indices are grouped together, but not
the selected control point for a given sample index.

## Coordinate And Scale Semantics

All fiber coordinates are kept in base-volume coordinates.

`base_volume_scale` selects:

- which Zarr group/level is read;
- the patch pixel spacing used for strip coordinate construction.

For scale `s`, one output pixel advances by `2**s` base voxels. Coordinates are
passed to the VC3D sampler in base-coordinate order `(z, y, x)`; the sampler
converts them to VC3D `(x, y, z)` and provides the selected level.

`strip_z_offset_count` and `strip_z_offset_step` generate offsets centered
around zero. The default count/step yields 16 selected-scale offsets:

```text
-7, -6, ..., -1, 0, 1, ..., 8
```

Each offset is sampled as a separate 2D patch in the batch tensor.

## Lasagna Normal Handling

Normals are decoded through the existing Lasagna normal decoder:

```python
lasagna.omezarr_pyramid._decode_normals
```

For each requested point:

1. `grad_mag` is trilinearly sampled first and must be positive.
2. The eight neighboring `nx`/`ny` encoded normal samples are decoded.
3. The normal sign ambiguity is handled by accumulating a weighted tensor
   `normal outer normal`.
4. A principal axis is solved and oriented using the weighted hint.

This preserves the Lasagna ambiguous normal representation instead of inventing
a separate `normal_xyz` storage format.

## Batch Shapes

`load_batch` returns `FiberStrip2DBatch` with:

- `images`: `[batch, strip_z, 1, height, width]`, float32;
- `coords_zyx`: `[batch, strip_z, height, width, 3]`, float32 base coords;
- `valid_mask`: `[batch, strip_z, height, width]`, bool;
- `strip_z_offsets`: `[strip_z]`, float32;
- `control_point_indices`: `[batch]`, int32;
- `record_indices`: `[batch]`, int32;
- `fiber_paths`: one path per batch sample;
- `samples`: flat tuple of `FiberStripSample`, ordered by sample then offset;
- `cache_stats`: cache trace object returned by the existing Zarr cache tracer.

When augmentation is enabled, the loader builds an oversized source strip,
maps the requested output patch into it, samples the volume at final augmented
coordinates, and then applies value augmentation.

## Prefetch

`chunk_requests_for_sample_index` builds the same final coordinate grids that
loading would use, then asks the sampler which chunks are needed.

`prefetch(start_sample_index, sample_count)`:

- deduplicates requests by store identity and chunk key;
- skips requests whose cache file or negative marker already exists;
- downloads missing chunks through the request store;
- uses at most 16 worker threads;
- prints progress, MiB/s, ETA, and error count.

Prefetch is only for addressed base-volume image chunks. Lasagna channels are
local manifest channels in the current Staticsheep config and are not part of
the VC3D base-volume prefetch path.

## Augmentation Contact Sheet

`runner.py --augment-vis` exports one contact sheet for the center strip offset
of a deterministic sample index.

The runner builds CP-local source geometry once:

- selected record and control point;
- oversized source shape;
- CP-local line window;
- local Lasagna normals;
- torch-vectorized source strip grid.

Every contact-sheet cell then reuses that source grid and applies its own
geometric coordinate mapping before volume sampling.

Layout:

- row 1: lower-limit examples;
- row 2: upper-limit examples;
- row 3: random combined training-style examples.

Each cell is a raw clipped image, not percentile-normalized. Invalid pixels are
black. A red 50 percent opacity line is drawn from transformed line coordinates
with fixed screen-space thickness.

The runner writes:

- `augment_contact_sheet.jpg`;
- `augment_summary.txt`.

It also prints:

- timing table with `descriptor`, `line_window`, `lasagna_normals`,
  `strip_coords`, `coord_augmentation`, `volume_sample`, `value_augmentation`,
  `line_coords`, `to_u8`, and `overlay`;
- volume sampler stats;
- raw image stats per contact-sheet cell.

## Batch Export

Without `--augment-vis`, exporting a loaded batch writes:

- one JPG per sampled strip offset;
- one valid-mask JPG per sampled strip offset;
- `contact_sheet.jpg`;
- `summary.txt` with tensor shapes, strip offsets, record/control-point indices,
  fiber paths, and control-point coordinates.

## Runner Commands

Smoke load and export a batch:

```bash
PYTHONPATH=vesuvius/src python -m vesuvius.neural_tracing.fiber_trace_2d.runner config.json --sample-index 0 --export-dir /tmp/fiber_trace_2d_batch
```

Prefetch addressed chunks:

```bash
PYTHONPATH=vesuvius/src python -m vesuvius.neural_tracing.fiber_trace_2d.runner config.json --sample-index 0 --prefetch --prefetch-samples 8
```

Export an augmentation contact sheet:

```bash
PYTHONPATH=vesuvius/src python -m vesuvius.neural_tracing.fiber_trace_2d.runner config.json --sample-index 0 --augment-vis --export-dir /tmp/fiber_trace_2d_aug
```

For this checkout, prefer the more specific local command in
`planning/local_development.md` because it includes the VC3D Python binding path.

## Tests

Focused tests live in:

```text
vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py
```

They cover:

- config parsing and offset generation;
- deterministic sample selection;
- fake/local-array coordinate sampling;
- side-strip coordinate generation;
- torch vectorized strip-grid equivalence to the NumPy path;
- vectorized line-coordinate augmentation behavior;
- prefetch request generation;
- runner export behavior where practical.

Run them with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py
```

The tests are expected to use fake/local data and not require network access.

## Local Caveats

- Normal local runner usage depends on the VC3D Python bindings. See
  `planning/local_development.md`.
- Do not run this checkout with `PYTHONNOUSERSITE=1`; on the current machine it
  selects the wrong zarr/numcodecs environment.
- After changing VC3D Python bindings, update the editable installed package
  with `python -m pip install -e volume-cartographer --no-deps --break-system-packages`.
- Production remote sampling should use the VC3D blocking sampler, not the
  fallback NumPy sampler.
