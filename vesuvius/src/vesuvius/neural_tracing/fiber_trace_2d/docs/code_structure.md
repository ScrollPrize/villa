# 2D Fiber Trace Loader Code Structure

This package implements a loader/debug runner and a V0 training path for 2D
fiber side-strip patches around VC3D fiber control points.

The important behavior is:

- read VC3D fiber JSONs and Lasagna manifests;
- select deterministic control-point samples;
- build VC3D-equivalent side-strip coordinate grids;
- sample image values through the VC3D blocking coordinate sampler;
- optionally apply coordinate-space geometric augmentation and torch value
  augmentation;
- train a small 2D direction model using CP-local Lasagna two-cos-channel
  direction targets;
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
- The dense grid also exposes the per-pixel strip offset axis so nearby
  strip-z patches can be derived from one CP-local source grid.

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
  - uses `Volume.collect_coords_dependencies(...)` for prefetch dependency
    discovery, without sampling image values;
  - exposes VC3D persistent-cache data and `.empty` marker paths for Python
    prefetch classification.
- `NumpyZarrCoordinateSampler` remains useful for tests and local fake arrays.
- `make_coordinate_sampler` currently returns the VC3D sampler for normal
  runtime.

`augmentation.py`

- Defines `FiberStripAugmentConfig` and `FiberStripAugmentParams`.
- Builds geometric augmentation maps in strip pixel coordinates. The image is
  never geometrically warped after loading; instead, output pixels map into an
  oversized source coordinate grid, and final 3D coordinates are sampled once.
- Affine shift is composed as an output-space translation after scale/flip, and
  the inverse sampling grid plus transformed line/control-point coordinates use
  that same order.
- Implements affine transforms, flips, smooth row offsets, value augmentation,
  line-coordinate mapping, and debug line overlays.
- Value augmentation runs as torch tensor operations on the configured device:
  brightness, contrast, gamma, noise, and separable Gaussian blur.
- Debug line overlays are drawn only as the final visualization step. The line
  coordinates themselves are transformed geometrically, not raster-warped.

`direction.py`

- Implements the Lasagna ambiguous two-cos-channel strip direction encoding:
  `0.5 + 0.5*cos(2*theta)` and `0.5 + 0.5*cos(2*theta + pi/4)`.
- Builds V0 training targets from transformed strip-line coordinates.
- Selects only the eight neighboring pixels around the rounded transformed CP
  location, filtered by valid image samples.
- Provides a visualization-only approximate decoder for drawing predicted
  direction arrows.

`model.py`

- Defines `FiberStripDirectionNet`, a deliberately small V0 2D CNN.
- Consumes flattened strip patches shaped `[patch_batch, 1, height, width]`.
- Outputs exactly two per-pixel direction channels in the Lasagna encoded
  representation.

`loader.py`

- Parses Vesuvius-style JSON configs into `FiberStrip2DConfig`.
- Opens the base volume and Lasagna manifest channels.
- Validates selected base-volume shape against the Lasagna manifest shape.
- Skips any fiber whose control points are outside the manifest/base-volume
  bounds.
- Loads only CP-local Lasagna normals needed for the requested strip window.
- Builds one CP-local source strip with the torch-vectorized augment-vis path,
  then derives all configured strip-z offsets from that source using the stored
  strip offset axis.
- Builds one sample as all configured strip-z offsets around one control point.
- Builds a batch by stacking deterministic samples.
- Implements `build_strip_source` / `build_strip_patch_from_source` as the
  shared path for training, runner loading, augment-vis, and prefetch.
- Computes prefetch envelopes from the same shared source geometry, asks the
  sampler for dependency-only chunk requests, deduplicates those requests, and
  fetches only chunks not already represented in the VC3D persistent cache.
- Keeps `build_augmented_center_strip_source` as a compatibility wrapper for
  the runner contact sheet.

`runner.py`

- Provides the command-line entry point:

  ```bash
  python -m vesuvius.neural_tracing.fiber_trace_2d.runner
  ```

- Loads a config, optionally prefetches chunks, loads a batch, exports JPGs, and
  exports augmentation contact sheets.
- Its prefetch mode is sample-count oriented:
  `--prefetch --prefetch-samples <control-point-samples>`.
- Prints augment-visualization timing rows and raw image stats.

`train.py`

- Provides the command-line entry point:

  ```bash
  python -m vesuvius.neural_tracing.fiber_trace_2d.train
  ```

- Loads the same JSON config as the runner, then reads a `training` section for
  optimizer, logging, and snapshot settings.
- Builds batches with `FiberStrip2DLoader`, so production training uses the
  same VC3D blocking sampler and coordinate-space augmentation path as
  augment-vis.
- Supports `--prefetch` for training-oriented prefetch-only runs. This maps
  training steps to deterministic control-point sample ranges, calls loader
  prefetch, and exits before model, optimizer, TensorBoard, run-directory, or
  snapshot setup.
- Flattens `[control_point_sample, strip_z_offset]` into one patch batch for the
  model.
- Computes direction targets from transformed line coordinates after geometric
  augmentation.
- Logs scalars/images to TensorBoard and writes `current.pt` / `best.pt`
  snapshots under the run directory.

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
- `training`: optional object used by `train.py`; ignored by the loader/debug
  runner config parser.

Training keys:

- `run_path`: parent directory for dated run directories.
- `run_name`: prefix for the run directory.
- `max_steps`: number of training steps.
- `learning_rate`: AdamW learning rate.
- `scalar_log_interval`: TensorBoard scalar/console interval.
- `tensorboard_image_interval`: TensorBoard batch-image interval.
- `checkpoint_interval`: interval for writing `snapshots/current.pt`.
- `control_points_per_step`: deterministic CP samples per step; default `4`.
- `device`: `auto`, `cpu`, or a torch device string.
- `tensorboard_enabled`: set false for smoke tests without TensorBoard.
- `model_hidden_channels` and `model_depth`: V0 CNN size knobs.

Training prefetch:

- `python -m vesuvius.neural_tracing.fiber_trace_2d.train config.json
  --prefetch --prefetch-steps N` prefetches the first `N` training steps.
- `--prefetch-steps 0` or omitting `--prefetch-steps` prefetches all configured
  `training.max_steps`.
- `--prefetch-start-step S` starts from the 1-based training step `S`.
- Sample count is `effective_steps * training.control_points_per_step`; start
  sample index is `(S - 1) * training.control_points_per_step`.
- Negative `--prefetch-steps` values are rejected.
- Training prefetch only fetches base-volume chunks; Lasagna manifest channels
  are opened for geometry/normal metadata but are not prefetched.
- Prefetch uses dependency-only chunk discovery and Python-side cache
  classification. It does not call the image-sampling path just to warm the
  cache.

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
detailed error instead of being replaced. Batch-oriented callers handle that
data-quality error differently: prefetch and training skip the invalid
deterministic sample, report the first reason, and continue with later sample
indices.

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

Each offset is returned as a separate 2D patch in the batch tensor, but the
coordinates are derived from one CP-local source grid rather than rebuilding
the side-strip frame and line window for every offset.

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

The loader builds the same CP-local source strip path used by augment-vis. When
augmentation is enabled, that source strip is oversized for the configured
augmentation envelope, output pixels map into it, the volume is sampled at final
augmented coordinates, and value augmentation is applied afterward.

Each `FiberStripSample` also stores `line_xy` and `control_point_xy` in final
output-pixel coordinates after geometric augmentation. Training labels and
debug overlays use those coordinates directly.

## V0 Training

The V0 trainer samples deterministic control-point groups by step:

```text
start_sample_index = (step - 1) * training.control_points_per_step
```

The default training shape is four control-point samples times 16 strip-z
offsets, producing 64 patches. `load_batch` returns `[4, 16, 1, H, W]`; the
trainer reshapes this to `[64, 1, H, W]` before the CNN forward pass.
If one deterministic sample cannot build its CP-local Lasagna normal window
for data reasons such as `grad_mag == 0`, `load_batch` skips it and advances
through following deterministic sample indices until the requested number of
control-point samples is loaded.

Images are normalized per patch over valid pixels. Invalid pixels are set to
zero after normalization.

Targets are built from `FiberStripSample.line_xy` and
`FiberStripSample.control_point_xy`, which are already transformed output-pixel
coordinates from the loader's augmentation path. The local tangent near the
transformed CP is encoded into Lasagna's ambiguous two-cos-channel format:

```text
cos2theta = (dx^2 - dy^2) / (dx^2 + dy^2 + eps)
sin2theta = 2*dx*dy / (dx^2 + dy^2 + eps)
dir0 = 0.5 + 0.5*cos2theta
dir1 = 0.5 + 0.5*(cos2theta - sin2theta)/sqrt(2)
```

Only the eight neighboring pixels around the rounded transformed CP location
are supervised. The model is not supervised on the whole line or full patch in
V0.

TensorBoard output is written under:

```text
<training.run_path>/<training.run_name>_<YYYYmmdd_HHMMSS>/
```

The trainer logs:

- `config/json` as text;
- `train/loss_direction`;
- `train/supervision_samples`;
- `timing/load_ms`;
- cache hit/download diagnostics where available;
- `train/batch_direction_overlay` images showing the transformed line, CP
  neighborhood, and predicted direction arrow.

Snapshots are written under:

```text
<run_dir>/snapshots/current.pt
<run_dir>/snapshots/best.pt
```

## Prefetch

`chunk_requests_for_sample_index` is a compatibility/test helper that derives
dependency requests from the same conservative source-envelope coordinates used
by prefetch.

`prefetch(start_sample_index, sample_count)`:

- builds the shared CP-local source once per deterministic sample index;
- derives each configured strip-z offset from that source;
- sends the source-envelope coordinates to `chunk_requests_for_coords`, which
  maps them to dependency-only base-volume chunk metadata;
- deduplicates globally by `(store_identity, key)`;
- treats existing VC3D persistent-cache data files as hits;
- treats existing `<cache>/level_<level>/<iz>/<iy>/<ix>.empty` files as
  known-missing hits;
- fetches only still-missing direct-source chunks with bounded Python workers,
  writes data through unique temp files followed by atomic rename, and writes
  zero-byte `.empty` markers for definitive missing chunks;
- prints sample/dependency progress only while dependency generation is still
  running, then switches to download-only progress. The live line includes
  unique chunks, cache hits, known-missing chunks, downloaded chunks, queued
  download futures, configured transfer workers, skipped samples, errors, and
  MiB/s. The download denominator counts chunks that were not cache hits or
  pre-existing `.empty` markers. While sample dependency generation is
  incomplete, download ETA extrapolates from observed chunks per sample and
  observed cache-hit/known-missing/download-needed ratios;
- reports invalid deterministic sample skips separately from download errors
  and includes the first skip reason.

For VC3D-backed remote volumes, dependency discovery returns the authoritative
remote chunk URL/key, final persistent-cache data path, `.empty` marker path,
persistent extension, and cache payload format. Python prefetch does not
reconstruct those paths. The current Python writer supports only uncompressed
direct-source chunks where the remote payload is exactly the `.bin` payload VC3D
expects; compressed, filtered, sharded, or byte-swapped payloads fail clearly
until explicit codec support is added.

Prefetch covers the configured augmentation envelope, not a single random
augmentation draw. It may fetch a conservative superset for one patch, but later
training draws within the configured extrema should be covered by the same
cached base-volume chunks.

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

Every contact-sheet cell then delegates to the same shared patch builder used
by training and applies its own geometric coordinate mapping before volume
sampling.

Layout:

- row 1: lower-limit examples;
- row 2: upper-limit examples;
- row 3: random combined training-style examples.

Each cell has a top label band naming the augmentation, then a raw clipped
image below it. Labels do not cover image pixels. Invalid pixels are black. A
red 50 percent opacity line is drawn from transformed line coordinates with
fixed screen-space thickness, and a final thin cyan vertical marker shows the
transformed control-point coordinate from `FiberStripSample.control_point_xy`
while leaving the CP pixel itself visible.

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

Run V0 direction training:

```bash
PYTHONPATH=vesuvius/src python -m vesuvius.neural_tracing.fiber_trace_2d.train config.json
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
- dependency-only prefetch, cache-hit / `.empty` marker handling, and
  augmentation-envelope dependency coverage;
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
