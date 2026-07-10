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
  `build_side_strip_patch_grid_tensor_from_line_window`, which keeps the same
  frame semantics but vectorizes per-pixel interpolation work and returns torch
  tensors on the requested device. The older
  `build_side_strip_patch_grid_from_line_window_torch` wrapper remains a
  NumPy-returning compatibility boundary.
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
- Exposes torch-native transformed line/control-point coordinate helpers for
  loader internals, with NumPy wrappers kept for public/debug callers.
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

- Defines `FiberStripDirectionNet`, the V0 2D residual CNN.
- The default model has 10 residual blocks and 64 hidden channels.
- The default normalization is `GroupNorm(8, 64)`; smaller explicit hidden
  widths use the largest valid group count up to 8.
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
- Optionally caches CP-local source strip coordinates under
  `strip_coord_cache_dir`. Cache hits skip local line-window normal sampling and
  source-grid construction; larger cached source grids are center-cropped for
  smaller requests. Cached entries also contain source-space line and
  control-point pixel coordinates for unaugmented patch use.
- Keeps source grids, strip-z offset grids, geometric coordinate augmentation,
  and transformed line/control-point coordinates as torch tensors until an
  explicit consumer needs NumPy.
- Converts final coordinates and validity masks to contiguous CPU NumPy once
  immediately before VC3D `sample_coords` or dependency discovery; runner/sample
  metadata converts line and control-point coordinates at assembly/export time.
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
- Non-prefetch runner/debug patch loading uses the configured `augment_device`
  for torch coordinate generation. With `augment_device: "auto"`, CUDA is used
  when available. Prefetch dependency generation is intentionally CPU-only.
- Its prefetch mode is sample-count oriented:
  `--prefetch --prefetch-samples <control-point-samples>`.
- `--augment-profile` enables augment-visualization timing rows with cold and
  warm passes, full total/average-per-patch summaries, and no-first
  total/average summaries for warm-path timing.
- Provides `--line-trace-vis --checkpoint <snapshot> --export-dir <dir>` for
  V0.1 patch line-tracing inspection. This mode loads the deterministic
  center side-strip patch for `--sample-index`, runs the checkpointed direction
  model, bilinearly traces the decoded direction field from the transformed CP
  in both directions with a default 4 px trace step and a default receptive
  field margin of `1 + 2 * model_depth`, and writes
  `line_trace_vis.jpg` plus `line_trace_summary.txt`.
- The line-trace JPG normally has two columns: the unaugmented trace view, then
  the original patch with a flock of traces from random combined geometric
  training-style test-time augmentations inverse-warped back into original
  patch coordinates. `--line-trace-tta-count` controls the TTA count and
  defaults to 100.
- `--line-trace-vis --med-tta` adds a third column. That trace stays in the
  original patch space and uses the same random TTA direction fields as the
  flock column. At each step it samples the reference and TTA direction fields,
  inverse-warps TTA orientations back to the original patch frame, resolves the
  ambiguous direction sign against the previous step, and steps along the
  normalized median direction.
- Provides `--dir-vis --checkpoint <snapshot> --export-dir <dir>` for
  direction-field inspection. This mode loads the same deterministic center
  side-strip patch, runs the checkpointed direction model, decodes the
  direction field, scales the patch image by 2x, and writes `dir_vis.jpg` with
  short direction segments drawn every second source pixel.

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
- Supports `--benchmark` for a 100-batch training-work benchmark that skips
  test evaluation, TensorBoard, run-directory creation, and snapshots. It
  reports throughput as CNN image patches per second, where patches are the
  flattened `[control point, strip-z offset]` images sent through the model.
- Supports `--profile` on the benchmark path. It prints per-batch rows and a
  final milliseconds-per-patch summary for aggregate coordinate generation,
  descriptor lookup, strip-coordinate cache load, source geometry generation,
  line-coordinate generation, coordinate augmentation, base-volume sampling,
  torch value augmentation, forward plus loss, and backward plus optimizer step.
  Loader-side stage timings come from
  the shared `load_batch` / `build_strip_source` / `build_strip_patch_from_source`
  profile hooks, so profiling uses the same sampling path as normal training.
- Supports `--load-only` on the benchmark path. It still performs deterministic
  CP selection, CP-local source construction, coordinate augmentation, and
  base-volume sampling, but skips value/image augmentation, normalization,
  supervision construction, model forward, backward, and optimizer work.
- Flattens `[control_point_sample, strip_z_offset]` into one patch batch for the
  model.
- Computes direction targets from transformed line coordinates after geometric
  augmentation.
- Logs scalars/images to TensorBoard and writes `current.pt` / `best.pt`
  snapshots under the run directory.
- When `test_datasets` is configured, evaluates a fixed deterministic held-out
  batch at `training.test_interval` and uses test loss for current/best
  snapshots.

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
- `test_datasets`: optional non-empty list of held-out dataset entries using
  the same schema as `datasets`.
- `batch_size`: number of control-point samples per loaded batch.
- `patch_shape_hw`: `[height, width]` output patch size.
- `strip_z_offset_count`: number of parallel strip-z offsets per sample.
- `strip_z_offset_step`: offset step in selected-scale voxels.
- `seed`: deterministic control-point sample seed.
- `prefetch_workers`: transfer-worker count for prefetch; values above 16 are
  allowed and are not clamped by the loader.
- `prefetch_sampler_workers`: dependency/sampler producer-worker count for
  prefetch. This limits CPU-side CP-local source generation separately from
  download concurrency.
- `volume_cache_dir`: optional cache directory for remote volume chunks.
- `volume_cache_offline`: passed to the Vesuvius Zarr cache opener.
- `volume_cache_retry_seconds`: passed to the Vesuvius Zarr cache opener.
- `strip_coord_cache_dir`: optional disk cache for CP-local source strip
  coordinates. It is separate from the base-volume chunk cache.
- `augment_*`: parsed into `FiberStripAugmentConfig`.
- `training`: optional object used by `train.py`; ignored by the loader/debug
  runner config parser.

Training keys:

- `run_path`: parent directory for dated run directories.
- `run_name`: prefix for the run directory.
- `max_steps`: number of training steps; `0` means indefinite mode, where
  training repeats deterministic pseudo-random full-dataset CP passes.
- `max_sample_index`: optional exclusive deterministic sample-index limit;
  `0` means unlimited. Positive values make training wrap global sample
  positions through that deterministic prefix, so many steps can reuse a
  prefetched subset.
- `learning_rate`: AdamW learning rate.
- `scalar_log_interval`: TensorBoard scalar/console interval.
- `tensorboard_image_interval`: TensorBoard batch-image interval.
- `checkpoint_interval`: interval for writing `snapshots/current.pt`.
- `test_interval`: interval for deterministic held-out evaluation and, when
  `test_datasets` is configured, current snapshot writes.
- `test_control_points`: number of deterministic held-out CP samples per test
  evaluation.
- `test_start_sample_index`: deterministic held-out sample start index.
- `control_points_per_step`: deterministic CP samples per step; default `4`.
- `device`: `auto`, `cpu`, or a torch device string.
- `tensorboard_enabled`: set false for smoke tests without TensorBoard.
- `model_hidden_channels` and `model_depth`: V0 ResNet size knobs. Defaults
  are 64 hidden channels and 10 residual blocks.

Training prefetch:

- `python -m vesuvius.neural_tracing.fiber_trace_2d.train config.json
  --prefetch --prefetch-steps N` prefetches the first `N` training steps and
  overrides `training.max_steps`.
- Explicit `--prefetch-steps 0` also overrides `training.max_steps` and
  prefetches every configured training CP once in deterministic pseudo-random
  order, or the `training.max_sample_index` prefix when configured, plus every
  configured `test_datasets` CP once when held-out data is present.
- Omitting `--prefetch-steps` uses `training.max_steps`; if that configured
  value is `0`, omitted prefetch also means every configured training/test CP
  once.
- `--prefetch-start-step S` starts from the 1-based training step `S`.
- For positive prefetch step counts, sample count is `effective_steps *
  training.control_points_per_step`; start sample index is `(S - 1) *
  training.control_points_per_step` in the deterministic pseudo-random stream.
- Prefetch progress prints `idx=<exclusive-index>`, the largest contiguous
  exclusive deterministic sample-index prefix whose required chunks are
  cache-complete: cache hits, known/new missing markers, or completed
  successful downloads. That value can be used as `training.max_sample_index`
  for a later run over the prefetched subset.
- Negative `--prefetch-steps` values are rejected.
- Training prefetch only fetches base-volume chunks; Lasagna manifest channels
  are opened for geometry/normal metadata but are not prefetched.
- Prefetch uses dependency-only chunk discovery and Python-side cache
  classification. It does not call the image-sampling path just to warm the
  cache.

Training benchmark/profile:

- `python -m vesuvius.neural_tracing.fiber_trace_2d.train config.json
  --benchmark` runs 100 training batches and prints a final
  `patches_per_second` summary.
- `--profile` implies the same 100-batch benchmark work and additionally prints
  table rows with milliseconds per CNN patch for:
  - `coord`: aggregate descriptor, cache, source geometry, and line-coordinate
    work;
  - `desc`: deterministic sample descriptor lookup;
  - `cache`: strip-coordinate cache lookup/load;
  - `source`: uncached CP-local line window, Lasagna normals, and strip
    coordinate grid generation;
  - `line`: transformed line/control-point coordinates;
  - `coord_aug`: coordinate-space geometric augmentation;
  - `load`: base-volume Zarr/VC3D coordinate sampling;
  - `img_aug`: torch image/value augmentation;
  - `fw`: model forward plus loss;
  - `bw_step`: backward pass plus optimizer step.
- Benchmark/profile mode intentionally does not create a run directory or write
  checkpoints.
- Add `--load-only` to isolate loader and volume-sampling cost:

  ```bash
  python -m vesuvius.neural_tracing.fiber_trace_2d.train config.json --profile --load-only
  ```

  In this mode `img_aug`, `fw`, and `bw_step` report zero because those stages
  are intentionally skipped.

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

- `_random_flat_index(sample_index)` maps the sample index into a dataset pass
  and offset within that pass;
- each pass sorts all flat control-point indices by seeded content-based random
  keys, so every configured CP appears once before the stream repeats;
- `_locate_flat_index` maps it back to `(record_index, control_point_index)`.

Changing max steps, batch size, or control points per step changes how much of
the stream is consumed or how indices are grouped, but not the selected control
point for a given sample index.

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

The V0 trainer samples deterministic pseudo-random control-point groups by
step:

```text
start_sample_index = (step - 1) * training.control_points_per_step
```

Each deterministic random dataset pass visits every configured CP once and then
wraps to another deterministic pass. For finite `training.max_steps > 0`,
training stops after that many steps. For `training.max_steps = 0`, training
continues indefinitely.

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
V0. The optimization loss is MSE in the encoded two-channel representation.
For readability, training also reports folded unoriented angular error in
degrees over the same supervised pixels.

TensorBoard output is written under:

```text
<training.run_path>/<training.run_name>_<YYYYmmdd_HHMMSS>/
```

The trainer logs:

- `config/json` as text;
- `train/loss_direction`;
- `train/angle_error_mean_deg`;
- `train/supervision_samples`;
- `timing/load_ms`;
- cache hit/download diagnostics where available;
- `train/batch_direction_overlay` images showing the transformed centerline
  behind one short network-predicted direction segment at the transformed CP.
  The contact sheet picks one center-offset representative from each loaded
  control-point sample before filling with additional strip-z offsets.
- when `test_datasets` is configured, `test/loss_direction`,
  `test/angle_error_mean_deg`, `test/supervision_samples`, test cache diagnostics, and
  `test/batch_direction_overlay` at test evaluation steps.

Console progress prints the same loss, mean angle error in degrees,
supervision, and load-time summary for every step whose deterministic
control-point sample range starts before sample index 100, then return to
`training.scalar_log_interval`.

Snapshots are written under:

```text
<run_dir>/snapshots/current.pt
<run_dir>/snapshots/best.pt
```

With `test_datasets`, `current.pt` is written at step 1, every
`training.test_interval`, and the final step; `best.pt` tracks the lowest
observed test loss. Without `test_datasets`, snapshots keep the train-only
behavior: `current.pt` follows `training.checkpoint_interval`, and `best.pt`
tracks the lowest observed training loss.

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
  download futures, configured transfer workers, configured sampler/dependency
  producer workers, skipped samples, errors, and MiB/s. The download
  denominator counts chunks that were not cache hits or pre-existing `.empty`
  markers. While sample dependency generation is
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

With `--augment-profile`, it also prints two timing tables:

- timing table with `descriptor`, `line_window`, `lasagna_normals`,
  `strip_coords`, `coord_augmentation`, `volume_sample`, `value_augmentation`,
  `line_coords`, `to_u8`, and `overlay`, plus `total`, `avg/patch`,
  `total/no-first`, and `avg/no-first` rows;
- volume sampler stats.

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

Add `--augment-profile` to print two timing passes for the same augment-vis
sample: pass 1 for cold/first-use costs and pass 2 for warmed costs.

Export a line-tracing inspection image:

```bash
PYTHONPATH=vesuvius/src python -m vesuvius.neural_tracing.fiber_trace_2d.runner config.json --sample-index 0 --line-trace-vis --checkpoint /path/to/current.pt --export-dir /tmp/fiber_trace_2d_trace
```

Export a direction-field inspection image:

```bash
PYTHONPATH=vesuvius/src python -m vesuvius.neural_tracing.fiber_trace_2d.runner config.json --sample-index 0 --dir-vis --checkpoint /path/to/current.pt --export-dir /tmp/fiber_trace_2d_dir
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
