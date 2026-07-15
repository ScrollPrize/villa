# 2D Fiber Trace Loader Code Structure

This package implements a loader/debug runner and a V0 training path for 2D
fiber side-strip patches around VC3D fiber control points.

The sibling package `vesuvius.neural_tracing.fiber_trace_3d` implements the V0
CP-centered 3D model path. It shares fiber JSON/NML parsing conventions and
Lasagna-manifest dataset semantics with this package, but it does not use 2D
side/top strip input loading.

## 3D CP Model Package

`fiber_trace_3d/direction.py`

- Implements Lasagna's 3D direction target layout as three ambiguous 2D
  double-angle projection pairs:
  `dir0_z,dir1_z`, `dir0_y,dir1_y`, and `dir0_x,dir1_x`.
- Decodes Lasagna 3x2 predictions analytically by decoding each two-channel
  projection with `theta = atan2(sin2theta, cos2theta) / 2` and applying the
  Lasagna three-plane reconstruction/sign-alignment logic. There is no
  unit-sphere candidate or grid-search decode path.
- Provides projection-magnitude weights for directions that are nearly
  degenerate in one projection plane.

`fiber_trace_3d/model.py`

- Wraps `Vesuvius3dUnetModel` for a seven-channel output layout.
- The first six channels are sigmoid Lasagna 3x2 direction channels.
- The seventh channel is sigmoid sheet/fiber presence.
- Config derives `features_per_stage`, `unet_base_channels`, `unet_depth`,
  `strides`, and `decoder_upsample_mode` the same way as the existing 3D fiber
  model helpers.
- The fiber 3D wrapper defaults to `BatchNorm3d` normalization. The trainer has
  no internal micro-batching, so the configured `batch_size` is the real batch
  used for BatchNorm statistics. Set `model_3d.normalization: "none"` to disable
  normalization explicitly.

`fiber_trace_3d/loader.py`

- Parses Vesuvius-style JSON configs into `FiberTrace3DConfig`.
- Uses `fiber_trace_2d.fiber_json.load_fiber_file`, so VC3D JSON, NML, and
  dataset-level XYZ affine transforms follow the same rules as the 2D loader.
- Requires Lasagna manifests for normal dataset entries and validates the base
  volume shape against the manifest.
- Samples ordinary CP-centered 3D ZYX patches from the selected Zarr level
  through explicit coordinates and the VC3D blocking coordinate sampler. It
  does not create fiber-aligned strips or slices for 3D input, and normal
  training no longer loads an oversized zarr block for torch `grid_sample`.
- Builds the final regular 3D patch with explicit coordinate maps.
  `backward_source_zyx` maps output voxels to source-volume coordinates for
  image sampling, while source fiber points are mapped into output patch
  coordinates by the matching analytic forward transform from the same
  augmentation parameters.
- Coordinate-space geometric augmentation supports CP-local shift, isotropic
  scale, arbitrary 3D rotation, independent axis flips, and opt-in smooth
  displacement. Smooth displacement modes (`1d`, `2d`, `3d`) are built as
  explicit paired maps; runtime paths do not invert one map direction by search
  or solve.
- Value augmentation includes normalization, brightness, contrast, gamma, noise,
  separable isotropic Gaussian blur, and opt-in anisotropic blur. Anisotropic
  blur is a torch value operation after volume sampling, not a geometric
  transform.
- Builds direction and presence targets in source-format-specific form.
  NML records supervise along the transformed fiber centerline by drawing
  rounded output-space line voxels; JSON/array records supervise only the
  sampled CP neighborhood. NML line targets are not radius-expanded tubes.
  Direction labels use the six Lasagna 3x2 channels, and presence negatives
  are balanced in the valid interior.
- Prefetch uses the same explicit coordinate path as training and collects
  VC3D chunk dependencies instead of conservative zarr crop bboxes.

`fiber_trace_3d/projection.py`

- Projects analytically decoded six-channel Lasagna predictions into
  caller-provided 2D strip-frame axes. The frame axes may be constant or
  per-pixel arrays from a Trace2CP segment source.

`fiber_trace_3d/trace2cp_bridge.py`

- Samples dense 3D checkpoint outputs at explicit 2D Trace2CP strip
  coordinates, projects the six Lasagna direction channels into the local strip
  frame, carries presence values through, and calls the existing 2D Trace2CP
  scorer.
- This is for test/metric integration with the existing 2D tracer; 3D training
  and loading remain CP-centered block loading.

`fiber_trace_3d/train.py`

- Command-line entry point:

  ```bash
  python -m vesuvius.neural_tracing.fiber_trace_3d.train <config.json>
  ```

- Supports `--prefetch`, `--prefetch-steps`, `--benchmark`, `--load-only`, and
  `--trace2cp-vis`.
- Writes snapshots to `<run_path>/<run_name>_<datestr>/snapshots/current.pt`
  and `best.pt`.
- Logs scalar losses/timings and the full training config JSON to TensorBoard
  when `training.tensorboard_enabled` is true.
  Direction reporting includes `train/angle_mean_deg` and, when held-out
  `test_datasets` are configured, `test/angle_mean_deg`.
- When `test_interval > 0`, configured test evaluation also runs once at step
  0 before the first optimizer step, so initial performance is visible.
- Logs `train_sample_3d/principal_slices` at `training.sample_vis_interval`.
  The sheet uses the sampled CP's three principal planes with three columns:
  volume image with projected GT line and predicted CP direction overlay, target
  presence, and predicted presence. The GT line overlay draws target-line
  portions within 2 voxels of the displayed slice plane. The target-presence
  panel is max-pooled in 3D for visualization only so one-voxel line targets are
  easier to see; the loss target is unchanged.
- `batch_size` is the actual CP-patch batch passed through the 3D U-Net. The
  trainer does not internally micro-batch.
- Normal training and `--benchmark --load-only` use
  `torch.utils.data.DataLoader` process workers when
  `training.loader_workers > 0`. Each worker lazily constructs its own
  `FiberTrace3DLoader` and VC3D coordinate sampler, and each DataLoader item is
  a complete `FiberTrace3DBatch` keyed by deterministic batch index.
- For 3D loading, omitted or `null` `volume_cache_memory_mib` resolves in
  Python to a 512 MiB VC3D decoded/hot-cache cap per loader/worker. Explicit
  positive values override this default.
- Worker batches are returned on CPU. The main process transfers the complete
  batch to the configured training device before model execution. Console
  `load_ms` includes DataLoader wait plus this main-process transfer;
  `to_device_ms` reports the transfer portion separately.
- Worker batches do not include full dense supervision tensors. They carry
  compact target descriptors instead: CP-only samples store local CP/tangent
  metadata, and NML dense-line samples store transformed output-space segment
  endpoints plus patch bboxes. The main process calls
  `fiber_trace_3d.targets.materialize_targets(...)` after transfer to create
  dense presence targets/masks on the training device. Direction supervision is
  sparse: the materializer builds `direction_indices_bzyx`,
  `direction_target_sparse`, and `direction_weight_sparse`, and the loss
  gathers the predicted six-channel direction output at those supervised
  centerline/CP voxels. The normal training path does not allocate a full
  dense `[B,6,Z,Y,X]` direction target.
- In `--benchmark` mode, `cpu_ms` and `cpu_x` report sampled CPU time for the
  main process plus DataLoader worker processes during each benchmark row where
  `/proc/<pid>/stat` is available. `cpu_x=1.0` is roughly one fully occupied
  CPU core for that row.
- The 3D load-only benchmark also exposes worker-side stage timings when using
  DataLoader workers: `worker_ms`, `worker_cpu`, `cpu/w`, loader construction,
  descriptor lookup, augmentation parameter sampling, coordinate-map creation,
  coordinate conversion/valid-mask generation, VC3D volume sampling, tensor
  conversion, value augmentation, compact target-spec generation, and batch
  stacking. Main-process target materialization is reported separately with
  `target_ms` plus sparse/dense target columns such as `line_idx`, `cp_idx`,
  `scatter`, `dir_enc`, `gpu_mask`, `linePts`, `dirPts`, and `posK`. `wait_ms`
  is the main process blocking on the next DataLoader item; it is distinct from
  `to_device_ms`, the CPU-to-training-device transfer. With worker processes,
  the first `loader_workers` benchmark rows can include worker-local loader
  construction and should be excluded from steady-state throughput summaries.
- When `training.test_trace2cp_enabled` is true, test evaluation builds
  Trace2CP side-strip geometry with the 2D loader, runs tiled dense 3D inference
  blocks over the requested strip coordinates, projects direction/presence into
  the 2D frame, logs `test/trace2cp_error`, and uses that value for `best.pt`
  selection. Dense 3D test loss remains a diagnostic.
- 3D prefetch uses the same explicit coordinate path as training, then asks
  VC3D for chunk dependency metadata. Because the VC3D dependency binding
  accepts 2D coordinate surfaces, rank-4 regular patch coordinates such as
  `[Z,Y,X,3]` are flattened to `[Z*Y,X,3]`, matching the regular training
  sampling adapter.
- 3D prefetch follows the 2D streaming prefetch architecture: bounded
  dependency producer workers controlled by `prefetch_sampler_workers`,
  bounded transfer workers controlled by `prefetch_workers`, immediate
  cache-hit / `.empty` / download classification, earliest-sample download
  priority, deterministic safe-prefix `idx`, and live progress while
  dependencies are still being generated. The 3D-only simplification is that
  each sample contributes one CP-centered 3D augmentation-envelope volume; no
  strip-z loop or top-view branch exists in 3D.

Example commands:

```bash
PYTHONPATH=vesuvius/src:. python -m vesuvius.neural_tracing.fiber_trace_3d.train vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/loader_example.json --benchmark --load-only
PYTHONPATH=vesuvius/src:. python -m vesuvius.neural_tracing.fiber_trace_3d.train vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/loader_example.json --prefetch --prefetch-steps 1
PYTHONPATH=vesuvius/src:. python -m vesuvius.neural_tracing.fiber_trace_3d.train vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/loader_example.json
PYTHONPATH=vesuvius/src:. python -m vesuvius.neural_tracing.fiber_trace_3d.train vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/loader_example.json --trace2cp-vis --checkpoint /path/to/best.pt --sample-index 0 --export-dir /tmp/fiber_trace_3d_trace2cp
```

The important behavior is:

- read VC3D fiber JSONs or NML fiber annotations plus Lasagna manifests;
- select deterministic control-point samples;
- build VC3D-equivalent side-strip coordinate grids;
- sample image values through the VC3D blocking coordinate sampler;
- optionally apply coordinate-space geometric augmentation and torch value
  augmentation;
- train a small 2D direction model using CP-local Lasagna two-cos-channel
  direction targets;
- optionally train a sheet/fiber-presence head from transformed CP pixels and
  reachable non-CP pixels;
- optionally train a second top-view direction plus distance-transform model
  from VC3D-style top-strip slices;
- optionally train a CP-local contrastive embedding head using cosine
  similarity;
- export JPG batches and augmentation contact sheets for inspection;
- prefetch addressed base-volume chunks into the configured cache.

## Module Map

`fiber_json.py`

- Re-exports the existing VC3D fiber JSON parser from
  `vesuvius.neural_tracing.fiber_trace.fiber_json`.
- Keeps control-point and line-point parsing semantics shared with the existing
  3D fiber trace code.
- Adds NML parsing for Knossos/WebKnossos `<thing>` graphs. NML nodes are
  ordered by `<edge>` connectivity; each open simple path component is
  normalized into one `Vc3dFiber`. Branching, closed, malformed, or singleton
  components are skipped or rejected with diagnostics instead of being guessed.
- Provides a shared `load_fiber_file` entrypoint that returns one or more
  normalized `Vc3dFiber` objects for `.json` or `.nml` sources.

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
- `CoordinateSampler.sample_coord_batch` loads a stack of coordinate patches
  and returns `[patches,H,W]` image and valid-mask arrays. The default
  implementation flattens `[patches,H,W,3]` into one larger coordinate image,
  calls ordinary `sample_coords` once, and reshapes back.
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
- Training, augment-vis, line tracing, Trace2CP, labels, TTA, and core loader
  paths keep geometric image-space augmentation helpers absent. Any geometric
  change on those paths must be represented as coordinate manipulation before
  sampling/slicing the patch. Only value-only image changes such as brightness,
  contrast, gamma, noise, and blur run after image sampling. The runner's
  `--dir-vis` mode is the only diagnostic exception: it applies pixel-perfect
  image-space flips and 90-degree rotations to an already sampled center patch
  for checkpoint robustness inspection.
- Provides `StripAugmentTransform`, the shared paired transform for geometric
  augmentation. At construction it bakes the whole geometric stack into two
  concrete tensors: `backward_map_xy` maps output pixels to source pixels for
  image/coordinate sampling, and `forward_map_xy` maps source pixels to output
  pixels for line/CP lookup. Runtime point mapping samples these cached maps; it
  does not re-run affine/smooth formulas or invert one direction from the
  other.
- Exposes torch-native transformed line/control-point coordinate helpers for
  loader internals, with NumPy wrappers kept for public/debug callers.
- Affine shift is composed as an output-space translation after scale/flip, and
  the inverse sampling grid plus transformed line/control-point coordinates use
  that same order.
- Implements affine transforms, flips, smooth row offsets, value augmentation,
  line-coordinate mapping, and debug line overlays. Smooth line/CP mapping uses
  bilinear lookup against the prebuilt `forward_map_xy`; smooth interpolation is
  only used while constructing the maps.
- Provides a batched sparse bilinear gather for line/CP point lookup. Training
  batch construction stacks `forward_map_xy` tensors across strip-z offsets and
  avoids tiny sparse `grid_sample` calls.
- Loader patch construction reuses one transform object for coordinate
  augmentation and line/CP mapping. Line points and the CP are mapped together
  in one batched source-to-output call, and shared line/CP results can be reused
  across strip-z offsets with identical augmentation params.
- The loader also exposes coordinate-sampled TTA builders for runner
  inspection. They start from a base patch's `coords_zyx`, transform those
  coordinates through the same augmentation maps, sample the volume at the
  transformed coordinates, and return both the output-to-reference
  `source_xy_grid` and the reference-to-output map. Median-TTA tracing uses the
  reference-to-output map to locate a reference trace point inside a TTA field
  in constant time, then uses `source_xy_grid` to map sampled TTA directions
  back to the reference frame.
- Training `build_sample` batches compatible strip-z offset coordinate
  augmentation by stacking `backward_map_xy` tensors and running dense
  coordinate/valid-mask sampling over the stack. It then sends the whole
  strip-z stack through `CoordinateSampler.sample_coord_batch`, so normal VC3D
  loading is one flattened coordinate-sampling call per CP sample instead of
  one call per strip-z offset.
- Value augmentation runs as torch tensor operations on the configured device:
  brightness, contrast, gamma, noise, and separable Gaussian blur. Training
  `build_sample` applies value augmentation to the loaded image stack, while
  retaining per-patch noise seeds and blur parameters. Batched value
  augmentation applies variable per-patch Gaussian blur with grouped
  convolutions so CUDA training does not launch one blur convolution pair per
  patch.
- Debug line overlays are drawn only as the final visualization step. The line
  coordinates themselves are transformed geometrically, not raster-warped.

`direction.py`

- Implements the Lasagna ambiguous two-cos-channel strip direction encoding:
  `0.5 + 0.5*cos(2*theta)` and `0.5 + 0.5*cos(2*theta + pi/4)`.
- Builds V0 training targets from transformed strip-line coordinates.
- Selects only the eight neighboring pixels around the rounded transformed CP
  location, filtered by valid image samples.
- Decodes predicted direction channels with Lasagna's analytic inverse:
  recover `cos(2*theta)` and `sin(2*theta)` from the two channels, then use
  `atan2(...)/2`. There is no binned candidate-angle lookup decoder.

`model.py`

- Defines `FiberStripDirectionNet`, the V0 2D residual CNN.
- The default model has 10 residual blocks and 64 hidden channels.
- The default normalization is `BatchNorm2d`.
- Consumes flattened strip patches shaped `[patch_batch, 1, height, width]`.
- Outputs two per-pixel direction channels in the Lasagna encoded
  representation first. If enabled, one sigmoid sheet/fiber-presence channel
  follows. Contrastive embedding channels are appended after direction and any
  presence channel only when contrastive training/inference is explicitly
  configured with embedding channels.
- The same network class is also reused for the optional top-view auxiliary
  model. In that layout the scalar sigmoid head is interpreted as a
  fiber-center distance transform rather than side-strip presence.

`loader.py`

- Parses Vesuvius-style JSON configs into `FiberStrip2DConfig`.
- Opens the base volume and Lasagna manifest channels.
- Validates selected base-volume shape against the Lasagna manifest shape.
- Skips any fiber whose control points are outside the manifest/base-volume
  bounds.
- Builds a compact in-RAM fiber-line geometry store at loader startup. The
  loader first computes CP source-window line-index ranges, samples Lasagna
  normals only for those required ranges with batched channel reads, creates
  valid frame intervals, and shares compact line/frame arrays read-only by
  loader clones and threaded workers in the same process. Startup record
  construction uses process workers when `loader_workers > 1`; each process
  opens its own base-volume and Lasagna handles, returns compact geometry, and
  the parent assembles the single shared store in original record order.
- Builds one CP source strip from the compact geometry store with the
  torch-vectorized augment-vis path, then derives all configured strip-z
  offsets from that source using the stored strip offset axis.
- Can derive one top-view patch per loaded CP sample with
  `load_top_batch_for_batch`. That path uses the same VC3D-style top-strip
  `lineSurface` coordinate construction as Trace2CP visualization and reuses
  the CP sample's deterministic geometric/value augmentation parameters. Its
  source geometry uses the same compact geometry store as side-view source
  construction. Runtime preparation mirrors the side-view path: batched
  augmentation maps, batched coordinate resampling, and grouped
  `CoordinateSampler.sample_coord_batch` calls. The transformed line and CP
  pixel coordinates are reused from the center side-strip sample because those
  pixel-frame coordinates are identical for the same augmentation.
- Does not use a persistent dense strip-coordinate cache. The old
  `strip_coord_cache_dir` config key is rejected; dense source coordinates are
  generated from compact in-RAM frame arrays and then discarded after the
  batch/source consumer no longer needs them.
- Keeps source grids, strip-z offset grids, geometric coordinate augmentation,
  and transformed line/control-point coordinates as torch tensors until an
  explicit consumer needs NumPy.
- Converts final coordinates and validity masks to contiguous CPU NumPy once
  immediately before VC3D `sample_coords` or dependency discovery; runner/sample
  metadata converts line and control-point coordinates at assembly/export time.
- Builds one sample as all configured strip-z offsets around one control point.
- Builds a batch by stacking deterministic samples.
- Uses the bounded sample index only for CP/data selection; when training wraps
  a bounded `max_sample_index` prefix, augmentation parameters are seeded by
  the unbounded raw training stream index so repeated CPs do not replay the
  same transform.
- Builds contrastive training batches by selecting deterministic shuffled
  groups of `N` CPs from one fiber, concatenating consecutive same-fiber groups
  to fill the configured training batch, using independent geometric
  augmentation seeds per patch, and synchronizing value/image augmentation
  seeds within each group.
- Implements `build_strip_source` / `build_strip_patch_from_source` as the
  shared path for training, runner loading, augment-vis, and prefetch.
- Implements `build_trace2cp_segment_patch` for runner inspection of a segment
  between two control points in the same fiber. This path resolves the start CP
  from the deterministic sample stream, validates the target CP, builds a
  Lasagna/VC3D-style side-strip segment covering both CPs plus margin, and
  samples the center strip-z image through the normal coordinate sampler. The
  returned segment sample carries the original line indices and signed Lasagna
  normals used for that strip, plus the actual start/target strip row-axis
  vectors after frame construction. Whole-fiber Trace2CP can pass a shared-CP
  row-axis reference into this path so adjacent pair-local strips keep the same
  vertical row orientation despite Lasagna normal sign ambiguity.
- Implements `build_trace2cp_refined_segment_source` for iterative Trace2CP
  refinement. It samples a prior segment source at a smoothed fused trace
  `(x,y,z)` to recover volume-space centerline points and strip-normal axes,
  then builds a fresh side-strip segment from those points. The synthetic line
  keeps endpoint context on both sides of the CP pair so the next pass behaves
  like an independent Trace2CP run on a new line source. This path samples the
  volume again for the next pass; it does not warp the previous image.
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
  field margin of `model_depth`, and writes
  `line_trace_vis.jpg` plus `line_trace_summary.txt`.
- The line-trace JPG normally has two columns: the unaugmented trace view, then
  the original patch with a flock of traces from random combined geometric
  training-style test-time augmentations mapped back through each TTA
  output-to-reference coordinate grid. `--line-trace-tta-count` controls the
  TTA count and defaults to 100.
- `--line-trace-vis --med-tta` adds a third column. That trace stays in the
  original patch space and uses the same random TTA direction fields as the
  flock column. At each step it samples the reference and TTA direction fields,
  maps the current original-patch point into each TTA field through the
  prebuilt reference-to-output map, maps TTA orientations back to the original
  patch frame through the output-to-reference coordinate grid, resolves the
  ambiguous direction sign against the previous step, and steps along the
  normalized median direction.
- Provides `--trace2cp-vis --checkpoint <snapshot> --export-dir <dir>` for
  trace-to-next-control-point inspection. This mode resolves `--sample-index`
  through the deterministic sample order, uses that CP as the start, targets
  the next CP by default, and can use `--trace2cp-target-offset` or
  `--trace2cp-target-cp-index` for other same-fiber target CPs.
- `--trace2cp-vis --fiber-json <path>` runs the same Trace2CP path for every
  in-range CP pair in an explicit fiber. Before constructing the loader, the
  runner narrows a single-dataset config to `fiber_paths=[<path>]`, so the
  command loads only that fiber while keeping the same Lasagna manifest,
  volume scale, cache, and sampler context as normal training/inspection. With
  the default target offset `1`, pairs are adjacent CPs. This mode writes
  `trace2cp_fiber_vis.jpg` and `trace2cp_fiber_summary.txt`; it does not write
  the single-pair `trace2cp_vis.jpg`. Invalid pair segments, for example from
  zero Lasagna `grad_mag` samples in the local line window, are skipped and
  listed in the summary; the command fails only if no valid pairs remain.
- `--trace2cp-vis` loads a side-strip segment spanning both CPs, runs the same
  decoded direction-field model as line tracing, traces start-to-target and
  target-to-start on the same strip, and reports the public
  `trace2cp_error`: the mean y error at the opposite CP x-columns divided by
  the horizontal start-to-target CP span. The forward trace is compared at the
  target CP column; the reverse trace is compared at the start CP column. When
  a target-directed step crosses the opposite CP column, the trace appends an
  exact interpolated point at that column before metric computation. If a
  target-directed trace exhausts `max_steps`, the runner raises a visible error
  instead of scoring a missing target-column fallback. The center-biased
  closest approach remains a `refine_score` diagnostic for the fused/optimized
  visualization rows only. The trace2cp segment strip uses eight times the
  configured patch height for more vertical room before the RF margin.
- `--trace2cp-refine-iterations N` runs additional Trace2CP passes after the
  initial pass. Each pass smooths the previous selected fused trace with a
  finite Gaussian kernel controlled by `--trace2cp-refine-smooth-window`
  (default `5`), preserves the CP endpoints and x columns, builds a new
  volume-sampled side strip from the smoothed trace with endpoint context
  before/after the CP pair, and reruns the same scoring mode. Single-pair mode
  keeps `trace2cp_vis.jpg` for pass 0 and writes `trace2cp_vis_it1.jpg`,
  `trace2cp_summary_it1.txt`, etc. Whole-fiber mode similarly writes
  `trace2cp_fiber_vis_it1.jpg` and `trace2cp_fiber_summary_it1.txt`.
- Trace2CP uses `--med-tta` to decide whether to use TTA. Without it, the tool
  traces and scores both directions on the base direction field. With it,
  deterministic random geometric TTA direction fields are built by transforming
  the segment coordinate grid and sampling the volume at those coordinates.
  The TTA set uses the training geometric ranges except y-shift is forced to
  zero and scale to one. The tool then traces both median-direction lines in
  the reference segment strip, using each TTA reference-to-output coordinate
  grid for point lookup and each output-to-reference grid for direction
  mapping.
- `--trace2cp-vis --trace2cp-combined` switches the selected Trace2CP output to
  the regular stepwise candidate-fan combined tracer. It scores side direction
  at both the current/last point and candidate point, plus optional presence.
  The monotone-x DP backend is still available for experiments, but only when
  `--trace2cp-dp` is also supplied. In DP mode, the state is
  `(side_z_layer, y, prev_dy, prev_dz)`. Side DP uses fixed 4 px horizontal
  transitions, plus the exact target column. `--line-trace-step` controls
  output resampling density, not DP transition length. Direction scoring is
  angle-space: `theta = degrees(acos(abs(dot(path_tangent, direction))))`,
  then `(theta / 10)^2 * (1 + max(theta - knee, 0) / knee)`. The existing
  candidate-angle setting provides the knee, defaulting to 25 degrees; it must
  not cap global horizontal slope because valid local fibers can be steeper
  than 45 degrees.
  Transition scoring samples fractional row/z positions with bilinear
  interpolation and sign-aligns ambiguous direction-vector corners to the
  transition tangent before blending. Side DP uses no default per-step z
  movement penalty; it uses dz smoothness (`0.5 * (dz_current - dz_previous)^2`)
  to discourage abrupt z-step changes while allowing steady z motion. This
  combined path is an inspection/refinement path; the non-combined reference
  tracer remains the public target-column Trace2CP metric.
  `--trace2cp-combined-mode direction` is the only active combined mode.
  `--trace2cp-use-presence` adds `1 - sigmoid_presence` at sampled DP pixels,
  weighted by `--trace2cp-combined-presence-weight`. Embedding and image
  similarity modes are no longer active tracer modes and now fail clearly if
  requested.
  With z-search, side-model presence is read through the z-plane cache before
  scoring or display. By default this returns raw per-layer presence. Adding
  `--trace2cp-presence-blur` enables an experimental weighted Gaussian-smoothed
  stack: side-z radius 21, then a PyTorch batched direction-aligned anisotropic
  x/y gather using radius 5 along the local predicted side direction and radius
  1 across it. The x/y kernel is symmetric, so Lasagna direction sign ambiguity
  does not affect the blurred presence.
  When presence scoring is active, `trace2cp_vis.jpg` appends a fixed-scale
  presence column and whole-fiber `trace2cp_fiber_vis.jpg` appends a fixed-scale
  presence row: `0` is black, `1` is white, invalid pixels are black, and the
  fiber line, CPs, and selected traces are overlaid. With z-search, the z debug
  column also shows forward/reverse/fused z-corrected presence maps, built
  column-by-column from the same selected z layers as the z-corrected image;
  whole-fiber presence uses the fused z-corrected presence when available.
- Trace2CP visualizations also include VC3D-style top-strip output sampled
  from volume coordinates, not warped from the side-strip image. Single-pair
  `trace2cp_vis.jpg` appends a debug column and whole-fiber
  `trace2cp_fiber_vis.jpg` appends stitched rows. The first top strip is the
  original/init comparison from the segment line window and Lasagna/VC3D frames.
  The second top strip is reconstructed from the traced fused line projected to
  the central z slice: each output column samples the segment coordinate grid
  and Lasagna row-normal axis at the fused trace position, derives a side axis
  from traced tangent and row normal, then samples rows through the volume. If
  z-search is active, both modes also include a traced fused z-corrected top
  strip that uses the fused trace's per-column selected z offset along the
  side-strip out-of-plane side-z axis.
  With z-search and a side presence head, the same top-strip section appends
  side-presence z-pillar rows. Each column samples the inferred side-slice
  presence stack across z layers at the relevant trace y coordinate, so a
  `--trace2cp-z-max-layer 40` run produces 81-pixel-high z-pillar panels. These
  are side-stack projections, not true top-strip surface predictions and not
  top-model predictions; they can resemble a narrow side-presence slice when
  the side presence field is broad or similar across shifted layers. For the
  z-search fused trace panel, columns are shifted by the selected trace z layer
  so the center row is relative z=0 around the used layer.
- `--trace2cp-obj` can be added to single-pair `--trace2cp-vis` to write
  vertex-colored OBJ debug meshes under `trace2cp_obj/`. The meshes reuse the
  same sampled Trace2CP coordinate grids as the images: center side strip,
  z-search selected side-strip columns, original top strip, traced fused top
  strip, and z-corrected traced top strip when present. Separate OBJ files are
  written for volume intensity and available side-model presence values, with a
  `manifest.txt` listing vertex and face counts. This flag is intentionally not
  supported for stitched whole-fiber `--fiber-json` output yet.
- `--trace2cp-top-model-dir-vis` loads the checkpoint's top-view model and
  appends sparse predicted direction indicators over the traced fused top strip.
  It samples top-strip offsets `-4..+4` selected-scale voxels around the
  z-corrected fused trace when available, otherwise around the central-z fused
  trace. Each offset layer is inferred by the top model, and each displayed
  direction is the normalized median of valid layer directions within
  45 degrees of image-horizontal after aligning the Lasagna-ambiguous signs.
  The panel also draws two equally weighted top traces through that fused
  direction field, from each CP toward the other CP's x column. Those traces
  resolve ambiguous direction signs before bilinear interpolation by flipping
  each of the four neighboring pixel direction samples, if needed, to agree
  with the current trace direction. The same panel also draws a monotone-x
  dynamic-programming path from CP column to CP column on the top-strip center
  row. That path uses `(top_offset_layer, y, prev_dy, prev_dz)` state, can move
  between neighboring top-offset layers with a `0.1 * abs(delta_layer)`
  transition penalty, adds second-order smoothness penalties
  that default to zero, uses fixed 8 px horizontal transitions, and integrates
  `1 - abs(dot(path_tangent, layer_direction))` across each transition's
  crossed pixel columns using fractional row/z interpolation from the direction
  field. It uses no absolute-y row bias by default, and uses a fixed penalty,
  rather than a hard stop, for invalid/missing direction
  pixels. The visualization also appends diagnostics derived from the DP
  optimized line: an optimized top strip resliced around the DP top-row path
  and selected top-offset layers, a side slice reconstructed from the combined
  top-row plus selected-layer side displacement, and top z-pillar/side-column
  presence panels when side-model presence exists. The optimized top-strip
  panel draws the optimized line as the straight slice center. Side/presence
  panels use a visualization z-plane cache sized to that combined optimized
  side displacement, not just the raw selected top-offset layer range.
  Z-corrected debug helpers lazily infer the requested z-plane layers instead
  of assuming `plane_cache.layers` is already complete. These panels are
  diagnostic only and do not change scoring or z-layer selection.
- `--trace2cp-side-top-z-experiment` adds a separate single-pair diagnostic
  path. It is exclusive: when this flag is set, the command writes only
  `trace2cp_side_top_z_experiment.jpg`,
  `trace2cp_side_top_z_summary.txt`, and the local top-slice debug
  directories; it does not run the normal Trace2CP overlay/refinement chain or
  write `trace2cp_vis.jpg`. It uses the regular side candidate fan scoring for side x/y steps: side
  directions are interpolated at both the current/last point and candidate
  point from the current z-layer prediction with ambiguity-aware two-cos
  direction handling, and optional presence is scored when
  `--trace2cp-use-presence` is active. It carries a bounded selected-scale
  z/offset state, but top-model inference is run only once after a side
  candidate has been accepted. That per-step top patch updates only the
  z/lateral state; embedding/image scores and DP are not part of this
  diagnostic. The top patch is centered at the accepted traced side point. Its
  x axis is derived from the sampled side-view direction by tilting the
  side-strip tangent within the tangent/normal plane, and its second in-plane
  axis remains the side-strip lateral axis; this avoids optimizing roll around
  the fiber line. The top direction is an ambiguity-aligned weighted median
  over a normally weighted local neighborhood, default radius 20 px. Outputs
  are `trace2cp_side_top_z_experiment.jpg` and
  `trace2cp_side_top_z_summary.txt`. The experiment JPG is intentionally
  compact: it shows only forward/backward z-corrected side traces,
  forward/backward z-corrected presence, the input top strip, and the
  forward/backward z-corrected traced top strips, without per-step
  top-direction ticks. It also writes every local top slice used during the
  stepwise z update to
  `trace2cp_side_top_z_top_slices/`, plus native-size direction overlays to
  `trace2cp_side_top_z_top_overlays/`, with `fw_####.jpg` and `bw_####.jpg`
  names. The trace state is kept as floating-point xyz positions; z-layer
  rounding is only used when selecting side prediction layers or reconstructing
  display columns. During the slow repeated local top-patch inference loop,
  forward and backward traces print throttled `trace2cp side_top_z progress`
  rows with a small bar, accepted step count, top-patch counts, current z,
  elapsed time, ETA, and the final reason.
- `--trace2cp-vis --trace2cp-combined --trace2cp-z-search` runs the same
  regular stepwise z-search over side-strip z layers by default. Each accepted
  candidate-fan step may choose the current or neighboring z layer. Adding
  `--trace2cp-dp` switches the z-search to the experimental monotone DP over a
  bounded z-layer stack. The loader builds one aligned Trace2CP segment source
  from the CP-to-CP line window and Lasagna normals. In that side strip, image
  x follows the fiber tangent, image y follows the Lasagna row-normal axis, and
  z-search layers move along the out-of-plane frame side axis. State layer `k`
  represents `k * --trace2cp-z-step-voxels` selected-scale voxels along that
  axis. Inference is capped at one selected-scale voxel spacing: z steps of
  `1.0` or larger sample each requested state directly, while sub-voxel steps
  infer only bracketing integer side-z offsets and interpolate direction and
  optional presence fields for the requested state. Direction interpolation
  sign-aligns the ambiguous vectors before interpolation and normalizes the
  result. The default z step is `1.0` selected-scale voxel, with a bounded
  range controlled by `--trace2cp-z-max-layer`. Z-search never warps an
  already sampled image and never rebuilds unrelated planes. Direction and
  optional presence costs are sampled from the selected state layer at each
  step/transition.
- Single-pair z-search visualization appends a z column with separate forward,
  reverse, and fused z-corrected views plus a fused z-layer map row. These
  images are reconstructed per column by rounding the trace/fused z value to
  the nearest z-search state layer and copying that state's sampled image
  column. With sub-voxel z steps, interpolated states reuse the nearest
  integer-inferred side-z image. Columns outside the available trace/fused z
  path render black. This visualization does not re-sample the volume and does
  not interpolate image values between z layers.
- `--trace2cp-z-layers-tif` can be added to z-search Trace2CP runs to export
  the inferred layer cache as multilayer TIFF. Single-pair mode writes
  `trace2cp_z_layers.tif`; whole-fiber mode writes one pair-local TIFF per
  valid pair under `trace2cp_z_layers/`. Page order is all sampled slice images
  in sorted inferred z-layer order, followed by all available presence maps in
  the same sorted inferred z-layer order. With sub-voxel z steps, this export
  writes the actual integer-inferred layers, not the interpolated state layers.
- When embedding channels are present, single-pair `trace2cp_vis.jpg` appends a
  debug column of fixed-scale cosine similarity maps: start CP, target CP,
  same-fiber CP-bank/global similarity when a combined Trace2CP bank is
  available, forward trace-progress last-point columns, and reverse
  trace-progress last-point columns. The forward/reverse panels paint the
  column band around each newly placed trace point using the previous accepted
  trace point's embedding; the band radius is `ceil(step_px / 2)`, and
  unvisited columns render black. The column is for inspection only and does
  not affect Trace2CP scoring or metrics.
- `--trace2cp-vis --med-tta --vis-tta` writes `trace2cp_tta/reference.jpg`,
  one `trace2cp_tta/random_NNN.jpg` per generated TTA field, and
  `trace2cp_tta/contact_sheet.jpg`. Each image shows the sampled slice with the
  transformed base-strip corner outline and start/target CP markers.
- Trace2CP writes `trace2cp_vis.jpg`, writes `trace2cp_summary.txt`, and prints
  the selected public metric as a standalone first stdout line:
  `trace2cp_error=<value>`. A second `trace2cp details ...` line carries
  diagnostics such as target-column raw y error in pixels, horizontal CP span,
  `refine_score`, trace mode, endpoint diagnostics, and per-direction
  scores/statuses. The summary includes metric target-column interpolation
  values, the refinement closest x position, fused/optimized point counts,
  forward/reverse raw endpoint errors, normalized endpoint scores, target
  x-columns, target-column reach statuses, termination reasons, and trace point
  counts. The JPG is a labeled vertical stack with full traces, partial
  closest-approach traces, the fused CP-to-CP line, and the optimized
  refinement. With `--med-tta`, that stack is rendered as the first column and
  a second column shows the reference-only/base-inference result without TTA.
  With `--trace2cp-combined`, the selected combined stack is rendered alongside
  the reference-only/base-inference stack and the summary/stdout include
  candidate settings, weights, embedding-bank size/skips, and mean score
  components without duplicating the selected public metric label. Trace2CP
  stdout also ends with a compact timing table. Single-pair mode prints
  `trace2cp timings`; whole-fiber mode prints `trace2cp fiber timings`
  aggregated across valid pairs. Each row is one stage with count, total,
  mean, and max milliseconds for source sampling, inference, tracing, debug
  rendering, and file-output stages. Slow Trace2CP DP solves emit
  time-throttled `trace2cp dp ...` progress rows only when the explicit DP
  backend is selected; progress rows include solved columns, elapsed seconds,
  and `eta_s`.
  CLI side, side-z, and top-model DP calls use a torch-vectorized backend on
  the active model device. The column recurrence remains sequential, but each
  column is computed with tensor operations over all layers/rows and move
  chunks. Direct helper calls without a torch device still use the NumPy
  fallback. Side-z DP also caps the inferred layer stack to the center-anchored
  reachable range. The candidate-angle setting is the angle-space penalty knee,
  not a global side-DP move-lattice cap.
- The whole-fiber Trace2CP JPG is composed after pair scoring. Pair-local
  images and traced points are mapped into a shared fiber arc-length x
  coordinate system using each pair's local start/target CP columns and global
  start/target arc-length columns. Whole-fiber mode aligns each segment's
  actual strip row axis against already accepted shared-CP row axes before
  image sampling, so adjacent pair images do not randomly flip in y.
  Valid overlapping image pixels are averaged with dense rectangular masks for
  display, and the long strip uses the same four rows as the single-pair
  Trace2CP view: full traces, partial closest-approach traces, fused CP-to-CP
  line, and optimized line. It then appends the original/init top strip, the
  traced fused top strip projected to central z, and with z-search also appends
  the traced fused z-corrected top strip. If side presence is available, it
  appends side-presence z-pillar rows for the same trace variants. The summary
  includes requested, valid, and skipped pair counts plus mean/min/max
  trace2cp errors.
  Stdout prints the public whole-fiber metric on its own line as
  `trace2cp_error_mean=<value>`;
  `trace2cp_fiber_debug.txt` records per-pair strip CP vectors, row axes,
  frame vectors, and projected CP deltas for debugging.
- Provides `--dir-vis --checkpoint <snapshot> --export-dir <dir>` for
  direction-field inspection. This mode loads the same deterministic center
  side-strip patch, applies pixel-perfect image-space identity/flip/90-degree
  rotation variants, runs the checkpointed direction model for each variant,
  nearest-neighbor scales each augmented patch image by 4x for visualization
  only, and writes a single `dir_vis.jpg` natural-size horizontal strip with one
  shared top label band and anti-aliased 6-display-pixel direction segments
  drawn every second source pixel, i.e. one segment per 8x8 display-pixel cell.
  Model inference runs on each native-resolution augmented patch before display
  upsampling. If the loaded center patch is not square, dir-vis center-crops it
  to the largest square before applying pixel-space flips/rotations; it does not
  rescale the native patch before inference. The valid mask gates model
  normalization and arrow placement, but does not black out display pixels.
  `--dbg-dirs` adds a second row: the first cell is the raw unaugmented patch
  without direction arrows, and the remaining cells run inference on transformed
  patches whose center half-image region has been overwritten with the matching
  unaugmented center crop.

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
  torch value augmentation, pipeline wait time, forward plus loss, and backward
  plus optimizer step. Loader-side stage timings come from
  the shared `load_batch` / `build_strip_source` / `build_strip_patch_from_source`
  profile hooks, so profiling uses the same sampling path as normal training.
- Supports `--load-only` on the benchmark path. It still performs deterministic
  CP selection, CP-local source construction, coordinate augmentation, and
  base-volume sampling, but skips value/image augmentation, normalization,
  supervision construction, model forward, backward, and optimizer work. When
  `training.pipeline_enabled` is true, load-only benchmark mode also uses the
  bounded whole-batch queue so independent batch-load parallelism can be
  measured without model work.
- Flattens `[control_point_sample, strip_z_offset]` into one patch batch for the
  model.
- Computes direction targets from transformed line coordinates after geometric
  augmentation.
- When `training.presence_enabled` is true, configures a one-channel
  sheet/fiber-presence head and adds a balanced BCE loss: the rounded
  transformed CP pixel in each strip patch is positive, valid reachable non-CP
  pixels are negative, and unreachable shift-margin edges are ignored.
- Standard training uses direction supervision plus the optional sheet/fiber
  presence head. The standard example config enables presence and leaves
  contrastive embedding disabled.
- When `training.top_view_enabled` is true, training jointly builds one
  top-view patch per CP sample and trains a second model. The top model's
  direction loss uses the same transformed-line tangent target as side strips.
  Its scalar head is trained as a distance transform along the rounded
  cross-fiber line through the transformed CP: `1.0` at the CP, linearly down
  to `0.0` at `training.top_view_dt_radius_px`, and explicit zero targets
  beyond that radius.
- When `training.contrastive_enabled` is true, configures an embedding head,
  uses the loader's same-fiber grouped batch path, and adds a cosine embedding
  loss. Positive terms operate on rounded transformed CP pixels: for each
  anchor CP sample/strip-z offset, only other CPs from the same fiber are
  considered, across their loaded strip-z offsets, and the already most-similar
  candidate is trained toward cosine similarity `1`. Negative terms compare
  each positive to one deterministic valid non-CP pixel from the batch inside
  the CP-neighborhood reachable region derived from `augment_shift_x/y`, so
  unreachable patch edges are ignored rather than trained as always-negative.
  CP embeddings from other fibers are not used as contrastive negatives. The
  positive and valid-pixel negative means are balanced. A similarity-image
  sparsity term also compares
  the valid-pixel mean of each CP's normalized `0..1` embedding-similarity map
  over the same shift-reachable CP area against fixed target `0.1`, encouraging
  only a small reachable region to remain similar to the CP embedding. The
  balanced pair loss plus this sparsity term are multiplied by
  `training.contrastive_weight`.
- Logs `train/loss_total`, `train/loss_direction`, optional presence
  loss/components, TensorBoard presence maps, and, only when contrastive
  embedding is enabled, `train/loss_contrastive`, positive/negative
  contrastive components, and TensorBoard embedding-similarity images that
  compare every pixel in a selected patch against that patch's CP embedding.
- When top-view training is enabled, logs top-view direction/angle/DT scalars
  and TensorBoard images for the top-view line plus predicted direction and the
  top-view DT scalar map.
- On CUDA training runs, uses a bounded deterministic whole-batch pipeline when
  `training.pipeline_enabled` is true. Background workers build exact training
  steps with `FiberStrip2DLoader.load_batch`, defer image/value augmentation,
  prepare tensors on side CUDA streams, and return prepared batches for strict
  in-order forward/backward consumption. Load-only benchmark mode uses the
  same whole-batch queue without CUDA preparation.
  Coordinate generation and geometric coordinate augmentation remain on the
  configured `augment_device`.
- `training.pipeline_depth` controls queued future batches.
  `training.pipeline_workers` controls concurrent load+prepare tasks; `0`
  means use `pipeline_depth`. Completion may be out of order, but consumption
  remains ordered by training step.
- Logs scalars/images to TensorBoard and writes `current.pt` / `best.pt`
  snapshots under the run directory.
- When `test_datasets` is configured, evaluates a deterministic held-out test
  set at `training.test_interval` using either the configured fixed-size
  random window or all held-out CPs when `test_control_points` is `0`; it also
  evaluates Trace2CP on each selected held-out CP using the segment to the next
  CP, and uses the averaged `test/trace2cp_error` for current/best snapshots.
- `--resume <snapshot.pt>` restores model and optimizer state, creates a fresh
  timestamped run directory from the current config just like normal training,
  and continues from `checkpoint_step + 1`. `training.max_steps` remains the
  absolute target step, so increase it before resuming a completed finite run.
  The first resumed step is logged and flushed to TensorBoard immediately.

`configs/loader_example.json`

- Example local Staticsheep config using:
  - PHercParis4 78keV masked base volume through the public Vesuvius S3 path;
  - local Lasagna manifest;
  - 64x64 strips;
  - one strip-z offset for CP-only training experiments;
  - measured load-only tuning with 4 whole-batch workers, queue depth 16, and
    4 CP-prep workers;
  - current augmentation extrema.

`configs/train_s1a_nml_all.json`

- S1A NML training config whose `datasets` entry uses
  `/home/hendrik/business/aiconsulting/vesuviuschallenge/data/train_fibers/fiber_vols/fibers_s1a_*.nml`.
- Loads all matching S1A NML files as the training dataset. It intentionally
  omits `test_datasets`; add a separate S1A held-out config if test evaluation
  should run on a split of this source.
- Keeps the PHercParis4 78keV base volume, base-volume scale, and Lasagna
  manifest settings from the normal example config.
- Includes the existing S1A/source-to-current PHercParis4 transform from
  `/home/hendrik/business/aiconsulting/vesuviuschallenge/villa2/lasagna/configs/tifxyz_train_s3_dbg.json`:
  inline `transform` plus `transform_invert: true`.

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
  download concurrency. During prefetch, PyTorch CPU intra-op threads are
  temporarily forced to `1` and restored afterwards, so producer workers do not
  each expand over the full machine.
- `loader_workers`: worker count for startup compact-geometry record
  construction and CP-sample construction in `load_batch`. The default is the
  logical CPU count. Set `loader_workers: 0` to explicitly use all logical CPU
  cores from the config, or `loader_workers: 1` for serial debugging. Parallel
  startup uses worker processes that open their own volume/Lasagna handles and
  return compact geometry for parent-owned storage by original record index.
  Parallel warm batch workers evaluate candidates concurrently while accepted
  batch output remains in deterministic sample-index order. For warm
  `load_batch` work with `loader_workers > 1`, the loader keeps a lazy
  persistent thread executor and reuses it across batches; call
  `FiberStrip2DLoader.close()` to shut it down explicitly in long-lived tools
  or tests.
- `volume_cache_dir`: optional cache directory for remote volume chunks.
- `volume_cache_memory_mib`: optional per-VC3D-sampler decoded/hot cache budget
  in MiB. `null` or omission leaves VC3D's default behavior intact. Use a
  positive cap mainly when `training.pipeline_isolated_loaders` is enabled and
  each worker creates its own VC3D sampler.
- `volume_io_threads`: optional positive VC3D I/O thread count, forwarded to
  the VC3D volume binding when available.
- `volume_cache_offline`: passed to the Vesuvius Zarr cache opener.
- `volume_cache_retry_seconds`: passed to the Vesuvius Zarr cache opener.
- `strip_coord_cache_dir`: removed. CP-local dense source-coordinate files are
  no longer read or written; compact fiber-line geometry is built in RAM during
  loader construction.
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
  prefetched subset. Augmentation seeds still use the unbounded training stream
  position, so the subset is deterministic but its augmentations keep changing
  across repeats.
- `learning_rate`: AdamW learning rate.
- `scalar_log_interval`: TensorBoard scalar/console interval.
- `tensorboard_image_interval`: TensorBoard batch-image interval.
- `checkpoint_interval`: interval for writing `snapshots/current.pt`.
- `test_interval`: interval for deterministic held-out evaluation and, when
  `test_datasets` is configured, current snapshot writes.
- `test_control_points`: number of deterministic held-out CP samples per test
  evaluation. Positive values use the deterministic random window starting at
  `test_start_sample_index`; `0` evaluates every held-out CP sample once in
  flat CP order starting at zero.
- `test_start_sample_index`: deterministic held-out sample start index.
- `control_points_per_step`: deterministic CP samples per step; default `4`.
  Training and benchmark modes require this to match top-level `batch_size`.
  Changing `strip_z_offset_count` changes the flattened CNN patch count without
  any default-shape warning.
- `contrastive_enabled`: explicitly enables experimental same-fiber
  contrastive embedding training. It defaults to false; standard training does
  not create an embedding head or loss.
- `contrastive_embedding_channels`: number of raw embedding channels appended
  after the two direction channels and optional presence channel; must be
  positive when contrastive training is enabled. A positive value is ignored
  while `contrastive_enabled` is false.
- `contrastive_control_points_per_fiber`: same-fiber CP group size `N`.
  `control_points_per_step` must be divisible by this value so the batch is an
  exact number of same-fiber CP groups.
- `contrastive_weight`: multiplier for the balanced cosine contrastive loss.
- `contrastive_negative_margin`: cosine margin for negative pairs.
  The margin applies to valid non-CP pixel negatives.
- `top_view_enabled`: enables the jointly trained top-view auxiliary model.
  The top model uses one top-strip patch per loaded CP sample and outputs
  direction plus a DT scalar channel.
- `top_view_direction_weight`: multiplier for the top-view direction MSE.
- `top_view_dt_weight`: multiplier for the top-view distance-transform MSE.
- `top_view_dt_radius_px`: pixel distance where the top-view DT target reaches
  zero; default `30.0`.
- `device`: `auto`, `cpu`, or a torch device string.
- `tensorboard_enabled`: set false for smoke tests without TensorBoard.
- `pipeline_enabled`: enables CUDA training batch pipelining and load-only
  benchmark batch pipelining; default `true`. CPU training keeps the
  synchronous path.
- `pipeline_depth`: queued whole-batch futures for the pipeline; default `16`.
- `pipeline_workers`: concurrent whole-batch loader calls for the pipeline.
  Default `8`; `0` means use `pipeline_depth`.
- `pipeline_isolated_loaders`: default `false`. The normal path shares the base
  loader and VC3D sampler/cache across whole-batch futures. `true` gives each
  pipeline worker a cloned loader with shared parsed records/order cache but a
  fresh VC3D sampler; combine this with `volume_cache_memory_mib` if the clone
  path is required.
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

Fiber paths may point to VC3D `.json` fibers or `.nml` files. A single NML file
can contribute multiple records when it contains multiple usable simple path
components. JSON and NML sources are normalized into `Vc3dFiber` before all
sampling, caching, prefetch, training, and Trace2CP paths.

Optional dataset keys:

- `base_volume_auth_json` / `volume_auth_json`;
- `lasagna_auth_json`;
- legacy `volume_path` / `volume_scale` aliases.
- `fiber_transform_json` / `fiber_transform_json_path`: Vesuvius registration
  `transform.json` with `p_fixed = M @ p_moving` in XYZ.
- `fiber_transform`: inline XYZ affine matrix, either 3x4 or homogeneous 4x4.
- `fiber_transform_invert`: invert the selected fiber transform before
  applying it.
- `transform` / `transform_invert`: Lasagna-compatible inline aliases for
  `fiber_transform` / `fiber_transform_invert` on fiber coordinates.

Fiber transforms apply only to parsed fiber coordinates. The matrix must map
source/moving fiber XYZ coordinates into current/fixed base-volume XYZ
coordinates unless the corresponding invert flag is set. The transform is
applied once before control-point bounds checks, deterministic sample identity,
strip-coordinate cache identity, prefetch, training, and Trace2CP use the
fiber. Lasagna normals are still read from the configured current
`lasagna_manifest_path` after coordinates are transformed.

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
6. Resolve fiber paths/globs and parse each VC3D JSON or NML source.
7. Apply any dataset-level fiber-coordinate transform in XYZ.
8. Skip the whole normalized fiber if any control point is outside the
   base-volume bounds.
9. Store a `_Record` containing fiber, volume, sampler, manifest channels, and
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
- random pass orders are cached by pass index. `load_batch` prewarms the
  attempted batch window before submitting parallel CP workers so warm workers
  do not serialize on random-order construction locks;
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

For each accepted CP sample, all configured strip-z offsets are sampled through
one sampler batch call. With the current VC3D path this uses flattened explicit
coordinate sampling: the stack is reshaped into one larger 2D coordinate image
for `Volume.sample_coords(..., blocking=True)` and reshaped back afterward.
This changes request grouping, not sampled values.

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
offsets, producing 64 patches. Non-default shapes are valid. `load_batch`
returns `[control_points_per_step, strip_z_offset_count, 1, H, W]`; the trainer
reshapes this to `[control_points_per_step * strip_z_offset_count, 1, H, W]`
before the CNN forward pass.
If one deterministic sample cannot build its CP-local Lasagna normal window
for data reasons such as `grad_mag == 0`, `load_batch` skips it and advances
through following deterministic sample indices until the requested number of
control-point samples is loaded.

`load_batch` can evaluate CP candidates in parallel via `loader_workers`.
Results are consumed by raw deterministic sample index, so changing
`loader_workers` should affect latency but not the accepted output sequence.
The parallel path reuses one loader-owned executor across batches instead of
creating and destroying a thread pool per training step. `loader_workers=1`
uses a direct serial path with no futures.
Whole-batch training and load-only benchmark pipelines can run several
`load_batch` calls concurrently. The default path shares loaded records and the
VC3D sampler/cache; isolated loader clones are opt-in and only duplicate the
mutable sampler handles, not fiber or Lasagna metadata.
In benchmark/profile output, `total` is batch wall time, `cpu` is whole-process
CPU time measured with `time.process_time()`, and `ctf` is `cpu / total`.
Compare `ctf` to system CPU-utilization monitors when checking whether the
process is really using cores. `wall` is real `load_batch` wall time, `work` is
summed per-candidate worker elapsed time, and `tf` is `work / wall`. The
individual stage columns remain summed worker timings under parallel loading,
so `tf` can be high even when actual process CPU usage is lower.
Cold deterministic random-order setup is included in the descriptor column;
warm batches should mostly avoid that cost.

Images are normalized per patch over valid pixels. Invalid pixels are set to
zero after normalization.

On CUDA training runs with `training.pipeline_enabled`, the trainer keeps two
overlap stages active:

- Background load+prepare workers run deterministic whole training steps
  concurrently. Each worker uses the loader's CP-level worker pool for patch
  construction and then submits deferred image/value augmentation, image
  normalization, and supervision-tensor construction on a side CUDA stream.
- The prepared image tensor stays on CUDA. The main training stream consumes
  prepared batches in step order and waits on each preparation event
  immediately before the forward pass.

`prep_submit_ms` measures only main-thread queue refill overhead. The actual
preparation work is represented by `prep_ms` and, on CUDA, `prep_gpu_ms`.
Training prints the effective pipeline enable flag, queue depth, whole-batch
loader worker count, and CP-level loader worker count once at startup.

In load-only benchmark mode, the same whole-batch queue is used when
`training.pipeline_enabled` is true, but workers stop after
`FiberStrip2DLoader.load_batch`. This isolates whether independent batch loads
can keep CPU/I/O resources busy before adding CUDA preparation or model work.

This training-only prepared-batch path avoids the runner/debug
`apply_batch_image_augmentation()` NumPy round trip. It also requests only the
CP-local tangent line needed for direction supervision and does not retain
post-sampling coordinate arrays in the returned training batch. Runner exports
and debug loader APIs still keep full transformed strip lines and coordinates.

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
- `timing/pipeline_wait_ms`;
- `timing/prep_enqueue_ms`;
- `timing/prep_gpu_ms`;
- `timing/prep_wait_ms`;
- `timing/prep_submit_ms`;
- cache hit/download diagnostics where available;
- `train/batch_direction_overlay` images showing the transformed centerline
  behind one short network-predicted direction segment at the transformed CP.
  The contact sheet picks one center-offset representative from each loaded
  control-point sample before filling with additional strip-z offsets.
- when `test_datasets` is configured, `test/loss_direction`,
  `test/angle_error_mean_deg`, `test/supervision_samples`,
  `test/trace2cp_error`, `test/trace2cp_raw_y_error_mean_px`,
  `test/trace2cp_segments`, `test/trace2cp_skipped_segments`, test cache
  diagnostics, and `test/batch_direction_overlay` at test evaluation steps.

Console progress prints the same loss, mean angle error in degrees,
supervision, load-time summary, and CUDA preparation timing for every step whose
deterministic control-point sample range starts before sample index 100, then
return to `training.scalar_log_interval`.

Snapshots are written under:

```text
<run_dir>/snapshots/current.pt
<run_dir>/snapshots/best.pt
```

With `test_datasets`, `current.pt` is written at step 1, every
`training.test_interval`, and the final step; `best.pt` tracks the lowest
observed averaged `test/trace2cp_error` over the effective test sample count
(`test_control_points`, or all held-out CP samples when it is `0`). Without
`test_datasets`, snapshots keep the train-only
behavior: `current.pt` follows `training.checkpoint_interval`, and `best.pt`
tracks the lowest observed training loss.

Resume an existing run:

```bash
PYTHONPATH=vesuvius/src python -m vesuvius.neural_tracing.fiber_trace_2d.train config.json --resume /path/to/run/snapshots/current.pt
```

Training writes the resumed run to a new timestamped run directory from the
current config's `training.run_path` and `training.run_name`. If that name
collides because two runs start in the same second, a numeric suffix is added.
The resumed run writes its own `snapshots/current.pt`, `snapshots/best.pt`, and
TensorBoard event file.

## Prefetch

`chunk_requests_for_sample_index` is a compatibility/test helper that derives
dependency requests from the same conservative source-envelope coordinates used
by prefetch.

`prefetch(start_sample_index, sample_count)`:

- builds the shared CP-local source once per deterministic sample index;
- derives each configured strip-z offset from that source;
- temporarily caps PyTorch CPU intra-op fanout during dependency generation so
  `prefetch_sampler_workers` is the effective generation concurrency limit;
- sends the source-envelope coordinates to `chunk_requests_for_coords`, which
  maps them to dependency-only base-volume chunk metadata;
- deduplicates globally by `(store_identity, key)`;
- may run dependency producers in parallel, but consumes their completed
  results in raw deterministic sample-index order before classifying chunks;
- treats existing VC3D persistent-cache data files as hits;
- treats existing `<cache>/level_<level>/<iz>/<iy>/<ix>.empty` files as
  known-missing hits;
- fetches only still-missing direct-source chunks with bounded Python workers,
  writes data through unique temp files followed by atomic rename, and writes
  zero-byte `.empty` markers for definitive missing chunks;
- keeps not-yet-submitted downloads in a priority queue keyed by the earliest
  raw deterministic sample index that needs each chunk. This prevents later
  dependency producers from flooding the download executor ahead of chunks that
  would advance the reported `idx` prefix. Already-active transfers are allowed
  to finish instead of being cancelled for reprioritization;
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

- timing table with `descriptor`, `compact_geometry`, `strip_coords`,
  `coord_augmentation`, `volume_sample`, `value_augmentation`, `line_coords`,
  `to_u8`, and `overlay`, plus finer training profile keys such
  as `map_build`, `line_lookup`, `line_filter`, `coord_aug_batch`, and
  `value_aug_batch` where those paths are used;
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

Export a trace-to-control-point inspection image and score:

```bash
PYTHONPATH=vesuvius/src python -m vesuvius.neural_tracing.fiber_trace_2d.runner config.json --sample-index 0 --trace2cp-vis --checkpoint /path/to/current.pt --export-dir /tmp/fiber_trace_2d_trace2cp
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
