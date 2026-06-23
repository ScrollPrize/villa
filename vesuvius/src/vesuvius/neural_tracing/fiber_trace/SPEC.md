# Fiber Tracing Spec And Status

This document records the intended fiber tracing training design and the current
MVP mapping. Keep it updated when the implementation changes so the training
semantics do not drift into undocumented defaults.

## Intended Training Spec

The model is a 3D U-Net or generic Vesuvius model with a
direction-conditioned decoder head.

Inputs:

- CT crop from the base scan OME-Zarr.
- Direction conditioning as one normalized `xyz` vector:
  - `fw`: forward direction.

Head outputs:

- N-dimensional embedding.
- Predicted `fw` vector.

Losses:

- Pairwise supervised contrastive loss for embeddings.
- Direct direction-vector losses are deferred.

Ground-truth positives:

- Training starts from VC3D line annotation control points.
- All control points on the same line can be compared against the other control
  points on that line as positive embedding samples.
- Within a crop, dense output voxels can also contribute positives when their
  geometry matches the same local fiber sample.

Negatives:

- The fixed GT control-point Lasagna normal defines the local sheet plane for
  GT crops.
- Around control points, a double cone along the Lasagna normal identifies
  regions above and below the local sheet that are probably not on the line.
  The cone apex starts at least `K = 30` voxels away from the normal-defined
  plane and widens away from the fiber.
- Random valid volume samples used as random-negative crops label all valid
  voxels as explicit negatives.
- Random-negative centers are precomputed before batch construction in a
  deterministic pool. `random_negative_pool_size` defaults to `1000`, and
  batch sampling selects from that pool by modulo.
- Voxels outside the positive and negative zones are ignored.

Direction conditioning:

- `fw` is the local fiber tangent from the line geometry.
- GT control-point crops use the control-point tangent with up to 30 degrees
  of forward-direction augmentation.
- Random-negative crops may choose conditioning independently.
- Conditioning direction is an input only; it does not change voxel labels.

Jitter semantics:

- Position jitter is part of pair labeling, not a separate data generation
  requirement.
- Same-sample jitter target: up to `+/-40` voxels in the plane of the Lasagna
  normal and up to `+/-10` voxels perpendicular to that plane.
- GT crop placement is deterministic per iteration. The selected control point
  is placed at `control_point_margin_voxels` from one crop side, or the same
  margin from the opposite side, independently per axis. The default margin is
  `10` voxels, clamped only when a smaller crop cannot fit it.
- Because the model is fully convolutional, the crop should generate many
  labeled samples within a crop and across crops by checking output voxels
  against the ground-truth fiber/control-point geometry and the per-output-voxel
  direction encoding.

## Adjusted Data Convention

The training config should not require raw paths for every derivative zarr.
The intended config inputs are:

- `base_volume_path`: base scan OME-Zarr.
- `base_volume_scale`: scan OME-Zarr level used as the training grid.
- `lasagna_manifest_path`: Lasagna `.lasagna.json` manifest.
- `fiber_paths` or `fiber_glob`: VC3D fiber JSON files.
- optional `test_fiber_paths` or `test_fiber_glob`: held-out VC3D fiber JSON
  files for test-loss logging.

The Lasagna manifest must provide:

- `base_shape_zyx`, matching `base_volume_path` level 0.
- `grad_mag` channel group.
- `nx` and `ny` normal channel groups.

Manifest channel shape and spacing follow Lasagna's loader convention:
`base_shape_zyx` with group `scaledown` validates zarr shape, and
`group.sd_fac * source_to_base` gives channel spacing in base voxels.

The `grad_mag`/mask validity convention is binary. Value `0` is invalid and
any value `> 0` is valid; there is no configurable validity threshold.

Coordinate convention:

- VC3D fiber JSON `line_points` and `control_points` are canonical `xyz`.
- Geometry, tangents, conditioning vectors, target directions, normals, and up
  vectors are computed in `xyz`.
- Dense arrays, zarr reads, crop origins, masks, and tensor spatial axes are
  `zyx`.
- Conversion from `xyz` to `zyx` happens only at dense indexing/crop boundaries.

Zarr convention:

- Training reads use `vesuvius.neural_tracing.datasets.common.open_zarr()`
  and `_read_volume_crop_from_patch()`.
- Remote HTTP/S3 paths require explicit `volume_cache_dir`.
- `python -m vesuvius.neural_tracing.fiber_trace.train --prefetch <config>`
  enumerates the deterministic train/test zarr chunk keys implied by the
  config, deduplicates them, and downloads the chunks into the cache with up to
  16 parallel workers.
- Direct `zarr.open()` fallbacks are not allowed in the fiber tracing training
  path.
- Manifest-less derivative channel keys such as `grad_mag_path`, `mask_path`,
  `nx_path`, and `ny_path` are rejected.

Normal convention:

- Lasagna `nx`/`ny` are required by default.
- Decode normals with Lasagna's normal decoder.
- The fixed sampled control-point normal defines GT label geometry and
  visualization planes, not model conditioning.
- Normal channels are required; there is no arbitrary perpendicular fallback in
  the training path.

## Current MVP Mapping

Implemented:

- Manifest-based dataset config via `base_volume_path`,
  `base_volume_scale`, `lasagna_manifest_path`, and fibers.
- Shape validation of base OME-Zarr level 0 against manifest `base_shape_zyx`.
- Lasagna `grad_mag`, `nx`, and `ny` lookup through manifest channel groups and
  group `scaledown`.
- Dense crop training with:
  - `N - floor(N / 4)` GT control-point crops with mixed
    positive/ignore/cone-negative labels;
  - `floor(N / 4)` random valid `grad_mag` crops labeled as explicit negatives.
- `augmentation_crop_size` reads a larger outer crop before post-augmentation
  center trimming to the final `crop_size`, keeping padded/interpolated borders
  out of the model input and label crop.
- Sampling is deterministic by purpose. Record choice, control-point choice,
  GT crop offset, direction jitter, and random-negative pool lookup are keyed by
  `seed`, iteration, slot, record, and control index where relevant instead of
  mutable RNG state.
- Direction-conditioned U-Net head outputting embedding and `fw`; training
  applies this head only to sampled contrastive-pair voxel features, while the
  dense head path remains available for visualization/debug output.
- U-Net model sizing via `unet_base_channels`, `unet_depth`,
  `conditioned_feature_channels`, `head_channels`, and `embedding_dim`. The
  default training shape is base 16, depth 7, conditioned feature width 64,
  head width 64, and 16D embeddings.
- Pairwise supervised contrastive embedding loss.
- Dense `labels` and `target_id` tensors:
  - positive voxels carry the selected fiber-line identity;
  - explicit negatives carry `NEGATIVE_ONLY_ID`;
  - ignored voxels carry `IGNORE_ID`.
- Pairwise contrastive loss:
  - same-identity positives attract across control-point crops after a
    deterministic pseudo-random shuffle over the whole flattened batch;
  - positive-negative pairs repel after independently shuffling the positive and
    explicit-negative lists over the whole flattened batch;
  - negative-negative pairs are not used.
- Per-voxel labels from normal-frame positive zones, cone negative zones, and
  random-negative crops.
- `normal_plane_jitter_voxels = 40`,
  `normal_perpendicular_jitter_voxels = 10`, and
  `positive_along_fiber_limit_voxels = 40` define the positive zone around the
  rounded CP. Voxels past the along-fiber limit default to ignore unless they
  independently satisfy the cone-negative rule.
- `control_point_margin_voxels` controls GT crop offset. Omit it to use
  `min(10, floor((min(crop_size) - 1) / 2))`; explicit values must fit every
  crop axis.
- `negative_cone_distance_voxels = 30` defines the minimum distance from the
  normal-defined plane where the normal-axis cone negatives start.
- `positive_direction_jitter_degrees = 30.0` controls GT control-point forward
  conditioning augmentation. Direction conditioning does not alter labels.
- Future folded direction bands may use 30/60/90 degree boundaries, but the
  current hard labels are geometry-derived only.
- `test_every` controls deterministic test evaluation and snapshot cadence.
  Each test evaluation averages the fixed sample ordinals
  `test_start_iteration .. test_start_iteration + test_sample_count - 1`.
- `run_path` and `run_name` create per-run directories with TensorBoard scalar
  and config-text logging plus `snapshots/current.pt` and `snapshots/best.pt`.
- `sample_visualization_every = 10000` logs up to two GT/control-point training
  samples to TensorBoard as side/top/cross oriented slices through each sampled
  point, stitched side by side per view. Each view has one normalized image
  slice, one fused label image using negative/undefined/positive values
  `0/127/255`, a fixed-scale predicted embedding cosine image against that
  sample's rounded CP embedding, and an `other_cp` cosine view against the other
  selected CP when available. `test_visualization_every` logs the first fixed
  test batch the same way under `test_sample/...`. Out-of-crop slice samples are
  black in image views and a coarse `63/191` checkerboard in label/cosine views.
- Label geometry uses the explicit normal-frame fields only; legacy radius
  fallbacks are not part of the config/API.

Not implemented yet:

- Soft contrastive weighting by continuous geometric distance.

## Configs

Smoke config:

```bash
PYTHONPATH=vesuvius/src:. python -m vesuvius.neural_tracing.fiber_trace.train vesuvius/src/vesuvius/neural_tracing/configs/fiber_trace_lasagna_smoke.json
```

Training starter config:

```bash
PYTHONPATH=vesuvius/src:. python -m vesuvius.neural_tracing.fiber_trace.train vesuvius/src/vesuvius/neural_tracing/configs/fiber_trace_lasagna_train.json
```

Before running either config, replace:

- `base_volume_path`
- `lasagna_manifest_path`
- `fiber_glob` or `fiber_paths`
- optional `test_fiber_glob` or `test_fiber_paths`
- `test_every`, `test_sample_count`, `test_start_iteration`, and
  `test_visualization_every`
- `sample_visualization_every`
- `volume_cache_dir` when using remote zarrs
- `random_negative_pool_size`; the default `1000` precomputes deterministic
  random-negative centers before batch sampling
- optionally `sample_limit`; when set, deterministic sample ordinals wrap after
  that many training samples so longer debug runs reuse the same bounded sample
  set and prefetch only downloads that bounded set
- optionally `control_point_margin_voxels`; omit it or set `10` to let the CP
  move as close as 10 voxels from the final crop border
- optionally `augmentation_crop_size`; the starter configs use a 16 voxel
  per-side trim margin

## Next Implementation Plan

1. Add soft contrastive weighting by continuous geometric distance if future
   experiments need it.
2. Add broader multi-fiber batching once cross-fiber sampling semantics are
   specified.
