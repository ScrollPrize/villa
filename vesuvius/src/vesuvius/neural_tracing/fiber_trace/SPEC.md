# Fiber Tracing Spec And Status

This document records the intended fiber tracing training design and the current
MVP mapping. Keep it updated when the implementation changes so the training
semantics do not drift into undocumented defaults.

## Intended Training Spec

The model is a 3D U-Net or generic Vesuvius model with a
direction-conditioned decoder head.

Inputs:

- CT crop from the base scan OME-Zarr.
- Direction conditioning as two normalized `xyz` vectors:
  - `fw`: forward direction.
  - `up`: right-angle up direction derived from Lasagna normals.

Head outputs:

- N-dimensional embedding.
- Predicted `fw` vector.
- Predicted `up` vector.

Losses:

- Multi-positive InfoNCE / supervised contrastive / soft contrastive loss for
  embeddings.
- Cosine similarity loss for predicted direction vectors.
- Up-vector loss must be sign ambiguous where appropriate: `up` and `-up` are
  equivalent unless a later task explicitly breaks that ambiguity.

Ground-truth positives:

- Training starts from VC3D line annotation control points.
- All control points on the same line can be compared against the other control
  points on that line as positive embedding samples.
- Within a crop, dense output voxels can also contribute positives when their
  geometry and conditioning match the same local fiber sample.

Negatives:

- Lasagna normal information defines the local sheet plane.
- Around control points, two 90 degree cones identify regions that are probably
  not on the line when those regions are also away from the normal-defined
  plane by at least `K = 30` voxels.
- Random valid volume samples only contribute labels where the explicit
  positive-zone, cone-negative, or direction-disagreement rules apply.
- Conditioning frames with folded frame error from 60 to 90 degrees should
  create negative samples for the same patch.
- Voxels outside the positive and negative zones are ignored.

Direction conditioning:

- `fw` is the local fiber tangent from the line geometry.
- `up` is derived from the Lasagna normal by projecting the normal away from
  `fw` and normalizing.
- Direction labels use folded frame-equivalent angular error, not raw `fw`
  angle. Lasagna normal/up sign ambiguity makes both target up signs valid;
  after also folding the equivalent pair `(fw, up) == (-fw, -up)`, the usable
  error range is 0 to 90 degrees.
- Conditioning frames with folded error up to 30 degrees should still supervise
  the model to output the correct target direction, up vector, and positive
  embedding for the local fiber.
- Conditioning frames with folded error from 30 to 60 degrees are ignored for
  direction conditioning.
- Conditioning frames with folded error from 60 to 90 degrees supervise
  negatives for the same local positive zone. There is no 90 to 180 degree
  negative band; raw angles above 90 fold back toward positive-equivalent
  frames.
- For a simple raw `fw` rotation around a stable up axis: raw 0..30 and
  150..180 degrees are positive-equivalent, raw 30..60 and 120..150 degrees
  are ignored, raw 60..120 degrees are negative-conditioning candidates, and
  raw 90 degrees is maximally wrong.

Jitter semantics:

- Position jitter is part of pair labeling, not a separate data generation
  requirement.
- Same-sample jitter target: up to `+/-40` voxels in the plane of the Lasagna
  normal and up to `+/-10` voxels perpendicular to that plane.
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
- Direct `zarr.open()` fallbacks are not allowed in the fiber tracing training
  path.
- Manifest-less derivative channel keys such as `grad_mag_path`, `mask_path`,
  `nx_path`, and `ny_path` are rejected.

Normal convention:

- Lasagna `nx`/`ny` are required by default.
- Decode with `(value - 128) / 127`.
- Reconstruct `nz = sqrt(1 - nx^2 - ny^2)` with `nz >= 0`.
- Compute `up_xyz = normalize(normal_xyz - dot(normal_xyz, fw_xyz) * fw_xyz)`.
- Degenerate projected normals are invalid by default or raise when configured;
  arbitrary perpendicular fallback requires explicit `allow_arbitrary_up_fallback:
  true`.

## Current MVP Mapping

Implemented:

- Manifest-based dataset config via `base_volume_path`,
  `base_volume_scale`, `lasagna_manifest_path`, and fibers.
- Shape validation of base OME-Zarr level 0 against manifest `base_shape_zyx`.
- Lasagna `grad_mag`, `nx`, and `ny` lookup through manifest channel groups and
  group `scaledown`.
- Dense crop training with:
  - paired positive and negative conditioning variants for every selected crop;
  - multiple GT control-point crops from the selected line when possible;
  - random valid `grad_mag` crops when the batch has room after GT pairs.
- Direction-conditioned U-Net head outputting embedding, `fw`, and `up`.
- Supervised contrastive loss plus cosine `fw` and sign-ambiguous `up` losses.
- Dense `labels` and `target_id` tensors:
  - positive voxels carry the selected fiber-line identity;
  - explicit negatives carry `NEGATIVE_ONLY_ID`;
  - ignored voxels carry `IGNORE_ID`.
- Hard supervised contrastive loss:
  - positive anchors only;
  - same-identity positives attract across control-point crops;
  - negative-only voxels are denominator-only and are never pulled together.
- Per-voxel labels from normal-frame positive zones, cone negative zones, and
  paired direction disagreement.
- `normal_plane_jitter_voxels = 40` and
  `normal_perpendicular_jitter_voxels = 10` define the positive zone.
- `negative_cone_distance_voxels = 30` defines the minimum distance from the
  normal-defined plane for cone negatives.
- `positive_direction_jitter_degrees = 30.0`,
  `positive_cosine = cos(30 degrees) = 0.8660254037844386`,
  `negative_direction_min_degrees = 60.0`,
  `negative_direction_max_degrees = 90.0`, and
  `negative_cosine = cos(60 degrees) = 0.5` implement the folded 30/60/90
  degree direction thresholds.
- `run_path` and `run_name` create per-run directories with TensorBoard scalar
  and config-text logging plus `snapshots/current.pt` and `snapshots/best.pt`.
- `sample_visualization_every = 10000` logs one training sample to TensorBoard
  as side/top/cross oriented slices. Each view has one normalized image slice
  and separate positive, undefined, and negative label-mask images.
- `positive_radius` and `ignore_radius` remain accepted only as legacy/debug
  fallback values when the named normal-frame fields are omitted.

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
- `sample_visualization_every`
- `volume_cache_dir` when using remote zarrs

## Next Implementation Plan

1. Add soft contrastive weighting by continuous geometric distance if future
   experiments need it.
2. Add broader multi-fiber batching once cross-fiber sampling semantics are
   specified.
