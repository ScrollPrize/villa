# Fiber Trace MVP

This package contains the first training path for tracing VC3D fiber JSON
records with direction-conditioned 3D crops. It is intentionally separate from
the row/column surface tracing and Lasagna sparse-solve samplers.

See [SPEC.md](SPEC.md) for the original fiber tracing training spec, the
Lasagna/Vesuvius convention decisions, and the current implementation status.

## Data Schema

Training records are `vc3d_fiber` version 1 JSON files:

```json
{
  "type": "vc3d_fiber",
  "version": 1,
  "line_points": [[x, y, z], "..."],
  "control_points": [[x, y, z], "..."]
}
```

The JSON points are interpreted in VC3D `x, y, z` order and remain `xyz`
through all geometry calculations. Volume crops, masks, zarr reads, and tensor
spatial axes are `z, y, x`; the loader converts `xyz -> zyx` only when indexing
dense arrays.

The preferred dataset config entry provides:

- `base_volume_path`: base image OME-Zarr store
- `base_volume_scale`: base image OME-Zarr pyramid level used as the training
  grid, default `0`; level `L` has voxel spacing `2**L` in base voxels
- `lasagna_manifest_path`: `.lasagna.json` manifest with `base_shape_zyx` and
  `grad_mag`, `nx`, and `ny` channel groups
- `fiber_paths` or `fiber_glob`: one or more VC3D fiber JSON files
- optional `test_fiber_paths` or `test_fiber_glob`: held-out VC3D fiber JSON
  files for TensorBoard test-loss logging

The loader validates that `base_volume_path` level 0 has shape equal to the
manifest `base_shape_zyx`. Image crops are read at `base_volume_scale`.
`grad_mag`, `nx`, and `ny` are opened through the manifest, using each group's
`scaledown` and channel index. Shape validation follows Lasagna's convention:
`base_shape_zyx` plus group `scaledown` determine the zarr shape, while
`group.sd_fac * source_to_base` determines the channel spacing in base voxels.
These Lasagna channels do not have to be the same shape as the selected base
image level; they are sampled into the selected training grid according to the
manifest scale convention. Mask/grad-mag validity is binary: voxels with value
`> 0` are valid and value `0` is invalid. There is no configurable validity
threshold.

Remote zarr paths use the existing `vesuvius.neural_tracing.datasets.common`
zarr/cache support. Missing mask or grad-mag data is an error, and mask shape
must align with the selected image volume level through the manifest. Normal
channel shapes must align the same way.

Manifest-less derivative channel configuration is not supported. Dataset
entries with legacy `volume_path` or raw `grad_mag_path`, `mask_path`,
`nx_path`, or `ny_path` keys are rejected because they bypass Lasagna manifest
scale, shape, channel, and coordinate metadata.

## Batch Construction

Batches are single-fiber batches. For batch size `N`, the builder samples one
fiber record, selects `N / 2` crop centers, and emits two conditioning variants
for each selected crop:

- a positive-conditioned copy jittered 0 to 30 folded frame degrees from the
  local tangent/up frame
- a negative-conditioned copy jittered 60 to 90 folded frame degrees from the
  local tangent/up frame

Selected centers prefer GT control points from the same fiber line, using
multiple distinct control points when possible, and then random valid
mask/grad-mag voxels when the batch has room. The batch stores crop kind and
direction kind metadata so tests and trainers can verify the composition. Batch
size must be even for this MVP.

## Direction Conditioning

Each crop receives normalized `fw(3)` and `up(3)` conditioning in `xyz`
component order. The local GT forward vector is derived from the nearest fiber
line tangent. Each selected crop receives a positive-conditioned and
negative-conditioned copy. Positive copies use
`positive_direction_jitter_degrees`, normally 30 folded frame degrees. Negative
copies use `negative_direction_min_degrees` and
`negative_direction_max_degrees`, normally 60 and 90 folded frame degrees.

Voxel direction labels use folded frame-equivalent angular error, not raw `fw`
angle. Lasagna normal/up sign ambiguity makes both target up signs valid; after
also folding the equivalent pair `(fw, up) == (-fw, -up)`, the usable error
range is 0 to 90 degrees. For a simple raw `fw` rotation around a stable up
axis, raw 0..30 and 150..180 degrees are positive-equivalent, raw 30..60 and
120..150 degrees are ignored, raw 60..120 degrees are negative-conditioning
candidates, and raw 90 degrees is maximally wrong.

The `up` vector is built by decoding Lasagna `nx`/`ny` channels as
`(value - 128) / 127`, reconstructing `nz = sqrt(1 - nx^2 - ny^2)` with
`nz >= 0`, and projecting that normal away from the local `fw`:

```text
up_xyz = normalize(normal_xyz - dot(normal_xyz, fw_xyz) * fw_xyz)
```

Degenerate projected normals mark up supervision invalid by default. Set
`degenerate_up_policy: "raise"` to fail the crop instead. The only arbitrary
perpendicular fallback is the explicit opt-in `allow_arbitrary_up_fallback:
true`; it is disabled by default. The loss treats `up` and `-up` as equivalent.

## Voxel Classification

For every output voxel:

- invalid mask/grad-mag voxels are `ignore`
- voxels in the normal-frame positive zone with folded frame error up to
  30 degrees are `positive`
- voxels in that same positive zone become `negative` only when the
  folded frame error is from 60 to 90 degrees
- cone negatives come only from the two lateral 90 degree cone zones and only
  when the absolute distance to the Lasagna-normal plane is at least
  `negative_cone_distance_voxels`
- all other valid voxels are `ignore`

The classifier also emits target local `fw` and `up` vector fields in `xyz`
component order plus a `target_up_valid` mask and dense `target_id` tensor.
Positive voxels carry the selected fiber-line identity, explicit negatives
carry `NEGATIVE_ONLY_ID`, and ignored voxels carry `IGNORE_ID`.

`normal_plane_jitter_voxels` and `normal_perpendicular_jitter_voxels` are
measured in the selected training grid, not always base-level voxels. If
`base_volume_scale` is `2`, then one training voxel spans `4` base voxels. The
fiber coordinates remain base `xyz`; the loader divides them by
`2**base_volume_scale` only for voxel classification. `positive_radius` and
`ignore_radius` are accepted as legacy/debug fallback config, but they are not
the documented training path.

## Model And Losses

`DirectionConditionedFiberTraceModel` uses the existing
`Vesuvius3dUnetModel` as a compact 3D U-Net feature backbone and adds a
direction-conditioned head. The head concatenates broadcast normalized `fw`
and `up` conditioning with the U-Net features and outputs:

- L2-normalized embedding
- normalized predicted `fw`
- normalized predicted `up`

Losses are:

- hard supervised contrastive / InfoNCE over classified positives and explicit
  negatives
- cosine forward-vector loss on positive voxels
- sign-ambiguous cosine up-vector loss on positive voxels

The contrastive loss anchors only on positive voxels. Positives with the same
fiber-line `target_id` attract each other across control-point crops, while
explicit negatives are denominator-only and are never pulled together. Soft
distance-weighted contrastive loss is intentionally deferred.

`loss.max_contrastive_samples` caps the number of positive and negative voxels
used by the contrastive term per batch. Positives and negatives are sampled
separately so explicit negatives do not crowd out all anchors.

## Config Knobs

Use the runnable smoke template for a one-step import/data-path check:

```bash
PYTHONPATH=vesuvius/src python -m vesuvius.neural_tracing.fiber_trace.train vesuvius/src/vesuvius/neural_tracing/configs/fiber_trace_lasagna_smoke.json
```

Use the starter training template for a longer run:

```bash
PYTHONPATH=vesuvius/src:. python -m vesuvius.neural_tracing.fiber_trace.train vesuvius/src/vesuvius/neural_tracing/configs/fiber_trace_lasagna_train.json
```

Replace only the dataset paths first:

- `base_volume_path`: the scan OME-Zarr
- `base_volume_scale`: which scan OME-Zarr level to train on
- `lasagna_manifest_path`: the Lasagna `.lasagna.json`
- `fiber_glob` or `fiber_paths`: VC3D fiber JSON files
- `test_fiber_glob` or `test_fiber_paths`: optional held-out VC3D fiber JSON
  files for test-loss logging; use top-level `test_datasets` for a separate
  test volume/manifest

`image_normalization: "zscore"` normalizes each CT crop with the same helper
used by the Lasagna training zarr path. Use `"unit"` only for uint8 smoke tests
where `value / 255` is intended.

`positive_direction_probability` is a legacy/debug knob from the earlier
single-conditioning MVP and is not used by the paired-conditioning training
path.

`positive_direction_jitter_degrees: 30.0` in the training template comes from
the spec: conditioning frames up to 30 folded degrees wrong are still
supervised to output the correct direction, up vector, and positive embedding.

`negative_direction_min_degrees: 60.0` and
`negative_direction_max_degrees: 90.0` define the paired negative-conditioning
range in folded frame degrees.

`positive_cosine: 0.8660254037844386` is `cos(30 degrees)`.
`negative_cosine: 0.5` is `cos(60 degrees)`, matching the folded negative
threshold.

`normal_plane_jitter_voxels: 40.0` and
`normal_perpendicular_jitter_voxels: 10.0` define the normal-frame positive
zone. `negative_cone_distance_voxels: 30.0` defines the cone-negative minimum
distance away from the Lasagna-normal plane.

`positive_radius` and `ignore_radius` are kept only for legacy/debug fallback
behavior when the named normal-frame fields are omitted.

`sample_visualization_every: 10000` logs one training sample to TensorBoard on
that interval. The trainer writes three oriented crop-center slices:

- `side`: fiber direction by sampled up/normal direction
- `top`: fiber direction by binormal
- `cross`: binormal by sampled up/normal direction, perpendicular to the fiber

Each view is logged under `train_sample/<view>/` as one normalized `image`
slice plus separate `positive`, `undef`, and `negative` class-mask images.

## Entrypoint

Run a training smoke or short job with:

```bash
python -m vesuvius.neural_tracing.fiber_trace.train /path/to/config.json
```

Each run creates `run_path/run_name_YYYYmmdd_HHMMSS/`. TensorBoard event files
are written directly in that run directory, including scalar losses every
`log_every` steps, sample-slice images every `sample_visualization_every` steps,
and a `config/json` text entry with the training config. Model snapshots are
written to:

- `snapshots/current.pt`: most recent logged model
- `snapshots/best.pt`: best logged model by `test/total` when a test split is
  configured, otherwise by `train/total`

## Current Status

Implemented:

- VC3D fiber JSON parser and validation
- tangent, conditioning direction, Lasagna-normal-derived up-vector, and voxel
  classification helpers
- fixed-size single-fiber batch builder using zarr image, mask/grad-mag, and
  `nx`/`ny` normal data
- direction-conditioned U-Net model/head
- contrastive and direction losses
- JSON-config train entrypoint with TensorBoard scalar/config logging and
  sample-slice visualization plus current/best snapshots
- unit and smoke tests on synthetic volumes

Out of scope for this MVP:

- cross-fiber negatives
- inference/trace integration
- distributed dataloading
- Lasagna sampler or sparse-solve sampler integration
