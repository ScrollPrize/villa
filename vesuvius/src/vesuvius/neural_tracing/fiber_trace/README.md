# Fiber Trace MVP

This package contains the first training path for tracing VC3D fiber JSON
records with direction-conditioned 3D crops. It is intentionally separate from
the row/column surface tracing and Lasagna sparse-solve samplers.

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

The loader validates that `base_volume_path` level 0 has shape equal to the
manifest `base_shape_zyx`. Image crops are read at `base_volume_scale`.
`grad_mag`, `nx`, and `ny` are opened through the manifest, using each group's
`scaledown` and channel index. These Lasagna channels do not have to be the
same shape as the selected base image level; they are sampled into the selected
training grid according to the manifest scale convention.

Remote zarr paths use the existing `vesuvius.neural_tracing.datasets.common`
zarr/cache support. Missing mask or grad-mag data is an error, and mask shape
must align with the selected image volume level through the manifest. Normal
channel shapes must align the same way.

Legacy raw-path config with `volume_path`, `grad_mag_path` or `mask_path`, and
`nx_path`/`ny_path` is still accepted for tests and direct debugging, but the
manifest path is the intended training spec.

## Batch Construction

Batches are single-fiber batches. For batch size `N`, the builder samples one
fiber record and emits:

- `N / 2` crops centered on random GT control points from that fiber
- `N / 2` crops centered on random valid mask/grad-mag voxels

The batch stores crop kind metadata so tests and trainers can verify the
composition. Batch size must be even for this MVP.

## Direction Conditioning

Each crop receives normalized `fw(3)` and `up(3)` conditioning in `xyz`
component order. The local GT forward vector is derived from the nearest fiber
line tangent. With probability
`positive_direction_probability` the conditioning direction is a small angular
jitter around the tangent; otherwise it is a random unit vector. Random
directions are still classified against the GT tangent and are not assumed to
be negative.

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
- voxels within `positive_radius` of the fiber polyline and aligned with the
  conditioning direction are `positive`
- voxels far enough from the polyline, or close but direction-disagreeing, are
  `negative`
- voxels in the geometric or angular tolerance band are `ignore`

The classifier also emits target local `fw` and `up` vector fields in `xyz`
component order plus a `target_up_valid` mask.

`positive_radius` and `ignore_radius` are measured in the selected training
grid, not always base-level voxels. If `base_volume_scale` is `2`, then one
training voxel spans `4` base voxels. The fiber coordinates remain base `xyz`;
the loader divides them by `2**base_volume_scale` only for voxel
classification.

## Model And Losses

`DirectionConditionedFiberTraceModel` uses the existing
`Vesuvius3dUnetModel` as a compact 3D U-Net feature backbone and adds a
direction-conditioned head. The head concatenates broadcast normalized `fw`
and `up` conditioning with the U-Net features and outputs:

- L2-normalized embedding
- normalized predicted `fw`
- normalized predicted `up`

Losses are:

- supervised contrastive / InfoNCE over classified positive and negative voxels
- cosine forward-vector loss on positive voxels
- sign-ambiguous cosine up-vector loss on positive voxels

`loss.max_contrastive_samples` caps the number of classified voxels used by the
contrastive term per batch. It limits memory/time for dense crops and does not
change which voxels are labeled positive, negative, or ignored.

## Config Knobs

Use the runnable smoke template:

```bash
PYTHONPATH=vesuvius/src python -m vesuvius.neural_tracing.fiber_trace.train vesuvius/src/vesuvius/neural_tracing/configs/fiber_trace_lasagna_smoke.json
```

Replace only the dataset paths first:

- `base_volume_path`: the scan OME-Zarr
- `base_volume_scale`: which scan OME-Zarr level to train on
- `lasagna_manifest_path`: the Lasagna `.lasagna.json`
- `fiber_glob` or `fiber_paths`: VC3D fiber JSON files

`image_normalization: "zscore"` normalizes each CT crop with the same helper
used by the Lasagna training zarr path. Use `"unit"` only for uint8 smoke tests
where `value / 255` is intended.

`positive_direction_probability: 0.5` means half of crop conditioning directions
are jittered from the true local fiber tangent and half are random directions.
This tests the direction-conditioned spec: the same crop can be positive or
negative depending on the requested forward direction.

`positive_direction_jitter_degrees: 10.0` is the angular perturbation applied to
the true tangent when sampling a positive conditioning direction. It makes the
positive branch tolerant to small orientation noise without changing the target
fiber tangent.

`positive_radius: 1.5` marks voxels close enough to the fiber centerline to be
eligible positive, after direction agreement. `ignore_radius: 3.0` creates a
buffer around the fiber so near-miss voxels are ignored instead of trained as
hard negatives. Both are in selected training-grid voxels.

## Entrypoint

Run a training smoke or short job with:

```bash
python -m vesuvius.neural_tracing.fiber_trace.train /path/to/config.json
```

The current trainer is intentionally minimal: it samples batches directly from
`FiberTraceBatchBuilder`, runs one model forward/backward per step, logs scalar
losses, and optionally writes a checkpoint.

## Current Status

Implemented:

- VC3D fiber JSON parser and validation
- tangent, conditioning direction, Lasagna-normal-derived up-vector, and voxel
  classification helpers
- fixed-size single-fiber batch builder using zarr image, mask/grad-mag, and
  `nx`/`ny` normal data
- direction-conditioned U-Net model/head
- contrastive and direction losses
- JSON-config train entrypoint
- unit and smoke tests on synthetic volumes

Out of scope for this MVP:

- cross-fiber negatives
- inference/trace integration
- distributed dataloading
- Lasagna sampler or sparse-solve sampler integration
