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

Each dataset config entry must provide:

- `volume_path`: image zarr store
- `volume_scale`: pyramid level, default `0`
- `mask_path` or `grad_mag_path`: mandatory valid-data source
- `nx_path` and `ny_path`: mandatory Lasagna hemisphere-encoded normal channels
- `fiber_paths` or `fiber_glob`: one or more VC3D fiber JSON files

Remote zarr paths use the existing `vesuvius.neural_tracing.datasets.common`
zarr/cache support. Missing mask or grad-mag data is an error, and mask shape
must match the selected image volume level. Normal channel shapes must also
match the selected image volume level.

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
