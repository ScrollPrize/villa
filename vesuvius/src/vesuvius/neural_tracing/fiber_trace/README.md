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
fiber record and emits:

- `N / 2` GT control-point crops with mixed positive/ignore/negative labels
  derived from the selected fiber line and Lasagna normal geometry
- `N / 2` random valid crops whose valid voxels are explicit negatives

GT crops choose control points with deterministic per-iteration random streams;
duplicates are allowed when the sampled slots collide. The crop is offset so the
control point is not forced to the crop center while still retaining the
configured `control_point_margin_voxels` around it. Random-negative centers are
precomputed before batch sampling from valid `grad_mag > 0` voxels with valid
Lasagna normal channels. `random_negative_pool_size` controls the pool size,
defaulting to `1000`; batch sampling then selects from that pool by
deterministic modulo. The batch stores crop kind and direction kind metadata so
tests and trainers can verify the composition. Batch size must be even for this
MVP.

## Direction Conditioning

Each crop receives one normalized `fw(3)` conditioning vector in `xyz`
component order. GT control-point crops condition on the local fiber tangent at
the sampled control point with up to `positive_direction_jitter_degrees` of
forward-direction augmentation, normally 30 degrees. Random-negative crops use
an independent random forward condition.

Conditioning direction does not change voxel labels. It is only an input to the
model head. The label masks come from fiber-line geometry, the Lasagna normal
channels, and the random-negative crop kind.

Lasagna normals are decoded from `nx`/`ny` channels through Lasagna's own normal
decoder. For GT crops, the selected control-point normal is used for label
geometry and slice visualization, not as model conditioning. Normal channels are
required.

## Voxel Classification

For every output voxel:

- invalid mask/grad-mag voxels are `ignore`
- GT-control crop voxels in the normal-frame positive zone are `positive`
- cone negatives come only from the normal-axis double cone above/below the
  local sheet, starting at `negative_cone_distance_voxels`
- random-negative crop valid voxels are explicit `negative`
- all other valid voxels are `ignore`

The classifier also emits target local `fw` vectors in `xyz` component order
plus a dense `target_id` tensor. Positive voxels carry the selected fiber-line
identity, explicit negatives carry `NEGATIVE_ONLY_ID`, and ignored voxels carry
`IGNORE_ID`.

`normal_plane_jitter_voxels` and `normal_perpendicular_jitter_voxels` are
measured in the selected training grid, not always base-level voxels. If
`base_volume_scale` is `2`, then one training voxel spans `4` base voxels. The
fiber coordinates remain base `xyz`; the loader divides them by
`2**base_volume_scale` only for voxel classification. `positive_radius` and
`ignore_radius` are accepted as legacy/debug fallback config, but they are not
the documented training path.

`crop_size` is the final model/input label crop size. `augmentation_crop_size`
is the larger outer crop read before post-augmentation center trimming; it must
be at least `crop_size` and differ by an even number on each axis. The starter
configs trim 16 voxels per side (`64 -> 96`, `128 -> 160`), which removes
modest padded or interpolated borders without the larger read cost of a
32-voxel margin.

## Model And Losses

`DirectionConditionedFiberTraceModel` uses the existing
`Vesuvius3dUnetModel` as a compact 3D U-Net feature backbone and adds a
direction-conditioned head. The head concatenates broadcast normalized `fw`
conditioning with the U-Net features and outputs:

- L2-normalized embedding
- normalized predicted `fw`

Losses are:

- pairwise embedding contrastive loss over classified positives and explicit
  negatives

The contrastive loss samples positive-positive pairs from matching fiber-line
`target_id`s and positive-negative pairs against explicit negatives. Negative
voxels are never paired with each other. Soft distance-weighted contrastive loss
is intentionally deferred.

`loss.max_contrastive_samples` caps the sampled pair budget per batch.

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

`positive_direction_jitter_degrees: 30.0` controls the GT control-point forward
conditioning augmentation. Direction conditioning does not change labels.

`control_point_margin_voxels` controls the deterministic GT crop offset. The
selected control point is placed at either that margin or the opposite-side
margin on each crop axis. If omitted, the builder uses `min(40, floor((size -
1) / 2))` per the smallest crop axis; explicit values that cannot fit the crop
are rejected.

Legacy label-conditioning keys such as `positive_direction_probability`,
`negative_direction_min_degrees`, `negative_direction_max_degrees`,
`positive_cosine`, and `negative_cosine` are rejected.

`normal_plane_jitter_voxels: 40.0` and
`normal_perpendicular_jitter_voxels: 10.0` define the normal-frame positive
zone. `negative_cone_distance_voxels: 30.0` defines the cone-negative minimum
distance away from the Lasagna-normal plane, where the normal-axis double cone
starts.

`random_negative_pool_size: 1000` precomputes the deterministic random-negative
center pool before training batches are sampled. Training batches select from
that pool by modulo instead of probing random valid voxels during batch
construction.

Sampling streams are deterministic by purpose and keyed from `seed`, iteration,
batch slot, record index, and control index where relevant. Re-running a given
training iteration samples the same record, crop offsets, conditioning vectors,
and random-negative pool entries independent of previous sampler calls.

`positive_radius` and `ignore_radius` are kept only for legacy/debug fallback
behavior when the named normal-frame fields are omitted.

`sample_visualization_every: 10000` logs one training sample to TensorBoard on
that interval. The trainer writes three oriented slices through the sampled
point:

- `side`: fiber direction by sampled up/normal direction
- `top`: fiber direction by binormal
- `cross`: binormal by sampled up/normal direction, perpendicular to the fiber

Each view is logged under `train_sample/<view>/` as one normalized `image`
slice plus one fused `labels` image using negative/undefined/positive values
`0/127/255`, and one fixed-scale `cos_emb_cp` image for predicted embedding
cosine against the sampled-point embedding. Slice samples outside the crop are
black in `image` and shown as a coarse `63/191` checkerboard in label/cosine
views.

## Entrypoint

Run a training smoke or short job with:

```bash
python -m vesuvius.neural_tracing.fiber_trace.train /path/to/config.json
```

To prefetch the zarr chunks that the same training config will touch, run:

```bash
python -m vesuvius.neural_tracing.fiber_trace.train --prefetch /path/to/config.json
```

`--prefetch` builds the deterministic train/test crop chunk list for
`num_steps`, deduplicates chunk keys, then downloads the chunks into
`volume_cache_dir` with up to 16 parallel workers. Use `--prefetch-workers N`
to lower the worker count.

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
- tangent, forward-conditioning direction, Lasagna-normal geometry, and voxel
  classification helpers
- fixed-size single-fiber batch builder using zarr image, mask/grad-mag, and
  fixed sampled-point `nx`/`ny` normal data decoded through Lasagna
- direction-conditioned U-Net model/head
- pairwise embedding contrastive loss
- JSON-config train entrypoint with TensorBoard scalar/config logging and
  sample-slice visualization plus current/best snapshots
- one debug timing/cache row per training step when `debug_sampling` or
  `debug_cache` is enabled
- unit and smoke tests on synthetic volumes

Out of scope for this MVP:

- cross-fiber negatives
- inference/trace integration
- distributed dataloading
- Lasagna sampler or sparse-solve sampler integration
