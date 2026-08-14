# Native phase caches

`infer_winding_volume.py` can stop after neural inference and save the model's
native phase field. The normal full-resolution point archive and raster volumes
can then be reconstructed later without rerunning the model or extracting the
source image volume.

This mode currently supports headless phase models (`use_crossing_head: false`).
The cache is lossless: phase is stored as float32, validity masks are bit-packed,
and each slab's geometric frame and seed winding are retained.

## Create the cache

Use the normal inference command and add `--native-phase-only`. `OUTPUT` is the
phase-cache Zarr rather than the final winding Zarr:

```bash
python infer_winding_volume.py \
  /ephemeral/sean/spiral_new/dataset/spiral_output/2026-07-27_s1_slice-7000-17500_4610-patch_track-walk-30k/checkpoint_fitted.ckpt \
  /ephemeral/sean/winding_inferences/PHercParis4/winding_native_phase.zarr \
  --model-ckpt /ephemeral/sean/villa/vesuvius/src/vesuvius/neural_tracing/winding_models/runs/winding_model_3d_large_9/ckpt_015000.pth \
  --reference-zarr /ephemeral/sean/winding_train_data/PHercParis4/volumes/s1_2um_ds2.zarr \
  --volume-scale 0 \
  --z-range 7500 9500 \
  --winding-range 10 130 \
  --winding-step 3 \
  --seed-spacing 60 \
  --compile \
  --gpus 0,1,2,3,4,5,6 \
  --native-phase-only
```

Do not pass output-resolution, interpolation, decode, raster, archive, or
probability options to this command. They have no meaning until reconstruction,
and the CLI rejects them in `--native-phase-only` mode. The stored phase is
always the model's native `32 x 32 x 384` float32 output.

Cache creation uses only:

- seed selection (`--z-range`, `--winding-range`, `--winding-step`,
  `--seed-spacing`, `--seed-source`, or `--seed-rays-npz`);
- model and volume extraction (`--model-ckpt`, `--reference-zarr`,
  `--volume-scale`, `--batch-size`, `--extract-threads`, and
  `--volume-cache-bytes`);
- execution (`--gpus`, `--compile`, and optional `--max-slabs`).

The cache contains:

- `phase/shard_*`: losslessly compressed native phase;
- `valid/shard_*`: bit-packed full `128 x 128 x 384` validity masks;
- `frame/shard_*`: slab origins and axes;
- `available/shard_*`: commit markers for successfully extracted slabs;
- `rays/*`: seeds, directions, and absolute seed windings;
- root attributes describing the model geometry and suggested reconstruction
  parameters.

Do not reconstruct from a cache whose root `complete` attribute is false.

For the 139,205-slab PHercParis4 workload, the phase payload is approximately
219 GB uncompressed and 144-150 GB losslessly compressed. Validity and frame
metadata are small by comparison.

## Export compact Spiral supervision

`export_spiral_supervision.py` converts a complete cache into an uncompressed,
GPU-resident-friendly crossing store. It decodes the centre ray of every slab,
uses only the contiguous valid interval containing the seed anchor, and drops
the first and last two crossings. Shards are independently restartable under
`OUTPUT.partial`; the final output directory appears only after all shards
complete and their NumPy-array checksums are recorded.

For the PHercParis4 ft7 cache:

```bash
python export_spiral_supervision.py \
  /ephemeral/sean/winding_inferences/PHercParis4/winding_native_phase_large11_ws3_ss60.zarr \
  /ephemeral/sean/winding_inferences/PHercParis4/winding_spiral_supervision_large11_ws3_ss60 \
  --workers 32 \
  --edge-trim 2
```

The output stores ray origins/directions and ragged subvoxel crossing
positions rather than repeated XYZ points. `fit_spiral.py` reconstructs pairs
on CUDA and derives adjacent-pair density as one winding divided by the
physical crossing gap.

The exporter divides every source shard into 256-ray tasks aligned to the
cache's 32-ray physical files, so worker count is not limited by the number of
GPU/source shards. Its aggregate `tqdm` bar reports source rays, retained rays,
decoded crossings, throughput, and ETA. Each completed task is atomic and
restartable under `OUTPUT.partial/parts`; use `--rays-per-task` to trade finer
progress/restart granularity for scheduling overhead.

## Reconstruct the normal full-resolution output

Run the same script with a new `OUTPUT` and pass the cache via `--phase-cache`.
The requested decode options are applied at reconstruction time:

```bash
python infer_winding_volume.py \
  /ephemeral/sean/spiral_new/dataset/spiral_output/2026-07-27_s1_slice-7000-17500_4610-patch_track-walk-30k/checkpoint_fitted.ckpt \
  /ephemeral/sean/winding_inferences/PHercParis4/winding_fullres.zarr \
  --model-ckpt /ephemeral/sean/villa/vesuvius/src/vesuvius/neural_tracing/winding_models/runs/winding_model_3d_large_9/ckpt_015000.pth \
  --reference-zarr /ephemeral/sean/winding_train_data/PHercParis4/volumes/s1_2um_ds2.zarr \
  --volume-scale 0 \
  --max-level 4 \
  --prob-volume \
  --output-downsample 1 \
  --column-upsample 4 \
  --column-step 1 \
  --prob-column-step 1 \
  --prob-volume-floor 0 \
  --prob-combine phase \
  --gpus 0,1,2,3,4,5,6 \
  --phase-cache /ephemeral/sean/winding_inferences/PHercParis4/winding_native_phase.zarr
```

Reconstruction performs the same `align_corners=True` transverse interpolation,
phase-passage decode, point/strip archiving, winding voting, and probability
merge as direct full-resolution inference. It writes the standard arrays:

- `points/{xyz,winding,prob}`;
- `strips/{offsets,slab}`;
- `winding` and `confidence`;
- `crossing_prob` when `--prob-volume` is supplied;
- `rays/*`.

The reconstruction GPU count does not need to match the cache-generation GPU
count. The model checkpoint and reference path are validated against the cache.
`--compile` has no effect during reconstruction because no neural forward pass
is executed.

If the cache and reference volume were copied to a machine with different
absolute paths, pass `--phase-cache-allow-relocated-inputs`. Cached model
geometry then replaces the model-checkpoint read, and the supplied reference
Zarr is used for output coordinates and shape. This is deliberately opt-in so
an accidental path mismatch is still rejected by default.

To reproduce a particular output, preserve the reconstruction command. Changing
`--column-upsample`, column steps, probability margins/combine mode, decode
thresholds, maximum level, or output downsample intentionally creates a
different derived product from the same phase cache.

For headless models, `--prob-combine phase` is the recommended crossing-volume
mode. Instead of averaging already-rendered unit kernels, it registers every
slab's fractional phase at its seed, circularly merges phase per output voxel,
and renders one crossing kernel from the consensus. Observations are weighted
by distance from the seed anchor and by a smooth taper inside the hard
probability margins. Circular concentration attenuates locations where slabs
disagree. Relevant controls are:

- `--prob-phase-level-half-life` (default 2 windings);
- `--prob-phase-max-level` (default `--max-level + 0.5`);
- `--prob-phase-edge-taper` (default 8 retained samples);
- `--prob-phase-agreement-power` (default 1);
- `--prob-phase-min-observations` (default 2 slabs).
- `--prob-phase-band-sigma` (default 4; lossless after uint8 quantization for
  an isolated Gaussian passage).

Legacy `mean` and `max` modes remain available and unchanged. In all modes,
`crossing_prob` is a crossing-evidence raster rather than a calibrated neural
probability for a headless checkpoint.

## Topology-free overlap registration

The legacy decoder fixes every slab's phase gauge by assuming the fitted seed
midpoint is an exact integer crossing. This can duplicate or shift sheets when
the fitted spiral is locally wrong. Reconstruction can instead add:

```bash
--phase-registration overlap \
--prob-combine phase-label
```

`overlap` runs a prepass over the selected cached slabs. It finds nearby seeds
in world XYZ, samples each local phase field at neighboring seed positions,
and solves one continuous additive correction per slab from those overlap
constraints. The graph has no edges derived from winding adjacency or from a
spiral parameterization. Cached seed winding is retained only as a weak robust
gauge prior for disconnected components. The solution and diagnostics are
cached under `OUTPUT.tmp/phase_overlap_sync/offsets.npz`; a compatible rerun
reuses them unless `--phase-sync-recompute` is supplied.

`phase-label` carries the synchronized integer winding through probability
aggregation. Different integer sheets cannot be circularly averaged together,
and each proposal must pass both a distinct-slab count and Kish effective
weighted-support gate. The most important controls are:

- `--phase-sync-radius` and `--phase-sync-neighbors`: world-space graph reach;
- `--phase-sync-prior-weight`: strength of the fitted-seed gauge prior;
- `--phase-sync-huber`: rejection threshold for inconsistent overlap edges;
- `--phase-sync-max-correction`: safety limit per slab, in windings;
- `--prob-phase-min-effective-observations`: weighted support gate;
- `--prob-phase-min-weight`: minimum total anchor/edge-taper weight.

This mode does not force inferred sheets to close or follow the fitted spiral.
It cannot, however, recover a physical region through which the cached rays
never passed. Inspect the `phase_registration.synchronization` output metadata,
especially supported-node fraction and edge-residual quantiles, before treating
a full-volume reconstruction as final.

## Fast full-resolution reconstruction

Native caches deliberately overlap adjacent 128-voxel slabs heavily. Reading
and projecting every complete footprint repeats the same sheet area many times;
the phase payload, not the neural model, then becomes the bottleneck. The
optional fast path has two independent controls:

- `--slab-center-width W` retains only the central `W x W` scale-0-voxel tile
  owned by each seed. The implementation crops the native phase before
  interpolation while retaining the bracketing grid lines, so values and world
  coordinates inside the selected tile are identical to full-field
  interpolation. Choose `W` larger than the cache's seed spacing so neighboring
  tiles overlap.
- `--phase-cache-winding-stride N` reads every Nth cached anchor sheet. Increase
  `--max-level` to the resulting anchor gap and set
  `--prob-phase-max-level` half a winding beyond it so adjacent retained anchors
  both support crossings between them. This is an accuracy/speed tradeoff:
  predictions from the nearer anchor keep the largest merge weight, while the
  farther anchor supplies consensus support.

For the PHercParis4 cache made with 48-voxel seed spacing and 3-winding anchor
spacing, the measured 1-2 hour configuration is:

```bash
python infer_winding_volume.py \
  /ephemeral/sean/spiral_new/dataset/spiral_output/2026-07-27_s1_slice-7000-17500_4610-patch_track-walk-30k/checkpoint_fitted.ckpt \
  /ephemeral/sean/winding_inferences/PHercParis4/winding_fullres_fast.zarr \
  --model-ckpt /ephemeral/sean/villa/vesuvius/src/vesuvius/neural_tracing/winding_models/runs/winding_model_3d_large_9/ckpt_015000.pth \
  --reference-zarr /ephemeral/sean/winding_train_data/PHercParis4/volumes/s1_2um_ds2.zarr \
  --volume-scale 0 \
  --phase-cache /ephemeral/sean/winding_inferences/PHercParis4/winding_native_phase.zarr \
  --phase-cache-winding-stride 3 \
  --slab-center-width 72 \
  --output-downsample 1 \
  --column-upsample 4 \
  --column-step 4 \
  --max-level 9 \
  --prob-volume \
  --prob-column-step 1 \
  --prob-combine phase \
  --prob-phase-max-level 9.5 \
  --prob-volume-floor 0 \
  --batch-size 4 \
  --archive-workers 4 \
  --merge-workers 16 \
  --gpus 0,1,2,3,4,5,6
```

This still creates scale-0 `winding`, `confidence`, and `crossing_prob` arrays.
`--column-step 4` applies only to decoded points and winding voting;
`--prob-column-step 1` retains the dense scale-0 probability sampling requested
for the consensus volume. The faster settings are opt-in; omitting both new
controls preserves the original all-slab, full-footprint reconstruction.
