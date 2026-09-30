# Automated Fiber Volumes from CT

`vesuvius.afv_spline_generator` predicts papyrus fibers in a zone of a CT
volume, fits smooth polylines to them and writes an Automated Fiber Volume
(`.afv`): the read-only fiber collection that VC3D displays over the volume
(format: [`volume-cartographer/docs/fiber-collections.md`](../../volume-cartographer/docs/fiber-collections.md)).
VC3D runs it from **Create volume…** in its **Automated Fiber Volume** dock;
it can also be run on its own.

Install the model dependencies and run, here on a 512³ zone of PHerc0813:

```bash
pip install "vesuvius[models]"
vesuvius.afv_spline_generator \
  --volume https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0813/volumes/20250821151723-9.362um-1.2m-113keV-masked.zarr \
  --origin 3720 4776 6592 --size 512 512 512 \
  --coordinate-space PHerc0813/20250821151723 --voxel-size 9.362 \
  --output PHerc0813-fibers.afv
```

The volume is a local path or an `http(s)://` or `s3://` URL of a Zarr array
or OME-Zarr group (`--level` selects the array of a group, `0` by default).
The zone is given in voxels of that array by its first voxel (`--origin`) and
its size (`--size`), both in X Y Z order, and is read into memory.

VC3D opens the file only on a volume with the same coordinate identity, the
`vc-open-data-coordinate-space` tag of the volume, given by
`--coordinate-space`. Points are written in native L0 voxels: when the
volume is a downsampled version of the native scan, `--native-scale` is the
number of native voxels per voxel of the volume (`2^level`). `--voxel-size`
is the native voxel size in micrometres; it adds fiber lengths in
millimetres. An existing output is never replaced, and the output appears
only once it is complete.

| Option | Default | |
| --- | --- | --- |
| `--model` | `Qualzz20/afv_fiber_9um` | nnU-Net fiber model: local export folder or Hugging Face repository, downloaded once into the Hugging Face cache |
| `--no-mirror` | mirroring on | disables test-time mirroring: about 8× faster, visibly worse predictions |
| `--threshold` | `60` | minimum fiber probability, in percent |
| `--device` | `auto` | `cuda`, `cuda:N`, `mps` or `cpu`; `auto` prefers CUDA, then MPS |
| `--source-path` | | native source volume recorded in the file |
| `--progress` | `text` | `json` writes one event per line on stdout, as VC3D reads it |

## Method

1. **Prediction.** The zone is z-scored with its own mean and standard
   deviation (at least 10 grey levels) and predicted in windows of the model
   patch overlapping by half, blended with Gaussian weights. Test-time
   mirroring averages the network outputs for the 8 axis flips of each
   window.
   The model gives, per voxel, the probabilities of a vertical fiber, a
   horizontal fiber and an intersection.
2. **Polylines.** For each family, the probability plus half of the
   intersection probability is thresholded and skeletonized; the skeleton is
   cut at its junctions into simple chains of at least 6 voxels. Each chain is
   fitted with a cubic smoothing spline, from the smoothest to the closest
   fit, and the first fit that stays within 1.5 voxels of the skeleton and
   inside the predicted fiber is kept. A chain without such a fit is split at
   half its length and each half is tried again. Pieces of skeleton are never
   bridged: a fiber interrupted in the prediction gives several polylines,
   and fibers leaving the zone stop at its border.
3. **Output.** Polylines at least 8 voxels long, sampled about every half voxel,
   are written longest first as `V` (vertical) and `H` (horizontal) fibers.
   The `generator` metadata key records the model and its revision, the
   options, the volume and the zone.

## Model

The default model, [`Qualzz20/afv_fiber_9um`](https://huggingface.co/Qualzz20/afv_fiber_9um),
was trained on 8.64 and 9.362 µm scans; a warning is shown when the voxel
size given by `--voxel-size` and `--native-scale` is outside 8–10 µm. Any
nnU-Net v2 model with the classes `background`, `vt-fiber`, `hz-fiber` and
`intersection`, z-score normalization and no axis transposition can be used
instead.

On one RTX 3090, the 512³ zone above takes 148 s with mirroring and 52 s
without, reading included, with a peak of 3.5 GB of RAM without mirroring.
A GPU is strongly recommended: predicting on the CPU is very slow.
