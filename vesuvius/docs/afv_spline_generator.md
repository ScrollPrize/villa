# Automated Fiber Volumes from CT

`vesuvius.afv_spline_generator` predicts papyrus fibers in a zone of a CT
volume block by block, fits smooth polylines to them, stitches the blocks
together, optionally extends the fibers across gaps, cleans them and writes
an Automated Fiber Volume
(`.afv`): the read-only fiber collection that VC3D displays over the volume
(format: [`volume-cartographer/docs/fiber-collections.md`](../../volume-cartographer/docs/fiber-collections.md)).
VC3D runs it from **Generate automated fibers…** in its **Automated Fiber
Volume** dock; it can also be run on its own.

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
its size (`--size`), both in X Y Z order, of any size: it is read one block
at a time.

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
| `--mirror` | off | test-time mirroring: better predictions, about 8× slower prediction |
| `--threshold` | `60` | minimum fiber probability, in percent |
| `--device` | `auto` | `cuda`, `cuda:N`, `mps` or `cpu`; `auto` prefers CUDA, then MPS |
| `--block-size` | `512` | side of the cubes the zone is processed in, in voxels |
| `--extend` | off | join fibers across gaps inside the zone with the gap model: longer fibers, several times slower |
| `--max-join-angle` | `45` | inferred joins turning more than this many degrees are cut; `180` keeps them all |
| `--min-length` | `100` | fibers shorter than this many voxels are removed; `0` keeps them all |
| `--black-distance` | `16` | fibers passing within this many voxels of the black outside the papyrus are removed; `0` keeps them all |
| `--preview-dir` | | folder where a preview `.afv` of the fibers stitched so far is written after each block |
| `--source-path` | | native source volume recorded in the file |
| `--progress` | `text` | `json` writes one event per line on stdout, as VC3D reads it: the block plan, the stage of each block, previews and progress |

## Method

1. **Blocks.** The zone is cut into cubes of `--block-size` voxels, X
   varying fastest; the last cubes along an axis are shorter. Each block is
   read with a margin of 32 voxels, clipped to the volume, so that the
   polylines of neighbouring blocks overlap; fibers can therefore reach up
   to 32 voxels beyond the zone.
2. **Prediction.** The block is z-scored with its own mean and standard
   deviation (at least 10 grey levels) and predicted in windows of the model
   patch overlapping by half, blended with Gaussian weights. With `--mirror`,
   the network outputs for the 8 axis flips of each window are averaged. The
   network is loaded once for all blocks.
   The model gives, per voxel, the probabilities of a vertical fiber, a
   horizontal fiber and an intersection.
3. **Polylines.** For each family, the probability plus half of the
   intersection probability is thresholded and skeletonized; the skeleton is
   cut at its junctions into simple chains of at least 6 voxels. Each chain is
   fitted with a cubic smoothing spline, from the smoothest to the closest
   fit, and the first fit that stays within 1.5 voxels of the skeleton and
   inside the predicted fiber is kept. A chain without such a fit is split at
   half its length and each half is tried again.
4. **Stitching.** The polylines of each block enter one catalogue as soon as
   the block is done. The catalogue drops polylines duplicated by a
   neighbouring block, chains polylines that overlap across a block face, and
   joins aligned ends up to 30 voxels apart when their directions and lateral
   offset agree. Polylines too short for the catalogue are kept once, by the
   block that owns their middle.
5. **Extension**, with `--extend`. Each chain, longest first, is grown from
   both ends across gaps of up to 40 voxels, between fibers already found in
   the zone and its margin. Every candidate join is scored by the gap
   model below; it is kept only when its score is at least 0.9, it beats the
   next candidate by at least 0.15, the other end picks it back, it neither
   crosses another fiber of the same family nor runs back into the fiber, and
   the CT around it is inside the volume and not empty.
   Chains absorbed into a longer fiber are not grown again.
6. **Cleanup.** At every inferred join, the directions entering and leaving
   it are measured over 8 voxels on each side; a join turning more than
   `--max-join-angle` is removed and its fiber split in two. Fibers shorter
   than `--min-length` are then removed, and so are fibers passing within
   `--black-distance` of the black outside the papyrus. That black is
   found in the CT read for prediction, on cubes of 4 voxels: the cubes
   without any non-zero voxel connected to the border of the zone, and the
   cubes with a zero voxel next to them. A fiber is removed when one of its
   points is inside such a cube or within the distance of its centre, so the
   distance holds to about 3.5 voxels. Enclosed holes and isolated zero
   voxels inside the papyrus are ignored. Finally the number of points of
   each fiber is reduced, keeping every original point within 0.05 voxel
   and at least 99.9 % of the length; the ends of every join are kept.
7. **Output.** Fibers are written longest first as `V` (vertical) and `H`
   (horizontal) fibers; the point ranges bridged by joins are recorded in
   each fiber's `inferred_gap_point_ranges` annotation. The `generator`
   metadata key records the model and its revision, the options, the volume,
   the zone, the blocks and, with `--extend`, the gap model.

## Model

The default model, [`Qualzz20/afv_fiber_9um`](https://huggingface.co/Qualzz20/afv_fiber_9um),
was trained on 8.64 and 9.362 µm scans; a warning is shown when the voxel
size given by `--voxel-size` and `--native-scale` is outside 8–10 µm. Any
nnU-Net v2 model with the classes `background`, `vt-fiber`, `hz-fiber` and
`intersection`, z-score normalization and no axis transposition can be used
instead.

The gap model used by `--extend`, `extend/gap_model.pt`, ships with the package: a network of
9,409 parameters that scores a join from 48 measurements of the geometry of
the two ends (gap length, alignment and lateral offset at three scales,
curvature, and the fibers around the gap),
without CT. Its distances are converted to micrometres with the voxel size
given by `--voxel-size` and `--native-scale`, 8.64 µm when it is not given.

On one RTX 3090, the 512³ zone above gives 1,055 fibers in 1 min 21 s with
a preview, as VC3D runs it, reading over HTTPS included: 32 s of prediction,
30 s of polylines and 2 s of cleanup, with a peak of 4.3 GB of RAM. With
`--extend`, it takes 6 min 18 s, 5 min of which is extension. A GPU is
strongly recommended: predicting on the CPU is very slow.
