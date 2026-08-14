# Ink Detection

This package provides ink-model training, flat surface-volume
inference, native tifxyz-guided 3D inference, label conversion, and aligned
21-slice input preparation.

For the complete config and CLI reference, see
[`docs/ink_detection.md`](../../../docs/ink_detection.md).

## Environment

From `villa/vesuvius`:

```bash
uv sync --extra models --extra tests
```

## Data

Each labeled segment is one directory holding its label zarrs beside its
surface volume, for example:

```text
/data/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc0139-w016/
├── pherc0139-w016_inklabels.zarr
├── pherc0139-w016_supervision_mask.zarr
├── pherc0139-w016_validation_mask.zarr
└── surface-volume.zarr
```

Runtime code is layout-agnostic: JSON supplies the segment and volume paths,
and label assets are discovered beside each segment. Sampling identifiers are
free strings and do not encode paths.

Paste volume paths in their correct local, `s3://`, or `https://` form; the
package does not rewrite URLs. The optional compressed-chunk disk cache requires
Zarr 3; uncached reads support Zarr 2.18.7 and Zarr 3. Its budget is per volume:
an over-budget volume subdirectory is swept oldest-first to 90% when opened,
then Zarr CacheStore applies a process-local LRU. Concurrent workers use atomic
LocalStore installs, but their independent accounting can cause a brief shared
cache-directory overshoot.

## Shipped configs

```text
configs/aligned21_hybrid_3d2d.json
configs/aligned21_fixed_scroll_prior.json
```

The hybrid recipe uses a 17-of-21 Z window, robust-MAD normalization, a local
3D stem feeding a 2D U-Net, and fixed per-batch scroll quotas. Its `datasets`
block lists the full 29-representation corpus; replace the `/path/to/ink_9um`
placeholder with a local copy of the ink_9um dataset and set `out_dir` before
training. The fixed-prior manifest describes the same 29 representations and
the 29/22/11/2 batch counts.

## Commands

Prepare a 2.399 µm surface volume as a 21-slice approximately 9.6 µm input:

```bash
uv run --extra models python -m \
  vesuvius.ink_detection.preprocessing.prepare_9um_isotropic_input \
  /data/raw/pherc0139-w016.ome.zarr \
  /data/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc0139-w016/surface-volume.zarr \
  --level 2
```

Convert edited label images into OME-Zarr pyramids:

```bash
uv run --extra models python -m \
  vesuvius.ink_detection.preprocessing.create_label_zarrs \
  /data/ink_9um/labels/aligned-scrollprizeorg-21slices
```

Curation commands use the same preprocessing module namespace:

```bash
uv run --extra models python -m vesuvius.ink_detection.preprocessing.validate_segments ROOT
uv run --extra models python -m vesuvius.ink_detection.preprocessing.clean_labels ROOT
uv run --extra models python -m vesuvius.ink_detection.preprocessing.merge_predictions ROOT
uv run --extra models python -m vesuvius.ink_detection.preprocessing.composite_from_zarr --input-root ROOT --method max
uv run --extra models python -m vesuvius.ink_detection.preprocessing.download_required_zarr_chunks --datasets-root DATASETS --volumes-json VOLUMES.json --output-root OUTPUT --dry-run
```

The downloader always creates an explicit Zarr-v2 store and records its source,
array schemas, exact chunk plan, and completed chunks in root attributes before
allowing resume. `--stored-grid-pad` and `--patch-finding-workers` are accepted
inert flags; they do not affect the plan. Prediction merging
retains the reference `betti`/`ema`/`640` stem filter and fails when it produces
no output. Label cleaning and composite TIFF publication are transactional.
Every long option also accepts the opposite hyphen/underscore spelling.

Train:

```bash
uv run --extra models python -m vesuvius.ink_detection.training.train \
  /path/to/aligned21_hybrid_3d2d.json
```

Train with two Accelerate workers:

```bash
uv run --extra models accelerate launch --num_processes 2 --module \
  vesuvius.ink_detection.training.train /path/to/aligned21_hybrid_3d2d.json
```

Run flat inference:

```bash
uv run --extra models python -m vesuvius.ink_detection.inference.infer \
  /data/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc0139-w016/surface-volume.zarr \
  /data/ink_9um/checkpoints/hybrid-best.pth \
  /data/predictions/pherc0139-w016.tif
```

Run native inference:

```bash
uv run --extra models python -m vesuvius.ink_detection.inference.infer_full3d_tifxyz \
  /data/segments/w035 \
  /data/ink_9um/checkpoints/native-best.pth \
  /data/predictions/w035.ome.zarr
```

The native tifxyz directory must contain `x.tif`, `y.tif`, `z.tif`, and
`volume_source.txt`. Use `--plan-only` to inspect occupied-chunk and patch
counts without creating output.

## Checkpoints and metrics

Training embeds the resolved JSON config in each checkpoint. Inference
rebuilds the model, normalization, and input geometry from that embedded config
and prefers EMA weights when present.

Training writes periodic checkpoints, append-only `validation_metrics.jsonl`,
and train/validation TIFF previews below `out_dir`. Online `val_loss` is the
smoothed objective. Use `val_bce_unsmoothed` from the JSONL file for
calibration-comparable BCE.

Flat inference writes a tiled LZW uint8 TIFF. Native inference writes a sparse
six-level uint8 OME-Zarr pyramid in scroll coordinates. Label conversion writes
65-plane, six-level OME-Zarr labels with the flat label at Z=32. The 9 µm
preparer writes one `(21,Y,X)` array tagged `level2-zmean4-21slice-v1` and
refuses to overwrite either a destination or its `.partial` path.

## Labeling loop

Train, infer, inspect the probability TIFF, extend the segment's ink and
supervision images, run `create_label_zarrs`, and retrain. Versioned label
assets use `_v<N>` before the extension; an explicit `label_version` requires
matching ink and supervision versions.

## Tests

```bash
uv run --no-sync pytest tests/ink_detection -q
```

The supported modes are `flat`, `full_3d`, and `full_3d_single_wrap`.
ResNet3D/container inference, `normal_pooled_3d`, mean-teacher, and Betti
matching are outside this package. Positive native dilation requires the
CUDA/cuCIM capability.
