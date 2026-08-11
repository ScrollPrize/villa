# Training configs

An ink-detection recipe for the aligned 21-slice surface-volume corpus
(2.399 µm public renders Z-mean-pooled 4× plus native ~9.4 µm renders of the
same segments, all layer-aligned):

| Config | Model | Input |
|---|---|---|
| `aligned21_hybrid_3d2d.json` | local 3D stem → 2D U-Net (`vesuvius_unet_3d_stem_2d`) | 17-of-21 Z window, jittered ±2 during training |

`aligned21_fixed_scroll_prior.json` is the batch-sampling contract used by the
hybrid recipe: per-batch scroll quotas (29/22/11/2 of 64 for PHerc0139 /
Scroll 1667 / Paris4 / PHerc0814) and the full list of 29 training
representations with the `sampling_*` keys each `datasets` entry must declare.

## Running

```
uv run python -m koine_machines.training.train <config.json>
```

Before running, set `out_dir` and expand the `datasets` block to your data:
one entry per source family, each listing its segment directories,
`surface_volume_paths` (OME-Zarr), and the three `sampling_*` key maps. Labels
(`*_inklabels.zarr`, `*_supervision_mask.zarr`, optional
`*_validation_mask.zarr`) are discovered next to each segment. The patch index
is built automatically under `out_dir` on first run.

Checkpoints save every 5,000 steps plus a rolling
`best_val_balanced_accuracy.pth`; note that the online validation loss uses
the configured (possibly smoothed) objective — `val_bce_unsmoothed` in
`validation_metrics.jsonl` is the calibration-comparable number.

## Inference

```
uv run python -m koine_machines.inference.infer \
  <input.zarr> <checkpoint.pth> <output.tif> \
  --overlap 0.5 --blend-mode hann
```

Checkpoints embed their training config; inference rebuilds the model and its
normalization contract from the checkpoint automatically. The inference CLI
defaults to 50% overlap with Hann blending; the flags above state that contract
explicitly for reproducible commands.

The models expect ~9 µm isotropic surface volumes. Native ~9 µm renders work
directly, local or by URL:

```
uv run python -m koine_machines.inference.infer \
  https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/segments/20260112000000-w043_2026011217/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr \
  <checkpoint.pth> w043.tif \
  --overlap 0.5 --blend-mode hann
```

2.4 µm surface volumes must first be pooled to the ~9.6 µm isotropic 21-slice
representation the models were trained on (XY pyramid level 2, 4× Z mean
pooling):

```
uv run python scripts/prepare_9um_isotropic_input.py <surface-volume-2p4um.zarr> <pooled.zarr>
```
