# Kaggle surface decoder evaluation

Run the commands below from the **dinovol repository root**, the directory containing
`.venv`, `dinovol_2`, and `evals`. All paths in `train.json` are relative to that
working directory. The scripts use the existing dinovol environment and the sibling
Vesuvius source checkout; no package installation is required in this workspace.

## Prepare data

```bash
.venv/bin/python -m evals.kaggle_surfaces.prepare \
  --source /mnt/raid_nvme/datasets/kaggle_ves \
  --output evals/kaggle_surfaces/data --workers 4
```

The supplied `images/labels` and `val/images/labels` directories define the split:
806 training cases and 106 validation cases. The script checks paired names and
rejects overlapping IDs. It does not reinterpret `splits_final.json` or resplit cases.
Each `<split>/<id>.zarr` group contains `image` and `label` ZYX arrays in Zarr v2,
using 64³ chunks and Zstd compression. The CUDA reader requires uint8 images. Original image values and label values
(`0=background`, `1=surface`, `2=ignore`) are preserved. Image padding is zero;
label padding is ignore. No image resampling is performed.

Conversion reads every voxel back through `vc_reader` and compares it to the TIFF.
It writes source fingerprints, array SHA256 hashes, occupancy blocks, and a manifest.
Completed cases are reused on rerun; changed source files or chunk settings are
rejected. Incomplete `.partial` conversions are retried. Use a different destination
for `--limit N` smoke datasets; the limit applies independently to each split.
Do not run two conversions into the same destination concurrently.

## Train

```bash
.venv/bin/python -m evals.kaggle_surfaces.train \
  --config evals/kaggle_surfaces/train.json
```

Defaults: checkpoint step 640000, teacher branch, 192³ patches, batch size 2,
50,000 AdamW steps at learning rate 1e-4, bf16, validation every 500 steps.
CLI flags override the JSON. For example, `--steps 40000`.
TorchInductor compilation is enabled by default for the combined frozen encoder and
trainable decoder forward, using static shapes and `--compile-mode default`. The
first training and validation calls include compilation overhead. Use `--no-compile`
for eager execution, or `--compile-mode max-autotune-no-cudagraphs` for additional
kernel autotuning. Compilation is in-place, so decoder checkpoint keys and optimizer
parameter identities stay compatible with eager runs. Backward compilation explicitly
uses the same non-autocast context as the training loop. Batch size 2 uses one optimizer
step per batch, without gradient accumulation.

The training path requires CUDA in both PyTorch and `vc_reader`; it fails explicitly
when unavailable. `--device cpu --precision float32` is for small smoke tests.

The complete 63,917,696-parameter CNN/transformer encoder is frozen, stays in eval
mode, and executes under `no_grad`. The checkpoint is loaded strictly. Its normalized
480-channel transformer tokens feed the bottleneck, with 128-, 64-, and 32-channel
CNN skips at quarter, half, and full resolution. The actual upstream Vesuvius
`Decoder` uses transposed convolutions, two `BasicBlockD` residual blocks per stage,
InstanceNorm3d, LeakyReLU, and a two-class segmentation head. There are 3,522,914
trainable decoder parameters. The three spatial patch dimensions must be divisible
by the encoder's patch size, 8 for this checkpoint.

Image and label crops use `vc_reader` CUDA readers with `source="chunks"` and bounded
reader caches. Paired native flips and 90-degree rotations preserve label alignment.
Brightness, contrast, gamma, and noise affect images only. Normalization also uses
`vc_reader`: the default `robust` matches this checkpoint's pretraining default;
`--normalization zscore` selects the Vesuvius surface configuration's normalization.
Training mixes uniform crops with foreground/valid occupancy anchors, rejects
nearly all-ignore crops, and uses an explicitly seeded RNG.

The loss is Vesuvius's **unmodified** `DC_SkelREC_and_CE_loss` (`MedialSurfaceRecall`):
Dice + CE + medial recall with weights 1/1/1, foreground-only Dice/recall, smooth
1e-5, and ignore label 2. Negative combined loss values are normal. `vc_reader`'s
`MedialSurface` generates the targets **after spatial augmentation**, using slice-wise
thinning, two dilations, closing, and restriction to foreground. Native thinning
is not bit-identical to Vesuvius's skimage target generator: a synthetic thick-strip
comparison differed at 42 of 4096 voxels. This evaluation intentionally uses the
requested native `vc_reader` transform, while retaining the upstream loss itself.

`--vesuvius-source ../vesuvius/src` loads model/loss modules directly from that
checkout through a namespace package, bypassing Vesuvius's unrelated top-level
volume imports. Omitting it uses a normally installed `vesuvius` package. Do not
mix different Vesuvius checkouts within one Python process.

## Validation and outputs

Validation uses the same two deterministic valid-region crops per case on every
evaluation, without random augmentation. All 106 validation cases are used by
default. The metrics are **crop-based foreground Dice and medial recall**, not
full-volume inference or the official Kaggle leaderboard metric. `--val-case-limit`
and `--val-crops-per-case` control smoke-test evaluation size. Add
`--preview-all-cases` to save one three-plane soft-probability panel from the first
evaluated crop of every validation volume under `previews/all_cases/`.
`preview_manifest.json` records each panel's case ID and crop coordinates.

Under `evals/kaggle_surfaces/runs/frozen_640000/`:

- `previews/step_*.png`: Z/Y/X center planes with CT, ground truth, surface probability,
  and a continuous softmax probability overlay on CT. Cyan opacity increases with
  surface probability; outputs are not thresholded in these images. Green is target
  surface and purple is ignored ground truth. The probability overlay remains visible
  in ignored regions. Six crop panels are written at each validation by default.
- `metrics.jsonl`: training loss and validation metrics.
- `validation_crops.json`: evaluated case IDs and crop coordinates.
- `run.json`: resolved arguments, encoder architecture/provenance, and dataset signature.
- `last.pt`, `best.pt`: decoder, optimizer, scaler, RNG states, and encoder reference.
  Best is selected by validation Dice. The frozen encoder weights are not duplicated.

```bash
# Resume the same run (the original encoder checkpoint is still required).
.venv/bin/python -m evals.kaggle_surfaces.train \
  --config evals/kaggle_surfaces/train.json \
  --resume evals/kaggle_surfaces/runs/frozen_640000/last.pt --steps 40000

# Evaluate a saved decoder and regenerate validation images.
.venv/bin/python -m evals.kaggle_surfaces.train \
  --config evals/kaggle_surfaces/train.json \
  --resume evals/kaggle_surfaces/runs/frozen_640000/best.pt --eval-only
```

Resume validates the data manifest, encoder file identity, and training settings.
Keep the checkpoint and dataset available at their recorded locations. Precision,
patch size, branch, normalization, seed, and optimizer settings must match. Total
steps and validation frequency can change. CUDA kernels may still introduce
small nondeterminism despite fixed seeds.

## Checks

```bash
.venv/bin/python -m unittest evals.kaggle_surfaces.test_surfaces -v

# Small real-data pipeline check with the supplied encoder.
.venv/bin/python -m evals.kaggle_surfaces.train \
  --config evals/kaggle_surfaces/train.json \
  --device cpu --precision float32 --no-compile --patch-size 32 32 32 \
  --steps 1 --val-case-limit 1 --val-crops-per-case 2 --preview-count 2 \
  --output evals/kaggle_surfaces/runs/cpu_check
```

Tests cover TIFF/Zarr equality including partial edge chunks and ignore padding,
split leakage, deterministic validation, native target semantics, ignore gradients,
and frozen-encoder updates. A CUDA reader/transform parity test runs when a device
and CUDA-enabled reader are available. CPU smoke metrics are only execution checks;
they are not trained-model evaluation results.
