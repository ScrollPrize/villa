---
title: "Tutorial: Ink Detection"
sidebar_label: "Ink Detection"
---

<head>
  <html data-theme="dark" />

  <meta
    name="description"
    content="Vesuvius Challenge ink detection tutorial: train a machine learning model to find carbon ink in Herculaneum scroll segments, from dataset to training to inference."
  />

  <meta property="og:type" content="website" />
  <meta property="og:url" content="https://scrollprize.org" />
  <meta property="og:title" content="Vesuvius Challenge" />
  <meta
    property="og:description"
    content="Vesuvius Challenge ink detection tutorial: train a machine learning model to find carbon ink in Herculaneum scroll segments, from dataset to training to inference."
  />
  <meta
    property="og:image"
    content="https://scrollprize.org/img/social/opengraph.jpg"
  />

  <meta property="twitter:card" content="summary_large_image" />
  <meta property="twitter:url" content="https://scrollprize.org" />
  <meta property="twitter:title" content="Vesuvius Challenge" />
  <meta
    property="twitter:description"
    content="Vesuvius Challenge ink detection tutorial: train a machine learning model to find carbon ink in Herculaneum scroll segments, from dataset to training to inference."
  />
  <meta
    property="twitter:image"
    content="https://scrollprize.org/img/social/opengraph.jpg"
  />
</head>

import { TutorialsTop } from '@site/src/components/TutorialsTop';

<TutorialsTop highlightId={5} />

Ink detection is the last step of the pipeline: taking the flattened surface of a papyrus sheet ([segmented](/segmentation) from the 3D X-ray scan) and identifying where the ink is, so that the text can be read.

This is where one of the core difficulties of the Herculaneum Papyri comes in: the carbon ink and the carbonized papyrus have very similar densities, so the ink is mostly invisible to the naked eye in the scans. But the data is there — machine learning models can detect it, and humans can sometimes see subtle textural patterns directly:

<figure className="">
  <img src="/img/tutorials/ink2-alpha.webp" />
</figure>

<figure className="">
  <img src="/img/tutorials/ink1-alpha.webp" />

  <figcaption className="mt-0">Ink visible in 3D surface volumes (left: 3D volume slice; right: infrared photo), found by Stephen Parsons</figcaption>
</figure>

In the electron microscope images below (from [From invisibility to readability: Recovering the ink of Herculaneum](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0215775)), you can clearly see the difference between the inked and non-inked regions. Machine learning models appear to learn some of these features from the 3D X-ray scans:

<figure>
  <a href="/img/tutorials/sem.webp" target="_blank"><img src="/img/tutorials/sem-alpha.webp"  className="w-[100%]"/></a>
  <figcaption className="mt-0">Electron microscope pictures from the top (A and B) and the side (C) <a href="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0215775">(source)</a></figcaption>
</figure>

### How ink detection works

An ink detection model is not reading letters — it is doing signal recovery. The model looks at a small local patch of the surface volume (the stack of slices sampled around the papyrus surface) and predicts, for each pixel, the probability that there is ink at that location. Stitching these predictions together produces an image of the segment where the writing becomes visible to a human reader.

We train the model by picking a pixel in a binary label image, sampling a subvolume around the same coordinates from the surface volume, and backpropagating the known label to update the model weights:

<figure>
  <video autoPlay playsInline loop muted className="w-[100%] " poster="/img/tutorials/ink-training-anim3-dark.webp">
    <source src="/img/tutorials/ink-training-anim3-dark.webm" type="video/webm"/>
  </video>
</figure>

We can then use the model to predict what a label image would have looked like, on data it has never seen:

<figure>
  <video autoPlay playsInline loop muted className="w-[100%]" poster="/img/tutorials/ink-detection-anim3-dark.webp">
    <source src="/img/tutorials/ink-detection-anim3-dark.webm" type="video/webm"/>
  </video>
</figure>

Where do the labels come from? The first ink labels came from detached fragments, where the exposed writing can be photographed in infrared and aligned with the surface volume. For the intact scrolls, labels are made **iteratively**: an existing model is run on a scroll segment, a human inspects the predictions, labels the regions where letter strokes are clearly visible, and the model is retrained on the enlarged dataset. Repeating this loop is how ink detection has improved from isolated letters to entire scrolls.

This process recently achieved the complete virtual unwrapping and reading of PHerc. 1667 — the first Herculaneum scroll to be fully digitally unrolled and read without physical opening. The methods, including the labeling and validation methodology this tutorial is based on, are described in detail in [the paper](https://arxiv.org/abs/2606.29085).

Because the labels come from model predictions, the process is designed to avoid reinforcing the model's own errors:

* The model only sees small local patches — smaller than a full letter — so it cannot learn to "draw" plausible letterforms.
* Labeling is conservative: only regions where strokes are clearly and repeatably visible get labeled.
* **Validation regions** are held out and never labeled from predictions, so you can measure whether the model generalizes.
* Final readings are always reviewed by papyrologists — machine output is never treated as a substitute for reading.

Now let's train a model. The rest of this tutorial is hands-on: you will set up the training pipeline, download a labeled dataset, train an ink detection model, and run inference on a scroll segment. It is written for Linux (Windows users are advised to use WSL2) and assumes an NVIDIA GPU with a working CUDA installation.

### The dataset

The tutorial uses the [`ink-labels` dataset](/data_datasets#ink-labels-2026-07), which lives in the [`scrollprize/datasets` storage bucket](https://huggingface.co/buckets/scrollprize/datasets/tree/ink) on Hugging Face, organized by scroll. The full dataset is hundreds of GB, so the whole tutorial runs end-to-end on **one segment** of PHerc. Paris 4 (Scroll 1) — about 25 GB:

```bash
uvx --from huggingface_hub hf buckets sync \
  hf://buckets/scrollprize/datasets/ink/phercparis4/w00_20231016151002 \
  ./ink-dataset/phercparis4/w00_20231016151002
```

`hf buckets sync` works like `rsync`: re-running it resumes interrupted downloads and only transfers what changed. If you hit rate limits, create a free account, generate a read token under **Settings → Access Tokens**, and either run `uvx --from huggingface_hub hf auth login` once or set `HF_TOKEN=hf_...` in your environment.

Each segment is a folder in the layout the training pipeline expects, containing the surface geometry (`.tifxyz`), the surface volume, and the labels:

```
ink-dataset/phercparis4/
└── w00_20231016151002/
    ├── x.tif                                    # surface geometry: 3D coordinates
    ├── y.tif                                    #   of every surface pixel
    ├── z.tif
    ├── meta.json
    ├── w00_20231016151002.zarr                  # surface volume (image data)
    ├── w00_20231016151002_inklabels.zarr        # binary ink labels
    ├── w00_20231016151002_inklabels.tif         #   (and the editable TIFF original)
    ├── w00_20231016151002_supervision_mask.zarr # where the labels are trustworthy
    └── w00_20231016151002_supervision_mask.tif
```

The label files work together, and understanding them is the key to the whole pipeline:

* **Ink labels** — a binary image aligned with the segment: white where there is ink, black where there is not.
* **Supervision mask** — marks the regions where the labels can be trusted. Only pixels inside the supervision mask contribute to the training loss: white pixels there are positive (ink) examples, black pixels are negative (no ink) examples. Everything outside the mask is ignored, so unlabeled or ambiguous areas don't teach the model anything wrong.
* **Validation mask** — some segments also have a `<segment>_validation_mask.zarr`: a held-out region, labeled the same way as the rest, but excluded from training and used only to measure the model's accuracy. A segment without one (like the tutorial segment) still trains — you just get no validation metrics for it.

Here is what that looks like on a crop of the tutorial segment. First, the ink labels: strokes that a human labeler could clearly and repeatably see, painted on top of the surface volume:

<figure>
  <a href="/img/tutorials/ink-labels-overlay-w00.webp" target="_blank"><img src="/img/tutorials/ink-labels-overlay-w00.webp" /></a>
  <figcaption className="mt-0">A crop of the tutorial segment's surface volume with its ink labels overlaid in red</figcaption>
</figure>

The supervision mask covers those strokes *plus* the clean papyrus around them — the background pixels are the negative examples, and they matter just as much as the ink:

<figure>
  <a href="/img/tutorials/ink-supervision-overlay-w00.webp" target="_blank"><img src="/img/tutorials/ink-supervision-overlay-w00.webp" /></a>
  <figcaption className="mt-0">The supervision mask (green) marks where labels are trustworthy: both the ink strokes (red) and the unlabeled background inside the green region are used for training</figcaption>
</figure>

:::info
The filename prefixes must exactly match the segment folder name: a segment folder named `w00_20231016151002` must contain `w00_20231016151002_inklabels.zarr`, `w00_20231016151002_supervision_mask.zarr`, etc. The pipeline discovers segments and their labels by these names.
:::

### Setting up the pipeline

The ink detection pipeline lives in the [villa repository](https://github.com/ScrollPrize/villa), under `ink-detection/`. It uses [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage its Python environment, so there is nothing to install system-wide.

```bash
git clone https://github.com/ScrollPrize/villa.git
cd villa/ink-detection
git checkout merge-ink-pipelines
uv sync
```

:::info
The pipeline described here currently lives on the `merge-ink-pipelines` branch: run `git checkout merge-ink-pipelines` after cloning.
:::

`uv sync` creates a virtual environment and installs the exact locked dependencies (PyTorch, zarr, and friends). Verify that PyTorch sees your GPU:

```bash
uv run python -c "import torch; print(torch.__version__, '| cuda:', torch.cuda.is_available())"
```

:::info
If you later enable native-3D label dilation (`full_3d.label_dilation_distance` or `full_3d.supervision_dilation_distance`), install cuCIM/CuPy too. Pick the build matching your CUDA major version: `uv pip install --extra-index-url https://pypi.nvidia.com "cucim-cu12==26.6.0"` for CUDA 12.x, or the corresponding `cucim-cu13` build for CUDA 13.x.
:::

### Training

Training runs are configured with a single JSON file. Create `configs/ink_tutorial.json` (the `configs` folder doesn't exist yet — create it too), pointing `segments_path` at the folder containing your downloaded segments:

```json
{
  "out_dir": "runs/ink_tutorial",
  "seed": 42,

  "mode": "flat",
  "model_type": "vesuvius_unet",
  "in_channels": 1,
  "model_config": { "autoconfigure": true, "z_projection_mode": "max" },
  "targets": { "ink": { "out_channels": 1, "activation": "none", "z_projection_mode": "max" } },

  "patch_size": [64, 256, 256],
  "patch_overlap": 0.5,
  "patch_min_labeled_coverage": 0.05,

  "batch_size": 2,
  "num_iterations": 20000,
  "learning_rate": 0.01,
  "mixed_precision": "fp16",
  "dataloader_workers": 4,

  "val_every": 500,
  "save_every": 1000,

  "datasets": [
    {
      "segments_path": "/path/to/ink-dataset/phercparis4",
      "volume_scale": "0"
    }
  ]
}
```

The important options:

* `mode: "flat"` trains directly on the pre-rendered surface volume zarrs — this is the standard **2.5D** setup: the model takes a 3D patch of the surface volume as input and predicts a 2D ink image as output. Nothing is rendered on the fly. (The pipeline also has native 3D modes — `full_3d`, `full_3d_single_wrap` — which instead sample patches on the fly from the original scroll volume using the `.tifxyz` coordinates; they get [their own section](#native-3d-training-and-inference) below.)
* `z_projection_mode: "max"` is what makes it 2.5D — the network processes the patch in 3D, then the ink head collapses the depth axis with a max-projection to produce the 2D prediction. (`mean`, `logsumexp`, and `learned_mlp` are alternative projections to experiment with.)
* `patch_size` is the `[z, y, x]` size of the patches sampled around the surface: 64 slices deep, 256×256 pixels across. Each dimension must be divisible by the network's pooling factors — the trainer prints the required factors and adjusts or complains if they don't match.
* `patch_overlap: 0.5` means training patches are sampled with a half-patch stride across each segment.
* `patch_min_labeled_coverage: 0.05` skips training patches whose ink labels cover less than 5% of the patch, so training focuses on labeled regions.
* `val_every` controls how often validation metrics are computed on the validation-mask regions, and `save_every` how often checkpoints are written.

Then start training:

```bash
uv run python -m koine_machines.training.train configs/ink_tutorial.json
```

The trainer discovers your segments, finds all training patches inside the supervision masks (excluding the validation regions), and starts training. Patch discovery can take a while on large datasets; the result is cached as a JSON file in `out_dir`, keyed by patch size, overlap, and label version, so re-runs with the same settings skip it. While it runs you will see the loss printed to the console, and in `runs/ink_tutorial/` you will find:

* `ckpt_001000.pth`, `ckpt_002000.pth`, ... — checkpoints, saved every `save_every` iterations.
* `train_previews/` (and `val_previews/`, when there is a validation set) — periodic image previews of the model's predictions next to the labels. Watching the previews sharpen from noise into letter strokes is the most satisfying part of the process.

Note that `segments_path` points at the *folder of segments*, not a single segment — the trainer picks up every valid segment it finds there, so the same config keeps working as you add more. One segment is enough for a real first model.

If your dataset includes segments with validation masks, the model is also evaluated on those held-out regions at each validation step, reporting balanced accuracy — how well it detects ink in areas it was never trained on. If training loss keeps dropping while validation accuracy stalls, the model is starting to overfit your labels. (The tutorial segment has no validation mask, so this first run reports training loss only.) You can stop training at any time with `ctrl+c` and use the most recently saved checkpoint.

:::tip
If you run out of GPU memory, reduce `batch_size` to 1, or reduce the `patch_size` to `[64, 128, 128]`. For multi-GPU training, launch through Accelerate instead: `uv run accelerate launch --num_processes 2 --module koine_machines.training.train configs/ink_tutorial.json`.
:::

:::tip
To log metrics and previews to Weights & Biases, add `"wandb_project": "ink-detection"` and `"wandb_entity": "your-username"` to the config and run `uv run wandb login` once.
:::

### Inference

To run your trained model on the same segment and produce an ink prediction image:

```bash
uv run python -m koine_machines.inference.infer \
  /path/to/ink-dataset/phercparis4/w00_20231016151002/w00_20231016151002.zarr \
  runs/ink_tutorial/ckpt_019000.pth \
  predictions/w00_20231016151002.tif \
  --batch-size 4
```

The three positional arguments are the segment's surface volume, the checkpoint (here the last one written by the 20,000-iteration run above), and the output path. The model slides across the whole segment in overlapping windows, blends the overlapping predictions, and writes a grayscale TIFF where each pixel's brightness (0–255) is the predicted probability of ink. Expect this to take on the order of an hour on a single GPU for a full segment. Open the result in any image viewer — if all went well, you'll see letters, including outside the regions you had labels for.

:::tip
For a faster first look, pass `--mask-path region.tif` — a grayscale TIFF the size of the segment where nonzero pixels mark the region to predict — to limit inference to an area of interest.
:::

Useful options:

* `--gpus 0,1` — run on multiple GPUs.
* `--tta-mirror` — average predictions over mirrored versions of each patch (slower, slightly better).
* `--layer-start` / `--layer-end` — restrict which depth layers of the surface volume are used.

Here is the result on the tutorial segment — the model's prediction in white, with the ink labels it was trained on overlaid in red:

<figure>
  <a href="/img/tutorials/ink-prediction-w00.webp" target="_blank"><img src="/img/tutorials/ink-prediction-w00.webp" /></a>
  <figcaption className="mt-0">The trained model's ink prediction for the tutorial segment. Red: the handful of letters it was trained on. Everything else it found on its own.</figcaption>
</figure>

### Native 3D: training and inference

Everything above is the 2.5D path: pre-rendered surface volume zarr in, 2D ink image out. The pipeline can also work **natively in 3D**, skipping the rendered surface volume entirely: for every training patch it uses the `.tifxyz` coordinates to find where the segment passes through the original scroll volume, samples a 3D crop there on the fly, and projects the 2D labels into the crop around the surface. The model then predicts ink directly in scroll space.

Two native 3D modes exist. `full_3d` trains on the raw crops; `full_3d_single_wrap` additionally feeds the model a second input channel marking which voxels belong to this segment's own wrap of papyrus, so the model isn't confused where neighboring wraps pass through the same crop — this is the mode to prefer.

#### Native 3D training

Create `configs/ink_full3d.json`. It is the same shape as the 2.5D config with three changes: the mode, no z-projection (the prediction stays 3D), and the dataset entry gains a `volume_path` pointing at the original scroll volume:

```json
{
  "out_dir": "runs/ink_full3d",
  "seed": 42,

  "mode": "full_3d_single_wrap",
  "model_type": "vesuvius_unet",
  "in_channels": 1,
  "model_config": { "autoconfigure": true },
  "targets": { "ink": { "out_channels": 1, "activation": "none" } },

  "patch_size": [80, 128, 128],
  "patch_overlap": 0.5,
  "patch_min_labeled_coverage": 0.05,

  "batch_size": 1,
  "num_iterations": 20000,
  "learning_rate": 0.01,
  "mixed_precision": "fp16",
  "dataloader_workers": 1,

  "val_every": 500,
  "save_every": 1000,

  "datasets": [
    {
      "segments_path": "/path/to/ink-dataset/phercparis4",
      "volume_path": "s3://vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/",
      "volume_scale": "0"
    }
  ]
}
```

* `volume_path` is where the 3D crops come from. The public `vesuvius-challenge-open-data` S3 bucket is read anonymously — no AWS account needed — but training streams many small random reads, so if you have local disk to spare, downloading the volume (or just the chunks your segments touch, with `koine_machines.preprocessing.download_required_zarr_chunks`) and pointing `volume_path` at the local copy is considerably faster.
* Native 3D patches are much heavier than 2.5D surface-volume patches. The `[80, 128, 128]` patch size and batch size 1 are conservative first-run defaults; increase them only after you have confirmed your GPU has enough memory and I/O headroom.
* `in_channels` is set automatically in the native 3D modes (2 for `full_3d_single_wrap`: image + wrap mask), so you don't need to change it.
* How far above and below the surface the 2D labels are projected is controlled by an optional `"full_3d": { "projection_half_thickness": ... }` block (default 1 voxel).

Training starts the same way:

```bash
uv run python -m koine_machines.training.train configs/ink_full3d.json
```

Expect a native 3D run to take several times longer than the flat run — every batch is sampled from the full-resolution scroll volume instead of a compact pre-rendered zarr.

#### Native 3D inference

Native 3D checkpoints use a different inference script, `koine_machines.inference.infer_full3d_tifxyz`, which samples patches the same way and writes a sparse 3D OME-Zarr prediction volume instead of a 2D image.

For inference the segment folder must contain a `volume_source.txt` — a single line with the path or URL of the original scroll volume:

```bash
echo "s3://vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/" \
  > /path/to/ink-dataset/phercparis4/w00_20231016151002/volume_source.txt
```

Then, with your native-3D checkpoint:

```bash
uv run python -m koine_machines.inference.infer_full3d_tifxyz \
  /path/to/ink-dataset/phercparis4/w00_20231016151002 \
  runs/ink_full3d/ckpt_019000.pth \
  predictions/w00_20231016151002_ink.ome.zarr \
  --write-region occupied --chunk-halo 0 \
  --batch-size 1 --overwrite
```

`--write-region occupied --chunk-halo 0` restricts the output to just the chunks that actually contain surface points, which is much faster for a first run than the default (which also writes a halo of neighboring chunks). Even so, a single segment can plan hundreds of thousands of native-3D patches — add `--plan-only` to preview the chunk/patch plan first, and only launch the full command when the printed patch count fits your compute budget. For `full_3d_single_wrap` checkpoints, the script reconstructs the surface-mask input channel from the `.tifxyz` geometry automatically.

### Scaling up: the full dataset

Everything above ran on one segment; scaling up is mostly a matter of downloading more. Sync a whole scroll (or several) into the same folder:

```bash
uvx --from huggingface_hub hf buckets sync hf://buckets/scrollprize/datasets/ink/phercparis4 ./ink-dataset/phercparis4
```

The training config doesn't change — `segments_path` already points at the folder, and the trainer picks up every segment in it on the next run. More segments means more diverse training data, which is the single most reliable way to improve the model.

For inference across many segments, use folder mode — it runs the checkpoint on every segment in the folder and writes each prediction into a `preds/` directory inside that segment:

```bash
uv run python -m koine_machines.inference.infer \
  --folder /path/to/ink-dataset/phercparis4 \
  --checkpoint-path runs/ink_tutorial/ckpt_019000.pth \
  --batch-size 4
```

### Improving the model: iterative labeling

A first model trained on a small dataset will reveal some letters clearly, others faintly, and miss some entirely. The way to make it better is the same loop that scaled ink detection to entire scrolls:

1. **Run inference** on your training segments (and new, unlabeled ones).
2. **Inspect the predictions.** Look for regions where letter strokes are clearly and unambiguously visible.
3. **Extend the labels.** In those regions, paint the visible strokes white in the ink label image, and extend the supervision mask to cover the region — both the strokes *and* the clean background around them, since the background pixels are the negative examples the model learns from.
4. **Retrain** on the enlarged labels, starting fresh or from your last checkpoint (add `"checkpoint": "runs/ink_tutorial/ckpt_019000.pth"` and `"weights_only": true` to the config).
5. **Repeat** until the model stops improving.

Labels are ordinary image files, so you can edit them in any image editor that handles large images (e.g. GIMP or Photoshop). If you edit or create labels as TIFF/PNG files, convert them to the `.zarr` format the trainer expects with:

```bash
uv run python -m koine_machines.preprocessing.create_label_zarrs /path/to/ink-dataset/phercparis4
```

### What's next

With segmentation and ink detection you now have the complete pipeline: from a 3D X-ray scan of an intact scroll to readable text. This exact loop — better segments, more careful labels, retrained models — is what produced the [complete reading of PHerc. 1667](https://arxiv.org/abs/2606.29085), and there are hundreds of scrolls to go.

Join the [Discord](https://discord.gg/V4fJhvtaQn) to see what the community is working on, check the open [prizes](/prizes), and help us read the rest of the library.
