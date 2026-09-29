---
library_name: vesuvius
tags:
  - fiber-tracing
  - 3d
  - pytorch
  - ome-zarr
---

# S1A Fiber-Tracing Models (SD1 and SD2)

> Draft Hugging Face model card and portable inference metadata for
> [`scrollprize/lasagna-fiber`](https://huggingface.co/scrollprize/lasagna-fiber).

## Model summary

This repository contains two historical S1A single-branch 3D fiber-tracing
checkpoints. Both predict fiber direction and presence from cubic 3D CT
patches and use the same inference entry point. They are separate variants,
because they were trained at different OME-Zarr input scales and have
different weights. Select the checkpoint and input group belonging to the
same variant; do not treat the two checkpoints as interchangeable weights.

The checkpoint is a PyTorch `.pt` file and contains the model configuration
needed by the current inference entry point; no local training-data or cache
paths are required for inference.

| Variant | Suggested model id | Input group | Atlas identity |
|---|---|---:|---|
| SD2 historical S1A | `s1a-128-sd2-single-25k` | `2` | none |
| SD1 Atlas fiber model | `s1a-128-sd1-atlas-20260801084232` | `1` | `20260801084232` |

| Field | Value |
|---|---|
| Suggested model id | `s1a-128-sd2-single-25k` |
| Checkpoint file | `s1a_128_2_single_8x8_20260728_094259_best_25_9k.pt` |
| Original checkpoint path | `s1a_128_2_single_8x8_20260728_094259/snapshots/best_25_9k.pt` |
| Checkpoint step | `25,900` updates |
| SHA-256 | `2528a46e38844c06e591084f1dc1af9442ccb45a17aae9b2a68441e6028bbdd6` |
| File size | 344,225,753 bytes |
| Input channels | 1 |
| Output channels | 7 |
| Direction branches | 1 |
| U-Net | depth 5, base width 16, pixel-shuffle decoder |
| Decoder conditioning | disabled |
| Normalization | BatchNorm |
| Training precision | BF16 autocast |
| Input normalization | z-score |

The seven raw head channels are six encoded direction values (two signed
options for each of Z, Y, and X) plus one presence channel. The inference
adapter decodes these into the standard Lasagna fiber prediction products;
consumers should use the generated `.lasagna.json` manifest rather than
interpreting raw head channels themselves.

The SD1 variant is the checkpoint recorded by Atlas model `20260801084232`.
Its canonical snapshot is `best91_5k.pt`; `best.pt` in the same training run
is byte-identical, but the Atlas provenance should continue to use the
canonical `best91_5k.pt` name and its recorded SHA-256.

| SD1 Atlas field | Value |
|---|---|
| Training run | `s1a_128_1_single_8x8_20260801_084232` |
| Checkpoint file in this repository | `s1a_128_1_single_8x8_20260801_084232_best91_5k.pt` |
| Original snapshot | `s1a_128_1_single_8x8_20260801_084232/snapshots/best91_5k.pt` |
| Checkpoint step | `91,500` updates |
| SHA-256 | `f389da7914a6da34506f92204bf5441964e96599339dfe79dfc9c48b67165e17` |
| Atlas model ID | `20260801084232` |
| Atlas output channels | `presence`, `nx`, `ny` |

## Training provenance

The SD2 variant was trained on S1A NML fibers registered to the PHercParis4
2.400&nbsp;µm, 78&nbsp;keV volume, using OME-Zarr input group 2. Training patches
were `128 × 128 × 128` and the recorded learning rate was `0.01`. The SD1
variant comes from the separate `s1a_128_1_single_8x8_20260801_084232` run
and is the checkpoint already referenced by Atlas model `20260801084232`.

Each checkpoint embeds its complete historical training configuration,
including augmentation and data paths; those paths are not needed by
inference and are not part of this portable card.

The historical benchmark used Villa revision
`07451e4eb0f7e610a43deaa94d51669ded65daca`, the legacy `python-sd2` protocol,
and `metric_sd2_s1_single.json`. Results below are retained as provenance, not
as a guarantee for every volume:

| Evaluation set | err/kvx | err/m | Mean trace length |
|---|---:|---:|---:|
| `paul4` | 1.0 | 105.0 | 9.2 mm |
| `fiber1` | 0.4 | 45.8 | 19.1 mm |

These measurements used the historical Python Trace2CP evaluator and are not
directly comparable to the later native five-scroll benchmark tables.

## Recommended Hugging Face repository layout

```text
README.md                         # this model card
s1a_128_2_single_8x8_20260728_094259_best_25_9k.pt
s1a_128_1_single_8x8_20260801_084232_best91_5k.pt
inference_metadata.json
```

Suggested `inference_metadata.json`:

```json
{
  "model_family": "s1a-fiber3d-unet",
  "variants": [
    {
      "model_id": "s1a-128-sd2-single-25k",
      "checkpoint": "s1a_128_2_single_8x8_20260728_094259_best_25_9k.pt",
      "checkpoint_sha256": "2528a46e38844c06e591084f1dc1af9442ccb45a17aae9b2a68441e6028bbdd6",
      "checkpoint_step": 25900,
      "training_input_ome_scale": 2,
      "atlas_model_id": null,
      "historical_inference_config": "metric_sd2_s1_single.json"
    },
    {
      "model_id": "s1a-128-sd1-atlas-20260801084232",
      "checkpoint": "s1a_128_1_single_8x8_20260801_084232_best91_5k.pt",
      "checkpoint_sha256": "f389da7914a6da34506f92204bf5441964e96599339dfe79dfc9c48b67165e17",
      "checkpoint_step": 91500,
      "training_input_ome_scale": 1,
      "atlas_model_id": "20260801084232",
      "historical_inference_config": null,
      "input_group_for_2p4um_source": 1
    }
  ],
  "framework": "pytorch",
  "architecture": {
    "family": "fiber_trace_3d_unet",
    "input_channels": 1,
    "output_channels": 7,
    "direction_branch_count": 1,
    "conditioned_decoder_enabled": false,
    "unet_depth": 5,
    "unet_base_channels": 16,
    "decoder_upsample_mode": "pixelshuffle",
    "normalization": "batch"
  },
  "preprocessing": {
    "image_normalization": "zscore",
    "training_patch_shape_zyx": [128, 128, 128]
  },
  "historical_inference_protocol": {
    "inference_scaledown_power": 2,
    "villa_revision": "07451e4eb0f7e610a43deaa94d51669ded65daca"
  }
}
```

## Portable installation

The supported installation currently uses the Villa monorepo because the
inference stack depends on the sibling Volume Cartographer bindings:

```bash
git clone https://github.com/ScrollPrize/villa.git
cd villa/lasagna
python3 scripts/bootstrap_venv.py --venv .venv --backend cu128
source .venv/bin/activate
```

Use `--backend cu130` when appropriate, or `--backend cpu` for a smoke test.
The bootstrap installs the sibling Vesuvius and Lasagna packages, the native
`vc` reader, and the CUDA-compatible PyTorch build. Verify the environment:

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

## Download the model

```bash
MODEL_DIR="$PWD/models/s1a-fiber"
mkdir -p "$MODEL_DIR"
hf download scrollprize/lasagna-fiber \
  s1a_128_2_single_8x8_20260728_094259_best_25_9k.pt \
  s1a_128_1_single_8x8_20260801_084232_best91_5k.pt \
  --local-dir "$MODEL_DIR"
MODEL_SD2="$MODEL_DIR/s1a_128_2_single_8x8_20260728_094259_best_25_9k.pt"
MODEL_SD1="$MODEL_DIR/s1a_128_1_single_8x8_20260801_084232_best91_5k.pt"
sha256sum "$MODEL_SD2" "$MODEL_SD1"
```

The hashes must match the values recorded in the variant table and metadata.

## Obtain an input volume

The public open-data bucket is readable without AWS credentials. Download or
cache an OME-Zarr volume with `lasagna-download`, then point inference at one
of its numeric OME-Zarr arrays (`/0`, `/1`, `/2`, ...), not at the root:

```bash
lasagna-download \
  s3://vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr
```

For very large volumes, a local lazy cache containing the root `_download`
metadata is also supported; the inference command will fetch missing chunks
unless `--no-download` is supplied.

## Inference: 2.4 µm source at SD2

Here “SD2” means that the selected input array is OME-Zarr group `2` of the
2.400&nbsp;µm source, matching the historical model protocol. The output
downscale is a separate setting; `2` below preserves the historical factor-4
output reduction relative to the selected input array.

```bash
VOLUME=/data/PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr
python -m vesuvius.neural_tracing.fiber_trace_3d.infer \
  --input "$VOLUME/2" \
  --output /data/predictions/PHercParis4-s1a-sd2.lasagna.json \
  --checkpoint "$MODEL_SD2" \
  --devices all \
  --tile-size 512 --overlap 96 --border 32 \
  --inference-scaledown-power 2 \
  --download-workers 64
```

## Inference: 9 µm source at SD0

Here “SD0” means full-resolution input group `0` of a 9&nbsp;µm scan. The model
was trained on the 2.4&nbsp;µm/SD2 distribution, so this is a transfer-use case;
it is not the original training distribution. The command keeps the
historical factor-4 output reduction (`--inference-scaledown-power 2`):

```bash
lasagna-download \
  s3://vesuvius-challenge-open-data/PHerc0125/volumes/20250821151825-9.362um-1.2m-113keV-masked.zarr

VOLUME=/data/PHerc0125/20250821151825-9.362um-1.2m-113keV-masked.zarr
python -m vesuvius.neural_tracing.fiber_trace_3d.infer \
  --input "$VOLUME/0" \
  --output /data/predictions/PHerc0125-s1a-sd0.lasagna.json \
  --checkpoint "$MODEL_SD2" \
  --devices all \
  --tile-size 512 --overlap 96 --border 32 \
  --inference-scaledown-power 2 \
  --download-workers 64
```

If full-resolution **output** is required as well as full-resolution input,
replace the final value with `--inference-scaledown-power 0`. That mode is
more expensive and was not the historical benchmark protocol.

## Inference: 2.4 µm source at SD1 with the Atlas-linked checkpoint

Use the SD1 checkpoint with input group `1`. This is the variant associated
with Atlas model `20260801084232` and with the previously published large
cohort of fiber predictions:

```bash
VOLUME=/data/PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr
python -m vesuvius.neural_tracing.fiber_trace_3d.infer \
  --input "$VOLUME/1" \
  --output /data/predictions/PHercParis4-s1a-sd1-atlas.lasagna.json \
  --checkpoint "$MODEL_SD1" \
  --devices all \
  --tile-size 512 --overlap 96 --border 32 \
  --inference-scaledown-power 2 \
  --download-workers 64
```

The command shape is the same for both variants; only the checkpoint and
matching input group change. Each output's `inference.json` records the exact
checkpoint hash, so downstream Atlas registration remains unambiguous.

Each successful run writes the Lasagna prediction manifest and a portable
`inference.json` containing the checkpoint hash, selected input, effective
scales, and Villa code provenance. Keep both files together when sharing the
prediction.

## Limitations

- This is a fiber direction/presence model, not a Lasagna surface-normal
  model.
- The published benchmark covers the historical `paul4` and `fiber1` test
  sets only.
- Running on other scrolls, voxel sizes, or OME scales is supported by the
  inference code but should be treated as transfer use and validated before
  quantitative comparison.
- The checkpoint is a legacy single-direction-branch model and has no
  embedding channels.
