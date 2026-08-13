# fiber_5class — 4-class fiber/ink self-distillation trainer

Trains a 3D UNet for **4-class semantic segmentation** on a CT volume, with no
ground-truth labels. Pseudo-labels are generated on the fly (entirely on GPU)
from two frozen teacher UNets plus a GPU watershed, and the student is trained
to reproduce them (self-distillation).

Classes: `0` background · `1` vertical fiber · `2` horizontal/angular fiber · `3` ink.

## Pseudo-label pipeline (per crop, GPU-resident)

1. `fiber_prob = sigmoid(fiber_teacher(image))` — frozen fiber UNet (FG channel).
2. `ink_prob   = sigmoid(ink_teacher(image))` — frozen ink UNet.
3. `fiber_mask = fiber_prob > fiber_thr`.
4. Watershed-from-minima (`cuws`) on the distance transform of `fiber_mask`
   (`ws_image_mode="distance"`, `ws_h_merge`, `ws_min_voxels`) → fiber instances.
5. Per-instance PCA on ZYX voxel coords: `|principal_axis · ẑ| > pca_cos_threshold`
   → class 1 (vertical), else class 2 (horizontal/angular). Sub-`ws_min_voxels`
   instances default to class 2.
6. **Ink overrides fibers:** `label[ink_prob > ink_thr] = 3`.
7. **Dark-voxel guard (final):** `label[raw < dark_voxel_thr] = 0`.

Augmentations (flips, 90° rotations, intensity jitter) are applied *after* label
generation; spatial transforms are applied jointly to image and label, and a
90° rotation in a plane containing Z swaps the vertical/horizontal classes.

## Files

| file | role |
|------|------|
| `train.py` | DDP trainer (CE + multiclass Dice, EMA, wandb, debug PNGs) |
| `label_generator.py` | `FiveClassLabelGenerator` (teachers + cuws + PCA + overrides) |
| `model.py` | `build_fiber_unet` — vesuvius `NetworkFromConfig` UNet |
| `dataset.py` | `RandomFiberCropDataset` — random 256³ crops, rejection-sampled |
| `visualization.py` | fixed categorical palette + debug figure |
| `inspect_labels.py` | render pseudo-label previews without training |
| `launch_ddp8.sh` | tmux + `torchrun` launcher |
| `configs/train.example.json` | config template (copy to `train.json`, fill `<...>`) |

## Environment

Runs in the `vesuvius` uv project. The CUDA stack must match the host driver
(CUDA 12.x here), which needs a one-time manual install — the resolver's
defaults can pull a mismatched CUDA build:

```bash
cd <repo>/vesuvius
uv sync --extra models
uv pip install --force-reinstall cupy-cuda12x cucim-cu12   # match the host CUDA
uv pip install cuws --no-deps                              # GPU watershed-from-minima
uv pip install glasbey                                     # categorical palette
```

Launch with `uv run --no-sync …` so these manual installs aren't reverted
(`launch_ddp8.sh` already does this).

## Usage

```bash
cp configs/train.example.json configs/train.json
# edit configs/train.json: set out_dir, volume_url, fiber_teacher, ink_teacher

# preview the pseudo-labels first (writes PNGs you can eyeball):
cd <repo>/vesuvius && uv run --no-sync python <path>/inspect_labels.py \
    --config <path>/configs/train.json --n 6 --out ./previews

# full 8-GPU run:
bash launch_ddp8.sh configs/train.json
```

`out_dir`, `volume_url`, and both teacher checkpoint paths are deployment-specific
and are **not** committed — fill them into your local `configs/train.json`
(git-ignored).
