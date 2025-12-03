# 2D TIFF layer UNet trainer

## Design decisions

- Minimal dependencies: PyTorch, tifffile, TensorBoard, single in-repo U-Net implementation.
- Use multi-layer TIFF stacks where each layer is treated as an independent 2D sample.
- Supervision from label TIFFs with three-valued encoding:
  - `0` → target intensity 0, contributes to loss.
  - `1` → target intensity 1, contributes to loss.
  - `2` → ignored (no loss contribution).
- Train on random 256×256 crops (patches) sampled per layer to normalize varying input sizes; the patch size is configurable in [`TiffLayerDataset`](train_unet.py:11).

## Project structure

- [`train_unet.py`](train_unet.py)
  - [`TiffLayerDataset`](train_unet.py:11): iterates over all layers in each multi-layer TIFF pair from `images/` and `labels/`, returning a random square patch (default `256×256`) per sample.
  - [`UNet`](train_unet.py:94): small 2D U-Net for single-channel input and output (values in `[0, 1]`).
  - [`masked_mse_loss`](train_unet.py:143): implements the label semantics (0 → 0, 1 → 1, 2 → ignore).
  - [`train`](train_unet.py:170): basic training loop with Adam, masked MSE, and TensorBoard logging.
  - [`main`](train_unet.py:215): CLI entry point for configuring paths and hyperparameters.

Expected data layout (relative to the project root):

```text
images/
  sample_001.tif
  sample_002.tif
labels/
  sample_001_surface.tif
  sample_002_surface.tif
```

Each image TIFF in `images/` must have a matching label TIFF in `labels/` with identical number of layers and filename pattern:

- image: `sample_XYZ.tif`
- label: `sample_XYZ_surface.tif`

## Supervision utilities (gen_post_data)

- [`gen_post_data.py`](gen_post_data.py) provides:
  - A CLI tool to generate various TIFF visualizations (`vis.tif`, `vis_monotone*.tif`, `vis_labels_cc*.tif`, `vis_frac_pos*.tif`) for a single label layer, useful for debugging geometry and supervision.
  - A planned importable API that computes the same fractional-order supervision used in `vis_frac_pos.tif`, plus connected-component masks, directly from tensors.

Planned module API (for later integration into training):

- A single function (name TBD) that, given a batch of label maps as a PyTorch tensor of shape `(N, H, W)` (values in `{0, 1, 2}` with `2` = ignore), will compute:

  - `frac_pos`: float32 tensor of shape `(N, H, W)`
    - Per-pixel fractional order along the inferred chain inside each valid large CC.
    - Pixels not participating in a valid chain are set to a negative sentinel (e.g. `-1`), matching the current `frac_pos` TIFF semantics.

  - `outer_cc_idx`: integer tensor of shape `(N, H, W)`
    - Encodes the *large outer* connected components that passed the current validity checks.
    - Each such CC is eroded by 16 pixels (in the 2D plane) before being written into `outer_cc_idx`.
    - Outside these eroded outer CCs, `outer_cc_idx` is `0`.
    - Inside them, `outer_cc_idx` takes values `1..K`, where indices are strictly increasing with no gaps: if a candidate CC is skipped by the geometric checks, its index is not used and the next valid CC reuses the next consecutive index.

  - `max_cc_idx`: integer scalar
    - The maximum CC index used across the entire batch, i.e. `max(outer_cc_idx)` over all `N` samples.
    - This allows downstream code to reason about the global number of outer CCs present in the batch.

The existing CLI behavior of [`gen_post_data.py`](gen_post_data.py) (reading a single TIFF, computing all intermediate fields, and writing visualization TIFFs next to the input) will be preserved by calling this function internally when the module is executed as a script.

## Dependencies

- `torch`
- `tifffile`
- `tensorboard` (via `torch.utils.tensorboard`)

## Running training

Example command:

```bash
python train_unet.py \
  --images-dir images \
  --labels-dir labels \
  --log-dir runs/unet \
  --run-name unet_baseline
```

Logs and checkpoints will be written into a timestamped subdirectory of `--log-dir`, for example:

```text
runs/unet/20251124_121207_unet_baseline/
```