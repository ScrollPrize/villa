from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
import wandb


def to_uint8_image(image_2d):
    image_2d = np.nan_to_num(np.asarray(image_2d, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(image_2d.min())
    max_value = float(image_2d.max())
    if max_value > min_value:
        image_2d = (image_2d - min_value) / (max_value - min_value)
    else:
        image_2d = np.zeros_like(image_2d, dtype=np.float32)
    return np.clip(np.rint(image_2d * 255.0), 0, 255).astype(np.uint8)


def to_uint8_label(label_2d, ignore_mask_2d=None):
    label_2d = np.asarray(label_2d, dtype=np.float32)
    label_vis = np.zeros(label_2d.shape, dtype=np.uint8)
    if ignore_mask_2d is not None:
        ignore_mask_2d = np.asarray(ignore_mask_2d, dtype=np.float32) > 0
        label_vis[ignore_mask_2d] = 127
    label_vis[label_2d == 0] = 0
    label_vis[label_2d > 0] = 255
    if ignore_mask_2d is not None:
        label_vis[ignore_mask_2d] = 127
    return label_vis


def to_uint8_probability(probability_2d, lower_percentile=1.0, upper_percentile=99.0):
    probability_2d = np.nan_to_num(np.asarray(probability_2d, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    probability_2d = np.clip(probability_2d, 0.0, 1.0)
    lo = float(np.percentile(probability_2d, lower_percentile))
    hi = float(np.percentile(probability_2d, upper_percentile))
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        probability_2d = np.clip(probability_2d, lo, hi)
        probability_2d = (probability_2d - lo) / (hi - lo)
    return np.clip(np.rint(probability_2d * 255.0), 0, 255).astype(np.uint8)


def build_preview_montage(input_tiles, label_tiles, probability_tiles, gap_size=4):
    if not input_tiles:
        return None

    rows = []
    for input_tile, label_tile, probability_tile in zip(input_tiles, label_tiles, probability_tiles):
        column_gap = np.zeros((input_tile.shape[0], gap_size), dtype=np.uint8)
        rows.append(
            np.concatenate(
                [input_tile, column_gap, label_tile, column_gap, probability_tile],
                axis=1,
            )
        )

    row_gap = np.zeros((gap_size, rows[0].shape[1]), dtype=np.uint8)
    montage = []
    for row_idx, row in enumerate(rows):
        if row_idx > 0:
            montage.append(row_gap)
        montage.append(row)
    return np.concatenate(montage, axis=0)


def save_val_preview_tif(output_path, input_tiles, label_tiles, probability_tiles, gap_size=4):
    montage = build_preview_montage(
        input_tiles,
        label_tiles,
        probability_tiles,
        gap_size=gap_size,
    )
    if montage is None:
        return
    tifffile.imwrite(output_path, montage, compression="lzw")


@dataclass
class PreviewAccumulator:
    accelerator: object
    get_model_input: Callable[[dict], torch.Tensor]
    inputs: list[np.ndarray] = field(default_factory=list)
    labels: list[np.ndarray] = field(default_factory=list)
    probabilities: list[np.ndarray] = field(default_factory=list)

    def add_batch(self, batch, preds, targets, ignore_mask):
        input_batch = self.get_model_input(batch)
        input_mid_slice = input_batch[:, :, input_batch.shape[2] // 2]
        if input_mid_slice.shape[-2:] != preds.shape[-2:]:
            input_mid_slice = F.interpolate(
                input_mid_slice,
                size=preds.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        gathered_inputs = self.accelerator.gather_for_metrics(input_mid_slice)
        gathered_targets = self.accelerator.gather_for_metrics(targets)
        gathered_ignore_masks = self.accelerator.gather_for_metrics(ignore_mask)
        gathered_probabilities = self.accelerator.gather_for_metrics(torch.sigmoid(preds.float()))

        if not self.accelerator.is_main_process:
            return

        input_tiles = gathered_inputs[:, 0].detach().cpu().numpy()
        label_tiles = gathered_targets[:, 0].detach().cpu().numpy()
        ignore_mask_tiles = gathered_ignore_masks[:, 0].detach().cpu().numpy()
        probability_tiles = gathered_probabilities[:, 0].detach().cpu().numpy()

        for input_tile, label_tile, ignore_mask_tile, probability_tile in zip(
            input_tiles,
            label_tiles,
            ignore_mask_tiles,
            probability_tiles,
        ):
            self.inputs.append(to_uint8_image(input_tile))
            self.labels.append(to_uint8_label(label_tile, ignore_mask_tile))
            self.probabilities.append(to_uint8_probability(probability_tile))

    def montage(self):
        return build_preview_montage(self.inputs, self.labels, self.probabilities)

    def save(self, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_val_preview_tif(str(output_path), self.inputs, self.labels, self.probabilities)

    def wandb_image(self, caption):
        montage = self.montage()
        if montage is None:
            return None
        return wandb.Image(montage, caption=caption)


def build_validation_preview_log(
    *,
    step,
    train_preview,
    val_preview,
    train_preview_dir,
    val_preview_dir,
    mean_val_loss,
    mean_ema_val_loss=None,
    include_wandb_images=True,
):
    train_preview.save(Path(train_preview_dir) / f"train_preview_{step:06}.tif")
    val_preview.save(Path(val_preview_dir) / f"val_preview_{step:06}.tif")

    log_dict = {"val/loss": mean_val_loss}
    if mean_ema_val_loss is not None:
        log_dict["val/loss_ema"] = mean_ema_val_loss
    if not include_wandb_images:
        return log_dict

    train_image = train_preview.wandb_image(f"step {step} train preview")
    if train_image is not None:
        log_dict["train/preview"] = train_image

    val_image = val_preview.wandb_image(f"step {step} val preview")
    if val_image is not None:
        log_dict["val/preview"] = val_image

    return log_dict
