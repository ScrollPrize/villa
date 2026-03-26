from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
import wandb

from koine_machines.training.normal_pooling import local_points_zyx_to_normalized_grid


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


def _pad_image_bottom_right(image_2d, *, target_height=None, target_width=None):
    image_2d = np.asarray(image_2d, dtype=np.uint8)
    height, width = image_2d.shape
    if target_height is None:
        target_height = height
    if target_width is None:
        target_width = width
    pad_h = max(0, int(target_height) - height)
    pad_w = max(0, int(target_width) - width)
    if pad_h == 0 and pad_w == 0:
        return image_2d
    return np.pad(image_2d, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)


def stack_preview_tiles(tiles, gap_size=4):
    if not tiles:
        return None

    max_width = max(int(tile.shape[1]) for tile in tiles)
    stacked_tiles = []
    row_gap = np.zeros((gap_size, max_width), dtype=np.uint8)
    for tile_idx, tile in enumerate(tiles):
        padded_tile = _pad_image_bottom_right(tile, target_width=max_width)
        if tile_idx > 0:
            stacked_tiles.append(row_gap)
        stacked_tiles.append(padded_tile)
    return np.concatenate(stacked_tiles, axis=0)


def build_panel_grid(rows, gap_size=4):
    row_tiles = []
    for row in rows:
        if not row:
            continue
        row_height = max(int(tile.shape[0]) for tile in row)
        padded_row_tiles = []
        for tile_idx, tile in enumerate(row):
            padded_row_tiles.append(_pad_image_bottom_right(tile, target_height=row_height))
            if tile_idx + 1 < len(row):
                padded_row_tiles.append(np.zeros((row_height, gap_size), dtype=np.uint8))
        row_tiles.append(np.concatenate(padded_row_tiles, axis=1))
    return stack_preview_tiles(row_tiles, gap_size=gap_size)


def build_preview_montage(input_tiles, label_tiles, probability_tiles, gap_size=4):
    if not input_tiles:
        return None

    sample_tiles = [
        build_panel_grid([[input_tile, label_tile, probability_tile]], gap_size=gap_size)
        for input_tile, label_tile, probability_tile in zip(input_tiles, label_tiles, probability_tiles)
    ]
    return stack_preview_tiles(sample_tiles, gap_size=gap_size)


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
    sample_tiles: list[np.ndarray] = field(default_factory=list)
    gap_size: int = 4

    def _add_standard_batch(self, batch, preds, targets, ignore_mask):
        input_batch = self.get_model_input(batch)
        input_mid_slice = input_batch[:, :1, input_batch.shape[2] // 2]
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
            sample_tile = build_panel_grid(
                [
                    [
                        to_uint8_image(input_tile),
                        to_uint8_label(label_tile, ignore_mask_tile),
                        to_uint8_probability(probability_tile),
                    ]
                ],
                gap_size=self.gap_size,
            )
            if sample_tile is not None:
                self.sample_tiles.append(sample_tile)

    def _sample_flat_crop(self, image_batch, batch):
        flat_points_local_zyx = batch["flat_points_local_zyx"].to(
            device=image_batch.device,
            dtype=image_batch.dtype,
        )
        flat_grid = local_points_zyx_to_normalized_grid(
            flat_points_local_zyx,
            image_batch.shape[-3:],
            align_corners=True,
        ).unsqueeze(1)
        flat_crop = F.grid_sample(
            image_batch,
            flat_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(2)
        flat_valid = batch["flat_valid"].to(device=image_batch.device) > 0
        if flat_valid.ndim == 3:
            flat_valid = flat_valid.unsqueeze(1)
        return torch.where(flat_valid, flat_crop, torch.zeros_like(flat_crop))

    def _add_normal_pooled_batch(self, batch, pooled_logits, volume_logits):
        image_batch = batch["image"].float()
        if image_batch.ndim == 4:
            image_batch = image_batch.unsqueeze(1)
        if image_batch.ndim != 5:
            raise ValueError(f"Expected 'image' with shape [B, 1, Z, Y, X], got {tuple(image_batch.shape)}")

        if volume_logits is None:
            raise ValueError("normal pooled previews require raw volume logits")
        volume_logits = volume_logits.float()
        if volume_logits.ndim != 5:
            raise ValueError(
                f"Expected raw volume_logits with shape [B, 1, Z, Y, X], got {tuple(volume_logits.shape)}"
            )
        if tuple(int(v) for v in volume_logits.shape[-3:]) != tuple(int(v) for v in image_batch.shape[-3:]):
            volume_logits = F.interpolate(
                volume_logits,
                size=image_batch.shape[-3:],
                mode="trilinear",
                align_corners=True,
            )

        pooled_logits = pooled_logits.float()
        if pooled_logits.ndim == 3:
            pooled_logits = pooled_logits.unsqueeze(1)
        if pooled_logits.ndim != 4:
            raise ValueError(
                f"Expected pooled_logits with shape [B, 1, H, W] or [B, H, W], got {tuple(pooled_logits.shape)}"
            )

        flat_crop = self._sample_flat_crop(image_batch, batch)
        mid_slice_idx = int(image_batch.shape[2] // 2)
        volume_crop_mid = image_batch[:, :1, mid_slice_idx]
        volume_logits_mid = volume_logits[:, :1, mid_slice_idx]

        gathered_volume_logits = self.accelerator.gather_for_metrics(volume_logits_mid)
        gathered_volume_crops = self.accelerator.gather_for_metrics(volume_crop_mid)
        gathered_pooled_logits = self.accelerator.gather_for_metrics(pooled_logits)
        gathered_flat_crops = self.accelerator.gather_for_metrics(flat_crop)

        if not self.accelerator.is_main_process:
            return

        volume_logit_tiles = gathered_volume_logits[:, 0].detach().cpu().numpy()
        volume_crop_tiles = gathered_volume_crops[:, 0].detach().cpu().numpy()
        pooled_logit_tiles = gathered_pooled_logits[:, 0].detach().cpu().numpy()
        flat_crop_tiles = gathered_flat_crops[:, 0].detach().cpu().numpy()

        for volume_logit_tile, volume_crop_tile, pooled_logit_tile, flat_crop_tile in zip(
            volume_logit_tiles,
            volume_crop_tiles,
            pooled_logit_tiles,
            flat_crop_tiles,
        ):
            sample_tile = build_panel_grid(
                [
                    [
                        to_uint8_image(volume_logit_tile),
                        to_uint8_image(volume_crop_tile),
                    ],
                    [
                        to_uint8_image(pooled_logit_tile),
                        to_uint8_image(flat_crop_tile),
                    ],
                ],
                gap_size=self.gap_size,
            )
            if sample_tile is not None:
                self.sample_tiles.append(sample_tile)

    def add_batch(self, batch, preds, targets, ignore_mask, *, volume_logits=None):
        if "flat_points_local_zyx" in batch and "flat_valid" in batch:
            self._add_normal_pooled_batch(batch, preds, volume_logits)
            return
        self._add_standard_batch(batch, preds, targets, ignore_mask)

    def montage(self):
        return stack_preview_tiles(self.sample_tiles, gap_size=self.gap_size)

    def save(self, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        montage = self.montage()
        if montage is None:
            return
        tifffile.imwrite(str(output_path), montage, compression="lzw")

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
