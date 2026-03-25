from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from koine_machines.common.common import (
    build_preview_montage,
    save_val_preview_tif,
    to_uint8_image,
    to_uint8_label,
    to_uint8_probability,
)


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
