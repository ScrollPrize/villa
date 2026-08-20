"""Standard three-panel TIFF previews for ink training validation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F


def to_uint8_image(image_HW) -> np.ndarray:
    """Min/max-normalize one tile and round it into uint8."""

    image_HW = np.nan_to_num(
        np.asarray(image_HW, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    minimum = float(image_HW.min())
    maximum = float(image_HW.max())
    if maximum > minimum:
        image_HW = (image_HW - minimum) / (maximum - minimum)
    else:
        image_HW = np.zeros_like(image_HW, dtype=np.float32)
    return np.clip(np.rint(image_HW * 255.0), 0, 255).astype(np.uint8)


def to_uint8_label(label_HW, ignore_mask_HW=None) -> np.ndarray:
    """Encode background, ink, and ignored pixels as 0, 255, and 127."""

    label_HW = np.asarray(label_HW, dtype=np.float32)
    result_HW = np.zeros(label_HW.shape, dtype=np.uint8)
    ignore_HW = None
    if ignore_mask_HW is not None:
        ignore_HW = np.asarray(ignore_mask_HW, dtype=np.float32) > 0
        result_HW[ignore_HW] = 127
    result_HW[label_HW == 0] = 0
    result_HW[label_HW > 0] = 255
    if ignore_HW is not None:
        result_HW[ignore_HW] = 127
    return result_HW


def to_uint8_probability(probability_HW) -> np.ndarray:
    """Clip and round one probability tile into uint8."""

    probability_HW = np.nan_to_num(
        np.asarray(probability_HW, dtype=np.float32),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    probability_HW = np.clip(probability_HW, 0.0, 1.0)
    return np.clip(np.rint(probability_HW * 255.0), 0, 255).astype(
        np.uint8
    )


def _pad_bottom_right(
    image_HW: np.ndarray,
    *,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    image_HW = np.asarray(image_HW, dtype=np.uint8)
    height, width = image_HW.shape
    target_height = int(target_height)
    target_width = int(target_width)
    padding = (
        (0, max(0, target_height - height)),
        (0, max(0, target_width - width)),
    )
    if padding == ((0, 0), (0, 0)):
        return image_HW
    return np.pad(image_HW, padding, mode="constant", constant_values=0)


def stack_preview_tiles(
    tiles: list[np.ndarray], gap_size: int = 4
) -> np.ndarray | None:
    """Stack sample rows vertically with zero-valued gaps."""

    if not tiles:
        return None
    maximum_width = max(int(tile.shape[1]) for tile in tiles)
    pieces = []
    gap = np.zeros((gap_size, maximum_width), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        if index:
            pieces.append(gap)
        pieces.append(
            _pad_bottom_right(
                tile,
                target_height=int(tile.shape[0]),
                target_width=maximum_width,
            )
        )
    return np.concatenate(pieces, axis=0)


def build_panel_grid(
    tiles: list[np.ndarray], gap_size: int = 4
) -> np.ndarray | None:
    """Join one row of preview panels with fixed zero gaps."""

    if not tiles:
        return None
    row_height = max(int(tile.shape[0]) for tile in tiles)
    pieces = []
    for index, tile in enumerate(tiles):
        pieces.append(
            _pad_bottom_right(
                tile,
                target_height=row_height,
                target_width=int(tile.shape[1]),
            )
        )
        if index + 1 < len(tiles):
            pieces.append(np.zeros((row_height, gap_size), dtype=np.uint8))
    return np.concatenate(pieces, axis=1)


@dataclass
class PreviewAccumulator:
    """Gather standard input/label/probability panels across processes."""

    accelerator: object
    get_model_input: Callable[[dict], torch.Tensor]
    sample_tiles: list[np.ndarray] = field(default_factory=list)
    gap_size: int = 4

    def add_batch(
        self,
        batch: dict,
        predictions_BCHW: torch.Tensor,
        targets_BCHW: torch.Tensor,
        ignore_mask_BCHW: torch.Tensor,
    ) -> None:
        input_BCZYX = self.get_model_input(batch)
        input_BCHW = input_BCZYX[:, :1, input_BCZYX.shape[2] // 2]
        if input_BCHW.shape[-2:] != predictions_BCHW.shape[-2:]:
            input_BCHW = F.interpolate(
                input_BCHW,
                size=predictions_BCHW.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        gathered_inputs = self.accelerator.gather_for_metrics(input_BCHW)
        gathered_targets = self.accelerator.gather_for_metrics(targets_BCHW)
        gathered_ignore = self.accelerator.gather_for_metrics(ignore_mask_BCHW)
        gathered_probabilities = self.accelerator.gather_for_metrics(
            torch.sigmoid(predictions_BCHW.float())
        )
        if not self.accelerator.is_main_process:
            return
        for input_HW, label_HW, ignore_HW, probability_HW in zip(
            gathered_inputs[:, 0].detach().cpu().numpy(),
            gathered_targets[:, 0].detach().cpu().numpy(),
            gathered_ignore[:, 0].detach().cpu().numpy(),
            gathered_probabilities[:, 0].detach().cpu().numpy(),
            strict=True,
        ):
            tile = build_panel_grid(
                [
                    to_uint8_image(input_HW),
                    to_uint8_label(label_HW, ignore_HW),
                    to_uint8_probability(probability_HW),
                ],
                gap_size=self.gap_size,
            )
            if tile is not None:
                self.sample_tiles.append(tile)

    def montage(self) -> np.ndarray | None:
        return stack_preview_tiles(self.sample_tiles, gap_size=self.gap_size)

    def save(self, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        montage = self.montage()
        if montage is not None:
            tifffile.imwrite(str(output_path), montage, compression="lzw")

    def wandb_image(self, caption: str):
        montage = self.montage()
        if montage is None:
            return None
        import wandb

        return wandb.Image(montage, caption=caption)


def central_full_3d_preview(
    batch: dict,
    predictions_BCZYX: torch.Tensor,
    targets_BCZYX: torch.Tensor,
    ignore_mask_BCZYX: torch.Tensor,
) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the central Z plane for a standard full-3D preview."""

    if (
        predictions_BCZYX.ndim != 5
        or targets_BCZYX.ndim != 5
        or ignore_mask_BCZYX.ndim != 5
    ):
        raise ValueError(
            "full_3d previews expect predictions, targets, and ignore mask "
            "with shape [B, 1, Z, Y, X]"
        )
    z_index = int(batch["supervision_mask"].shape[-3] // 2)
    preview_batch = dict(batch)
    image_BCZYX = batch["image"]
    if image_BCZYX.ndim == 4:
        image_BCZYX = image_BCZYX.unsqueeze(1)
    if image_BCZYX.ndim != 5:
        raise ValueError(
            "full_3d preview image must have shape [B, 1, Z, Y, X]"
        )
    preview_batch["image"] = image_BCZYX[:, :, z_index : z_index + 1]
    surface_mask = batch.get("surface_mask")
    if isinstance(surface_mask, torch.Tensor):
        if surface_mask.ndim == 4:
            surface_mask = surface_mask.unsqueeze(1)
        if surface_mask.ndim != 5:
            raise ValueError(
                "full_3d preview surface_mask must have shape [B, 1, Z, Y, X]"
            )
        preview_batch["surface_mask"] = surface_mask[
            :, :, z_index : z_index + 1
        ]
    return (
        preview_batch,
        predictions_BCZYX[:, :, z_index],
        targets_BCZYX[:, :, z_index],
        ignore_mask_BCZYX[:, :, z_index],
    )


def build_validation_preview_log(
    *,
    step: int,
    train_preview: PreviewAccumulator,
    val_preview: PreviewAccumulator,
    train_preview_dir: str | Path,
    val_preview_dir: str | Path,
    mean_val_loss: float,
    mean_ema_val_loss: float | None = None,
    include_wandb_images: bool = True,
) -> dict:
    """Write both TIFFs and optionally construct W&B image values."""

    train_preview.save(Path(train_preview_dir) / f"train_preview_{step:06}.tif")
    val_preview.save(Path(val_preview_dir) / f"val_preview_{step:06}.tif")
    result = {"val/loss": mean_val_loss}
    if mean_ema_val_loss is not None:
        result["val/loss_ema"] = mean_ema_val_loss
    if not include_wandb_images:
        return result
    train_image = train_preview.wandb_image(f"step {step} train preview")
    if train_image is not None:
        result["train/preview"] = train_image
    val_image = val_preview.wandb_image(f"step {step} val preview")
    if val_image is not None:
        result["val/preview"] = val_image
    return result
