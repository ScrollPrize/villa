from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from embedding.model import (
    InkPatchEmbedder,
    build_embedder_from_config,
    normalize_for_backbone,
    resolve_checkpoint_config,
)


def resolve_embedding_runtime_config(
    *,
    config_path: str | Path | None,
    checkpoint_path: str | Path | None,
) -> tuple[Path | None, Path | None, dict[str, Any]]:
    resolved_ckpt = Path(checkpoint_path).expanduser().resolve() if checkpoint_path else None
    resolved_cfg = Path(config_path).expanduser().resolve() if config_path else None

    checkpoint_config: dict[str, Any] | None = None
    if resolved_ckpt is not None:
        resolved_ckpt, checkpoint_config = resolve_checkpoint_config(resolved_ckpt)
        if resolved_cfg is None:
            adjacent_cfg = resolved_ckpt.parent / "config.json"
            if adjacent_cfg.exists():
                resolved_cfg = adjacent_cfg.resolve()

    if resolved_cfg is not None:
        with resolved_cfg.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        if not isinstance(config, dict):
            raise ValueError(f"Expected JSON object at {resolved_cfg}, got {type(config).__name__}")
    elif checkpoint_config is not None:
        config = checkpoint_config
    else:
        raise ValueError(
            "stitch embedding loss requires either training.embedding_model_config_path "
            "or a checkpoint with embedded/adjacent config.json"
        )

    return resolved_cfg, resolved_ckpt, config

def build_embedding_model_for_similarity(
    *,
    config: dict[str, Any],
    checkpoint_path: str | Path | None,
    device: torch.device,
) -> tuple[InkPatchEmbedder, dict[str, Any]]:
    resolved_checkpoint = Path(checkpoint_path).expanduser().resolve() if checkpoint_path else None
    model, runtime_config = build_embedder_from_config(
        config,
        device=device,
        checkpoint_path=resolved_checkpoint,
        freeze_parameters=True,
    )
    model.eval()
    return model, runtime_config


def resolve_covering_starts(length: int, patch_size: int) -> list[int]:
    length = int(length)
    patch_size = int(patch_size)
    if length <= 0 or patch_size <= 0:
        raise ValueError(f"Expected positive length/patch_size, got {length}, {patch_size}")
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, patch_size))
    last_start = length - patch_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def downsample_stitched_ink_map(image: torch.Tensor, factor: int) -> torch.Tensor:
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape={tuple(image.shape)!r}")
    factor = int(factor)
    if factor <= 1:
        return image
    return F.interpolate(
        image[None, None],
        size=(max(1, image.shape[0] // factor), max(1, image.shape[1] // factor)),
        mode="area",
    )[0, 0]


def resize_stitched_ink_map(image: torch.Tensor, size: int) -> torch.Tensor:
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape={tuple(image.shape)!r}")
    size = int(size)
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    if tuple(image.shape) == (size, size):
        return image
    mode = "area" if int(image.shape[0]) >= size and int(image.shape[1]) >= size else "bilinear"
    kwargs = {} if mode == "area" else {"align_corners": False}
    return F.interpolate(image[None, None], size=(size, size), mode=mode, **kwargs)[0, 0]


def extract_covering_embedding_patches(image: torch.Tensor, crop_size: int) -> torch.Tensor:
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape={tuple(image.shape)!r}")
    crop_size = int(crop_size)
    if crop_size < 1:
        raise ValueError(f"crop_size must be >= 1, got {crop_size}")

    starts_y = resolve_covering_starts(int(image.shape[0]), crop_size)
    starts_x = resolve_covering_starts(int(image.shape[1]), crop_size)
    patches = []
    for top in starts_y:
        for left in starts_x:
            patch = image[top : top + crop_size, left : left + crop_size]
            if patch.shape != (crop_size, crop_size):
                padded = image.new_zeros((crop_size, crop_size))
                padded[: patch.shape[0], : patch.shape[1]] = patch
                patch = padded
            patches.append(patch)
    return torch.stack(patches, dim=0).unsqueeze(1)


def _crop_to_valid_bbox(image: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    coords = torch.nonzero(valid_mask, as_tuple=False)
    if coords.numel() == 0:
        raise ValueError("valid_mask contains no covered pixels")
    y0 = int(coords[:, 0].min().item())
    y1 = int(coords[:, 0].max().item()) + 1
    x0 = int(coords[:, 1].min().item())
    x1 = int(coords[:, 1].max().item()) + 1
    return image[y0:y1, x0:x1]


def compute_stitch_embedding_similarity(
    *,
    embedding_model: InkPatchEmbedder,
    embedding_crop_size: int,
    input_downsample_factor: int,
    stitched_logits: torch.Tensor,
    stitched_targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    pred = torch.sigmoid(stitched_logits.float())
    target = stitched_targets.float().clamp_(0.0, 1.0)
    valid = valid_mask.to(dtype=pred.dtype)

    if int(input_downsample_factor) == -1:
        pred = pred * valid
        target = target * valid
        pred_patches = resize_stitched_ink_map(pred, embedding_crop_size)[None, None]
        target_patches = resize_stitched_ink_map(target, embedding_crop_size)[None, None]
    else:
        pred = _crop_to_valid_bbox(pred * valid, valid_mask)
        target = _crop_to_valid_bbox(target * valid, valid_mask)
        pred = downsample_stitched_ink_map(pred, input_downsample_factor)
        target = downsample_stitched_ink_map(target, input_downsample_factor)
        pred_patches = extract_covering_embedding_patches(pred, embedding_crop_size)
        target_patches = extract_covering_embedding_patches(target, embedding_crop_size)

    pred_inputs = normalize_for_backbone(pred_patches)
    target_inputs = normalize_for_backbone(target_patches)

    pred_embeddings, _ = embedding_model(pred_inputs)
    with torch.no_grad():
        target_embeddings, _ = embedding_model(target_inputs)

    return 1.0 - F.cosine_similarity(pred_embeddings.float(), target_embeddings.float(), dim=1).mean()
