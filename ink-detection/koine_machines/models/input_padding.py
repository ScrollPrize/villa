from __future__ import annotations

import torch
import torch.nn.functional as F


def configured_input_pad_depth(config) -> int | None:
    value = (config.get("model_config") or {}).get("input_pad_depth_to")
    if value is None:
        return None
    target_depth = int(value)
    if target_depth <= 0:
        raise ValueError(
            f"model_config.input_pad_depth_to must be positive, got {value!r}"
        )
    return target_depth


def center_pad_input_depth(image: torch.Tensor, target_depth: int | None) -> torch.Tensor:
    if target_depth is None:
        return image
    if image.ndim != 5:
        raise ValueError(
            f"Depth padding expects [batch, channel, z, y, x], got {tuple(image.shape)}"
        )
    source_depth = int(image.shape[2])
    target_depth = int(target_depth)
    if source_depth > target_depth:
        raise ValueError(
            f"Cannot pad input depth {source_depth} down to {target_depth}"
        )
    total = target_depth - source_depth
    before = total // 2
    after = total - before
    return F.pad(image, (0, 0, 0, 0, before, after))
