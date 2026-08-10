"""Centered zero padding along the depth axis of BCZYX model inputs."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vesuvius.ink_detection.config import InkConfig


def configured_input_pad_depth(config: InkConfig) -> int | None:
    """Return the validated target depth selected by the model configuration."""

    return config.model.input_pad_depth_to


def center_pad_input_depth(
    image_BCZYX: torch.Tensor,
    target_depth: int | None,
) -> torch.Tensor:
    """Pad BCZYX depth equally, placing an odd extra plane after the input."""

    if target_depth is None:
        return image_BCZYX
    if image_BCZYX.ndim != 5:
        raise ValueError(
            "Depth padding expects [batch, channel, z, y, x], "
            f"got {tuple(image_BCZYX.shape)}"
        )
    source_depth = int(image_BCZYX.shape[2])
    target_depth = int(target_depth)
    if source_depth > target_depth:
        raise ValueError(
            f"Cannot pad input depth {source_depth} down to {target_depth}"
        )
    total = target_depth - source_depth
    before = total // 2
    after = total - before
    return F.pad(image_BCZYX, (0, 0, 0, 0, before, after))
