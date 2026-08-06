from __future__ import annotations

import torch
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_trace_3d.direction import (
    decode_lasagna_direction_3x2_analytic,
)


_EPS = 1.0e-12


def direction_branch_count_from_channels(channels: int) -> int:
    channels_i = int(channels)
    if channels_i < 6:
        raise ValueError("fiber 3D prediction output has fewer than six channels")
    if channels_i == 6:
        return 1
    if channels_i < 7 or channels_i % 7 != 0:
        raise ValueError(
            "fiber 3D prediction output channels must be 6 or a positive "
            f"multiple of 7; got {channels_i}"
        )
    return channels_i // 7


def decode_grouped_direction_presence(
    sampled_nchannels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode raw fiber model samples into branch directions and presence.

    Input shape is ``N,C`` with either six Lasagna 3x2 direction channels or
    ``7 * branch_count`` grouped channels where each group stores six direction
    channels followed by one presence channel.
    """

    if sampled_nchannels.ndim != 2:
        raise ValueError("sampled_nchannels must have shape N,C")
    channels = int(sampled_nchannels.shape[1])
    branch_count = direction_branch_count_from_channels(channels)
    directions: list[torch.Tensor] = []
    presences: list[torch.Tensor] = []
    for branch in range(branch_count):
        start = 0 if channels == 6 else branch * 7
        axis_xyz = decode_lasagna_direction_3x2_analytic(
            sampled_nchannels[:, start : start + 6]
        )
        axis_zyx = axis_xyz[:, [2, 1, 0]].to(dtype=torch.float32)
        axis_zyx = F.normalize(axis_zyx, p=2.0, dim=1, eps=float(_EPS))
        directions.append(axis_zyx)
        if channels == 6:
            presences.append(
                torch.ones(
                    (int(sampled_nchannels.shape[0]),),
                    dtype=torch.float32,
                    device=sampled_nchannels.device,
                )
            )
        else:
            presences.append(
                sampled_nchannels[:, start + 6].to(dtype=torch.float32).clamp(0.0, 1.0)
            )
    direction_t = torch.stack(directions, dim=1)
    presence_t = torch.stack(presences, dim=1)
    valid_t = torch.isfinite(direction_t).all(dim=2) & torch.isfinite(presence_t)
    return direction_t, presence_t, valid_t


__all__ = [
    "decode_grouped_direction_presence",
    "direction_branch_count_from_channels",
]
