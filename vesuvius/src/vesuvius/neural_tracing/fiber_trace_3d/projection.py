from __future__ import annotations

import functools
import math

import torch

from vesuvius.neural_tracing.fiber_trace_3d.direction import (
    encode_lasagna_direction_3x2,
)


@functools.lru_cache(maxsize=8)
def _candidate_axes(device_type: str, candidate_count: int) -> torch.Tensor:
    count = max(64, int(candidate_count))
    # Fibonacci sphere, with half-sphere de-duplication because the Lasagna
    # direction code is sign ambiguous.
    indices = torch.arange(count * 2, dtype=torch.float32)
    golden = math.pi * (3.0 - math.sqrt(5.0))
    z = 1.0 - 2.0 * (indices + 0.5) / float(count * 2)
    radius = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    theta = indices * golden
    x = torch.cos(theta) * radius
    y = torch.sin(theta) * radius
    axes = torch.stack([x, y, z], dim=1)
    axes = axes[axes[:, 0] >= 0.0][:count]
    if axes.shape[0] < count:
        axes = torch.cat([axes, -axes[: count - axes.shape[0]]], dim=0)
    return axes.to(torch.device(device_type))


def decode_lasagna_direction_3x2_grid_search(
    encoded: torch.Tensor,
    *,
    candidate_count: int = 1024,
) -> torch.Tensor:
    """Approximate a 3D axis from six Lasagna direction channels.

    The 3x2 encoding stores three normalized 2D projections, so this helper
    uses a deterministic unit-sphere table and chooses the axis whose encoded
    channels best match. It is intended for evaluation/projection tooling, not
    for training loss.
    """

    values = torch.as_tensor(encoded, dtype=torch.float32)
    if values.shape[-1] != 6:
        raise ValueError("encoded must have final dimension 6")
    candidates = _candidate_axes(values.device.type, int(candidate_count)).to(values.device)
    candidate_codes = encode_lasagna_direction_3x2(candidates)
    flat = values.reshape(-1, 6)
    error = torch.mean((flat[:, None, :] - candidate_codes[None, :, :]) ** 2, dim=2)
    best = torch.argmin(error, dim=1)
    axes = candidates[best].reshape(*values.shape[:-1], 3)
    return axes


def project_3d_direction_to_2d_frame(
    encoded_3x2: torch.Tensor,
    *,
    frame_x_xyz: torch.Tensor,
    frame_y_xyz: torch.Tensor,
    candidate_count: int = 1024,
) -> torch.Tensor:
    """Project decoded 3D ambiguous directions into a 2D strip frame."""

    axis_xyz = decode_lasagna_direction_3x2_grid_search(
        encoded_3x2,
        candidate_count=candidate_count,
    )
    x_axis = torch.as_tensor(frame_x_xyz, dtype=torch.float32, device=axis_xyz.device)
    y_axis = torch.as_tensor(frame_y_xyz, dtype=torch.float32, device=axis_xyz.device)
    x_axis = x_axis / torch.linalg.vector_norm(x_axis).clamp_min(1.0e-12)
    y_axis = y_axis / torch.linalg.vector_norm(y_axis).clamp_min(1.0e-12)
    projected = torch.stack(
        [
            torch.sum(axis_xyz * x_axis, dim=-1),
            torch.sum(axis_xyz * y_axis, dim=-1),
        ],
        dim=-1,
    )
    norm = torch.linalg.vector_norm(projected, dim=-1, keepdim=True).clamp_min(1.0e-12)
    return projected / norm
