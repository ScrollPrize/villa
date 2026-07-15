from __future__ import annotations

import torch

from vesuvius.neural_tracing.fiber_trace_3d.direction import (
    decode_lasagna_direction_3x2_analytic,
)


def project_3d_direction_to_2d_frame(
    encoded_3x2: torch.Tensor,
    *,
    frame_x_xyz: torch.Tensor,
    frame_y_xyz: torch.Tensor,
) -> torch.Tensor:
    """Project analytically decoded 3D ambiguous directions into a 2D strip frame."""

    axis_xyz = decode_lasagna_direction_3x2_analytic(encoded_3x2)
    x_axis = torch.as_tensor(frame_x_xyz, dtype=torch.float32, device=axis_xyz.device)
    y_axis = torch.as_tensor(frame_y_xyz, dtype=torch.float32, device=axis_xyz.device)
    x_axis = x_axis / torch.linalg.vector_norm(x_axis, dim=-1, keepdim=True).clamp_min(1.0e-12)
    y_axis = y_axis / torch.linalg.vector_norm(y_axis, dim=-1, keepdim=True).clamp_min(1.0e-12)
    projected = torch.stack(
        [
            torch.sum(axis_xyz * x_axis, dim=-1),
            torch.sum(axis_xyz * y_axis, dim=-1),
        ],
        dim=-1,
    )
    norm = torch.linalg.vector_norm(projected, dim=-1, keepdim=True).clamp_min(1.0e-12)
    return projected / norm
