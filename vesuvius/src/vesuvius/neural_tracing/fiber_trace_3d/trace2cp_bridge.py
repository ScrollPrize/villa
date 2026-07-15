from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_trace_3d.projection import (
    project_3d_direction_to_2d_frame,
)


@dataclass(frozen=True)
class Trace2Cp3DProjectedFields:
    direction_xy: np.ndarray
    valid_mask: np.ndarray
    presence_hw: np.ndarray | None


@dataclass(frozen=True)
class Trace2Cp3DScore:
    trace2cp_error: float
    raw_y_error_px: float
    horizontal_span_px: float
    reached_target_columns: bool


def _sample_channel_volume_at_coords(
    volume_cdhw: torch.Tensor,
    coords_zyx: torch.Tensor,
) -> torch.Tensor:
    if volume_cdhw.ndim != 4:
        raise ValueError("volume_cdhw must have shape C,D,H,W")
    if coords_zyx.ndim != 3 or coords_zyx.shape[-1] != 3:
        raise ValueError("coords_zyx must have shape H,W,3")
    c, d, h, w = (int(v) for v in volume_cdhw.shape)
    del c
    coords = coords_zyx.to(dtype=torch.float32, device=volume_cdhw.device)
    gx = coords[..., 2] * (2.0 / float(max(w - 1, 1))) - 1.0 if w > 1 else torch.zeros_like(coords[..., 2])
    gy = coords[..., 1] * (2.0 / float(max(h - 1, 1))) - 1.0 if h > 1 else torch.zeros_like(coords[..., 1])
    gz = coords[..., 0] * (2.0 / float(max(d - 1, 1))) - 1.0 if d > 1 else torch.zeros_like(coords[..., 0])
    grid = torch.stack([gx, gy, gz], dim=-1).unsqueeze(0).unsqueeze(1)
    return F.grid_sample(
        volume_cdhw.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0, :, 0].permute(1, 2, 0)


def project_3d_output_to_trace2cp_fields(
    output: torch.Tensor,
    coords_zyx: np.ndarray | torch.Tensor,
    valid_mask: np.ndarray | torch.Tensor,
    *,
    frame_x_xyz: np.ndarray | torch.Tensor,
    frame_y_xyz: np.ndarray | torch.Tensor,
    volume_origin_zyx: np.ndarray | torch.Tensor | None = None,
) -> Trace2Cp3DProjectedFields:
    """Sample dense 3D prediction fields and project them into Trace2CP 2D fields.

    ``output`` is a 3D model output tensor shaped ``C,D,H,W`` or
    ``1,C,D,H,W``. ``coords_zyx`` are selected-level volume coordinates for the
    Trace2CP strip. If ``volume_origin_zyx`` is provided, it is subtracted before
    sampling, allowing callers to pass coordinates in global selected-level
    volume space while the model output is a local inference block.
    """

    values = output.detach()
    if values.ndim == 5:
        if int(values.shape[0]) != 1:
            raise ValueError("batched 3D Trace2CP projection expects batch size 1")
        values = values[0]
    if values.ndim != 4 or int(values.shape[0]) < 6:
        raise ValueError("output must have shape C,D,H,W with at least six direction channels")
    coords = torch.as_tensor(coords_zyx, dtype=torch.float32, device=values.device)
    if volume_origin_zyx is not None:
        origin = torch.as_tensor(volume_origin_zyx, dtype=torch.float32, device=values.device)
        coords = coords - origin.view(1, 1, 3)
    valid = torch.as_tensor(valid_mask, dtype=torch.bool, device=values.device)
    sampled = _sample_channel_volume_at_coords(values, coords)
    sampled_dir = sampled[..., :6]
    direction = project_3d_direction_to_2d_frame(
        sampled_dir,
        frame_x_xyz=torch.as_tensor(frame_x_xyz, dtype=torch.float32, device=values.device),
        frame_y_xyz=torch.as_tensor(frame_y_xyz, dtype=torch.float32, device=values.device),
    )
    d, h, w = (int(v) for v in values.shape[1:])
    in_bounds = (
        torch.isfinite(coords).all(dim=-1)
        & (coords[..., 0] >= 0.0)
        & (coords[..., 0] <= float(d - 1))
        & (coords[..., 1] >= 0.0)
        & (coords[..., 1] <= float(h - 1))
        & (coords[..., 2] >= 0.0)
        & (coords[..., 2] <= float(w - 1))
    )
    projected_valid = valid & in_bounds & torch.isfinite(direction).all(dim=-1)
    presence = sampled[..., 6] if int(values.shape[0]) >= 7 else None
    return Trace2Cp3DProjectedFields(
        direction_xy=direction.detach().cpu().numpy().astype(np.float32),
        valid_mask=projected_valid.detach().cpu().numpy().astype(bool),
        presence_hw=None
        if presence is None
        else presence.detach().cpu().numpy().astype(np.float32),
    )


def score_trace2cp_projected_fields(
    fields: Trace2Cp3DProjectedFields,
    *,
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    step_px: float = 1.0,
    rf_margin_px: float = 5.0,
) -> Trace2Cp3DScore:
    from vesuvius.neural_tracing.fiber_trace_2d.runner import (
        _trace_score_trace2cp_bidirectional,
    )

    result = _trace_score_trace2cp_bidirectional(
        fields.direction_xy,
        np.asarray(start_xy, dtype=np.float32),
        np.asarray(target_xy, dtype=np.float32),
        valid_mask=fields.valid_mask,
        step_px=float(step_px),
        rf_margin_px=float(rf_margin_px),
    )
    return Trace2Cp3DScore(
        trace2cp_error=float(result.metric.error),
        raw_y_error_px=float(result.metric.raw_y_error_px),
        horizontal_span_px=float(result.metric.horizontal_span_px),
        reached_target_columns=bool(result.metric.reached_target_columns),
    )
