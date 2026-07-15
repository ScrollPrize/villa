from __future__ import annotations

import math

import numpy as np
import torch


LASAGNA_3X2_CHANNELS: tuple[str, ...] = (
    "dir0_z",
    "dir1_z",
    "dir0_y",
    "dir1_y",
    "dir0_x",
    "dir1_x",
)


def encode_lasagna_direction_2d(
    direction_ab: np.ndarray | torch.Tensor,
    *,
    eps: float = 1.0e-12,
) -> np.ndarray | torch.Tensor:
    """Encode one projected ambiguous 2D direction using Lasagna's formula.

    The input final dimension is ``(a, b)``. The output final dimension is the
    two double-angle channels used by Lasagna. ``(a, b)`` and ``(-a, -b)``
    intentionally encode identically.
    """

    if isinstance(direction_ab, torch.Tensor):
        direction = direction_ab.to(dtype=torch.float32)
        if direction.shape[-1] != 2:
            raise ValueError("direction_ab must have final dimension 2")
        a = direction[..., 0]
        b = direction[..., 1]
        denom = a * a + b * b + float(eps)
        cos2t = (a * a - b * b) / denom
        sin2t = (2.0 * a * b) / denom
        dir0 = 0.5 + 0.5 * cos2t
        dir1 = 0.5 + 0.5 * (cos2t - sin2t) / math.sqrt(2.0)
        return torch.stack([dir0, dir1], dim=-1)

    direction_np = np.asarray(direction_ab, dtype=np.float32)
    if direction_np.shape[-1] != 2:
        raise ValueError("direction_ab must have final dimension 2")
    a = direction_np[..., 0]
    b = direction_np[..., 1]
    denom = a * a + b * b + np.float32(eps)
    cos2t = (a * a - b * b) / denom
    sin2t = (np.float32(2.0) * a * b) / denom
    dir0 = np.float32(0.5) + np.float32(0.5) * cos2t
    dir1 = np.float32(0.5) + np.float32(0.5 / math.sqrt(2.0)) * (cos2t - sin2t)
    return np.stack([dir0, dir1], axis=-1).astype(np.float32, copy=False)


def decode_lasagna_direction_2d(
    encoded: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Decode Lasagna's ambiguous two-channel 2D direction representation."""

    if isinstance(encoded, torch.Tensor):
        values = encoded.to(dtype=torch.float32)
        if values.shape[-1] != 2:
            raise ValueError("encoded must have final dimension 2")
        cos2t = 2.0 * values[..., 0] - 1.0
        sin2t = cos2t - math.sqrt(2.0) * (2.0 * values[..., 1] - 1.0)
        theta = 0.5 * torch.atan2(sin2t, cos2t)
        return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

    values_np = np.asarray(encoded, dtype=np.float32)
    if values_np.shape[-1] != 2:
        raise ValueError("encoded must have final dimension 2")
    cos2t = np.float32(2.0) * values_np[..., 0] - np.float32(1.0)
    sin2t = cos2t - np.float32(math.sqrt(2.0)) * (
        np.float32(2.0) * values_np[..., 1] - np.float32(1.0)
    )
    theta = np.float32(0.5) * np.arctan2(sin2t, cos2t)
    return np.stack([np.cos(theta), np.sin(theta)], axis=-1).astype(
        np.float32, copy=False
    )


def encode_lasagna_direction_3x2(
    tangent_xyz: np.ndarray | torch.Tensor,
    *,
    eps: float = 1.0e-12,
) -> np.ndarray | torch.Tensor:
    """Encode a 3D tangent as Lasagna's six ambiguous direction channels.

    Channel order:

    - ``dir0_z, dir1_z`` encode the XY projection ``(tx, ty)``.
    - ``dir0_y, dir1_y`` encode the XZ projection ``(tx, tz)``.
    - ``dir0_x, dir1_x`` encode the YZ projection ``(ty, tz)``.
    """

    if isinstance(tangent_xyz, torch.Tensor):
        tangent = tangent_xyz.to(dtype=torch.float32)
        if tangent.shape[-1] != 3:
            raise ValueError("tangent_xyz must have final dimension 3")
        norm = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True).clamp_min(
            float(eps)
        )
        tangent = tangent / norm
        tx = tangent[..., 0]
        ty = tangent[..., 1]
        tz = tangent[..., 2]
        z_pair = encode_lasagna_direction_2d(torch.stack([tx, ty], dim=-1), eps=eps)
        y_pair = encode_lasagna_direction_2d(torch.stack([tx, tz], dim=-1), eps=eps)
        x_pair = encode_lasagna_direction_2d(torch.stack([ty, tz], dim=-1), eps=eps)
        return torch.cat([z_pair, y_pair, x_pair], dim=-1)

    tangent_np = np.asarray(tangent_xyz, dtype=np.float32)
    if tangent_np.shape[-1] != 3:
        raise ValueError("tangent_xyz must have final dimension 3")
    norm = np.linalg.norm(tangent_np, axis=-1, keepdims=True)
    tangent_np = np.divide(
        tangent_np,
        np.maximum(norm, np.float32(eps)),
        out=np.zeros_like(tangent_np, dtype=np.float32),
        where=norm > np.float32(eps),
    )
    tx = tangent_np[..., 0]
    ty = tangent_np[..., 1]
    tz = tangent_np[..., 2]
    z_pair = encode_lasagna_direction_2d(np.stack([tx, ty], axis=-1), eps=eps)
    y_pair = encode_lasagna_direction_2d(np.stack([tx, tz], axis=-1), eps=eps)
    x_pair = encode_lasagna_direction_2d(np.stack([ty, tz], axis=-1), eps=eps)
    return np.concatenate([z_pair, y_pair, x_pair], axis=-1).astype(
        np.float32, copy=False
    )


def projection_magnitude_weights_3x2(
    tangent_xyz: np.ndarray | torch.Tensor,
    *,
    eps: float = 1.0e-12,
) -> np.ndarray | torch.Tensor:
    """Return per-channel projection-magnitude weights for 3x2 supervision."""

    if isinstance(tangent_xyz, torch.Tensor):
        tangent = tangent_xyz.to(dtype=torch.float32)
        if tangent.shape[-1] != 3:
            raise ValueError("tangent_xyz must have final dimension 3")
        norm = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True).clamp_min(
            float(eps)
        )
        tangent = tangent / norm
        tx, ty, tz = tangent[..., 0], tangent[..., 1], tangent[..., 2]
        wz = tx * tx + ty * ty
        wy = tx * tx + tz * tz
        wx = ty * ty + tz * tz
        return torch.stack([wz, wz, wy, wy, wx, wx], dim=-1)

    tangent_np = np.asarray(tangent_xyz, dtype=np.float32)
    if tangent_np.shape[-1] != 3:
        raise ValueError("tangent_xyz must have final dimension 3")
    norm = np.linalg.norm(tangent_np, axis=-1, keepdims=True)
    tangent_np = np.divide(
        tangent_np,
        np.maximum(norm, np.float32(eps)),
        out=np.zeros_like(tangent_np, dtype=np.float32),
        where=norm > np.float32(eps),
    )
    tx = tangent_np[..., 0]
    ty = tangent_np[..., 1]
    tz = tangent_np[..., 2]
    wz = tx * tx + ty * ty
    wy = tx * tx + tz * tz
    wx = ty * ty + tz * tz
    return np.stack([wz, wz, wy, wy, wx, wx], axis=-1).astype(
        np.float32, copy=False
    )
