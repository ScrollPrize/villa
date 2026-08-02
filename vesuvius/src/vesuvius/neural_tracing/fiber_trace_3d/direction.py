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


def decode_lasagna_direction_3x2_analytic(
    encoded: np.ndarray | torch.Tensor,
    *,
    eps: float = 1.0e-12,
) -> np.ndarray | torch.Tensor:
    """Decode Lasagna's six 3D direction channels without candidate search.

    The six-channel representation stores three sign-ambiguous 2D projections:
    XY, XZ, and YZ. This reconstructs the 3D ambiguous axis with the same
    analytic three-plane reconstruction/sign-alignment scheme used by the
    Lasagna preprocessing path. The returned axis is unit length; ``v`` and
    ``-v`` remain equivalent.
    """

    if isinstance(encoded, torch.Tensor):
        values = encoded.to(dtype=torch.float32)
        if values.shape[-1] != 6:
            raise ValueError("encoded must have final dimension 6")
        z_pair = decode_lasagna_direction_2d(values[..., 0:2])
        y_pair = decode_lasagna_direction_2d(values[..., 2:4])
        x_pair = decode_lasagna_direction_2d(values[..., 4:6])
        cz, sz = z_pair[..., 0], z_pair[..., 1]
        cy, sy = y_pair[..., 0], y_pair[..., 1]
        cx, sx = x_pair[..., 0], x_pair[..., 1]

        n1 = torch.stack([cz * cy, sz * cy, cz * sy], dim=-1)
        n2 = torch.stack([cz * cx, sz * cx, sz * sx], dim=-1)
        n3 = torch.stack([cy * sx, sy * cx, sy * sx], dim=-1)

        def align(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
            sign = torch.where(
                torch.sum(reference * candidate, dim=-1, keepdim=True) >= 0.0,
                torch.ones((), dtype=candidate.dtype, device=candidate.device),
                -torch.ones((), dtype=candidate.dtype, device=candidate.device),
            )
            return candidate * sign

        n2 = align(n1, n2)
        n3 = align(n1, n3)

        def enc_pair(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            r2 = a * a + b * b + float(eps)
            c2 = (a * a - b * b) / r2
            s2 = (2.0 * a * b) / r2
            return (
                0.5 + 0.5 * c2,
                0.5 + 0.5 * (c2 - s2) / math.sqrt(2.0),
            )

        scores: list[torch.Tensor] = []
        for candidate in (n1, n2, n3):
            nx, ny, nz = candidate[..., 0], candidate[..., 1], candidate[..., 2]
            pz0, pz1 = enc_pair(nx, ny)
            py0, py1 = enc_pair(nx, nz)
            px0, px1 = enc_pair(ny, nz)
            err_z = (pz0 - values[..., 0]) ** 2 + (pz1 - values[..., 1]) ** 2
            err_y = (py0 - values[..., 2]) ** 2 + (py1 - values[..., 3]) ** 2
            err_x = (px0 - values[..., 4]) ** 2 + (px1 - values[..., 5]) ** 2
            wz = nx * nx + ny * ny
            wy = nx * nx + nz * nz
            wx = ny * ny + nz * nz
            scores.append(1.0 / (wz * err_z + wy * err_y + wx * err_x + float(eps)))

        estimate = scores[0][..., None] * n1 + scores[1][..., None] * n2 + scores[2][..., None] * n3
        estimate = estimate / torch.linalg.vector_norm(estimate, dim=-1, keepdim=True).clamp_min(float(eps))

        ex, ey, ez = estimate[..., 0], estimate[..., 1], estimate[..., 2]
        wz2 = torch.sqrt(ex * ex + ey * ey + float(eps))
        wy2 = torch.sqrt(ex * ex + ez * ez + float(eps))
        wx2 = torch.sqrt(ey * ey + ez * ez + float(eps))

        rn1 = (wz2 * wy2)[..., None] * n1
        rn2 = (wz2 * wx2)[..., None] * n2
        rn3 = (wy2 * wx2)[..., None] * n3
        rn2 = align(rn1, rn2)
        rn3 = align(rn1, rn3)
        axis = rn1 + rn2 + rn3
        return axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(float(eps))

    values_np = np.asarray(encoded, dtype=np.float32)
    if values_np.shape[-1] != 6:
        raise ValueError("encoded must have final dimension 6")
    z_pair = decode_lasagna_direction_2d(values_np[..., 0:2])
    y_pair = decode_lasagna_direction_2d(values_np[..., 2:4])
    x_pair = decode_lasagna_direction_2d(values_np[..., 4:6])
    cz, sz = z_pair[..., 0], z_pair[..., 1]
    cy, sy = y_pair[..., 0], y_pair[..., 1]
    cx, sx = x_pair[..., 0], x_pair[..., 1]

    n1 = np.stack([cz * cy, sz * cy, cz * sy], axis=-1)
    n2 = np.stack([cz * cx, sz * cx, sz * sx], axis=-1)
    n3 = np.stack([cy * sx, sy * cx, sy * sx], axis=-1)

    def align_np(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
        sign = np.where(
            np.sum(reference * candidate, axis=-1, keepdims=True) >= 0.0,
            np.float32(1.0),
            np.float32(-1.0),
        )
        return candidate * sign

    n2 = align_np(n1, n2)
    n3 = align_np(n1, n3)

    def enc_pair_np(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r2 = a * a + b * b + np.float32(eps)
        c2 = (a * a - b * b) / r2
        s2 = (np.float32(2.0) * a * b) / r2
        return (
            np.float32(0.5) + np.float32(0.5) * c2,
            np.float32(0.5) + np.float32(0.5 / math.sqrt(2.0)) * (c2 - s2),
        )

    scores_np: list[np.ndarray] = []
    for candidate in (n1, n2, n3):
        nx, ny, nz = candidate[..., 0], candidate[..., 1], candidate[..., 2]
        pz0, pz1 = enc_pair_np(nx, ny)
        py0, py1 = enc_pair_np(nx, nz)
        px0, px1 = enc_pair_np(ny, nz)
        err_z = (pz0 - values_np[..., 0]) ** 2 + (pz1 - values_np[..., 1]) ** 2
        err_y = (py0 - values_np[..., 2]) ** 2 + (py1 - values_np[..., 3]) ** 2
        err_x = (px0 - values_np[..., 4]) ** 2 + (px1 - values_np[..., 5]) ** 2
        wz = nx * nx + ny * ny
        wy = nx * nx + nz * nz
        wx = ny * ny + nz * nz
        scores_np.append(np.float32(1.0) / (wz * err_z + wy * err_y + wx * err_x + np.float32(eps)))

    estimate_np = scores_np[0][..., None] * n1 + scores_np[1][..., None] * n2 + scores_np[2][..., None] * n3
    norm_e = np.linalg.norm(estimate_np, axis=-1, keepdims=True)
    estimate_np = estimate_np / np.maximum(norm_e, np.float32(eps))

    ex, ey, ez = estimate_np[..., 0], estimate_np[..., 1], estimate_np[..., 2]
    wz2 = np.sqrt(ex * ex + ey * ey + np.float32(eps))
    wy2 = np.sqrt(ex * ex + ez * ez + np.float32(eps))
    wx2 = np.sqrt(ey * ey + ez * ez + np.float32(eps))
    rn1 = (wz2 * wy2)[..., None] * n1
    rn2 = (wz2 * wx2)[..., None] * n2
    rn3 = (wy2 * wx2)[..., None] * n3
    rn2 = align_np(rn1, rn2)
    rn3 = align_np(rn1, rn3)
    axis_np = rn1 + rn2 + rn3
    norm_f = np.linalg.norm(axis_np, axis=-1, keepdims=True)
    axis_np = axis_np / np.maximum(norm_f, np.float32(eps))
    return axis_np.astype(np.float32, copy=False)


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
