"""Local tracing frames, oriented crops and polyline helpers.

World coordinates are trace-grid ``x, y, z`` (float). A frame is a 3x3
matrix whose columns are ``(u, v, f)`` in world xyz: ``f`` is the heading,
``u, v`` span the cross-section. Oriented crops are laid out
``C, D, H, W`` = channels, forward (f), v, u.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CropSpec:
    depth: int = 64  # samples along f
    width: int = 64  # samples along u and v
    behind: int = 16  # samples behind the current point
    spacing: float = 1.0
    @property
    def forward_coords(self) -> np.ndarray:
        return (np.arange(self.depth) - self.behind) * self.spacing

    @property
    def lateral_coords(self) -> np.ndarray:
        return (np.arange(self.width) - (self.width - 1) / 2.0) * self.spacing

    @property
    def center_offset(self) -> float:
        fc = self.forward_coords
        return 0.5 * float(fc[0] + fc[-1])

    @property
    def block_size(self) -> int:
        fc = self.forward_coords
        h = 0.5 * float(self.lateral_coords[-1] - self.lateral_coords[0])
        r = np.sqrt((fc - self.center_offset) ** 2 + 2 * h**2).max()
        return int(2 * np.ceil(r + 2.0))


def normalize(v: np.ndarray, axis: int = -1) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), 1e-9)


def frame_from_heading(f: np.ndarray, u_hint: np.ndarray | None = None) -> np.ndarray:
    """Frame with columns (u, v, f); ``u`` parallel-transported from ``u_hint``."""
    f = normalize(np.asarray(f, dtype=np.float64))
    if u_hint is not None:
        u = u_hint - np.dot(u_hint, f) * f
        if np.linalg.norm(u) < 1e-3:
            u_hint = None
    if u_hint is None:
        ref = np.array([0.0, 0.0, 1.0]) if abs(f[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(ref, f)
    u = normalize(u)
    v = np.cross(f, u)
    return np.stack([u, v, f], axis=1)


def crop_local_grid(spec: CropSpec) -> np.ndarray:
    """Local (a=u, b=v, c=f) coordinates, shape (D, H, W, 3)."""
    lc = spec.lateral_coords
    fc = spec.forward_coords
    c, b, a = np.meshgrid(fc, lc, lc, indexing="ij")
    return np.stack([a, b, c], axis=-1)


def block_start(pos_xyz: np.ndarray, frame: np.ndarray, spec: CropSpec) -> np.ndarray:
    """zyx start of the axis-aligned block that contains the oriented crop."""
    center = pos_xyz + spec.center_offset * frame[:, 2]
    half = spec.block_size // 2
    return np.floor(center[::-1]).astype(np.int64) - half + 1


# ---------------------------------------------------------------- polylines


def arclength(p: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])


def resample_polyline(p: np.ndarray, step: float) -> np.ndarray:
    s = arclength(p)
    if s[-1] < step:
        return p[[0, -1]].copy()
    t = np.arange(0.0, s[-1] + 1e-9, step)
    return np.stack([np.interp(t, s, p[:, i]) for i in range(3)], 1)


def interp_at(p: np.ndarray, s: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(t, s, p[:, i]) for i in range(3)], -1)


def tangent_at(p: np.ndarray, s: np.ndarray, t: float, half: float = 3.0) -> np.ndarray:
    a = interp_at(p, s, np.array([max(t - half, 0.0)]))[0]
    b = interp_at(p, s, np.array([min(t + half, s[-1])]))[0]
    return normalize(b - a)


def sample_oriented_fast(
    raw: torch.Tensor,
    starts_zyx: torch.Tensor,
    pos_xyz: torch.Tensor,
    frames: torch.Tensor,
    local_grid: torch.Tensor,
) -> torch.Tensor:
    """Oriented trilinear crop straight from raw uint8 (B, 1, S, S, S) blocks."""
    if raw.shape[1] != 1:
        raise ValueError('Oriented sampling reads one scalar channel')
    S = raw.shape[-1]
    world = pos_xyz[:, None, None, None, :] + torch.einsum("bij,dhwj->bdhwi", frames, local_grid)
    idx = world - starts_zyx.flip(-1)[:, None, None, None, :].to(world.dtype)
    grid = idx * (2.0 / (S - 1)) - 1.0
    return F.grid_sample(raw.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=True) * (1.0 / 255.0)
