"""Local tracing frames, oriented crops and polyline helpers.

World coordinates are trace-grid ``x, y, z`` (float). A frame is a 3x3
matrix whose columns are ``(u, v, f)`` in world xyz: ``f`` is the heading,
``u, v`` span the cross-section. Oriented crops are laid out
``C, D, H, W`` = channels, forward (f), v, u.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CropSpec:
    depth: int = 64  # samples along f
    width: int = 64  # samples along u and v
    behind: int = 16  # samples behind the current point
    spacing: float = 1.0
    # multiply the direction (axis-tensor) channels by presence so direction
    # is only "loud" on fibers (the field is dense noise in empty space)
    gate_direction: bool = False
    # Legacy checkpoints default to point splats; CT tube runs opt into segments.
    history_render: str = 'points'
    history_sigma: float = 1.0  # trace-grid voxels

    def __post_init__(self):
        if self.history_render not in ('points', 'segments') or not np.isfinite(self.history_sigma) or self.history_sigma <= 0:
            raise ValueError('Invalid history rendering mode or sigma')

    def replay_dict(self):
        values = asdict(self)
        # Keep legacy collectors compatible with training processes already running.
        if self.history_render == 'points' and self.history_sigma == 1.0:
            del values['history_render'], values['history_sigma']
        return values

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


def random_rotation_about(frame: np.ndarray, angle: float) -> np.ndarray:
    u, v, f = frame[:, 0], frame[:, 1], frame[:, 2]
    c, s = np.cos(angle), np.sin(angle)
    return np.stack([c * u + s * v, -s * u + c * v, f], axis=1)


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


def render_history(hist_local: torch.Tensor, hist_mask: torch.Tensor, local_grid: torch.Tensor,
                   sigma: float = 1.0, mode: str = 'points') -> torch.Tensor:
    """Gaussian history, optionally joining consecutive valid points to the current origin.

    Missing history stays empty. Masked gaps are never bridged. Width uses trace-grid units.
    """
    if mode not in ('points', 'segments') or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('Invalid history rendering mode or sigma')
    g = local_grid.reshape(-1, 3)
    B, H, _ = hist_local.shape
    if H == 0:
        return hist_local.new_zeros((B, 1, *local_grid.shape[:3]))
    if mode == 'points':
        d2 = torch.cdist(hist_local, g[None].expand(B, -1, -1)).square()
        d2 = d2.masked_fill(hist_mask[..., None] <= 0, float('inf')).amin(1)
    else:
        # Process one segment at a time to bound memory on full 64-cubed crops.
        d2 = hist_local.new_full((B, len(g)), float('inf'))
        for k in range(H):
            end = hist_local[:, k]
            start = hist_local[:, k-1] if k else torch.zeros_like(end)
            connected = hist_mask[:, k-1] > 0 if k else torch.ones(B, device=end.device, dtype=torch.bool)
            # Isolated valid points still contribute a spherical endpoint.
            start = torch.where(connected[:, None], start, end)
            v = end-start
            q = g[None]-start[:, None]
            t = ((q*v[:, None]).sum(-1) / v.square().sum(-1).clamp_min(1e-12)[:, None]).clamp(0, 1)
            candidate = (q-t[..., None]*v[:, None]).square().sum(-1)
            candidate = candidate.masked_fill(hist_mask[:, k, None] <= 0, float('inf'))
            d2 = torch.minimum(d2, candidate)
    return torch.exp(-d2 / (2*sigma*sigma)).view(B, 1, *local_grid.shape[:3])


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
    gate_direction: bool = False,
) -> torch.Tensor:
    """Oriented crop straight from raw uint8 blocks (presence, nx, ny[, ct]).

    Presence/CT are trilinear; the fiber axis is taken from the nearest voxel
    and decoded only at crop points, then expressed in the local frame as
    (uu, vv, ff, uv, uf, vf). Channels: presence, 6 axis-tensor terms, [ct].
    """
    B, S = raw.shape[0], raw.shape[-1]
    world = pos_xyz[:, None, None, None, :] + torch.einsum("bij,dhwj->bdhwi", frames, local_grid)
    idx = world - starts_zyx.flip(-1)[:, None, None, None, :].to(world.dtype)
    grid = idx * (2.0 / (S - 1)) - 1.0
    r = raw.float()
    if raw.shape[1] == 1:  # CT-only input
        return F.grid_sample(r, grid, mode="bilinear", padding_mode="zeros", align_corners=True) * (1.0 / 255.0)
    lin = [0] + ([3] if raw.shape[1] > 3 else [])
    lin_s = F.grid_sample(r[:, lin], grid, mode="bilinear", padding_mode="zeros", align_corners=True) * (1.0 / 255.0)
    dn = F.grid_sample(r[:, 1:3], grid, mode="nearest", padding_mode="border", align_corners=True)
    nx = (dn[:, 0] - 128.0) * (1.0 / 127.0)
    ny = (dn[:, 1] - 128.0) * (1.0 / 127.0)
    nz = torch.sqrt(torch.clamp(1.0 - nx * nx - ny * ny, min=0.0))
    n = torch.stack([nx, ny, nz], -1)
    n = n * torch.rsqrt(torch.clamp((n * n).sum(-1, keepdim=True), min=1e-12))
    nl = torch.einsum("bdhwi,bij->bdhwj", n, frames)  # local (u, v, f) components
    a, b, c = nl.unbind(-1)
    t6 = torch.stack([a * a, b * b, c * c, a * b, a * c, b * c], 1)
    if gate_direction:
        t6 = t6 * lin_s[:, :1]
    out = [lin_s[:, :1], t6]
    if len(lin) > 1:
        out.append(lin_s[:, 1:])
    return torch.cat(out, 1)
