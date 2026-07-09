from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_trace_2d.loader import FiberStripSample


_CP_NEIGHBOR_OFFSETS_YX: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


@dataclass(frozen=True)
class DirectionSupervision:
    patch_indices: torch.Tensor
    y: torch.Tensor
    x: torch.Tensor
    target: torch.Tensor
    cp_xy: torch.Tensor
    tangent_xy: torch.Tensor


def encode_lasagna_direction_xy(direction_xy: torch.Tensor, *, eps: float = 1.0e-12) -> torch.Tensor:
    """Encode strip-image directions using Lasagna's ambiguous two-cos channels."""

    device = getattr(direction_xy, "device", None)
    direction = torch.as_tensor(direction_xy, dtype=torch.float32, device=device)
    if direction.shape[-1] != 2:
        raise ValueError("direction_xy must have final dimension 2")
    dx = direction[..., 0]
    dy = direction[..., 1]
    denom = dx * dx + dy * dy + float(eps)
    cos2theta = (dx * dx - dy * dy) / denom
    sin2theta = (2.0 * dx * dy) / denom
    dir0 = 0.5 + 0.5 * cos2theta
    dir1 = 0.5 + 0.5 * (cos2theta - sin2theta) / (2.0**0.5)
    return torch.stack([dir0, dir1], dim=-1)


def decode_lasagna_direction_xy(encoded: torch.Tensor, *, bins: int = 180) -> torch.Tensor:
    """Approximate inverse for visualization. The sign is arbitrary by design."""

    device = getattr(encoded, "device", None)
    encoded_t = torch.as_tensor(encoded, dtype=torch.float32, device=device)
    if encoded_t.shape[-1] != 2:
        raise ValueError("encoded must have final dimension 2")
    theta = torch.linspace(0.0, torch.pi, int(bins), dtype=torch.float32, device=encoded_t.device)
    unit = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    table = encode_lasagna_direction_xy(unit)
    diff = encoded_t.reshape(-1, 1, 2) - table.reshape(1, int(bins), 2)
    best = torch.argmin(torch.sum(diff * diff, dim=2), dim=1)
    return unit[best].reshape(*encoded_t.shape[:-1], 2)


def cp_neighborhood_yx(cp_xy: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    cp = np.rint(np.asarray(cp_xy, dtype=np.float64)).astype(np.int64)
    if cp.shape != (2,):
        raise ValueError("cp_xy must have shape (2,)")
    height, width = (int(v) for v in shape_hw)
    points: list[tuple[int, int]] = []
    for dy, dx in _CP_NEIGHBOR_OFFSETS_YX:
        y = int(cp[1]) + dy
        x = int(cp[0]) + dx
        if 0 <= y < height and 0 <= x < width:
            points.append((y, x))
    return np.asarray(points, dtype=np.int64)


def line_cp_and_tangent_xy(line_xy: np.ndarray, cp_xy: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    line = np.asarray(line_xy, dtype=np.float32)
    if line.ndim != 2 or line.shape[1] != 2 or line.shape[0] < 2:
        return None
    finite = np.isfinite(line).all(axis=1)
    if not bool(finite.any()):
        return None
    line = line[finite]
    if line.shape[0] < 2:
        return None
    if cp_xy is None:
        cp = line[int(line.shape[0] // 2)]
        center = int(line.shape[0] // 2)
    else:
        cp = np.asarray(cp_xy, dtype=np.float32)
        if cp.shape != (2,) or not bool(np.isfinite(cp).all()):
            return None
        delta = line - cp.reshape(1, 2)
        center = int(np.argmin(np.sum(delta * delta, axis=1)))
    left = max(0, center - 1)
    right = min(line.shape[0] - 1, center + 1)
    if left == right:
        return None
    tangent = line[right] - line[left]
    norm = float(np.linalg.norm(tangent))
    if not np.isfinite(norm) or norm <= 1.0e-6:
        return None
    return cp.astype(np.float32), (tangent / norm).astype(np.float32)


def build_direction_supervision(
    samples: Sequence[FiberStripSample],
    valid_mask: np.ndarray | torch.Tensor,
    *,
    device: torch.device,
) -> DirectionSupervision:
    valid_np = np.asarray(valid_mask, dtype=bool)
    if valid_np.ndim != 3:
        raise ValueError("valid_mask must have shape N,H,W")
    if len(samples) != int(valid_np.shape[0]):
        raise ValueError("samples length must match valid_mask patch count")

    patch_indices: list[int] = []
    ys: list[int] = []
    xs: list[int] = []
    targets: list[np.ndarray] = []
    cp_rows: list[np.ndarray] = []
    tangent_rows: list[np.ndarray] = []
    height, width = int(valid_np.shape[1]), int(valid_np.shape[2])
    for patch_index, sample in enumerate(samples):
        cp_and_tangent = line_cp_and_tangent_xy(
            sample.line_xy,
            getattr(sample, "control_point_xy", None),
        )
        if cp_and_tangent is None:
            continue
        cp_xy, tangent_xy = cp_and_tangent
        positions = cp_neighborhood_yx(cp_xy, (height, width))
        if positions.size == 0:
            continue
        target = encode_lasagna_direction_xy(torch.as_tensor(tangent_xy, dtype=torch.float32)).cpu().numpy()
        for y, x in positions.tolist():
            if not bool(valid_np[patch_index, y, x]):
                continue
            patch_indices.append(patch_index)
            ys.append(int(y))
            xs.append(int(x))
            targets.append(target.astype(np.float32))
            cp_rows.append(cp_xy.astype(np.float32))
            tangent_rows.append(tangent_xy.astype(np.float32))

    if not patch_indices:
        empty_i = torch.zeros((0,), dtype=torch.long, device=device)
        return DirectionSupervision(
            patch_indices=empty_i,
            y=empty_i,
            x=empty_i,
            target=torch.zeros((0, 2), dtype=torch.float32, device=device),
            cp_xy=torch.zeros((0, 2), dtype=torch.float32, device=device),
            tangent_xy=torch.zeros((0, 2), dtype=torch.float32, device=device),
        )
    return DirectionSupervision(
        patch_indices=torch.as_tensor(patch_indices, dtype=torch.long, device=device),
        y=torch.as_tensor(ys, dtype=torch.long, device=device),
        x=torch.as_tensor(xs, dtype=torch.long, device=device),
        target=torch.as_tensor(np.stack(targets, axis=0), dtype=torch.float32, device=device),
        cp_xy=torch.as_tensor(np.stack(cp_rows, axis=0), dtype=torch.float32, device=device),
        tangent_xy=torch.as_tensor(np.stack(tangent_rows, axis=0), dtype=torch.float32, device=device),
    )


def gather_direction_predictions(prediction: torch.Tensor, supervision: DirectionSupervision) -> torch.Tensor:
    if prediction.ndim != 4 or int(prediction.shape[1]) != 2:
        raise ValueError("prediction must have shape N,2,H,W")
    return prediction[supervision.patch_indices, :, supervision.y, supervision.x]


def direction_mse_loss(prediction: torch.Tensor, supervision: DirectionSupervision) -> torch.Tensor:
    gathered = gather_direction_predictions(prediction, supervision)
    if gathered.numel() == 0:
        raise ValueError("no valid CP-local direction supervision samples")
    return torch.nn.functional.mse_loss(gathered, supervision.target)
