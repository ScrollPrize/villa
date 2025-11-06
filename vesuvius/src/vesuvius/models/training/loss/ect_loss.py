from __future__ import annotations

"""
ECT-based loss for volumetric segmentation.

Core computation follows the differentiable Euler Characteristic Transform
implementation in https://github.com/aidos-lab/dect (BSD-3-Clause).
"""

import math
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _generate_uniform_directions(
    num_directions: int,
    dim: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    v = torch.randn((dim, num_directions), device=device, dtype=dtype, generator=g)
    v /= v.pow(2).sum(dim=0, keepdim=True).sqrt().clamp_min(1e-12)
    return v


@dataclass
class _GridCacheEntry:
    coordinates: Tensor
    radius: Tensor


class ECTLoss(nn.Module):
    """
    Compute a differentiable loss by matching Euler Characteristic Transforms of
    predictions and targets.

    Args:
        num_directions: Number of projection directions (>= 4 recommended).
        resolution: Number of thresholds along the filtration axis.
        scale: Sigmoid sharpness for the filtration indicator.
        normalize: If True, normalize per (batch, class) before comparison.
        aggregation: One of {"mse", "l1", "smooth_l1"}.
        apply_activation: {"auto", "sigmoid", "softmax", "none"}.
        seed: Random seed for direction sampling.
        radius_multiplier: Scale applied to coordinate radius (>= 1).
        use_fast_ect: Enable histogram-based ECT approximation.
        fast_subsample_ratio: Optional ratio (0,1] of voxels to sample per batch element when using the fast path.
        fast_max_points: Optional hard cap on sampled voxels per batch element for the fast path.
    """

    def __init__(
        self,
        num_directions: int = 32,
        resolution: int = 64,
        scale: float = 8.0,
        *,
        normalize: bool = False,
        aggregation: str = "mse",
        apply_activation: str = "auto",
        seed: int = 17,
        radius_multiplier: float = 1.1,
        use_fast_ect: bool = False,
        fast_subsample_ratio: float | None = None,
        fast_max_points: int | None = None,
        learnable: bool = False,
        accumulation_mode: str = "auto",
        soft_sigma: float | None = None,
        soft_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        if num_directions < 1:
            raise ValueError("num_directions must be >= 1")
        if resolution < 2:
            raise ValueError("resolution must be >= 2")
        if aggregation not in {"mse", "l1", "smooth_l1"}:
            raise ValueError(f"Unsupported aggregation '{aggregation}'")
        if apply_activation not in {"auto", "sigmoid", "softmax", "none"}:
            raise ValueError(f"Unsupported apply_activation '{apply_activation}'")
        if radius_multiplier < 1.0:
            raise ValueError("radius_multiplier must be >= 1.0")

        if fast_subsample_ratio is not None:
            if not (0.0 < fast_subsample_ratio <= 1.0):
                raise ValueError("fast_subsample_ratio must lie in (0, 1]")
        if fast_max_points is not None and fast_max_points <= 0:
            raise ValueError("fast_max_points must be positive")
        if use_fast_ect and fast_subsample_ratio is None and fast_max_points is None:
            # default to using all voxels; explicit None is acceptable
            pass

        self.num_directions = int(num_directions)
        self.resolution = int(resolution)
        self.scale = float(scale)
        self.normalize = bool(normalize)
        self.aggregation = aggregation
        self.apply_activation = apply_activation
        self.seed = int(seed)
        self.radius_multiplier = float(radius_multiplier)
        self.use_fast_ect = bool(use_fast_ect)
        self.fast_subsample_ratio = float(fast_subsample_ratio) if fast_subsample_ratio is not None else None
        self.fast_max_points = int(fast_max_points) if fast_max_points is not None else None
        self.learnable = bool(learnable)

        self.accumulation_mode = accumulation_mode.lower()
        if self.accumulation_mode not in {"auto", "fast", "regular", "soft"}:
            raise ValueError("accumulation_mode must be one of: auto, fast, regular, soft")

        if self.accumulation_mode == "fast" and not self.use_fast_ect:
            self.accumulation_mode = "regular"

        if self.learnable and self.use_fast_ect and self.accumulation_mode in {"auto", "fast"}:
            print(
                "ECTLoss: learnable directions require differentiable accumulation; switching to soft-binned ECT."
            )
            self.accumulation_mode = "soft"

        self.soft_sigma = float(soft_sigma) if soft_sigma is not None else None
        if soft_chunk_size is None:
            self.soft_chunk_size = 8192
        else:
            if soft_chunk_size <= 0:
                raise ValueError("soft_chunk_size must be positive")
            self.soft_chunk_size = int(soft_chunk_size)

        self._grid_cache: Dict[Tuple[Tuple[int, ...], torch.device, torch.dtype], _GridCacheEntry] = {}
        self._direction_cache: Dict[Tuple[torch.device, torch.dtype], Tensor] = {}

        if self.learnable:
            init_dirs = _generate_uniform_directions(
                self.num_directions,
                dim=3,
                seed=self.seed,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            self.v_param = nn.Parameter(init_dirs)
        else:
            self.register_parameter("v_param", None)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        if prediction.dim() < 4:
            raise ValueError("prediction must be BxCx... with >=2 spatial dims")

        pred = prediction
        tgt = target

        if pred.dim() != tgt.dim() and tgt.dim() == pred.dim() - 1:
            tgt = tgt.unsqueeze(1)

        if pred.shape[0] != tgt.shape[0]:
            raise ValueError("prediction and target batch sizes must match")
        if pred.shape[2:] != tgt.shape[2:]:
            raise ValueError("prediction and target spatial shapes must match")

        num_classes = pred.shape[1]
        prepared_target = self._prepare_target(tgt, num_classes)

        ect_pred = self.compute_ect_volume(pred, apply_activation=True)
        ect_tgt = self.compute_ect_volume(prepared_target, apply_activation=False)

        if self.normalize:
            ect_pred = self._normalize_ect(ect_pred)
            ect_tgt = self._normalize_ect(ect_tgt)

        if self.aggregation == "mse":
            loss = torch.mean((ect_pred - ect_tgt) ** 2)
        elif self.aggregation == "l1":
            loss = torch.mean(torch.abs(ect_pred - ect_tgt))
        else:
            loss = F.smooth_l1_loss(ect_pred, ect_tgt, reduction="mean")

        return loss

    def _activate(self, logits: Tensor) -> Tensor:
        if self.apply_activation == "none":
            return logits
        if self.apply_activation == "sigmoid":
            return torch.sigmoid(logits)
        if self.apply_activation == "softmax":
            return torch.softmax(logits, dim=1)
        if self.apply_activation == "auto":
            if logits.shape[1] == 1:
                return torch.sigmoid(logits)
            return torch.softmax(logits, dim=1)
        raise RuntimeError("Invalid activation option encountered")

    def _prepare_target(self, target: Tensor, num_classes: int) -> Tensor:
        tgt = target
        if tgt.shape[1] == num_classes:
            if torch.is_floating_point(tgt):
                return tgt
            return tgt.to(dtype=target.dtype)

        if num_classes == 1:
            return tgt.to(dtype=target.dtype, device=target.device)

        if tgt.shape[1] != 1:
            raise ValueError(
                "ECTLoss expects a single-channel target (class indices) or one-hot target."
            )

        indices = tgt.squeeze(1)
        if indices.dtype.is_floating_point:
            rounded = torch.round(indices)
            if torch.max(torch.abs(indices - rounded)) > 1e-5:
                raise ValueError(
                    "ECTLoss received floating-point class indices that are not near integer values. "
                    "This often occurs when Deep Supervision is enabled. Please disable Deep Supervision "
                    "or provide discrete class indices."
                )
            indices = rounded.to(torch.long).clamp_(0, num_classes - 1)
        else:
            indices = indices.to(torch.long)
        one_hot = F.one_hot(indices, num_classes=num_classes).movedim(-1, 1)
        return one_hot.to(dtype=target.dtype)

    def _get_directions(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        if self.learnable:
            dirs = F.normalize(self.v_param, dim=0, eps=1e-12)
            return dirs.to(device=device, dtype=dtype)

        key = (device, dtype)
        if key not in self._direction_cache:
            self._direction_cache[key] = _generate_uniform_directions(
                self.num_directions,
                dim=3,
                seed=self.seed,
                device=device,
                dtype=dtype,
            )
        return self._direction_cache[key]

    def compute_ect_volume(self, tensor: Tensor, *, apply_activation: bool = True) -> Tensor:
        if tensor.dim() < 4:
            raise ValueError("ECT computation expects tensors of shape [B, C, ...]")

        if apply_activation:
            tensor = self._activate(tensor)

        tensor = tensor.to(torch.float32)
        batch_size, num_classes = tensor.shape[:2]
        spatial_shape = tensor.shape[2:]

        weights = tensor.view(batch_size, num_classes, -1).transpose(1, 2).contiguous()

        device = tensor.device
        directions = self._get_directions(device, torch.float32)
        grid_entry = self._get_grid(spatial_shape, device, torch.float32)

        mode = self.accumulation_mode
        if mode == "auto":
            mode = "fast" if self.use_fast_ect else "regular"
        if mode == "fast" and self.learnable:
            mode = "soft"

        if mode == "fast":
            ect = self._compute_weighted_ect_fast(
                grid_entry.coordinates,
                weights,
                directions,
                grid_entry.radius,
            )
        elif mode == "soft":
            ect = self._compute_weighted_ect_soft(
                grid_entry.coordinates,
                weights,
                directions,
                grid_entry.radius,
            )
        else:
            ect = self._compute_weighted_ect_regular(
                grid_entry.coordinates,
                weights,
                directions,
                grid_entry.radius,
            )

        return ect

    def _get_grid(
        self,
        spatial_shape: Sequence[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> _GridCacheEntry:
        key = (tuple(int(s) for s in spatial_shape), device, dtype)
        if key in self._grid_cache:
            return self._grid_cache[key]

        dims = len(spatial_shape)
        if dims not in (2, 3):
            raise ValueError(f"ECTLoss supports 2D or 3D tensors, got {dims}D")

        axes: Tuple[Tensor, ...] = tuple(
            torch.linspace(-(s - 1) / 2.0, (s - 1) / 2.0, s, device=device, dtype=dtype)
            for s in spatial_shape
        )
        mesh = torch.meshgrid(*axes, indexing="ij")
        coords = torch.stack([m.reshape(-1) for m in mesh], dim=1)
        if dims == 2:
            coords = torch.cat(
                [coords, torch.zeros(coords.shape[0], 1, device=device, dtype=dtype)],
                dim=1,
            )

        radius = coords.abs().amax() * self.radius_multiplier
        entry = _GridCacheEntry(coordinates=coords, radius=radius)
        self._grid_cache[key] = entry
        return entry

    def _compute_weighted_ect_regular(
        self,
        coords: Tensor,
        weights: Tensor,
        directions: Tensor,
        radius: Tensor,
    ) -> Tensor:
        batch_size = weights.shape[0]
        num_voxels = weights.shape[1]
        num_classes = weights.shape[2]

        if coords.shape[0] != num_voxels:
            raise ValueError(
                f"Cached coordinate grid has {coords.shape[0]} points but received {num_voxels}"
            )

        coords_expanded = coords.repeat(batch_size, 1)

        batch_index = torch.arange(batch_size, device=weights.device).repeat_interleave(num_voxels)

        weights_flat = weights.reshape(batch_size * num_voxels, num_classes)

        radius = radius.to(coords.device, coords.dtype)

        ect = self._weighted_ect_core(
            coords_expanded,
            weights_flat,
            batch_index,
            directions,
            radius,
            num_classes,
            batch_size,
        )
        ect = ect.view(batch_size, num_classes, self.num_directions, self.resolution)
        return ect

    def _weighted_ect_core(
        self,
        coords: Tensor,
        weights: Tensor,
        batch_index: Tensor,
        directions: Tensor,
        radius: float,
        num_classes: int,
        batch_size: int,
    ) -> Tensor:
        num_points = coords.shape[0]
        num_dirs = directions.shape[1]

        radius = radius.to(coords.device, coords.dtype)
        lin = (
            torch.linspace(-1.0, 1.0, self.resolution, device=coords.device, dtype=coords.dtype)
            .view(self.resolution, 1, 1)
            * radius
        )

        heights = coords @ directions  # (num_points, num_dirs)
        ecc = torch.sigmoid(self.scale * (lin - heights.unsqueeze(0)))  # (R, P, D)

        weighted = ecc.unsqueeze(-1) * weights.view(1, num_points, 1, num_classes)
        weighted = weighted.permute(0, 1, 3, 2).reshape(self.resolution, num_points * num_classes, num_dirs)

        batch_index = batch_index.to(torch.long)
        class_indices = torch.arange(num_classes, device=coords.device)
        expanded_index = (
            batch_index.unsqueeze(1) * num_classes + class_indices.unsqueeze(0)
        ).reshape(-1)

        batch_len = batch_size * num_classes
        output = torch.zeros(
            (self.resolution, batch_len, num_dirs),
            device=coords.device,
            dtype=coords.dtype,
        )
        if weighted.numel() and expanded_index.numel():
            output.index_add_(1, expanded_index, weighted)

        return output.permute(1, 2, 0).contiguous()

    def _compute_weighted_ect_fast(
        self,
        coords: Tensor,
        weights: Tensor,
        directions: Tensor,
        radius: Tensor,
    ) -> Tensor:
        batch_size, num_voxels, num_classes = weights.shape

        if coords.shape[0] != num_voxels:
            raise ValueError(
                f"Cached coordinate grid has {coords.shape[0]} points but received {num_voxels}"
            )

        outputs = []
        for b in range(batch_size):
            outputs.append(
                self._compute_weighted_ect_fast_single(
                    coords,
                    weights[b],
                    directions,
                    radius,
                )
            )

        return torch.stack(outputs, dim=0)

    def _compute_weighted_ect_fast_single(
        self,
        coords: Tensor,
        weights: Tensor,
        directions: Tensor,
        radius: Tensor,
    ) -> Tensor:
        num_voxels = coords.shape[0]
        num_classes = weights.shape[1]

        idx = self._select_fast_indices(num_voxels, coords.device)
        coords_sel = coords[idx]
        weights_sel = weights[idx]

        radius = radius.to(coords.device, coords.dtype)
        safe_radius = torch.where(radius > 0, radius, radius.new_tensor(1.0))
        heights = coords_sel @ directions  # (P, num_dirs)
        scale = (self.resolution - 1) / (2.0 * safe_radius)
        bins = torch.clamp(
            ((heights + safe_radius) * scale).round_().to(torch.long),
            0,
            self.resolution - 1,
        )

        ect = torch.zeros(
            num_classes,
            self.num_directions,
            self.resolution,
            device=coords.device,
            dtype=weights_sel.dtype,
        )

        for d in range(self.num_directions):
            direction_bins = bins[:, d]
            for c in range(num_classes):
                values = weights_sel[:, c]
                hist = torch.zeros(
                    self.resolution, device=coords.device, dtype=values.dtype
                )
                hist.index_add_(0, direction_bins, values)
                ect[c, d] = hist.cumsum(dim=-1)

        return ect

    def _compute_weighted_ect_soft(
        self,
        coords: Tensor,
        weights: Tensor,
        directions: Tensor,
        radius: Tensor,
    ) -> Tensor:
        batch_size, num_voxels, num_classes = weights.shape
        device = coords.device
        dtype = coords.dtype

        radius = radius.to(device, dtype)
        thresholds = torch.linspace(-1.0, 1.0, self.resolution, device=device, dtype=dtype)
        thresholds = thresholds.view(1, 1, -1) * radius

        if self.soft_sigma is None:
            if self.resolution > 1:
                sigma = torch.abs(thresholds[0, 0, 1] - thresholds[0, 0, 0]).clamp_min(1e-6)
            else:
                sigma = radius.new_tensor(1.0)
        else:
            sigma = torch.as_tensor(self.soft_sigma, device=device, dtype=dtype)

        chunk = min(self.soft_chunk_size, num_voxels)
        if chunk <= 0:
            chunk = num_voxels

        ect = torch.zeros(
            batch_size,
            num_classes,
            self.num_directions,
            self.resolution,
            device=device,
            dtype=dtype,
        )

        for start in range(0, num_voxels, chunk):
            end = min(start + chunk, num_voxels)
            coord_chunk = coords[start:end]  # (chunk, 3)
            height_chunk = coord_chunk @ directions  # (chunk, dirs)

            diff = (height_chunk.unsqueeze(-1) - thresholds) / sigma
            kernel = torch.softmax(-0.5 * diff.pow(2), dim=-1)  # (chunk, dirs, res)

            kernel = kernel.unsqueeze(0)  # (1, chunk, dirs, res)
            weight_chunk = weights[:, start:end, :].unsqueeze(2).unsqueeze(3)  # (B, chunk, 1, 1, C)
            contrib = kernel.unsqueeze(-1) * weight_chunk  # (B, chunk, dirs, res, C)
            contrib = contrib.sum(dim=1)  # (B, dirs, res, C)
            ect += contrib.permute(0, 3, 1, 2)  # B x C x dirs x res

        return ect

    def _select_fast_indices(self, num_points: int, device: torch.device) -> Tensor:
        sample_n = num_points
        if self.fast_subsample_ratio is not None:
            sample_n = max(1, int(math.ceil(num_points * self.fast_subsample_ratio)))
        if self.fast_max_points is not None:
            sample_n = min(sample_n, self.fast_max_points)

        if sample_n >= num_points:
            return torch.arange(num_points, device=device)

        perm = torch.randperm(num_points, device=device)
        return perm[:sample_n]

    def _normalize_ect(self, ect: Tensor) -> Tensor:
        max_val = ect.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-12)
        return ect / max_val
