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
from torch.utils.checkpoint import checkpoint as _checkpoint
from torch import Tensor, nn


def _maybe_checkpoint(function, *args):
    try:
        return _checkpoint(function, *args, use_reentrant=False)
    except TypeError:
        return _checkpoint(function, *args)


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


def _generate_fibonacci_directions(
    num_directions: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Fibonacci sphere sampling - optimal uniform coverage for any N.

    Uses the golden ratio to distribute points evenly on a unit sphere.
    This provides better coverage than random sampling, especially for
    3D curvilinear structures where uniform directional coverage matters.
    """
    indices = torch.arange(num_directions, device=device, dtype=dtype)
    phi = (1 + 5**0.5) / 2  # Golden ratio

    theta = 2 * math.pi * indices / phi
    z = 1 - 2 * (indices + 0.5) / num_directions
    r = torch.sqrt(1 - z**2)

    x = r * torch.cos(theta)
    y = r * torch.sin(theta)

    return torch.stack([x, y, z], dim=0)  # (3, num_directions)


_DTYPE_LOOKUP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def _resolve_dtype(value: torch.dtype | str | None, name: str) -> torch.dtype | None:
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        key = value.lower()
        if key in _DTYPE_LOOKUP:
            return _DTYPE_LOOKUP[key]
    raise ValueError(f"{name} must be a torch.dtype or one of {sorted(_DTYPE_LOOKUP)}")


@dataclass
class _GridCacheEntry:
    coordinates: Tensor
    radius: Tensor


class ECTLoss(nn.Module):
    """
    Compute a differentiable loss by matching Euler Characteristic Transforms of
    predictions and targets.

    Args:
        ect_variant: Which transform to match. "mass" (default) matches cumulative projected mass
            (fast, but not an Euler-characteristic transform). "chi" matches an (approximate)
            Euler-characteristic transform using expected Euler characteristic of a random cubical set.
        num_directions: Number of projection directions (>= 4 recommended).
        resolution: Number of thresholds along the filtration axis.
        scale: Sigmoid sharpness for the filtration indicator.
        normalize: If True, normalize per (batch, class) before comparison.
        aggregation: One of {"mse", "l1", "smooth_l1"}.
        seed: Random seed for direction sampling (only used with direction_mode="random").
        direction_mode: Direction sampling strategy. "fibonacci" (default) provides optimal
            uniform coverage on the sphere using Fibonacci spiral. "random" uses Gaussian
            sampling with the specified seed.
        radius_multiplier: Scale applied to coordinate radius (>= 1).
        direction_chunk_size: Chunk size for direction processing in the "mass" variant.
        fast_subsample_ratio: Optional ratio (0,1] of voxels to sample per batch element.
        fast_max_points: Optional hard cap on sampled voxels per batch element.
        ignore_label: Optional label value to ignore. Voxels with this label in the target will not contribute to the ECT.
        label_values: Optional class indices to include in the ECT (e.g., [1] for fg only).
        compute_dtype: Optional dtype for coordinates/weights (float32/float16/bfloat16). Histogram accumulation stays in fp32.
        chi_mc_directions: For ect_variant="chi", number of directions to sample per forward pass
            (Monte Carlo approximation). If None, defaults to min(num_directions, 8).
        chi_mc_thresholds: For ect_variant="chi", number of thresholds to sample per forward pass
            (Monte Carlo approximation). If None, defaults to min(resolution, 8).
        filtration_mode: For ect_variant="chi", how to apply the half-space filtration.
            "sigmoid" uses a smooth sigmoid with sharpness `scale`. "hard" uses an exact step.
    """

    def __init__(
        self,
        ect_variant: str = "mass",
        num_directions: int = 32,
        resolution: int = 64,
        scale: float = 8.0,
        *,
        normalize: bool = True,
        aggregation: str = "smooth_l1",
        seed: int = 17,
        direction_mode: str = "fibonacci",
        radius_multiplier: float = 1.1,
        direction_chunk_size: int = 16,
        chi_mc_directions: int | None = None,
        chi_mc_thresholds: int | None = None,
        filtration_mode: str = "sigmoid",
        fast_subsample_ratio: float | None = None,
        fast_max_points: int | None = None,
        ignore_label: int | None = None,
        label_values: Sequence[int] | int | None = None,
        compute_dtype: torch.dtype | str | None = None,
    ) -> None:
        super().__init__()
        if num_directions < 1:
            raise ValueError("num_directions must be >= 1")
        if resolution < 2:
            raise ValueError("resolution must be >= 2")
        if aggregation not in {"mse", "l1", "smooth_l1"}:
            raise ValueError(f"Unsupported aggregation '{aggregation}'")
        if direction_mode not in {"random", "fibonacci"}:
            raise ValueError(f"Unsupported direction_mode '{direction_mode}'")
        if ect_variant not in {"mass", "chi"}:
            raise ValueError("ect_variant must be one of {'mass', 'chi'}")
        if radius_multiplier < 1.0:
            raise ValueError("radius_multiplier must be >= 1.0")
        if direction_chunk_size < 1:
            raise ValueError("direction_chunk_size must be >= 1")
        if filtration_mode not in {"sigmoid", "hard"}:
            raise ValueError("filtration_mode must be one of {'sigmoid', 'hard'}")
        if chi_mc_directions is not None and chi_mc_directions < 1:
            raise ValueError("chi_mc_directions must be >= 1")
        if chi_mc_thresholds is not None and chi_mc_thresholds < 1:
            raise ValueError("chi_mc_thresholds must be >= 1")

        if fast_subsample_ratio is not None:
            if not (0.0 < fast_subsample_ratio <= 1.0):
                raise ValueError("fast_subsample_ratio must lie in (0, 1]")
        if fast_max_points is not None and fast_max_points <= 0:
            raise ValueError("fast_max_points must be positive")

        self.ect_variant = ect_variant
        self.num_directions = int(num_directions)
        self.resolution = int(resolution)
        self.scale = float(scale)
        self.normalize = bool(normalize)
        self.aggregation = aggregation
        self.seed = int(seed)
        self.direction_mode = direction_mode
        self.radius_multiplier = float(radius_multiplier)
        self.direction_chunk_size = int(direction_chunk_size)
        self.chi_mc_directions = int(chi_mc_directions) if chi_mc_directions is not None else None
        self.chi_mc_thresholds = int(chi_mc_thresholds) if chi_mc_thresholds is not None else None
        self.filtration_mode = filtration_mode
        self.fast_subsample_ratio = float(fast_subsample_ratio) if fast_subsample_ratio is not None else None
        self.fast_max_points = int(fast_max_points) if fast_max_points is not None else None
        self.ignore_label = int(ignore_label) if ignore_label is not None else None

        self._grid_cache: Dict[Tuple[Tuple[int, ...], torch.device, torch.dtype], _GridCacheEntry] = {}
        self._direction_cache: Dict[Tuple[torch.device, torch.dtype], Tensor] = {}

        self.accumulation_dtype = torch.float32
        self.compute_dtype = _resolve_dtype(compute_dtype, "compute_dtype") or torch.float32
        if not torch.is_floating_point(torch.empty((), dtype=self.compute_dtype)):
            raise ValueError("compute_dtype must be a floating point dtype")

        if label_values is None:
            self.label_values = None
        else:
            if isinstance(label_values, int):
                values = [int(label_values)]
            elif isinstance(label_values, str):
                raise ValueError("label_values must be a sequence of integers")
            else:
                values = [int(v) for v in label_values]
            if not values:
                raise ValueError("label_values must be non-empty")
            uniq = []
            for v in values:
                if v < 0:
                    raise ValueError("label_values must be non-negative")
                if v not in uniq:
                    uniq.append(v)
            self.label_values = tuple(uniq)

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
        label_indices = self._get_label_indices(num_classes)
        prepared_target = self._prepare_target(tgt, num_classes)

        # Create ignore mask if ignore_label is set
        # mask is True for voxels to KEEP (not ignored)
        if self.ignore_label is not None:
            if tgt.shape[1] == 1:
                # Single-channel class indices
                ignore_mask = (tgt.squeeze(1) != self.ignore_label)
            else:
                # Multi-channel (one-hot or soft labels) - use argmax
                ignore_mask = (tgt.argmax(dim=1) != self.ignore_label)
        else:
            ignore_mask = None

        if self.ect_variant == "mass":
            with torch.autocast("cuda"):
                ect_pred = self.compute_ect_volume(
                    pred,
                    apply_softmax=True,
                    ignore_mask=ignore_mask,
                    label_indices=label_indices,
                )

            # Target ECT doesn't need gradients - wrap in no_grad for memory efficiency
            with torch.no_grad(), torch.autocast("cuda"):
                ect_tgt = self.compute_ect_volume(
                    prepared_target,
                    apply_softmax=False,
                    ignore_mask=ignore_mask,
                    label_indices=label_indices,
                )

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

        # ect_variant == "chi": Monte Carlo approximation of ECT based on expected Euler characteristic
        return self._forward_chi(
            pred,
            prepared_target,
            ignore_mask=ignore_mask,
            label_indices=label_indices,
        )

    def _forward_chi(
        self,
        prediction: Tensor,
        prepared_target: Tensor,
        *,
        ignore_mask: Tensor | None,
        label_indices: Sequence[int] | None,
    ) -> Tensor:
        pred = prediction
        tgt = prepared_target

        device = pred.device
        compute_dtype = self.compute_dtype
        if device.type == "cpu" and compute_dtype in (torch.float16, torch.bfloat16):
            compute_dtype = torch.float32

        pred_logits = pred.to(dtype=compute_dtype)
        tgt_probs = tgt.to(dtype=compute_dtype)

        # Compute probabilities over ORIGINAL class dim, then optionally select classes.
        pred_probs_full = torch.softmax(pred_logits, dim=1)
        if label_indices is not None:
            sel = tuple(int(i) for i in label_indices)
            pred_probs = pred_probs_full[:, sel, ...]
            tgt_probs = tgt_probs[:, sel, ...]
        else:
            pred_probs = pred_probs_full

        spatial_shape = pred_probs.shape[2:]
        if len(spatial_shape) != 3:
            raise ValueError("ect_variant='chi' currently supports 3D tensors only")

        directions = self._get_directions(device, compute_dtype)
        grid_entry = self._get_grid(spatial_shape, device, compute_dtype)
        coords = grid_entry.coordinates
        radius = grid_entry.radius

        thresholds_full = torch.linspace(
            -radius,
            radius,
            self.resolution,
            device=device,
            dtype=compute_dtype,
        )

        mc_dirs = self.chi_mc_directions if self.chi_mc_directions is not None else min(self.num_directions, 8)
        mc_thrs = self.chi_mc_thresholds if self.chi_mc_thresholds is not None else min(self.resolution, 8)

        dir_idx = torch.randperm(self.num_directions, device=device)[:mc_dirs]
        thr_idx = torch.randperm(self.resolution, device=device)[:mc_thrs]

        # Optional global normalization to keep scale stable across crops.
        if self.normalize:
            denom = tgt_probs.sum(dim=(-3, -2, -1), keepdim=False).clamp_min(1.0)  # (B, C)
        else:
            denom = None

        num_samples = int(dir_idx.numel() * thr_idx.numel())
        if num_samples == 0:
            return pred_probs.new_zeros(())

        total = pred_probs.new_zeros((), dtype=torch.float32)

        # Compute loss by sampling a small (direction, threshold) grid per step.
        for d in dir_idx.tolist():
            v = directions[:, d]
            heights = (coords @ v).view(spatial_shape)

            for t_i in thr_idx.tolist():
                t_val = thresholds_full[t_i]

                with torch.no_grad():
                    if self.filtration_mode == "sigmoid":
                        gate_tgt = torch.sigmoid(self.scale * (t_val - heights))
                    else:
                        gate_tgt = (heights <= t_val).to(dtype=heights.dtype)
                    q_tgt = tgt_probs * gate_tgt
                    if ignore_mask is not None:
                        q_tgt = q_tgt * ignore_mask[:, None, ...]
                    chi_tgt = self._expected_euler_characteristic(q_tgt)
                    if denom is not None:
                        chi_tgt = chi_tgt / denom

                def _chi_pred_from_probs(probs: Tensor, *, heights=heights, t_val=t_val) -> Tensor:
                    if self.filtration_mode == "sigmoid":
                        gate = torch.sigmoid(self.scale * (t_val - heights))
                    else:
                        gate = (heights <= t_val).to(dtype=heights.dtype)

                    q = probs * gate
                    if ignore_mask is not None:
                        q = q * ignore_mask[:, None, ...]
                    chi = self._expected_euler_characteristic(q)
                    if denom is not None:
                        chi = chi / denom
                    return chi

                chi_pred = _maybe_checkpoint(_chi_pred_from_probs, pred_probs)

                if self.aggregation == "mse":
                    sample = torch.mean((chi_pred - chi_tgt) ** 2)
                elif self.aggregation == "l1":
                    sample = torch.mean(torch.abs(chi_pred - chi_tgt))
                else:
                    sample = F.smooth_l1_loss(chi_pred, chi_tgt, reduction="mean")
                total = total + sample

        return total / float(num_samples)

    def _get_label_indices(self, num_classes: int) -> Tuple[int, ...] | None:
        if self.label_values is None:
            return None
        if num_classes == 1:
            for v in self.label_values:
                if v not in (0, 1):
                    raise ValueError(
                        "label_values must be 0 or 1 for single-channel targets"
                    )
            return (0,)

        indices = []
        for v in self.label_values:
            if v < 0 or v >= num_classes:
                raise ValueError(
                    f"label_values contains {v}, but num_classes is {num_classes}"
                )
            if v not in indices:
                indices.append(v)
        return tuple(indices)

    def _get_class_indices(self, target: Tensor, num_classes: int) -> Tensor:
        tgt = target
        if tgt.shape[1] == num_classes:
            return tgt.argmax(dim=1)
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
        return indices

    def _prepare_target(self, target: Tensor, num_classes: int) -> Tensor:
        tgt = target
        if tgt.shape[1] == num_classes:
            if torch.is_floating_point(tgt):
                return tgt
            return tgt.to(dtype=target.dtype)

        if num_classes == 1:
            return tgt.to(dtype=target.dtype, device=target.device)

        indices = self._get_class_indices(tgt, num_classes)
        one_hot = F.one_hot(indices, num_classes=num_classes).movedim(-1, 1)
        return one_hot.to(dtype=target.dtype)

    def _get_directions(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        key = (device, dtype)
        if key not in self._direction_cache:
            if self.direction_mode == "fibonacci":
                self._direction_cache[key] = _generate_fibonacci_directions(
                    self.num_directions,
                    device=device,
                    dtype=dtype,
                )
            else:  # "random"
                self._direction_cache[key] = _generate_uniform_directions(
                    self.num_directions,
                    dim=3,
                    seed=self.seed,
                    device=device,
                    dtype=dtype,
                )
        return self._direction_cache[key]

    def compute_ect_volume(
        self,
        tensor: Tensor,
        *,
        apply_softmax: bool = False,
        ignore_mask: Tensor | None = None,
        label_indices: Sequence[int] | None = None,
    ) -> Tensor:
        if tensor.dim() < 4:
            raise ValueError("ECT computation expects tensors of shape [B, C, ...]")

        device = tensor.device
        compute_dtype = self.compute_dtype
        if device.type == "cpu" and compute_dtype in (torch.float16, torch.bfloat16):
            compute_dtype = torch.float32

        tensor = tensor.to(dtype=compute_dtype)

        batch_size, num_classes = tensor.shape[:2]
        spatial_shape = tensor.shape[2:]

        logits_view = tensor.view(batch_size, num_classes, -1)
        ignore_flat = ignore_mask.view(batch_size, -1) if ignore_mask is not None else None

        if apply_softmax:
            # Compute softmax over the ORIGINAL class dimension so selecting a single class
            # still yields gradients (softmax couples classes).
            weights_view = torch.softmax(logits_view, dim=1)
        else:
            weights_view = logits_view

        if label_indices is not None:
            weights_view = weights_view[:, tuple(int(i) for i in label_indices), :]

        directions = self._get_directions(device, compute_dtype)
        grid_entry = self._get_grid(spatial_shape, device, compute_dtype)

        ect = self._compute_ect(
            grid_entry.coordinates,
            weights_view,
            directions,
            grid_entry.radius,
            apply_softmax=False,
            ignore_flat=ignore_flat,
        )

        if ect.dtype != torch.float32:
            ect = ect.to(torch.float32)
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

        # Bound projections via ||x||_2 (since |<x, v>| <= ||x||_2 for ||v||_2 = 1).
        extents = torch.tensor(
            [(float(s) - 1.0) / 2.0 for s in spatial_shape],
            device=device,
            dtype=dtype,
        )
        if dims == 2:
            extents = torch.cat([extents, extents.new_zeros(1)], dim=0)
        radius = torch.sqrt((extents * extents).sum()) * self.radius_multiplier
        entry = _GridCacheEntry(coordinates=coords, radius=radius)
        self._grid_cache[key] = entry
        return entry

    def _subsample_indices(self, num_points: int, device: torch.device) -> Tensor | None:
        """Get random subset of indices for subsampling, or None if no subsampling needed."""
        sample_n = num_points
        if self.fast_subsample_ratio is not None:
            sample_n = max(1, int(math.ceil(num_points * self.fast_subsample_ratio)))
        if self.fast_max_points is not None:
            sample_n = min(sample_n, self.fast_max_points)

        if sample_n >= num_points:
            return None

        perm = torch.randperm(num_points, device=device)
        return perm[:sample_n]

    def _normalize_ect(self, ect: Tensor) -> Tensor:
        """Normalize ECT per (batch, class) before comparison."""
        # Use the final cumulative value (total mass) instead of amax to avoid
        # nondifferentiability/tie-breaking issues when many entries share the max.
        denom = ect[..., -1:].mean(dim=-2, keepdim=True).clamp_min(1e-12)
        return ect / denom

    def _expected_euler_characteristic(self, q: Tensor) -> Tensor:
        """Expected Euler characteristic for a random cubical set with cube probabilities `q`.

        Uses the standard cubical-complex cell count:
            chi = #V - #E + #F - #C   (3D)
            chi = #V - #E + #F       (2D)

        For each cell, P(cell present) = 1 - prod_{incident cubes}(1 - q_cube).

        Args:
            q: Tensor of shape [B, C, ...] with q in [0, 1].

        Returns:
            Tensor of shape [B, C] with expected chi for each (batch, class).
        """
        if q.dim() < 4:
            raise ValueError("q must be [B, C, ...] with >=2 spatial dims")

        spatial_dims = q.dim() - 2
        if spatial_dims == 2:
            return self._expected_euler_characteristic_2d(q)
        if spatial_dims == 3:
            return self._expected_euler_characteristic_3d(q)
        raise ValueError(f"Expected 2D or 3D spatial tensor, got {spatial_dims}D")

    def _expected_euler_characteristic_2d(self, q: Tensor) -> Tensor:
        q = q.clamp(0.0, 1.0)
        q_pad = F.pad(q, (1, 1, 1, 1), value=0.0)  # [B, C, H+2, W+2]
        r = 1.0 - q_pad
        H, W = q.shape[-2:]

        # Vertices: (H+1, W+1), incident squares: 4.
        prod_v = (
            r[..., 0 : H + 1, 0 : W + 1]
            * r[..., 1 : H + 2, 0 : W + 1]
            * r[..., 0 : H + 1, 1 : W + 2]
            * r[..., 1 : H + 2, 1 : W + 2]
        )
        p_v = 1.0 - prod_v

        # Horizontal edges: (H+1, W), incident squares: 2 (above/below).
        prod_eh = r[..., 0 : H + 1, 1 : W + 1] * r[..., 1 : H + 2, 1 : W + 1]
        p_eh = 1.0 - prod_eh

        # Vertical edges: (H, W+1), incident squares: 2 (left/right).
        prod_ev = r[..., 1 : H + 1, 0 : W + 1] * r[..., 1 : H + 1, 1 : W + 2]
        p_ev = 1.0 - prod_ev

        sum_v = p_v.to(self.accumulation_dtype).sum(dim=(-2, -1))
        sum_e = (p_eh.to(self.accumulation_dtype).sum(dim=(-2, -1)) + p_ev.to(self.accumulation_dtype).sum(dim=(-2, -1)))
        sum_f = q.to(self.accumulation_dtype).sum(dim=(-2, -1))
        return sum_v - sum_e + sum_f

    def _expected_euler_characteristic_3d(self, q: Tensor) -> Tensor:
        q = q.clamp(0.0, 1.0)
        q_pad = F.pad(q, (1, 1, 1, 1, 1, 1), value=0.0)  # [B, C, Z+2, Y+2, X+2]
        r = 1.0 - q_pad
        Z, Y, X = q.shape[-3:]

        # Vertices: (Z+1, Y+1, X+1), incident cubes: 8.
        prod_v = (
            r[..., 0 : Z + 1, 0 : Y + 1, 0 : X + 1]
            * r[..., 1 : Z + 2, 0 : Y + 1, 0 : X + 1]
            * r[..., 0 : Z + 1, 1 : Y + 2, 0 : X + 1]
            * r[..., 1 : Z + 2, 1 : Y + 2, 0 : X + 1]
            * r[..., 0 : Z + 1, 0 : Y + 1, 1 : X + 2]
            * r[..., 1 : Z + 2, 0 : Y + 1, 1 : X + 2]
            * r[..., 0 : Z + 1, 1 : Y + 2, 1 : X + 2]
            * r[..., 1 : Z + 2, 1 : Y + 2, 1 : X + 2]
        )
        p_v = 1.0 - prod_v

        # Edges: 3 orientations, each incident cubes: 4.
        # x-edges: (Z+1, Y+1, X)
        prod_ex = (
            r[..., 0 : Z + 1, 0 : Y + 1, 1 : X + 1]
            * r[..., 1 : Z + 2, 0 : Y + 1, 1 : X + 1]
            * r[..., 0 : Z + 1, 1 : Y + 2, 1 : X + 1]
            * r[..., 1 : Z + 2, 1 : Y + 2, 1 : X + 1]
        )
        p_ex = 1.0 - prod_ex

        # y-edges: (Z+1, Y, X+1)
        prod_ey = (
            r[..., 0 : Z + 1, 1 : Y + 1, 0 : X + 1]
            * r[..., 1 : Z + 2, 1 : Y + 1, 0 : X + 1]
            * r[..., 0 : Z + 1, 1 : Y + 1, 1 : X + 2]
            * r[..., 1 : Z + 2, 1 : Y + 1, 1 : X + 2]
        )
        p_ey = 1.0 - prod_ey

        # z-edges: (Z, Y+1, X+1)
        prod_ez = (
            r[..., 1 : Z + 1, 0 : Y + 1, 0 : X + 1]
            * r[..., 1 : Z + 1, 1 : Y + 2, 0 : X + 1]
            * r[..., 1 : Z + 1, 0 : Y + 1, 1 : X + 2]
            * r[..., 1 : Z + 1, 1 : Y + 2, 1 : X + 2]
        )
        p_ez = 1.0 - prod_ez

        # Faces: 3 orientations, each incident cubes: 2.
        # yz-faces (normal x): (Z, Y, X+1)
        prod_fx = r[..., 1 : Z + 1, 1 : Y + 1, 0 : X + 1] * r[..., 1 : Z + 1, 1 : Y + 1, 1 : X + 2]
        p_fx = 1.0 - prod_fx

        # xz-faces (normal y): (Z, Y+1, X)
        prod_fy = r[..., 1 : Z + 1, 0 : Y + 1, 1 : X + 1] * r[..., 1 : Z + 1, 1 : Y + 2, 1 : X + 1]
        p_fy = 1.0 - prod_fy

        # xy-faces (normal z): (Z+1, Y, X)
        prod_fz = r[..., 0 : Z + 1, 1 : Y + 1, 1 : X + 1] * r[..., 1 : Z + 2, 1 : Y + 1, 1 : X + 1]
        p_fz = 1.0 - prod_fz

        sum_v = p_v.to(self.accumulation_dtype).sum(dim=(-3, -2, -1))
        sum_e = (
            p_ex.to(self.accumulation_dtype).sum(dim=(-3, -2, -1))
            + p_ey.to(self.accumulation_dtype).sum(dim=(-3, -2, -1))
            + p_ez.to(self.accumulation_dtype).sum(dim=(-3, -2, -1))
        )
        sum_f = (
            p_fx.to(self.accumulation_dtype).sum(dim=(-3, -2, -1))
            + p_fy.to(self.accumulation_dtype).sum(dim=(-3, -2, -1))
            + p_fz.to(self.accumulation_dtype).sum(dim=(-3, -2, -1))
        )
        sum_c = q.to(self.accumulation_dtype).sum(dim=(-3, -2, -1))
        return sum_v - sum_e + sum_f - sum_c

    def _compute_ect(
        self,
        coords: Tensor,
        logits: Tensor,
        directions: Tensor,
        radius: Tensor,
        *,
        apply_softmax: bool = False,
        ignore_flat: Tensor | None = None,
    ) -> Tensor:
        """Compute ECT with masking and optional subsampling."""
        batch_size, num_classes, num_voxels = logits.shape

        if coords.shape[0] != num_voxels:
            raise ValueError(
                f"Cached coordinate grid has {coords.shape[0]} points but received {num_voxels}"
            )

        # Prepare mask indices for each batch element
        mask_indices = []
        for b in range(batch_size):
            mask_b = ignore_flat[b] if ignore_flat is not None else None

            if mask_b is not None and bool(mask_b.all()):
                mask_b = None

            if mask_b is None:
                idx = None
                num_points = num_voxels
            else:
                idx = torch.nonzero(mask_b, as_tuple=False).squeeze(1)
                num_points = idx.numel()

            # Apply subsampling if configured
            if num_points > 0:
                perm = self._subsample_indices(num_points, coords.device)
                if perm is not None:
                    if idx is None:
                        idx = perm
                    else:
                        idx = idx[perm]

            mask_indices.append(idx)

        # Use custom autograd function for memory-efficient computation
        return FastWeightedECT.apply(
            coords,
            logits,
            directions,
            radius,
            self.resolution,
            apply_softmax,
            mask_indices,
            self.accumulation_dtype,
            self.direction_chunk_size,
        )


class FastWeightedECT(torch.autograd.Function):
    """
    Memory-efficient weighted ECT with custom backward pass.

    Instead of storing all intermediate tensors for backprop (which explodes
    with millions of voxels), we only save the minimal tensors needed and
    recompute bins during backward.
    """

    @staticmethod
    def forward(
        ctx,
        coords: Tensor,         # (num_voxels, 3)
        logits: Tensor,         # (batch, num_classes, num_voxels)
        directions: Tensor,     # (3, num_dirs)
        radius: Tensor,         # scalar
        resolution: int,
        apply_softmax: bool,
        mask_indices: list,     # List of (indices tensor or None) per batch
        accumulation_dtype: torch.dtype,
        direction_chunk_size: int,
    ) -> Tensor:
        batch_size, num_classes, num_voxels = logits.shape
        num_dirs = directions.shape[1]
        device = coords.device

        # Compute softmax weights if needed
        if apply_softmax:
            # Softmax over classes (dim=1)
            weights = torch.softmax(logits, dim=1)
        else:
            weights = logits

        # Compute bin scale
        safe_radius = radius if radius > 0 else torch.tensor(1.0, device=device, dtype=coords.dtype)
        scale = (resolution - 1) / (2.0 * safe_radius)

        # Output histogram
        output = torch.zeros(
            batch_size, num_classes, num_dirs, resolution,
            device=device, dtype=accumulation_dtype,
        )

        direction_chunk_size = int(max(1, min(direction_chunk_size, num_dirs)))

        # Process each batch element
        for b in range(batch_size):
            idx = mask_indices[b] if mask_indices is not None else None

            if idx is not None:
                num_points = idx.numel()
                if num_points == 0:
                    continue
                coords_b = coords[idx]
                weights_b = weights[b, :, idx]  # (C, num_points)
            else:
                num_points = num_voxels
                coords_b = coords
                weights_b = weights[b]  # (C, num_voxels)

            weights_b = weights_b.to(accumulation_dtype)
            weights_bt = weights_b.transpose(0, 1).contiguous()  # (num_points, C)

            hist_flat = torch.zeros(
                num_dirs * resolution, num_classes,
                device=device, dtype=accumulation_dtype,
            )

            for dir_start in range(0, num_dirs, direction_chunk_size):
                dir_end = min(dir_start + direction_chunk_size, num_dirs)
                dirs_chunk = directions[:, dir_start:dir_end]  # (3, chunk)

                heights = coords_b @ dirs_chunk  # (num_points, chunk)
                bins = ((heights + safe_radius) * scale).round().to(torch.int32).clamp_(0, resolution - 1)

                for j in range(dir_end - dir_start):
                    d = dir_start + j
                    idx_d = bins[:, j].to(torch.int64) + d * resolution
                    hist_flat.index_add_(0, idx_d, weights_bt)

            # Reshape and cumsum
            hist = hist_flat.view(num_dirs, resolution, num_classes)
            output[b] = hist.cumsum(dim=1).permute(2, 0, 1).contiguous()

        # Save for backward - only essential tensors
        # Ensure float32 for coords/directions to avoid dtype issues with autocast
        ctx.save_for_backward(
            weights.float(),
            coords.float(),
            directions.float(),
            radius.float() if radius.dim() == 0 else radius,
        )
        ctx.resolution = resolution
        ctx.apply_softmax = apply_softmax
        ctx.mask_indices = mask_indices
        ctx.accumulation_dtype = accumulation_dtype
        ctx.direction_chunk_size = direction_chunk_size

        return output

    @staticmethod
    def backward(ctx, grad_ect: Tensor):
        weights, coords, directions, radius = ctx.saved_tensors
        resolution = ctx.resolution
        apply_softmax = ctx.apply_softmax
        mask_indices = ctx.mask_indices
        direction_chunk_size = int(getattr(ctx, "direction_chunk_size", directions.shape[1]))

        batch_size, num_classes, num_dirs, _ = grad_ect.shape
        num_voxels = coords.shape[0]
        device = coords.device

        # Reverse cumsum: gradient through cumsum
        # cumsum forward: out[i] = sum_{j<=i} x[j]
        # cumsum backward: grad_in[i] = sum_{j>=i} grad_out[j]
        # This is flip -> cumsum -> flip
        # Ensure float32 for numerical stability (grad_ect may be float16 under autocast)
        grad_hist = grad_ect.float().flip(-1).cumsum(-1).flip(-1)

        # Compute bin scale (same as forward)
        safe_radius = radius if radius > 0 else torch.tensor(1.0, device=device, dtype=coords.dtype)
        scale = (resolution - 1) / (2.0 * safe_radius)

        # Gradient w.r.t. weights
        grad_weights = torch.zeros_like(weights)

        # Process each batch element
        for b in range(batch_size):
            idx = mask_indices[b] if mask_indices is not None else None

            if idx is not None:
                num_points = idx.numel()
                if num_points == 0:
                    continue
                coords_b = coords[idx]
            else:
                num_points = num_voxels
                idx = None
                coords_b = coords

            # Gradient accumulator for this batch
            grad_w_b = torch.zeros(num_classes, num_points, device=device, dtype=grad_weights.dtype)

            direction_chunk_size = int(max(1, min(direction_chunk_size, num_dirs)))

            for dir_start in range(0, num_dirs, direction_chunk_size):
                dir_end = min(dir_start + direction_chunk_size, num_dirs)
                dirs_chunk = directions[:, dir_start:dir_end]  # (3, chunk)

                heights = coords_b @ dirs_chunk  # (num_points, chunk)
                bins = ((heights + safe_radius) * scale).round().to(torch.int64).clamp_(0, resolution - 1)

                for j in range(dir_end - dir_start):
                    d = dir_start + j
                    bin_indices = bins[:, j]  # (num_points,)
                    grad_w_b += grad_hist[b, :, d].index_select(1, bin_indices)

            # Store gradient back (handle masking)
            if idx is not None:
                grad_weights[b, :, idx] = grad_w_b
            else:
                grad_weights[b] = grad_w_b

        # If softmax was applied, compute gradient through softmax
        # softmax: s_c = exp(x_c) / sum_c' exp(x_c')
        # dsoftmax/dx_c = s_c * (1 - s_c) for same class
        # dsoftmax/dx_c' = -s_c * s_c' for different class
        # Combined: grad_logits[c] = weights[c] * (grad_weights[c] - sum_c'(weights[c'] * grad_weights[c']))
        if apply_softmax:
            weighted_grad_sum = (weights * grad_weights).sum(dim=1, keepdim=True)  # (B, 1, V)
            grad_logits = weights * (grad_weights - weighted_grad_sum)
        else:
            grad_logits = grad_weights

        # Return gradients for all forward inputs
        # (coords, logits, directions, radius, resolution, apply_softmax, mask_indices, accumulation_dtype, direction_chunk_size)
        return None, grad_logits, None, None, None, None, None, None, None
