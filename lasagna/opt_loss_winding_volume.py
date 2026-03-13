from __future__ import annotations

import math

import torch
import torch.nn.functional as F

import model as fit_model


# Module-level state set by compute_auto_offset(), persists for rest of optimization.
_winding_offset: float | None = None
_winding_direction: int = 1  # +1 or -1


def reset_auto_offset() -> None:
	"""Clear auto-offset state (e.g. at start of optimization)."""
	global _winding_offset, _winding_direction
	_winding_offset = None
	_winding_direction = 1


def _sample_winding_volume(*, res: fit_model.FitResult3D) -> torch.Tensor:
	"""Sample the winding volume at coarse mesh positions. Returns (D, 1, Hm, Wm)."""
	wv = res.data.winding_volume
	if wv is None:
		raise RuntimeError("winding_vol loss requested but FitData3D.winding_volume is None")

	D, Hm, Wm, _ = res.xyz_lr.shape
	Z, Y, X = wv.shape[2], wv.shape[3], wv.shape[4]

	ox, oy, oz = res.data.origin_fullres
	sx, sy, sz = res.data.spacing

	grid = res.xyz_lr.clone()  # (D, Hm, Wm, 3) — (x, y, z) in fullres
	grid[..., 0] = (grid[..., 0] - ox) / sx / max(1, X - 1) * 2 - 1
	grid[..., 1] = (grid[..., 1] - oy) / sy / max(1, Y - 1) * 2 - 1
	grid[..., 2] = (grid[..., 2] - oz) / sz / max(1, Z - 1) * 2 - 1

	sampled = F.grid_sample(
		wv, grid.unsqueeze(0),  # (1, 1, Z, Y, X), (1, D, Hm, Wm, 3)
		mode='bilinear', padding_mode='zeros', align_corners=True,
	)
	# sampled: (1, 1, D, Hm, Wm) → (D, 1, Hm, Wm)
	sampled = sampled[0].permute(1, 0, 2, 3)
	return sampled


def compute_auto_offset(*, res: fit_model.FitResult3D) -> tuple[float, int]:
	"""Compute best (offset, direction) to align depth indices with winding volume values.

	For each candidate: target(d) = offset + d * direction, where d = 0..D-1.
	Picks the (offset, direction) that minimizes MSE against sampled winding values
	at valid positions (winding >= 1 and mask_lr).

	Sets module-level _winding_offset and _winding_direction.
	"""
	global _winding_offset, _winding_direction

	sampled = _sample_winding_volume(res=res)
	D = sampled.shape[0]
	dev = sampled.device

	# Mask: valid positions
	wv_valid = (sampled.detach() >= 1.0).float()
	mask = res.mask_lr * wv_valid  # (D, 1, Hm, Wm)

	wsum = mask.sum().item()
	if wsum < 1:
		# No valid points — default to offset=1, direction=+1
		_winding_offset = 1.0
		_winding_direction = 1
		return _winding_offset, _winding_direction

	# Depth indices: (D, 1, 1, 1)
	d_idx = torch.arange(D, device=dev, dtype=torch.float32).view(D, 1, 1, 1)

	# Masked sampled values
	masked_sampled = sampled.detach() * mask  # (D, 1, Hm, Wm)

	# Determine search range from sampled values
	valid_vals = sampled.detach()[mask > 0]
	min_val = float(valid_vals.min())
	max_val = float(valid_vals.max())

	best_mse = float('inf')
	best_offset = 1.0
	best_dir = 1

	for direction in [+1, -1]:
		# For direction +1: target(d) = offset + d
		#   offset should be near min_val (when d=0) to max_val (when d=D-1)
		# For direction -1: target(d) = offset - d
		#   offset should be near max_val (when d=0) to min_val (when d=D-1)
		if direction == 1:
			lo = math.floor(min_val) - D
			hi = math.ceil(max_val) + D
		else:
			lo = math.floor(min_val) - D
			hi = math.ceil(max_val) + D

		# Clamp search range to something reasonable
		lo = max(lo, -1000)
		hi = min(hi, 1000)

		for off_int in range(lo, hi + 1):
			offset = float(off_int)
			target = offset + d_idx * direction  # (D, 1, 1, 1) broadcast
			diff = (masked_sampled - target * mask)
			mse = (diff ** 2).sum().item() / wsum
			if mse < best_mse:
				best_mse = mse
				best_offset = offset
				best_dir = direction

	_winding_offset = best_offset
	_winding_direction = best_dir
	return _winding_offset, _winding_direction


def winding_volume_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Loss penalizing mesh winding deviation from ground-truth winding volume.

	Samples the winding volume at coarse mesh positions via F.grid_sample
	(float32, natively differentiable).

	If auto-offset has been computed, uses target = offset + d * direction.
	Otherwise falls back to target = d + 1 (1-indexed).

	Returns (loss, (lm,), (mask,)).
	"""
	sampled = _sample_winding_volume(res=res)
	D = sampled.shape[0]
	dev = sampled.device

	# Build target based on auto-offset state
	if _winding_offset is not None:
		target = (_winding_offset + torch.arange(D, device=dev, dtype=torch.float32) * _winding_direction).view(D, 1, 1, 1).expand_as(sampled)
	else:
		# Default: depth d (0-indexed) → winding d+1 (1-indexed)
		target = torch.arange(1, D + 1, device=dev, dtype=torch.float32).view(D, 1, 1, 1).expand_as(sampled)

	lm = (sampled - target) ** 2

	# Mask: mesh validity AND winding >= 1 (0 = out-of-bounds / invalid from padding_mode='zeros')
	wv_valid = (sampled.detach() >= 1.0).float()
	mask = res.mask_lr * wv_valid

	wsum = mask.sum().clamp(min=1)
	loss = (lm * mask).sum() / wsum

	return loss, (lm,), (mask,)
