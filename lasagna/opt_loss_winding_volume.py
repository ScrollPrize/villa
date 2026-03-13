from __future__ import annotations

import torch
import torch.nn.functional as F

import model as fit_model


def winding_volume_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Loss penalizing mesh winding deviation from ground-truth winding volume.

	Samples the winding volume at coarse mesh positions via F.grid_sample
	(float32, natively differentiable). Target: depth d → winding d+1 (1-indexed).

	Returns (loss, (lm,), (mask,)).
	"""
	wv = res.data.winding_volume
	if wv is None:
		raise RuntimeError("winding_vol loss requested but FitData3D.winding_volume is None")

	dev = res.xyz_lr.device
	D, Hm, Wm, _ = res.xyz_lr.shape
	Z, Y, X = wv.shape[2], wv.shape[3], wv.shape[4]

	ox, oy, oz = res.data.origin_fullres
	sx, sy, sz = res.data.spacing

	# Transform fullres coords → normalized [-1, 1] for grid_sample
	grid = res.xyz_lr.clone()  # (D, Hm, Wm, 3) — (x, y, z) in fullres
	grid[..., 0] = (grid[..., 0] - ox) / sx / max(1, X - 1) * 2 - 1
	grid[..., 1] = (grid[..., 1] - oy) / sy / max(1, Y - 1) * 2 - 1
	grid[..., 2] = (grid[..., 2] - oz) / sz / max(1, Z - 1) * 2 - 1

	# F.grid_sample expects (N, C, D, H, W) input and (N, D_out, H_out, W_out, 3) grid
	# with grid last dim = (x, y, z) mapping to (W, H, D) axes of the volume.
	# Our winding volume is (1, 1, Z, Y, X) and grid xyz is (x→X, y→Y, z→Z),
	# so the grid ordering matches F.grid_sample's convention.
	sampled = F.grid_sample(
		wv, grid.unsqueeze(0),  # (1, 1, Z, Y, X), (1, D, Hm, Wm, 3)
		mode='bilinear', padding_mode='zeros', align_corners=True,
	)
	# sampled: (1, 1, D, Hm, Wm) → (D, 1, Hm, Wm)
	sampled = sampled[0].permute(1, 0, 2, 3)

	# Target: depth d (0-indexed) → winding d+1 (1-indexed)
	target = torch.arange(1, D + 1, device=dev, dtype=torch.float32).view(D, 1, 1, 1).expand_as(sampled)

	lm = (sampled - target) ** 2

	# Mask: mesh validity AND winding >= 1 (0 = out-of-bounds / invalid from padding_mode='zeros')
	wv_valid = (sampled.detach() >= 1.0).float()
	mask = res.mask_lr * wv_valid

	wsum = mask.sum().clamp(min=1)
	loss = (lm * mask).sum() / wsum

	return loss, (lm,), (mask,)
