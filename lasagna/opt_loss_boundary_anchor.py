from __future__ import annotations

import torch

import model as fit_model


def boundary_anchor_loss(
	*, res: fit_model.FitResult3D
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Directly anchor valid tifxyz vertices on hole boundaries."""
	if not res.boundary_anchors:
		z = torch.zeros((), device=res.xyz_lr.device, dtype=res.xyz_lr.dtype)
		return z, (z.unsqueeze(0),), (z.unsqueeze(0),)

	total = torch.zeros((), device=res.xyz_lr.device, dtype=res.xyz_lr.dtype)
	all_lm = []
	all_mask = []
	n_active = 0

	for target, mask in res.boundary_anchors:
		if target.shape != res.xyz_lr.shape[1:] or mask.shape != res.xyz_lr.shape[1:3]:
			continue
		target_d = target.unsqueeze(0).expand(res.xyz_lr.shape[0], -1, -1, -1)
		mask_d = mask.to(device=res.xyz_lr.device, dtype=res.xyz_lr.dtype)
		mask_d = mask_d.unsqueeze(0).expand(res.xyz_lr.shape[0], -1, -1).unsqueeze(1)
		lm = (res.xyz_lr - target_d).square().sum(dim=-1).unsqueeze(1)
		wsum = mask_d.sum()
		if float(wsum.detach().cpu()) <= 0.0:
			continue
		total = total + (lm * mask_d).sum() / wsum
		all_lm.append(lm)
		all_mask.append(mask_d)
		n_active += 1

	if n_active == 0:
		z = torch.zeros((), device=res.xyz_lr.device, dtype=res.xyz_lr.dtype)
		return z, (z.unsqueeze(0),), (z.unsqueeze(0),)

	return total / n_active, (sum(all_lm) / n_active,), (sum(all_mask).clamp(max=1.0),)
