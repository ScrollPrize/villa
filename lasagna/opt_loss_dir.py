from __future__ import annotations

import torch

import model as fit_model


def _vertex_normals(xyz_lr: torch.Tensor) -> torch.Tensor:
	"""Compute unit vertex normals from quad mesh grid.

	Central differences for interior vertices, one-sided for edges.
	xyz_lr: (D, Hm, Wm, 3) -> normals: (D, Hm, Wm, 3)
	"""
	# h-direction tangent
	dh = torch.cat([
		xyz_lr[:, 1:2, :, :] - xyz_lr[:, 0:1, :, :],
		xyz_lr[:, 2:, :, :] - xyz_lr[:, :-2, :, :],
		xyz_lr[:, -1:, :, :] - xyz_lr[:, -2:-1, :, :],
	], dim=1)
	# w-direction tangent
	dw = torch.cat([
		xyz_lr[:, :, 1:2, :] - xyz_lr[:, :, 0:1, :],
		xyz_lr[:, :, 2:, :] - xyz_lr[:, :, :-2, :],
		xyz_lr[:, :, -1:, :] - xyz_lr[:, :, -2:-1, :],
	], dim=2)
	n = torch.cross(dh, dw, dim=-1)
	return n * torch.rsqrt((n * n).sum(dim=-1, keepdim=True) + 1e-12)


def normal_loss_maps(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, torch.Tensor]:
	"""Compare per-vertex mesh normal to stored data normal.

	Vertex normals: central differences interior, one-sided at edges.
	Data normal is hemisphere-encoded (nx, ny); nz = sqrt(1 - nx² - ny²).

	Loss: 1 - dot²(vertex_normal, data_normal).
	Mask: grad_mag > 0 at vertex positions.

	Returns (lm, mask) both of shape (D, 1, Hm, Wm).
	"""
	normal = _vertex_normals(res.xyz_lr)  # (D, Hm, Wm, 3)

	# Sample data at vertex positions (detached — pure target)
	data = res.data.grid_sample_fullres(res.xyz_lr.detach())

	# Reconstruct data normal from hemisphere encoding
	data_nx = data.nx.squeeze(0).squeeze(0)  # (D, Hm, Wm)
	data_ny = data.ny.squeeze(0).squeeze(0)
	data_nz = torch.sqrt(torch.clamp(1.0 - data_nx * data_nx - data_ny * data_ny, min=1e-8))
	target = torch.stack([data_nx, data_ny, data_nz], dim=-1)  # (D, Hm, Wm, 3)

	# Loss: 1 - dot² = sin²(θ), sign-invariant
	dot = (normal * target).sum(dim=-1)
	lm = 1.0 - dot * dot

	# Mask: grad_mag > 0
	mask = (data.grad_mag.squeeze(0).squeeze(0) > 0.0).to(dtype=normal.dtype)

	return lm.unsqueeze(1), mask.unsqueeze(1)


def normal_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Normal direction loss: mesh normal vs stored data normal."""
	lm, mask = normal_loss_maps(res=res)
	wsum = mask.sum()
	loss = (lm * mask).sum() / wsum if float(wsum.detach().cpu()) > 0.0 else lm.mean()
	return loss, (lm,), (mask,)
