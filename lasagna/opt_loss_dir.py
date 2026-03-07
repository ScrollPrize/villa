from __future__ import annotations

import torch

import model as fit_model


def _quad_face_normals(xyz_lr: torch.Tensor) -> torch.Tensor:
	"""Compute unit face normals from quad mesh.

	xyz_lr: (D, Hm, Wm, 3) -> normals: (D, Hm-1, Wm-1, 3)
	"""
	# Edge vectors along height and width
	e_h = xyz_lr[:, 1:, :-1, :] - xyz_lr[:, :-1, :-1, :]  # along height
	e_w = xyz_lr[:, :-1, 1:, :] - xyz_lr[:, :-1, :-1, :]  # along width
	n = torch.cross(e_h, e_w, dim=-1)
	return n * torch.rsqrt((n * n).sum(dim=-1, keepdim=True) + 1e-12)


def normal_loss_maps(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, torch.Tensor]:
	"""Compare mesh surface normal to stored data normal.

	Data normal is stored as hemisphere-encoded (nx, ny); nz is reconstructed
	on the fly as sqrt(max(0, 1 - nx² - ny²)) ≥ 0.

	Loss: 1 - |dot(mesh_normal, data_normal)|
	Mask: grad_mag > 0 at face centers.

	Returns (lm, mask) both of shape (D, 1, Hm-1, Wm-1).
	"""
	normal = _quad_face_normals(res.xyz_lr)  # (D, Hm-1, Wm-1, 3)

	face_centers = 0.25 * (
		res.xyz_lr[:, :-1, :-1] + res.xyz_lr[:, 1:, :-1] +
		res.xyz_lr[:, :-1, 1:] + res.xyz_lr[:, 1:, 1:]
	)
	data = res.data.grid_sample_fullres(face_centers)

	# Reconstruct data normal: stored (nx, ny) → nz = sqrt(1 - nx² - ny²)
	data_nx = data.nx.squeeze(0).squeeze(0)  # (D, Hm-1, Wm-1)
	data_ny = data.ny.squeeze(0).squeeze(0)
	data_nz = torch.sqrt(torch.clamp(1.0 - data_nx * data_nx - data_ny * data_ny, min=0.0))
	target = torch.stack([data_nx, data_ny, data_nz], dim=-1)  # (D, Hm-1, Wm-1, 3)

	# Loss: 1 - dot² = sin²(θ), sign-invariant, ≈ θ² for small angles
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


