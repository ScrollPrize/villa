from __future__ import annotations

import torch

import fit_data
import model as fit_model


def _encode_dir(gx: torch.Tensor, gy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	"""Encode a 2D direction vector (gx, gy) into the (dir0, dir1) representation."""
	eps = 1e-8
	r2 = gx * gx + gy * gy + eps
	cos2 = (gx * gx - gy * gy) / r2
	sin2 = 2.0 * gx * gy / r2
	inv_sqrt2 = 1.0 / (2.0 ** 0.5)
	d0 = 0.5 + 0.5 * cos2
	d1 = 0.5 + 0.5 * ((cos2 - sin2) * inv_sqrt2)
	return d0, d1


def _quad_face_normals(xyz_lr: torch.Tensor) -> torch.Tensor:
	"""Compute unit face normals from quad mesh.

	xyz_lr: (D, Hm, Wm, 3) -> normals: (D, Hm-1, Wm-1, 3)
	"""
	# Edge vectors along height and width
	e_h = xyz_lr[:, 1:, :-1, :] - xyz_lr[:, :-1, :-1, :]  # along height
	e_w = xyz_lr[:, :-1, 1:, :] - xyz_lr[:, :-1, :-1, :]  # along width
	n = torch.cross(e_h, e_w, dim=-1)
	return n * torch.rsqrt((n * n).sum(dim=-1, keepdim=True) + 1e-12)


def dir_loss_maps(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, torch.Tensor]:
	"""Compute direction loss maps from quad face normals vs per-axis direction channels.

	Returns (lm, mask) both of shape (D, 1, Hm-1, Wm-1).
	"""
	normal = _quad_face_normals(res.xyz_lr)  # (D, Hm-1, Wm-1, 3)
	nx = normal[..., 0]
	ny = normal[..., 1]
	nz = normal[..., 2]

	# Sample dir channels at face centers (average of 4 corner positions)
	face_centers = 0.25 * (
		res.xyz_lr[:, :-1, :-1] + res.xyz_lr[:, 1:, :-1] +
		res.xyz_lr[:, :-1, 1:] + res.xyz_lr[:, 1:, 1:]
	)  # (D, Hm-1, Wm-1, 3)

	data_at_faces = res.data.grid_sample_fullres(face_centers)

	lm = torch.zeros_like(nx)
	w_total = torch.zeros_like(nx)

	# Weight each axis by the normal's in-plane projection magnitude:
	# high when the slicing plane cuts the surface edge-on (reliable direction data),
	# low when face-on (degenerate direction data).

	# z-axis (XY plane): project normal -> (nx, ny), weight by sqrt(nx²+ny²)
	if data_at_faces.dir0_z is not None:
		d0, d1 = _encode_dir(nx, ny)
		data_d0 = data_at_faces.dir0_z.squeeze(0).squeeze(0)  # (D, Hm-1, Wm-1)
		data_d1 = data_at_faces.dir1_z.squeeze(0).squeeze(0)
		w = nx * nx + ny * ny
		lm = lm + w * 0.5 * ((d0 - data_d0) ** 2 + (d1 - data_d1) ** 2)
		w_total = w_total + w

	# y-axis (XZ plane): project -> (nx, nz), weight by nx²+nz²
	if data_at_faces.dir0_y is not None:
		d0, d1 = _encode_dir(nx, nz)
		data_d0 = data_at_faces.dir0_y.squeeze(0).squeeze(0)
		data_d1 = data_at_faces.dir1_y.squeeze(0).squeeze(0)
		w = nx * nx + nz * nz
		lm = lm + w * 0.5 * ((d0 - data_d0) ** 2 + (d1 - data_d1) ** 2)
		w_total = w_total + w

	# x-axis (YZ plane): project -> (ny, nz), weight by ny²+nz²
	if data_at_faces.dir0_x is not None:
		d0, d1 = _encode_dir(ny, nz)
		data_d0 = data_at_faces.dir0_x.squeeze(0).squeeze(0)
		data_d1 = data_at_faces.dir1_x.squeeze(0).squeeze(0)
		w = ny * ny + nz * nz
		lm = lm + w * 0.5 * ((d0 - data_d0) ** 2 + (d1 - data_d1) ** 2)
		w_total = w_total + w

	lm = lm / (w_total + 1e-8)

	# Mask from valid channel at face centers
	if data_at_faces.valid is not None:
		mask = (data_at_faces.valid.squeeze(0).squeeze(0) > 0.5).to(dtype=torch.float32)
	else:
		mask = torch.ones_like(lm)

	return lm.unsqueeze(1), mask.unsqueeze(1)  # (D, 1, Hm-1, Wm-1)


def dir_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Direction loss: quad face normal alignment with per-axis direction channels."""
	lm, mask = dir_loss_maps(res=res)
	wsum = mask.sum()
	loss = lm.sum() / wsum if float(wsum.detach().cpu()) > 0.0 else lm.mean()
	return loss, (lm,), (mask,)
