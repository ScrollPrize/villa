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


def _decode_dir(d0: torch.Tensor, d1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	"""Decode (d0, d1) double-angle encoding to (cos θ, sin θ) direction.

	The encoding maps direction angle θ to:
	  d0 = 0.5 + 0.5 * cos(2θ)
	  d1 = 0.5 + 0.5 * cos(2θ + π/4)

	Returns decoded half-angle direction (cos θ, sin θ), with inherent π ambiguity.
	"""
	cos2 = 2.0 * d0 - 1.0
	sin2 = cos2 - (2.0 * d1 - 1.0) * (2.0 ** 0.5)
	angle = torch.atan2(sin2, cos2) * 0.5
	return torch.cos(angle), torch.sin(angle)


def normal_loss_maps(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, torch.Tensor]:
	"""Reconstruct target 3D normal from per-plane direction channels and compare
	to mesh surface normal.

	Each plane's direction defines a linear constraint on the normal: the cross
	product of the projected normal with the data direction should be zero.
	This gives one row per plane in a homogeneous system A·n = 0.  The target
	normal is the cross product of two constraint rows (= null vector of A).

	When 3 planes are available, picks the best-conditioned pair per pixel.

	Returns (lm, mask) both of shape (D, 1, Hm-1, Wm-1).
	"""
	normal = _quad_face_normals(res.xyz_lr)  # (D, Hm-1, Wm-1, 3)

	face_centers = 0.25 * (
		res.xyz_lr[:, :-1, :-1] + res.xyz_lr[:, 1:, :-1] +
		res.xyz_lr[:, :-1, 1:] + res.xyz_lr[:, 1:, 1:]
	)
	data = res.data.grid_sample_fullres(face_centers)

	has_y = data.dir0_y is not None
	has_x = data.dir0_x is not None

	if not has_y and not has_x:
		# Only z-plane: can't reconstruct 3D normal
		lm = torch.zeros_like(normal[..., 0])
		return lm.unsqueeze(1), torch.zeros_like(lm).unsqueeze(1)

	# Decode z-plane direction: (cosθ_z, sinθ_z) ∝ (nx, ny)
	cz, sz = _decode_dir(
		data.dir0_z.squeeze(0).squeeze(0),
		data.dir1_z.squeeze(0).squeeze(0),
	)

	# Constraint rows (null-space formulation A·n = 0):
	#   Z: (sinθ_z, -cosθ_z, 0)     — from (nx,ny) direction
	#   Y: (sinθ_y,  0, -cosθ_y)    — from (nx,nz) direction
	#   X: (0,  sinθ_x, -cosθ_x)    — from (ny,nz) direction
	# Target normal = cross product of two rows

	targets = []
	mags_sq = []

	if has_y:
		cy, sy = _decode_dir(
			data.dir0_y.squeeze(0).squeeze(0),
			data.dir1_y.squeeze(0).squeeze(0),
		)
		# Z×Y = (cz·cy, sz·cy, cz·sy)
		t = torch.stack([cz * cy, sz * cy, cz * sy], dim=-1)
		targets.append(t)
		mags_sq.append((t * t).sum(dim=-1))

	if has_x:
		cx, sx = _decode_dir(
			data.dir0_x.squeeze(0).squeeze(0),
			data.dir1_x.squeeze(0).squeeze(0),
		)
		# Z×X = (cz·cx, sz·cx, sz·sx)
		t = torch.stack([cz * cx, sz * cx, sz * sx], dim=-1)
		targets.append(t)
		mags_sq.append((t * t).sum(dim=-1))

	if has_y and has_x:
		# Y×X = (cy·sx, sy·cx, sy·sx)
		t = torch.stack([cy * sx, sy * cx, sy * sx], dim=-1)
		targets.append(t)
		mags_sq.append((t * t).sum(dim=-1))

	if len(targets) == 1:
		target = targets[0]
	else:
		# Pick best-conditioned pair per pixel (largest cross product magnitude)
		mags = torch.stack(mags_sq, dim=-1)
		best = mags.argmax(dim=-1)
		targets_stacked = torch.stack(targets, dim=-2)  # (..., N, 3)
		idx = best.unsqueeze(-1).unsqueeze(-1).expand(*best.shape, 1, 3)
		target = targets_stacked.gather(-2, idx).squeeze(-2)

	# Normalize target
	tn = torch.sqrt((target * target).sum(dim=-1, keepdim=True) + 1e-12)
	target_unit = target / tn

	# Loss: 1 - |dot(normal, target)|  via detached sign for clean gradients
	dot = (normal * target_unit).sum(dim=-1)
	sign = torch.where(dot >= 0, torch.ones_like(dot), -torch.ones_like(dot)).detach()
	lm = 1.0 - dot * sign

	# Mask: valid data + target well-conditioned
	mask = (tn.squeeze(-1) > 0.01).to(dtype=normal.dtype)
	if data.valid is not None:
		mask = mask * (data.valid.squeeze(0).squeeze(0) > 0.5).to(dtype=normal.dtype)

	return lm.unsqueeze(1), mask.unsqueeze(1)


def normal_loss(*, res: fit_model.FitResult3D) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
	"""Normal direction loss: mesh normal vs target normal reconstructed from direction data."""
	lm, mask = normal_loss_maps(res=res)
	wsum = mask.sum()
	loss = (lm * mask).sum() / wsum if float(wsum.detach().cpu()) > 0.0 else lm.mean()
	return loss, (lm,), (mask,)
