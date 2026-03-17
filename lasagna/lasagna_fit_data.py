"""Fit dense 3D data volumes (normals + winding) to a frozen model.

Given a fitted lasagna model, creates synthetic dense 3D data volumes that
match the model's geometry. This is the inverse of model fitting: the model
is frozen, the data is optimized. Useful for label conversion — generating
the normal/winding field that a given surface model implies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr

import fit_data
import model as fit_model
import opt_loss_dir
import opt_loss_winding_volume


# ---------------------------------------------------------------------------
# Normal visualization
# ---------------------------------------------------------------------------

def _write_normal_vis_slices(
	out: Path,
	final_normal: np.ndarray,
	final_winding: np.ndarray,
	*,
	validity: np.ndarray | None = None,
	density: np.ndarray | None = None,
	labels: np.ndarray | None = None,
	ds: int = 4,
	z_step: int = 32,
	line_step: int = 4,
	label_alpha: float = 0.3,
) -> None:
	"""Write JPEG images of reconstructed normals projected into XY slices.

	Composited at labels' fullres resolution if labels are provided,
	otherwise at ds * volume resolution.

	Background: winding volume (min/max normalized per slice).
	Original TIF labels overlaid transparently if provided.
	Reconstructed normals drawn as short red lines.

	final_normal: (3, Z, Y, X) float32 unit normals
	final_winding: (Z, Y, X) float32 winding values
	labels: (Zl, Yl, Xl) uint8 original labels at fullres (0=bg, 1=pred, 2=ignore)
	ds: downscale factor from fullres to volume resolution
	"""
	import cv2
	from skimage.measure import find_contours

	nx_vol = final_normal[0]  # (Z, Y, X) float [-1, 1]
	ny_vol = final_normal[1]  # (Z, Y, X) float [-1, 1]
	Z, Y, X = nx_vol.shape

	# Output resolution: labels res if available, else ds * volume res
	if labels is not None:
		Zl, Yl, Xl = labels.shape
	else:
		Zl, Yl, Xl = Z * ds, Y * ds, X * ds

	# Mask: normal magnitude > threshold
	norm_mag = np.sqrt(nx_vol**2 + ny_vol**2)

	# Scale factor from volume coords to output pixel coords
	sy_out = Yl / Y
	sx_out = Xl / X
	line_half = max(sy_out, sx_out) * 1.5

	for z in range(0, Z, z_step):
		# Map volume z to labels z
		zl = int(z * Zl / Z)

		# Background: winding volume normalized per-slice, upsampled to output res
		wslice = final_winding[z]  # (Y, X)
		wmin, wmax = float(wslice.min()), float(wslice.max())
		if wmax - wmin < 1e-8:
			wmax = wmin + 1.0
		bg = ((wslice - wmin) / (wmax - wmin) * 255.0).clip(0, 255).astype(np.uint8)

		img = cv2.resize(
			cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR),
			(Xl, Yl),
			interpolation=cv2.INTER_NEAREST,
		)

		# Overlay original labels transparently
		if labels is not None:
			lbl = labels[zl]  # (Yl, Xl) uint8: 0=bg, 1=pred, 2=ignore
			label_rgb = np.zeros((Yl, Xl, 3), dtype=np.uint8)
			label_rgb[lbl == 1] = [0, 255, 0]    # green = prediction
			label_rgb[lbl == 2] = [0, 0, 255]    # red = ignore
			mask3 = (lbl > 0)[:, :, None]
			blended = (img.astype(np.float32) * (1 - label_alpha)
					   + label_rgb.astype(np.float32) * label_alpha).clip(0, 255).astype(np.uint8)
			img = np.where(mask3, blended, img)

		# Draw iso-winding contour lines at integer levels
		wmin_int = int(np.ceil(wslice.min()))
		wmax_int = int(np.floor(wslice.max()))
		for level in range(wmin_int, wmax_int + 1):
			contours = find_contours(wslice, level)
			for contour in contours:
				pts = contour[:, ::-1]  # (N, 2) as (x, y)
				pts[:, 0] = pts[:, 0] * sx_out + sx_out / 2
				pts[:, 1] = pts[:, 1] * sy_out + sy_out / 2
				pts_int = pts.astype(np.int32).reshape(-1, 1, 2)
				cv2.polylines(img, [pts_int], isClosed=False, color=(255, 255, 0), thickness=1)

		# Draw reconstructed normals as red lines at output resolution
		nxf = nx_vol[z]
		nyf = ny_vol[z]
		mag = norm_mag[z]
		for y in range(0, Y, line_step):
			for x in range(0, X, line_step):
				if mag[y, x] < 0.1:
					continue
				cx = int(x * sx_out + sx_out / 2)
				cy = int(y * sy_out + sy_out / 2)
				dx = float(nxf[y, x]) * line_half
				dy = float(nyf[y, x]) * line_half
				pt1 = (int(cx - dx), int(cy - dy))
				pt2 = (int(cx + dx), int(cy + dy))
				cv2.line(img, pt1, pt2, (0, 0, 255), 1)

		fname = f"normals_z{z:04d}.jpg"
		cv2.imwrite(str(out / fname), img)
		print(f"[fit_data] wrote {fname}", flush=True)

		# Write validity TIF slice
		if validity is not None:
			v_slice = (validity[z].clip(0, 1) * 255).astype(np.uint8)
			v_fname = f"validity_z{z:04d}.tif"
			tifffile.imwrite(str(out / v_fname), v_slice)
			print(f"[fit_data] wrote {v_fname}", flush=True)

		# Write density TIF slice (float32)
		if density is not None:
			d_fname = f"density_z{z:04d}.tif"
			tifffile.imwrite(str(out / d_fname), density[z].astype(np.float32))
			print(f"[fit_data] wrote {d_fname}", flush=True)


# ---------------------------------------------------------------------------
# Residual pyramid for 3D volumes
# ---------------------------------------------------------------------------

def _pyramid_shapes(Z: int, Y: int, X: int) -> list[tuple[int, int, int]]:
	"""Compute pyramid level shapes until all dims reach 1."""
	shapes = [(Z, Y, X)]
	while shapes[-1] != (1, 1, 1):
		z, y, x = shapes[-1]
		shapes.append((max(1, (z + 1) // 2), max(1, (y + 1) // 2), max(1, (x + 1) // 2)))
	return shapes


class DataVolume(nn.Module):
	"""Residual pyramid representation of normal + winding + validity + density 3D volumes."""

	def __init__(self, normal_pyr: nn.ParameterList, winding_pyr: nn.ParameterList,
				 validity_pyr: nn.ParameterList, density_pyr: nn.ParameterList) -> None:
		super().__init__()
		self.normal_pyr = normal_pyr
		self.winding_pyr = winding_pyr
		self.validity_pyr = validity_pyr
		self.density_pyr = density_pyr

	@staticmethod
	def _integrate(pyr: nn.ParameterList) -> torch.Tensor:
		"""Integrate residual pyramid: coarsest → finest via upsample + add."""
		v = pyr[-1]
		for p in reversed(list(pyr[:-1])):
			v = F.interpolate(
				v.unsqueeze(0),
				size=(p.shape[1], p.shape[2], p.shape[3]),
				mode='trilinear', align_corners=True,
			).squeeze(0) + p
		return v

	def integrate_normal(self) -> torch.Tensor:
		"""Integrate normal pyramid and normalize per voxel. Returns (3, Z, Y, X)."""
		raw = self._integrate(self.normal_pyr)  # (3, Z, Y, X)
		return F.normalize(raw, dim=0, eps=1e-8)

	def integrate_winding(self) -> torch.Tensor:
		"""Integrate winding pyramid. Returns (1, Z, Y, X)."""
		return self._integrate(self.winding_pyr)

	def integrate_validity(self) -> torch.Tensor:
		"""Integrate validity pyramid. Returns (1, Z, Y, X)."""
		return self._integrate(self.validity_pyr)

	def integrate_density(self) -> torch.Tensor:
		"""Integrate density pyramid. Returns (1, Z, Y, X)."""
		return self._integrate(self.density_pyr)

	@classmethod
	def from_initial(
		cls,
		normal_vol: torch.Tensor,
		winding_vol: torch.Tensor,
		validity_vol: torch.Tensor,
		density_vol: torch.Tensor,
		count: torch.Tensor,
		device: torch.device,
	) -> 'DataVolume':
		"""Build from scattered mesh data via valid-mean-pool pyramid.

		normal_vol: (3, Z, Y, X) sum of normals at each voxel
		winding_vol: (1, Z, Y, X) sum of winding values
		validity_vol: (1, Z, Y, X) sum of validity values
		density_vol: (1, Z, Y, X) sum of density values
		count: (1, Z, Y, X) number of vertices per voxel
		"""
		valid = count > 0
		# Average where we have data
		normal_avg = torch.where(valid, normal_vol / count.clamp(min=1), torch.zeros_like(normal_vol))
		# Normalize normals where valid
		n_len = normal_avg.norm(dim=0, keepdim=True).clamp(min=1e-8)
		normal_avg = torch.where(valid, normal_avg / n_len, normal_avg)
		winding_avg = torch.where(valid, winding_vol / count.clamp(min=1), torch.zeros_like(winding_vol))
		validity_avg = torch.where(valid, validity_vol / count.clamp(min=1), torch.zeros_like(validity_vol))
		density_avg = torch.where(valid, density_vol / count.clamp(min=1), torch.zeros_like(density_vol))

		C_n, Z, Y, X = normal_avg.shape
		shapes = _pyramid_shapes(Z, Y, X)

		# Build target values at each pyramid level via valid mean pooling
		normal_targets = [normal_avg]
		winding_targets = [winding_avg]
		validity_targets = [validity_avg]
		density_targets = [density_avg]
		count_levels = [count]

		for lvl_shape in shapes[1:]:
			prev_n = normal_targets[-1]
			prev_w = winding_targets[-1]
			prev_v = validity_targets[-1]
			prev_d = density_targets[-1]
			prev_c = count_levels[-1]
			pz, py, px = prev_n.shape[1], prev_n.shape[2], prev_n.shape[3]
			tz, ty, tx = lvl_shape

			# Pad to even dims
			pad_z = pz % 2
			pad_y = py % 2
			pad_x = px % 2
			if pad_z or pad_y or pad_x:
				prev_n = F.pad(prev_n, (0, pad_x, 0, pad_y, 0, pad_z))
				prev_w = F.pad(prev_w, (0, pad_x, 0, pad_y, 0, pad_z))
				prev_v = F.pad(prev_v, (0, pad_x, 0, pad_y, 0, pad_z))
				prev_d = F.pad(prev_d, (0, pad_x, 0, pad_y, 0, pad_z))
				prev_c = F.pad(prev_c, (0, pad_x, 0, pad_y, 0, pad_z))

			# Reshape into 2x2x2 blocks and sum
			ez, ey, ex = prev_n.shape[1], prev_n.shape[2], prev_n.shape[3]
			hz, hy, hx = ez // 2, ey // 2, ex // 2

			def _pool_sum(t: torch.Tensor) -> torch.Tensor:
				C = t.shape[0]
				t = t.view(C, hz, 2, hy, 2, hx, 2)
				return t.sum(dim=(2, 4, 6))

			n_sum = _pool_sum(prev_n)
			w_sum = _pool_sum(prev_w)
			v_sum = _pool_sum(prev_v)
			d_sum = _pool_sum(prev_d)
			c_sum = _pool_sum(prev_c)

			# Trim to target size (in case of rounding)
			n_sum = n_sum[:, :tz, :ty, :tx]
			w_sum = w_sum[:, :tz, :ty, :tx]
			v_sum = v_sum[:, :tz, :ty, :tx]
			d_sum = d_sum[:, :tz, :ty, :tx]
			c_sum = c_sum[:, :tz, :ty, :tx]

			c_valid = c_sum > 0
			n_avg = torch.where(c_valid, n_sum / c_sum.clamp(min=1), torch.zeros_like(n_sum))
			n_len = n_avg.norm(dim=0, keepdim=True).clamp(min=1e-8)
			n_avg = torch.where(c_valid[:1], n_avg / n_len, n_avg)
			w_avg = torch.where(c_valid, w_sum / c_sum.clamp(min=1), torch.zeros_like(w_sum))
			v_avg = torch.where(c_valid, v_sum / c_sum.clamp(min=1), torch.zeros_like(v_sum))
			d_avg = torch.where(c_valid, d_sum / c_sum.clamp(min=1), torch.zeros_like(d_sum))

			normal_targets.append(n_avg)
			winding_targets.append(w_avg)
			validity_targets.append(v_avg)
			density_targets.append(d_avg)
			count_levels.append(c_sum)

		# Convert to residual encoding
		n_levels = len(normal_targets)
		normal_residuals = [None] * n_levels
		winding_residuals = [None] * n_levels
		validity_residuals = [None] * n_levels
		density_residuals = [None] * n_levels

		# Coarsest level is absolute
		normal_residuals[-1] = normal_targets[-1]
		winding_residuals[-1] = winding_targets[-1]
		validity_residuals[-1] = validity_targets[-1]
		density_residuals[-1] = density_targets[-1]

		# Reconstruct from coarsest and compute residuals
		n_recon = normal_targets[-1]
		w_recon = winding_targets[-1]
		v_recon = validity_targets[-1]
		d_recon = density_targets[-1]
		for i in range(n_levels - 2, -1, -1):
			target_size = (normal_targets[i].shape[1], normal_targets[i].shape[2], normal_targets[i].shape[3])
			n_up = F.interpolate(
				n_recon.unsqueeze(0), size=target_size,
				mode='trilinear', align_corners=True,
			).squeeze(0)
			w_up = F.interpolate(
				w_recon.unsqueeze(0), size=target_size,
				mode='trilinear', align_corners=True,
			).squeeze(0)
			v_up = F.interpolate(
				v_recon.unsqueeze(0), size=target_size,
				mode='trilinear', align_corners=True,
			).squeeze(0)
			d_up = F.interpolate(
				d_recon.unsqueeze(0), size=target_size,
				mode='trilinear', align_corners=True,
			).squeeze(0)

			# Residual only at valid voxels; 0 at empty (interpolation fills in)
			c_valid = count_levels[i] > 0
			normal_residuals[i] = torch.where(c_valid, normal_targets[i] - n_up, torch.zeros_like(n_up))
			winding_residuals[i] = torch.where(c_valid, winding_targets[i] - w_up, torch.zeros_like(w_up))
			validity_residuals[i] = torch.where(c_valid, validity_targets[i] - v_up, torch.zeros_like(v_up))
			density_residuals[i] = torch.where(c_valid, density_targets[i] - d_up, torch.zeros_like(d_up))

			n_recon = n_up + normal_residuals[i]
			w_recon = w_up + winding_residuals[i]
			v_recon = v_up + validity_residuals[i]
			d_recon = d_up + density_residuals[i]

		normal_pyr = nn.ParameterList([nn.Parameter(r.to(device)) for r in normal_residuals])
		winding_pyr = nn.ParameterList([nn.Parameter(r.to(device)) for r in winding_residuals])
		validity_pyr = nn.ParameterList([nn.Parameter(r.to(device)) for r in validity_residuals])
		density_pyr = nn.ParameterList([nn.Parameter(r.to(device)) for r in density_residuals])

		return cls(normal_pyr, winding_pyr, validity_pyr, density_pyr)


# ---------------------------------------------------------------------------
# Scatter mesh data into volumes
# ---------------------------------------------------------------------------

def _scatter_mesh_to_volume(
	xyz_lr: torch.Tensor,
	mask_lr: torch.Tensor,
	normals: torch.Tensor,
	winding_targets: torch.Tensor,
	validity_targets: torch.Tensor,
	density_targets: torch.Tensor,
	origin: tuple[float, float, float],
	spacing: tuple[float, float, float],
	vol_shape: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Scatter mesh vertex data into volume grids.

	Returns (normal_sum, winding_sum, validity_sum, density_sum, count) each at vol_shape.
	"""
	D, Hm, Wm, _ = xyz_lr.shape
	Z, Y, X = vol_shape
	device = xyz_lr.device

	# Convert fullres xyz to voxel indices
	ox, oy, oz = origin
	sx, sy, sz = spacing
	vx = ((xyz_lr[..., 0] - ox) / sx).round().long()
	vy = ((xyz_lr[..., 1] - oy) / sy).round().long()
	vz = ((xyz_lr[..., 2] - oz) / sz).round().long()

	# Validity: mask and in-bounds
	mask = mask_lr.squeeze(1) > 0.5  # (D, Hm, Wm)
	in_bounds = (vx >= 0) & (vx < X) & (vy >= 0) & (vy < Y) & (vz >= 0) & (vz < Z)
	valid = mask & in_bounds

	# Flatten valid entries
	flat_idx = vz[valid] * Y * X + vy[valid] * X + vx[valid]  # linear index
	flat_normals = normals[valid]  # (N, 3)
	flat_winding = winding_targets.view(D, Hm, Wm)[valid]  # (N,)
	flat_validity = validity_targets.view(D, Hm, Wm)[valid]  # (N,)
	flat_density = density_targets.view(D, Hm, Wm)[valid]  # (N,)

	# Scatter into volumes
	normal_sum = torch.zeros(3, Z * Y * X, device=device, dtype=torch.float32)
	winding_sum = torch.zeros(1, Z * Y * X, device=device, dtype=torch.float32)
	validity_sum = torch.zeros(1, Z * Y * X, device=device, dtype=torch.float32)
	density_sum = torch.zeros(1, Z * Y * X, device=device, dtype=torch.float32)
	count = torch.zeros(1, Z * Y * X, device=device, dtype=torch.float32)

	for c in range(3):
		normal_sum[c].scatter_add_(0, flat_idx, flat_normals[:, c])
	winding_sum[0].scatter_add_(0, flat_idx, flat_winding)
	validity_sum[0].scatter_add_(0, flat_idx, flat_validity)
	density_sum[0].scatter_add_(0, flat_idx, flat_density)
	count[0].scatter_add_(0, flat_idx, torch.ones_like(flat_winding))

	return (
		normal_sum.view(3, Z, Y, X),
		winding_sum.view(1, Z, Y, X),
		validity_sum.view(1, Z, Y, X),
		density_sum.view(1, Z, Y, X),
		count.view(1, Z, Y, X),
	)


# ---------------------------------------------------------------------------
# Sampling and losses
# ---------------------------------------------------------------------------

def _build_sample_grid(
	xyz_lr: torch.Tensor,
	origin: tuple[float, float, float],
	spacing: tuple[float, float, float],
	vol_shape: tuple[int, int, int],
) -> torch.Tensor:
	"""Build normalized [-1,1] grid for F.grid_sample from mesh positions.

	Returns grid: (1, D, Hm, Wm, 3) matching F.grid_sample 5D convention.
	"""
	Z, Y, X = vol_shape
	ox, oy, oz = origin
	sx, sy, sz = spacing

	grid = xyz_lr.clone()  # (D, Hm, Wm, 3) in fullres (x, y, z)
	grid[..., 0] = (grid[..., 0] - ox) / sx / max(1, X - 1) * 2 - 1
	grid[..., 1] = (grid[..., 1] - oy) / sy / max(1, Y - 1) * 2 - 1
	grid[..., 2] = (grid[..., 2] - oz) / sz / max(1, Z - 1) * 2 - 1
	return grid.unsqueeze(0)  # (1, D, Hm, Wm, 3)


def _sample_volume(vol: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
	"""Sample a (C, Z, Y, X) volume at grid positions.

	Returns (D, C, Hm, Wm).
	"""
	out = F.grid_sample(
		vol.unsqueeze(0), grid,
		mode='bilinear', padding_mode='border', align_corners=True,
	)
	# out: (1, C, D, Hm, Wm) -> (D, C, Hm, Wm)
	return out[0].permute(1, 0, 2, 3)


def _normal_data_loss(
	sampled_n: torch.Tensor,
	mesh_normals: torch.Tensor,
	mask_lr: torch.Tensor,
) -> torch.Tensor:
	"""1 - dot^2(sampled_normal, mesh_normal) at valid positions.

	sampled_n: (D, 3, Hm, Wm), mesh_normals: (D, Hm, Wm, 3), mask_lr: (D, 1, Hm, Wm)
	"""
	# Reshape sampled to (D, Hm, Wm, 3)
	sn = sampled_n.permute(0, 2, 3, 1)
	sn = F.normalize(sn, dim=-1, eps=1e-8)
	mn = F.normalize(mesh_normals, dim=-1, eps=1e-8)
	dot = (sn * mn).sum(dim=-1)  # (D, Hm, Wm)
	loss_map = 1.0 - dot * dot  # (D, Hm, Wm)
	mask = mask_lr.squeeze(1)  # (D, Hm, Wm)
	wsum = mask.sum().clamp(min=1)
	return (loss_map * mask).sum() / wsum


def _winding_data_loss(
	sampled_w: torch.Tensor,
	winding_target: torch.Tensor,
	mask_lr: torch.Tensor,
) -> torch.Tensor:
	"""MSE between sampled winding and target at valid positions.

	sampled_w: (D, 1, Hm, Wm), winding_target: (D, 1, 1, 1) or broadcastable, mask_lr: (D, 1, Hm, Wm)
	"""
	diff = sampled_w - winding_target
	loss_map = diff * diff  # (D, 1, Hm, Wm)
	wsum = mask_lr.sum().clamp(min=1)
	return (loss_map * mask_lr).sum() / wsum


def _validity_data_loss(
	sampled_v: torch.Tensor,
	validity_target: torch.Tensor,
	mask_lr: torch.Tensor,
) -> torch.Tensor:
	"""MSE between sampled validity and target at valid positions.

	sampled_v: (D, 1, Hm, Wm), validity_target: (D, 1, Hm, Wm), mask_lr: (D, 1, Hm, Wm)
	"""
	diff = sampled_v - validity_target
	loss_map = diff * diff
	wsum = mask_lr.sum().clamp(min=1)
	return (loss_map * mask_lr).sum() / wsum


def _density_integral_loss(
	density_field: torch.Tensor,
	strip_grid: torch.Tensor,
	strip_len: torch.Tensor,
	conn_mask: torch.Tensor,
	strip_samples: int,
	D: int, Hm: int, Wm: int,
) -> torch.Tensor:
	"""Log² loss on density integral along strip. Target = 1.0.

	density_field: (1, Z, Y, X)
	strip_grid: (1, D, Hm, Wm*S, 3) normalized grid
	strip_len: (D, Hm, Wm) physical length of each strip
	conn_mask: (D, Hm, Wm) validity mask
	"""
	sampled = F.grid_sample(
		density_field.unsqueeze(0), strip_grid,
		mode='bilinear', padding_mode='border', align_corners=True,
	)
	# (1, 1, D, Hm, Wm*S) → (D, Hm, Wm, S)
	mag = sampled.squeeze(0).squeeze(0).reshape(D, Hm, Wm, strip_samples)
	integral = mag.mean(dim=-1) * strip_len  # (D, Hm, Wm)
	lm = torch.log(integral.clamp(min=1e-4)) ** 2
	wsum = conn_mask.sum().clamp(min=1)
	return (lm * conn_mask).sum() / wsum


def _density_average_loss(
	density_field: torch.Tensor,
	strip_grid: torch.Tensor,
	strip_len: torch.Tensor,
	conn_mask: torch.Tensor,
	strip_samples: int,
	D: int, Hm: int, Wm: int,
) -> torch.Tensor:
	"""MSE loss pushing each density sample toward 1/strip_length (uniform density)."""
	sampled = F.grid_sample(
		density_field.unsqueeze(0), strip_grid,
		mode='bilinear', padding_mode='border', align_corners=True,
	)
	mag = sampled.squeeze(0).squeeze(0).reshape(D, Hm, Wm, strip_samples)
	target = (1.0 / strip_len.clamp(min=0.1)).unsqueeze(-1)  # (D, Hm, Wm, 1)
	diff = mag - target
	mse = (diff * diff).mean(dim=-1)  # (D, Hm, Wm) — avg over S samples
	wsum = conn_mask.sum().clamp(min=1)
	return (mse * conn_mask).sum() / wsum


def _positivity_loss(vol: torch.Tensor) -> torch.Tensor:
	"""Mean squared penalty on negative values: mean(relu(-vol)²)."""
	neg = F.relu(-vol)
	return (neg * neg).mean()


def _validity_boundary_loss(validity_field: torch.Tensor) -> torch.Tensor:
	"""Penalize non-zero validity at all 6 faces of the volume.

	validity_field: (1, Z, Y, X) → scalar MSE of boundary voxels.
	"""
	# Collect all boundary slices (may overlap at corners/edges, that's fine)
	parts = [
		validity_field[:, 0, :, :],   # z=0
		validity_field[:, -1, :, :],  # z=-1
		validity_field[:, :, 0, :],   # y=0
		validity_field[:, :, -1, :],  # y=-1
		validity_field[:, :, :, 0],   # x=0
		validity_field[:, :, :, -1],  # x=-1
	]
	total = sum((p * p).mean() for p in parts)
	return total / len(parts)


def _finite_diff_loss(vol: torch.Tensor) -> torch.Tensor:
	"""Mean squared finite differences along Z, Y, X.

	vol: (C, Z, Y, X) → scalar.
	"""
	dz = vol[:, 1:] - vol[:, :-1]
	dy = vol[:, :, 1:] - vol[:, :, :-1]
	dx = vol[:, :, :, 1:] - vol[:, :, :, :-1]
	return (dz * dz).mean() + (dy * dy).mean() + (dx * dx).mean()


def _curvature_loss(vol: torch.Tensor) -> torch.Tensor:
	"""Mean squared 2nd-order finite differences (curvature) along Z, Y, X.

	Penalizes changes in the gradient, encouraging linear extrapolation
	beyond supervised regions.

	vol: (C, Z, Y, X) → scalar.
	"""
	dz = vol[:, 1:] - vol[:, :-1]
	dy = vol[:, :, 1:] - vol[:, :, :-1]
	dx = vol[:, :, :, 1:] - vol[:, :, :, :-1]
	d2z = dz[:, 1:] - dz[:, :-1]
	d2y = dy[:, :, 1:] - dy[:, :, :-1]
	d2x = dx[:, :, :, 1:] - dx[:, :, :, :-1]
	return (d2z * d2z).mean() + (d2y * d2y).mean() + (d2x * d2x).mean()


def _normal_smooth_loss(normal_field: torch.Tensor) -> torch.Tensor:
	"""Smoothing loss on normalized normals: finite diffs of unit normals."""
	return _finite_diff_loss(normal_field)


def _winding_smooth_loss(winding_field: torch.Tensor) -> torch.Tensor:
	"""Smoothing loss on winding: 1st-order finite differences only."""
	return _finite_diff_loss(winding_field)


# ---------------------------------------------------------------------------
# Main optimization
# ---------------------------------------------------------------------------

def fit_data_to_model(
	*,
	model_path: str,
	input_path: str,
	output_path: str | None = None,
	tif_output_dir: str | None = None,
	stats_json_path: str | None = None,
	n_steps: int = 1000,
	lr: float = 1e-6,
	w_normal: float = 1.0,
	w_winding: float = 10.0,
	w_smooth_n: float = 10.0,
	w_smooth_w: float = 10.0,
	w_curv_n: float = 10.0,
	w_curv_w: float = 10.0,
	w_validity: float = 10.0,
	w_smooth_v: float = 10.0,
	w_density: float = 10.0,
	w_smooth_d: float = 10.0,
	w_curv_d: float = 10.0,
	w_pos_d: float = 10.0,
	w_avg_d: float = 1.0,
	lr_exponent: float = 4.0,
	device_str: str = "cuda",
	log_interval: int = 20,
	normal_vis_dir: str | None = None,
	labels_path: str | None = None,
) -> None:
	dev = torch.device(device_str)

	# --- Load model (frozen) ---
	print(f"[fit_data] loading model from {model_path}", flush=True)
	st = torch.load(model_path, map_location=dev, weights_only=False)
	mdl = fit_model.Model3D.from_checkpoint(st, device=dev)
	mdl.eval()
	for p in mdl.parameters():
		p.requires_grad_(False)

	# --- Load data for spatial metadata ---
	print(f"[fit_data] loading spatial metadata from {input_path}", flush=True)
	data = fit_data.load_3d_for_model(path=input_path, device=dev, model=mdl,
									  cuda_gridsample=dev.type == "cuda")
	Z, Y, X = data.size
	origin = data.origin_fullres
	spacing = data.spacing
	print(f"[fit_data] volume shape: Z={Z}, Y={Y}, X={X}", flush=True)
	print(f"[fit_data] origin={origin}, spacing={spacing}", flush=True)

	# --- Erode volume validity (match optimizer setting) ---
	fit_cfg = st.get("_fit_config_", {})
	fit_args = fit_cfg.get("args", {}) if isinstance(fit_cfg, dict) else {}
	erode_valid_mask = int(fit_args.get("erode-valid-mask", fit_args.get("erode_valid_mask", 4)))
	if erode_valid_mask > 0:
		fit_data.erode_grad_mag(data, radius=erode_valid_mask)
		print(f"[fit_data] eroded valid mask by {erode_valid_mask} voxels", flush=True)

	# --- Forward pass through frozen model ---
	print("[fit_data] running model forward pass", flush=True)
	with torch.no_grad():
		xyz_lr = mdl._grid_xyz()  # (D, Hm, Wm, 3)
		mask_lr = (data.grid_sample_fullres(xyz_lr).grad_mag.squeeze(0).squeeze(0) > 0.0).to(dtype=torch.float32).unsqueeze(1)
		# (D, 1, Hm, Wm)

	D, Hm, Wm, _ = xyz_lr.shape
	print(f"[fit_data] mesh: D={D}, Hm={Hm}, Wm={Wm}", flush=True)

	# --- Compute mesh normals ---
	with torch.no_grad():
		mesh_normals = opt_loss_dir._vertex_normals(xyz_lr)  # (D, Hm, Wm, 3)

	# --- Compute winding targets ---
	# Restore winding offset from checkpoint
	if "_winding_offset_" in st:
		winding_offset = float(st["_winding_offset_"])
		winding_direction = int(st["_winding_direction_"])
		print(f"[fit_data] restored winding offset={winding_offset}, direction={winding_direction}", flush=True)
	else:
		winding_offset = 1.0
		winding_direction = 1
		print(f"[fit_data] using default winding offset={winding_offset}, direction={winding_direction}", flush=True)

	with torch.no_grad():
		d_idx = torch.arange(D, device=dev, dtype=torch.float32)
		winding_target = (winding_offset + d_idx * winding_direction).view(D, 1, 1, 1)
		# (D, 1, 1, 1)

	# --- Compute eroded model validity ---
	with torch.no_grad():
		kernel_2d = torch.ones(1, 1, 3, 3, device=dev)
		validity_target_mesh = (F.conv2d(mask_lr, kernel_2d, padding=1) >= 9).float()
		# (D, 1, Hm, Wm) — valid only if all 3x3 spatial neighbors valid

	# --- Crop model depth to layers present in volume ---
	with torch.no_grad():
		valid_per_d = validity_target_mesh.sum(dim=(1, 2, 3))  # (D,)
		has_data = valid_per_d >= 10
		if has_data.any():
			d_indices = torch.where(has_data)[0]
			d_lo = int(d_indices[0])
			d_hi = int(d_indices[-1]) + 1
		else:
			d_lo, d_hi = 0, D  # fallback: keep all

	if d_lo > 0 or d_hi < D:
		print(f"[fit_data] cropping model depth [{d_lo}, {d_hi}) — "
			  f"{D} -> {d_hi - d_lo} layers (removed {D - (d_hi - d_lo)} empty layers)")
		mdl.crop_depth(d_lo, d_hi)
		winding_offset = winding_offset + d_lo * winding_direction

		# Recompute all D-dependent tensors from cropped model
		with torch.no_grad():
			xyz_lr = mdl._grid_xyz()
			mask_lr = (data.grid_sample_fullres(xyz_lr).grad_mag.squeeze(0).squeeze(0) > 0.0
					   ).to(dtype=torch.float32).unsqueeze(1)
		D, Hm, Wm, _ = xyz_lr.shape
		print(f"[fit_data] cropped mesh: D={D}, Hm={Hm}, Wm={Wm}")

		with torch.no_grad():
			mesh_normals = opt_loss_dir._vertex_normals(xyz_lr)
			d_idx = torch.arange(D, device=dev, dtype=torch.float32)
			winding_target = (winding_offset + d_idx * winding_direction).view(D, 1, 1, 1)
			validity_target_mesh = (F.conv2d(mask_lr, kernel_2d, padding=1) >= 9).float()

	# --- Compute density strip geometry ---
	strip_samples = 4
	ox, oy, oz = origin
	sx, sy, sz = spacing
	with torch.no_grad():
		xy_conn, mask_conn, sign_conn, _ = mdl._xyz_conn(xyz_lr, data)
		prev_pt = xy_conn[:, :, :, :, 0]    # (D, Hm, Wm, 3)
		center_pt = xy_conn[:, :, :, :, 1]
		next_pt = xy_conn[:, :, :, :, 2]

		# Upsample strip endpoints to 4x mesh resolution in H, W for denser density supervision
		density_upsample = 4
		Hd, Wd = Hm * density_upsample, Wm * density_upsample
		prev_pt_dense = F.interpolate(
			prev_pt.permute(0, 3, 1, 2), size=(Hd, Wd),
			mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
		center_pt_dense = F.interpolate(
			center_pt.permute(0, 3, 1, 2), size=(Hd, Wd),
			mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
		next_pt_dense = F.interpolate(
			next_pt.permute(0, 3, 1, 2), size=(Hd, Wd),
			mode='bilinear', align_corners=True).permute(0, 2, 3, 1)

		# Midpoint positions (exclude endpoints on winding surfaces)
		t = (0.5 + torch.arange(strip_samples, device=dev)) / strip_samples

		def _build_strip_grid(start, end, H, W):
			"""Build normalized grid + strip lengths for F.grid_sample."""
			diff = end - start
			strip = start.unsqueeze(-2) + t.view(1, 1, 1, -1, 1) * diff.unsqueeze(-2)
			strip_flat = strip.reshape(D, H, W * strip_samples, 3)
			# Convert to normalized [-1,1] grid coords
			grid = strip_flat.clone()
			grid[..., 0] = (grid[..., 0] - ox) / sx / max(1, X - 1) * 2 - 1
			grid[..., 1] = (grid[..., 1] - oy) / sy / max(1, Y - 1) * 2 - 1
			grid[..., 2] = (grid[..., 2] - oz) / sz / max(1, Z - 1) * 2 - 1
			strip_len = torch.sqrt((diff * diff).sum(dim=-1) + 1e-12)  # (D, H, W)
			return grid.unsqueeze(0), strip_len  # (1, D, H, W*S, 3), (D, H, W)

		strip_grid_prev, strip_len_prev = _build_strip_grid(prev_pt, center_pt, Hm, Wm)
		strip_grid_next, strip_len_next = _build_strip_grid(center_pt, next_pt, Hm, Wm)

		# Dense grids for density losses
		strip_grid_prev_d, strip_len_prev_d = _build_strip_grid(prev_pt_dense, center_pt_dense, Hd, Wd)
		strip_grid_next_d, strip_len_next_d = _build_strip_grid(center_pt_dense, next_pt_dense, Hd, Wd)

		# Connection validity masks (D, Hm, Wm)
		conn_mask_prev = mask_conn[:, 0, :, :, 0] * mask_conn[:, 0, :, :, 1]
		conn_mask_next = mask_conn[:, 0, :, :, 1] * mask_conn[:, 0, :, :, 2]

		# Dense masks (D, Hd, Wd) — nearest upsample of boolean-like mask
		conn_mask_prev_d = F.interpolate(
			conn_mask_prev.unsqueeze(1), size=(Hd, Wd),
			mode='nearest').squeeze(1)
		conn_mask_next_d = F.interpolate(
			conn_mask_next.unsqueeze(1), size=(Hd, Wd),
			mode='nearest').squeeze(1)

		# Strip validity: all midpoint samples must be inside volume (grad_mag > 0)
		gm_vol = data.grad_mag.float()  # (1, 1, Z, Y, X)
		for direction_grid, direction_name in [(strip_grid_prev, 'prev'), (strip_grid_next, 'next')]:
			sv = F.grid_sample(gm_vol, direction_grid, mode='nearest',
							   padding_mode='zeros', align_corners=True)
			sv = sv.squeeze(0).squeeze(0).reshape(D, Hm, Wm, strip_samples)
			sv = (sv > 0).all(dim=-1).float()
			if direction_name == 'prev':
				conn_mask_prev = conn_mask_prev * sv
			else:
				conn_mask_next = conn_mask_next * sv

		for direction_grid, direction_name, H, W in [
			(strip_grid_prev_d, 'prev', Hd, Wd),
			(strip_grid_next_d, 'next', Hd, Wd),
		]:
			sv = F.grid_sample(gm_vol, direction_grid, mode='nearest',
							   padding_mode='zeros', align_corners=True)
			sv = sv.squeeze(0).squeeze(0).reshape(D, H, W, strip_samples)
			sv = (sv > 0).all(dim=-1).float()
			if direction_name == 'prev':
				conn_mask_prev_d = conn_mask_prev_d * sv
			else:
				conn_mask_next_d = conn_mask_next_d * sv

		# Initial density: uniform 1/avg_strip_length (so initial integral ≈ 1.0)
		all_lens = torch.cat([strip_len_prev[conn_mask_prev > 0.5],
							  strip_len_next[conn_mask_next > 0.5]])
		avg_strip_len = all_lens.mean().clamp(min=0.1) if len(all_lens) > 0 else torch.tensor(5.0)
		density_init_val = 1.0 / float(avg_strip_len)
		density_initial = torch.full((D, 1, Hm, Wm), density_init_val, device=dev)

	print(f"[fit_data] density strips: {int(conn_mask_prev.sum())} prev, {int(conn_mask_next.sum())} next, "
		  f"dense: {int(conn_mask_prev_d.sum())} prev, {int(conn_mask_next_d.sum())} next, "
		  f"avg_len={float(avg_strip_len):.2f}, init_val={density_init_val:.4f}", flush=True)

	# --- Scatter mesh data into volume ---
	print("[fit_data] scattering mesh data into volume", flush=True)
	with torch.no_grad():
		normal_sum, winding_sum, validity_sum, density_sum, count = _scatter_mesh_to_volume(
			xyz_lr, mask_lr, mesh_normals, winding_target.expand(D, 1, Hm, Wm),
			validity_target_mesh, density_initial,
			origin, spacing, (Z, Y, X),
		)
		n_scattered = int((count > 0).sum())
		print(f"[fit_data] scattered to {n_scattered}/{Z*Y*X} voxels", flush=True)

	# --- Build initial pyramid ---
	print("[fit_data] building initial pyramid", flush=True)
	data_vol = DataVolume.from_initial(normal_sum, winding_sum, validity_sum, density_sum, count, device=dev)
	n_levels = len(data_vol.normal_pyr)
	print(f"[fit_data] pyramid levels: {n_levels}", flush=True)
	for i, p in enumerate(data_vol.normal_pyr):
		print(f"  normal level {i}: {p.shape}", flush=True)

	# --- Supersample supervision grid 4x in H, W ---
	with torch.no_grad():
		Hs, Ws = 4 * Hm - 3, 4 * Wm - 3
		xyz_hr = F.interpolate(xyz_lr.permute(0, 3, 1, 2), size=(Hs, Ws),
							   mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
		normals_hr = F.interpolate(mesh_normals.permute(0, 3, 1, 2), size=(Hs, Ws),
								   mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
		normals_hr = F.normalize(normals_hr, dim=-1, eps=1e-8)
		mask_hr = F.interpolate(mask_lr, size=(Hs, Ws), mode='nearest')
		validity_hr = F.interpolate(validity_target_mesh, size=(Hs, Ws), mode='nearest')
	print(f"[fit_data] supersampled supervision: {Hm}x{Wm} -> {Hs}x{Ws}", flush=True)
	mesh_normals = normals_hr
	mask_lr = mask_hr
	validity_target_mesh = validity_hr
	del normals_hr, mask_hr, validity_hr

	# --- Build sample grid (frozen) ---
	with torch.no_grad():
		grid = _build_sample_grid(xyz_hr, origin, spacing, (Z, Y, X))

	# --- Per-level LR scaling: coarser levels get higher LR ---
	# Level 0 = finest (full res), level n_levels-1 = coarsest (1x1x1)
	# lr_level = lr * lr_exponent^(n_levels - 1 - i)
	# So finest gets lr*1, coarsest gets lr * lr_exponent^(n_levels-1)
	param_groups = []
	for i, (np_, wp_, vp_, dp_) in enumerate(zip(
			data_vol.normal_pyr, data_vol.winding_pyr, data_vol.validity_pyr, data_vol.density_pyr)):
		lr_level = lr * (lr_exponent ** (n_levels - 1 - i))
		param_groups.append({"params": [np_], "lr": lr_level})
		param_groups.append({"params": [wp_], "lr": lr_level})
		param_groups.append({"params": [vp_], "lr": lr_level})
		param_groups.append({"params": [dp_], "lr": lr_level})

	optimizer = torch.optim.Adam(param_groups, lr=lr)

	# --- Free data tensors we no longer need ---
	del data, normal_sum, winding_sum, validity_sum, density_sum, count

	# --- Optimization loop ---
	print(f"[fit_data] optimizing {n_steps} steps", flush=True)
	loss_history = []
	best_loss = float('inf')
	best_state = None
	best_step = -1
	for step in range(n_steps):
		normal_field = data_vol.integrate_normal()      # (3, Z, Y, X)
		winding_field = data_vol.integrate_winding()    # (1, Z, Y, X)
		validity_field = data_vol.integrate_validity()  # (1, Z, Y, X)
		density_field = data_vol.integrate_density()    # (1, Z, Y, X)

		# Sample at mesh vertices
		sampled_n = _sample_volume(normal_field, grid)    # (D, 3, Hs, Ws)
		sampled_w = _sample_volume(winding_field, grid)   # (D, 1, Hs, Ws)
		sampled_v = _sample_volume(validity_field, grid)  # (D, 1, Hs, Ws)

		# Data losses
		l_normal = _normal_data_loss(sampled_n, mesh_normals, mask_lr)
		l_winding = _winding_data_loss(sampled_w, winding_target, mask_lr)
		l_validity = _validity_data_loss(sampled_v, validity_target_mesh, mask_lr)

		# Density integral loss (dense sampling)
		l_density_prev = _density_integral_loss(
			density_field, strip_grid_prev_d, strip_len_prev_d, conn_mask_prev_d,
			strip_samples, D, Hd, Wd)
		l_density_next = _density_integral_loss(
			density_field, strip_grid_next_d, strip_len_next_d, conn_mask_next_d,
			strip_samples, D, Hd, Wd)
		l_density = 0.5 * (l_density_prev + l_density_next)

		# Density average (uniformity) loss (dense sampling)
		l_avg_d_prev = _density_average_loss(
			density_field, strip_grid_prev_d, strip_len_prev_d, conn_mask_prev_d,
			strip_samples, D, Hd, Wd)
		l_avg_d_next = _density_average_loss(
			density_field, strip_grid_next_d, strip_len_next_d, conn_mask_next_d,
			strip_samples, D, Hd, Wd)
		l_avg_d = 0.5 * (l_avg_d_prev + l_avg_d_next)

		# Smoothing losses
		l_smooth_n = _normal_smooth_loss(normal_field)
		l_smooth_w = _winding_smooth_loss(winding_field)
		l_smooth_v = _finite_diff_loss(validity_field)
		l_smooth_d = _finite_diff_loss(density_field)

		# Boundary loss: volume edges → validity = 0
		l_boundary_v = _validity_boundary_loss(validity_field)

		# Curvature (gradient-extension) losses
		l_curv_n = _curvature_loss(normal_field) if w_curv_n > 0 else torch.zeros(1, device=dev)
		l_curv_w = _curvature_loss(winding_field) if w_curv_w > 0 else torch.zeros(1, device=dev)
		l_curv_d = _curvature_loss(density_field) if w_curv_d > 0 else torch.zeros(1, device=dev)
		l_pos_d = _positivity_loss(density_field) if w_pos_d > 0 else torch.zeros(1, device=dev)

		loss = (w_normal * l_normal + w_winding * l_winding
				+ w_smooth_n * l_smooth_n + w_smooth_w * l_smooth_w
				+ w_curv_n * l_curv_n + w_curv_w * l_curv_w
				+ w_validity * l_validity + w_smooth_v * l_smooth_v
				+ w_validity * l_boundary_v
				+ w_density * l_density + w_smooth_d * l_smooth_d
				+ w_curv_d * l_curv_d + w_pos_d * l_pos_d
				+ w_avg_d * l_avg_d)

		# Track best
		loss_val = loss.item()
		if loss_val < best_loss:
			best_loss = loss_val
			best_step = step
			best_state = {k: v.detach().cpu().clone() for k, v in data_vol.state_dict().items()}

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if step % log_interval == 0 or step == n_steps - 1:
			print(f"  [step {step:4d}] loss={loss_val:.6f}  "
				  f"normal={l_normal.item():.6f}  winding={l_winding.item():.6f}  "
				  f"validity={l_validity.item():.6f}  boundary_v={l_boundary_v.item():.6f}  "
				  f"density={l_density.item():.6f}  avg_d={l_avg_d.item():.6f}  pos_d={l_pos_d.item():.6f}  "
				  f"smooth_n={l_smooth_n.item():.6f}  smooth_w={l_smooth_w.item():.6f}  "
				  f"smooth_v={l_smooth_v.item():.6f}  smooth_d={l_smooth_d.item():.6f}  "
				  f"curv_n={l_curv_n.item():.6f}  curv_w={l_curv_w.item():.6f}  "
				  f"curv_d={l_curv_d.item():.6f}",
				  flush=True)
			loss_history.append({
				"step": step,
				"total": loss_val,
				"normal_data": l_normal.item(),
				"winding_data": l_winding.item(),
				"validity_data": l_validity.item(),
				"validity_boundary": l_boundary_v.item(),
				"density": l_density.item(),
				"density_avg": l_avg_d.item(),
			"density_pos": l_pos_d.item(),
				"normal_smooth": l_smooth_n.item(),
				"winding_smooth": l_smooth_w.item(),
				"validity_smooth": l_smooth_v.item(),
				"density_smooth": l_smooth_d.item(),
				"curv_n": l_curv_n.item(),
				"curv_w": l_curv_w.item(),
				"curv_d": l_curv_d.item(),
			})

	# --- Restore best state ---
	print(f"[fit_data] restoring best state from step {best_step} (loss={best_loss:.6f})", flush=True)
	data_vol.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
	del best_state

	# --- Final integration ---
	print("[fit_data] integrating final volumes", flush=True)
	with torch.no_grad():
		final_normal = data_vol.integrate_normal().cpu().numpy()     # (3, Z, Y, X)
		final_winding = data_vol.integrate_winding().cpu().numpy()   # (1, Z, Y, X)
		final_winding = final_winding[0]  # (Z, Y, X)
		final_validity = data_vol.integrate_validity().cpu().numpy()  # (1, Z, Y, X)
		final_validity = final_validity[0]  # (Z, Y, X)
		final_density = data_vol.integrate_density().cpu().numpy()   # (1, Z, Y, X)
		final_density = final_density[0]  # (Z, Y, X)

	# Clamp validity to 0 outside the model's winding range
	w_end = winding_offset + (D - 1) * winding_direction
	w_min, w_max = min(winding_offset, w_end), max(winding_offset, w_end)
	outside_winding = (final_winding < w_min) | (final_winding > w_max)
	final_validity[outside_winding] = 0.0
	final_validity = (final_validity >= 0.9).astype(np.float32)
	n_valid = int(final_validity.sum())
	print(f"[fit_data] validity: clamped {outside_winding.sum()} outside winding [{w_min}, {w_max}], "
		  f"thresholded at 0.9 -> {n_valid}/{final_validity.size} valid voxels", flush=True)

	# --- Normal visualization ---
	if normal_vis_dir is not None:
		vis_dir = Path(normal_vis_dir)
		vis_dir.mkdir(parents=True, exist_ok=True)
		labels_vol = None
		vis_ds = int(round(spacing[0]))
		if labels_path is not None:
			print(f"[fit_data] loading labels from {labels_path}", flush=True)
			labels_full = tifffile.imread(str(labels_path))  # (Z_full, Y_full, X_full)
			ox, oy, oz = origin
			sx, sy, sz = spacing
			# Crop in fullres coords (no downsample — compose at fullres)
			x0, y0, z0 = int(ox), int(oy), int(oz)
			x1 = x0 + int(X * sx)
			y1 = y0 + int(Y * sy)
			z1 = z0 + int(Z * sz)
			labels_vol = labels_full[z0:z1, y0:y1, x0:x1]
			print(f"[fit_data] labels fullres crop: [{z0}:{z1}, {y0}:{y1}, {x0}:{x1}] "
				  f"-> {labels_vol.shape}", flush=True)
		_write_normal_vis_slices(vis_dir, final_normal, final_winding,
							validity=final_validity, density=final_density,
							labels=labels_vol, ds=vis_ds)

	# --- Write output zarr ---
	if output_path is not None:
		print(f"[fit_data] writing zarr output to {output_path}", flush=True)
		out_path = Path(output_path)
		root = zarr.open_group(str(out_path), mode="w", zarr_format=2)
		ds_n = root.create_array(
			"normal", shape=final_normal.shape,
			chunks=(3, min(64, Z), min(64, Y), min(64, X)),
			dtype=np.float32, overwrite=True,
		)
		ds_n[:] = final_normal.astype(np.float32)
		ds_w = root.create_array(
			"winding", shape=final_winding.shape,
			chunks=(min(64, Z), min(64, Y), min(64, X)),
			dtype=np.float32, overwrite=True,
		)
		ds_w[:] = final_winding.astype(np.float32)
		ds_v = root.create_array(
			"validity", shape=final_validity.shape,
			chunks=(min(64, Z), min(64, Y), min(64, X)),
			dtype=np.float32, overwrite=True,
		)
		ds_v[:] = final_validity.astype(np.float32)
		ds_d = root.create_array(
			"density", shape=final_density.shape,
			chunks=(min(64, Z), min(64, Y), min(64, X)),
			dtype=np.float32, overwrite=True,
		)
		ds_d[:] = final_density.astype(np.float32)
		root.attrs["origin_fullres"] = list(origin)
		root.attrs["spacing"] = list(spacing)
		root.attrs["size"] = [Z, Y, X]
		print(f"[fit_data] wrote normal {final_normal.shape}, winding {final_winding.shape}, "
			  f"validity {final_validity.shape}, density {final_density.shape}", flush=True)

	# --- Write output TIFs ---
	if tif_output_dir is not None:
		tif_dir = Path(tif_output_dir)
		tif_dir.mkdir(parents=True, exist_ok=True)
		# Normal: (3, Z, Y, X) -> (Z, 3, Y, X) multi-page, 3 channels per page
		normal_tif = final_normal.astype(np.float32).transpose(1, 0, 2, 3)  # (Z, 3, Y, X)
		normal_path = tif_dir / "normal.tif"
		tifffile.imwrite(str(normal_path), normal_tif, compression="lzw")
		print(f"[fit_data] wrote {normal_path}  shape={normal_tif.shape}", flush=True)
		# Winding: (Z, Y, X) multi-page float32
		winding_tif = final_winding.astype(np.float32)
		winding_path = tif_dir / "winding.tif"
		tifffile.imwrite(str(winding_path), winding_tif, compression="lzw")
		print(f"[fit_data] wrote {winding_path}  shape={winding_tif.shape}", flush=True)
		# Validity: (Z, Y, X) multi-page float32
		validity_tif = final_validity.astype(np.float32)
		validity_path = tif_dir / "validity.tif"
		tifffile.imwrite(str(validity_path), validity_tif, compression="lzw")
		print(f"[fit_data] wrote {validity_path}  shape={validity_tif.shape}", flush=True)
		# Density: (Z, Y, X) multi-page float32
		density_tif = final_density.astype(np.float32)
		density_path = tif_dir / "density.tif"
		tifffile.imwrite(str(density_path), density_tif, compression="lzw")
		print(f"[fit_data] wrote {density_path}  shape={density_tif.shape}", flush=True)

	# --- Stats JSON ---
	if stats_json_path is not None:
		# Compute final loss details
		with torch.no_grad():
			normal_field = data_vol.integrate_normal()
			winding_field = data_vol.integrate_winding()
			validity_field = data_vol.integrate_validity()
			density_field = data_vol.integrate_density()
			sampled_n = _sample_volume(normal_field, grid)
			sampled_w = _sample_volume(winding_field, grid)
			sampled_v = _sample_volume(validity_field, grid)

			# Per-vertex normal loss
			sn = sampled_n.permute(0, 2, 3, 1)
			sn = F.normalize(sn, dim=-1, eps=1e-8)
			mn = F.normalize(mesh_normals, dim=-1, eps=1e-8)
			dot = (sn * mn).sum(dim=-1)
			n_loss_map = 1.0 - dot * dot
			n_mask = mask_lr.squeeze(1)
			n_valid = n_loss_map[n_mask > 0.5]

			# Per-vertex winding loss
			w_diff = sampled_w - winding_target
			w_loss_map = (w_diff * w_diff).squeeze(1)
			w_valid = w_loss_map[n_mask > 0.5]

			# Per-vertex validity loss
			v_diff = sampled_v - validity_target_mesh
			v_loss_map = (v_diff * v_diff).squeeze(1)
			v_valid = v_loss_map[n_mask > 0.5]

			# Density integral loss (dense sampling)
			l_dp = _density_integral_loss(
				density_field, strip_grid_prev_d, strip_len_prev_d, conn_mask_prev_d,
				strip_samples, D, Hd, Wd)
			l_dn = _density_integral_loss(
				density_field, strip_grid_next_d, strip_len_next_d, conn_mask_next_d,
				strip_samples, D, Hd, Wd)

			# Density average (uniformity) loss (dense sampling)
			l_ap = _density_average_loss(
				density_field, strip_grid_prev_d, strip_len_prev_d, conn_mask_prev_d,
				strip_samples, D, Hd, Wd)
			l_an = _density_average_loss(
				density_field, strip_grid_next_d, strip_len_next_d, conn_mask_next_d,
				strip_samples, D, Hd, Wd)

		stats = {
			"losses": {
				"normal_data": {
					"avg": float(n_valid.mean()) if len(n_valid) > 0 else 0.0,
					"max": float(n_valid.max()) if len(n_valid) > 0 else 0.0,
				},
				"winding_data": {
					"avg": float(w_valid.mean()) if len(w_valid) > 0 else 0.0,
					"max": float(w_valid.max()) if len(w_valid) > 0 else 0.0,
				},
				"validity_data": {
					"avg": float(v_valid.mean()) if len(v_valid) > 0 else 0.0,
					"max": float(v_valid.max()) if len(v_valid) > 0 else 0.0,
				},
				"density_integral": {
					"avg": float(0.5 * (l_dp.item() + l_dn.item())),
					"prev": float(l_dp.item()),
					"next": float(l_dn.item()),
				},
				"density_avg": {
					"avg": float(0.5 * (l_ap.item() + l_an.item())),
					"prev": float(l_ap.item()),
					"next": float(l_an.item()),
				},
				"normal_smooth": {"avg": loss_history[-1]["normal_smooth"] if loss_history else 0.0},
				"winding_smooth": {"avg": loss_history[-1]["winding_smooth"] if loss_history else 0.0},
				"validity_smooth": {"avg": loss_history[-1]["validity_smooth"] if loss_history else 0.0},
				"density_smooth": {"avg": loss_history[-1]["density_smooth"] if loss_history else 0.0},
			},
			"volume": {"Z": Z, "Y": Y, "X": X, "n_pyramid_levels": n_levels},
			"mesh": {"D": D, "Hm": Hm, "Wm": Wm},
			"loss_history": loss_history,
		}
		stats_path = Path(stats_json_path)
		stats_path.parent.mkdir(parents=True, exist_ok=True)
		stats_path.write_text(json.dumps(stats, indent=2))
		print(f"[fit_data] wrote stats to {stats_json_path}", flush=True)

	print("[fit_data] done.", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
	parser = argparse.ArgumentParser(
		description="Fit dense 3D data volumes to a frozen lasagna model",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--model", required=True, help="Path to model.pt checkpoint")
	parser.add_argument("--input", required=True, help="Path to normals.zarr (for spatial metadata)")
	parser.add_argument("--output", default=None, help="Output zarr path")
	parser.add_argument("--tif-output", default=None,
						help="Output directory for multi-layer TIFs (normal.tif + winding.tif)")
	parser.add_argument("--stats-json", default=None, help="Stats JSON output path")
	parser.add_argument("--steps", type=int, default=1000, help="Optimization steps")
	parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
	parser.add_argument("--w-normal", type=float, default=1.0, help="Normal data loss weight")
	parser.add_argument("--w-winding", type=float, default=10.0, help="Winding data loss weight")
	parser.add_argument("--w-smooth-n", type=float, default=10.0, help="Normal smoothing weight")
	parser.add_argument("--w-smooth-w", type=float, default=10.0, help="Winding smoothing weight")
	parser.add_argument("--w-curv-n", type=float, default=10.0,
						help="Curvature (gradient-extension) penalty on normals")
	parser.add_argument("--w-curv-w", type=float, default=10.0,
						help="Curvature (gradient-extension) penalty on winding")
	parser.add_argument("--w-validity", type=float, default=10.0,
						help="Validity data loss weight")
	parser.add_argument("--w-smooth-v", type=float, default=10.0,
						help="Validity smoothing weight")
	parser.add_argument("--w-density", type=float, default=10.0,
						help="Density integral loss weight")
	parser.add_argument("--w-smooth-d", type=float, default=10.0,
						help="Density smoothing weight")
	parser.add_argument("--w-curv-d", type=float, default=10.0,
						help="Density curvature penalty")
	parser.add_argument("--w-pos-d", type=float, default=10.0,
						help="Density positivity penalty")
	parser.add_argument("--w-avg-d", type=float, default=1.0,
						help="Density uniformity (average) penalty")
	parser.add_argument("--lr-exponent", type=float, default=4.0,
						help="LR multiplier per pyramid level (coarser=higher LR)")
	parser.add_argument("--device", default="cuda", help="Torch device")
	parser.add_argument("--log-interval", type=int, default=20, help="Log every N steps")
	parser.add_argument("--normal-vis-dir", default=None,
						help="Write per-Z-slice JPEG of reconstructed normals to DIR")
	parser.add_argument("--labels", default=None,
						help="Original TIF labels for overlay in normal-vis (0=bg, 1=pred, 2=ignore)")
	args = parser.parse_args()

	if args.output is None and args.tif_output is None:
		parser.error("at least one of --output or --tif-output must be given")

	fit_data_to_model(
		model_path=args.model,
		input_path=args.input,
		output_path=args.output,
		tif_output_dir=args.tif_output,
		stats_json_path=args.stats_json,
		n_steps=args.steps,
		lr=args.lr,
		w_normal=args.w_normal,
		w_winding=args.w_winding,
		w_smooth_n=args.w_smooth_n,
		w_smooth_w=args.w_smooth_w,
		w_curv_n=args.w_curv_n,
		w_curv_w=args.w_curv_w,
		w_validity=args.w_validity,
		w_smooth_v=args.w_smooth_v,
		w_density=args.w_density,
		w_smooth_d=args.w_smooth_d,
		w_curv_d=args.w_curv_d,
		w_pos_d=args.w_pos_d,
		w_avg_d=args.w_avg_d,
		lr_exponent=args.lr_exponent,
		device_str=args.device,
		log_interval=args.log_interval,
		normal_vis_dir=args.normal_vis_dir,
		labels_path=args.labels,
	)


if __name__ == "__main__":
	main()
