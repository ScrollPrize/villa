from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zarr


@dataclass(frozen=True)
class CorrPoints3D:
	points_xyz_winda: torch.Tensor  # (K, 4) — x, y, z, winda in fullres
	collection_idx: torch.Tensor    # (K,) — integer collection ID per point


@dataclass(frozen=True)
class FitData3D:
	cos: torch.Tensor              # (1, 1, Z, Y, X)
	grad_mag: torch.Tensor         # (1, 1, Z, Y, X)
	dir0_z: torch.Tensor           # (1, 1, Z, Y, X) — z-axis direction
	dir1_z: torch.Tensor
	dir0_y: torch.Tensor | None    # (1, 1, Z, Y, X) — y-axis direction
	dir1_y: torch.Tensor | None
	dir0_x: torch.Tensor | None    # (1, 1, Z, Y, X) — x-axis direction
	dir1_x: torch.Tensor | None
	valid: torch.Tensor | None     # (1, 1, Z, Y, X)
	pred_dt: torch.Tensor | None
	corr_points: CorrPoints3D | None
	origin_fullres: tuple[float, float, float]  # (x0, y0, z0) in fullres voxels
	spacing: tuple[float, float, float]          # (sx, sy, sz) voxel size in fullres units

	@property
	def size(self) -> tuple[int, int, int]:
		"""(Z, Y, X) spatial dimensions."""
		if self.cos.ndim != 5:
			raise ValueError("FitData3D.cos must be (1,1,Z,Y,X)")
		_, _, z, y, x = self.cos.shape
		return int(z), int(y), int(x)

	def grid_sample_fullres(self, xyz_fullres: torch.Tensor) -> "FitData3D":
		"""Sample at fullres positions. xyz: (D, H, W, 3) where last dim is (x, y, z)."""
		Z, Y, X = self.size
		grid = xyz_fullres.clone()
		# Convert fullres -> volume voxel coords
		grid[..., 0] = (grid[..., 0] - self.origin_fullres[0]) / self.spacing[0]
		grid[..., 1] = (grid[..., 1] - self.origin_fullres[1]) / self.spacing[1]
		grid[..., 2] = (grid[..., 2] - self.origin_fullres[2]) / self.spacing[2]
		# Normalize to [-1, 1] for grid_sample
		# PyTorch 5D grid_sample: input is (N,C,D_in,H_in,W_in), grid last dim is (x->W, y->H, z->D)
		# Our volume is (1,C,Z,Y,X) -> D_in=Z, H_in=Y, W_in=X
		# grid (x,y,z) maps to (W=X, H=Y, D=Z) which is correct order for PyTorch
		grid[..., 0] = grid[..., 0] / max(1, X - 1) * 2 - 1  # x -> W
		grid[..., 1] = grid[..., 1] / max(1, Y - 1) * 2 - 1  # y -> H
		grid[..., 2] = grid[..., 2] / max(1, Z - 1) * 2 - 1  # z -> D
		# grid_sample expects (N, D_out, H_out, W_out, 3)
		grid_5d = grid.unsqueeze(0)  # (1, D, H, W, 3)

		def _gs(t: torch.Tensor | None) -> torch.Tensor | None:
			if t is None:
				return None
			return F.grid_sample(t, grid_5d, mode="bilinear", padding_mode="zeros", align_corners=True)

		return FitData3D(
			cos=_gs(self.cos),
			grad_mag=_gs(self.grad_mag),
			dir0_z=_gs(self.dir0_z),
			dir1_z=_gs(self.dir1_z),
			dir0_y=_gs(self.dir0_y),
			dir1_y=_gs(self.dir1_y),
			dir0_x=_gs(self.dir0_x),
			dir1_x=_gs(self.dir1_x),
			valid=_gs(self.valid),
			pred_dt=_gs(self.pred_dt),
			corr_points=self.corr_points,
			origin_fullres=self.origin_fullres,
			spacing=self.spacing,
		)


def auto_crop_for_mesh(
	mesh_bbox: tuple[float, float, float, float, float, float],
	volume_extent_fullres: tuple[int, int, int],
	margin: float = 3.0,
) -> tuple[int, int, int, int, int, int]:
	"""Compute crop = margin x mesh extent, centered on mesh, clamped to volume.

	Returns (x0, y0, z0, w, h, d) in fullres voxels.
	"""
	x_min, y_min, z_min, x_max, y_max, z_max = mesh_bbox
	vol_x, vol_y, vol_z = volume_extent_fullres

	cx = (x_min + x_max) / 2.0
	cy = (y_min + y_max) / 2.0
	cz = (z_min + z_max) / 2.0

	ex = max((x_max - x_min) * margin / 2.0, 100.0)
	ey = max((y_max - y_min) * margin / 2.0, 100.0)
	ez = max((z_max - z_min) * margin / 2.0, 100.0)

	x0 = max(0, int(cx - ex))
	y0 = max(0, int(cy - ey))
	z0 = max(0, int(cz - ez))
	x1 = min(vol_x, int(cx + ex))
	y1 = min(vol_y, int(cy + ey))
	z1 = min(vol_z, int(cz + ez))

	return (x0, y0, z0, x1 - x0, y1 - y0, z1 - z0)


def load_3d_for_model(
	*, path: str, device: torch.device, model: object,
	blur_sigma: float = 1.0,
) -> FitData3D:
	"""Load 3D data auto-cropped around model mesh bbox. Optionally blurs."""
	scaledown = float(model.params.scaledown)
	with torch.no_grad():
		xyz = model._grid_xyz()
		mesh_bbox = (xyz[..., 0].min().item(), xyz[..., 1].min().item(), xyz[..., 2].min().item(),
					 xyz[..., 0].max().item(), xyz[..., 1].max().item(), xyz[..., 2].max().item())
	print(f"[fit_data] mesh bbox: "
		  f"min=({mesh_bbox[0]:.0f},{mesh_bbox[1]:.0f},{mesh_bbox[2]:.0f}) "
		  f"max=({mesh_bbox[3]:.0f},{mesh_bbox[4]:.0f},{mesh_bbox[5]:.0f})", flush=True)
	prep = get_preprocessed_params(path)
	crop = auto_crop_for_mesh(mesh_bbox, prep["volume_extent_fullres"]) if prep else None
	if crop is not None:
		print(f"[fit_data] auto-crop: x={crop[0]} y={crop[1]} z={crop[2]} "
			  f"w={crop[3]} h={crop[4]} d={crop[5]}", flush=True)
	data = load_3d(path=path, device=device, downscale=scaledown, crop=crop)
	if blur_sigma > 0:
		blur_3d(data, sigma=blur_sigma)
		print(f"[fit_data] blurred data sigma={blur_sigma}", flush=True)
	return data


def blur_3d(data: FitData3D, sigma: float) -> None:
	"""Apply separable 3D Gaussian blur in-place to all data channels.

	Erodes the valid mask by the blur radius so blurred edge artifacts are excluded.
	"""
	radius = int(math.ceil(2 * sigma))
	ks = 2 * radius + 1
	# 1D Gaussian kernel
	x = torch.arange(ks, dtype=torch.float32, device=data.cos.device) - radius
	k1d = torch.exp(-0.5 * (x / sigma) ** 2)
	k1d = k1d / k1d.sum()

	def _blur_separable(t: torch.Tensor) -> torch.Tensor:
		# t: (1, 1, Z, Y, X)
		# Apply along Z (dim=2), Y (dim=3), X (dim=4) using conv3d with 1D kernels
		v = t
		# Z axis: kernel shape (1,1,ks,1,1)
		kz = k1d.view(1, 1, ks, 1, 1)
		v = F.conv3d(F.pad(v, (0, 0, 0, 0, radius, radius), mode='reflect'), kz)
		# Y axis: kernel shape (1,1,1,ks,1)
		ky = k1d.view(1, 1, 1, ks, 1)
		v = F.conv3d(F.pad(v, (0, 0, radius, radius, 0, 0), mode='reflect'), ky)
		# X axis: kernel shape (1,1,1,1,ks)
		kx = k1d.view(1, 1, 1, 1, ks)
		v = F.conv3d(F.pad(v, (radius, radius, 0, 0, 0, 0), mode='reflect'), kx)
		return v

	# Blur all signal channels in-place
	for name in ("cos", "grad_mag", "dir0_z", "dir1_z",
				 "dir0_y", "dir1_y", "dir0_x", "dir1_x"):
		t = getattr(data, name)
		if t is not None:
			t.copy_(_blur_separable(t))

	# Erode valid mask by blur radius using 3D min-pool
	if data.valid is not None:
		data.valid.copy_(
			-F.max_pool3d(-data.valid, kernel_size=ks, stride=1, padding=radius)
		)


def get_preprocessed_params(path: str) -> dict | None:
	"""Probe preprocessed zarr metadata for scaledown and volume extent.

	Returns dict with keys 'scaledown', 'volume_extent_fullres', or None.
	"""
	p = Path(path)
	s = str(p)
	is_omezarr = (
		s.endswith(".zarr")
		or s.endswith(".ome.zarr")
		or (".zarr/" in s)
		or (".ome.zarr/" in s)
	)
	if not is_omezarr:
		return None
	try:
		zsrc = zarr.open(s, mode="r")
	except Exception:
		return None
	if not (isinstance(zsrc, zarr.Array) and int(len(zsrc.shape)) == 4 and int(zsrc.shape[0]) >= 4):
		return None
	params = dict(getattr(zsrc, "attrs", {}).get("preprocess_params", {}) or {})
	if not params:
		return None
	ds = float(params["scaledown"])
	ds_i = max(1, int(round(ds)))
	C_all, Z_all, Y_all, X_all = (int(v) for v in zsrc.shape)
	volume_extent_fullres = (X_all * ds_i, Y_all * ds_i, Z_all * ds_i)
	return {"scaledown": ds, "volume_extent_fullres": volume_extent_fullres}


def load_3d(
	*,
	path: str,
	device: torch.device,
	downscale: float = 4.0,
	crop: tuple[int, int, int, int, int, int] | None = None,
) -> FitData3D:
	"""Load 3D sub-volume from preprocessed zarr.

	crop: (x0, y0, z0, w, h, d) in fullres voxel coords, or None for full volume.
	"""
	p = Path(path)
	s = str(p)
	is_omezarr = (
		s.endswith(".zarr")
		or s.endswith(".ome.zarr")
		or (".zarr/" in s)
		or (".ome.zarr/" in s)
	)
	if not is_omezarr:
		raise ValueError(f"load_3d requires zarr input, got: {s}")

	zsrc = zarr.open(s, mode="r")
	if not isinstance(zsrc, zarr.Array):
		raise ValueError(f"expected zarr.Array, got {type(zsrc)}")
	if int(len(zsrc.shape)) != 4:
		raise ValueError(f"expected 4D CZYX zarr, got shape={zsrc.shape}")

	params = dict(getattr(zsrc, "attrs", {}).get("preprocess_params", {}) or {})
	if not params:
		raise ValueError("preprocessed zarr missing preprocess_params")

	channels = [str(v) for v in (params.get("channels", []) or [])]
	if not channels:
		channels = ["cos", "grad_mag", "dir0", "dir1"]
	ci = {name: i for i, name in enumerate(channels)}

	req = ["cos", "grad_mag", "dir0", "dir1"]
	miss = [k for k in req if k not in ci]
	if miss:
		raise ValueError(f"preprocessed zarr missing required channels: {miss}; available={channels}")

	ds_meta = float(params["scaledown"])
	ds_i = max(1, int(round(ds_meta)))
	gmag_enc = float(params.get("grad_mag_encode_scale", 255.0))
	if gmag_enc <= 0.0:
		raise ValueError(f"invalid grad_mag_encode_scale: {gmag_enc}")

	shape_czyx = tuple(int(v) for v in zsrc.shape)
	C_all, Z_all, Y_all, X_all = shape_czyx

	# Determine read bounds (crop is in fullres voxels, zarr is downscaled by ds_i)
	if crop is not None:
		x0, y0, z0, cw, ch, cd = (int(v) for v in crop)
		x0v = max(0, x0 // ds_i)
		y0v = max(0, y0 // ds_i)
		z0v = max(0, z0 // ds_i)
		x1v = min(X_all, (x0 + cw + ds_i - 1) // ds_i)
		y1v = min(Y_all, (y0 + ch + ds_i - 1) // ds_i)
		z1v = min(Z_all, (z0 + cd + ds_i - 1) // ds_i)
		origin_fullres = (float(x0v * ds_i), float(y0v * ds_i), float(z0v * ds_i))
	else:
		x0v, y0v, z0v = 0, 0, 0
		x1v, y1v, z1v = X_all, Y_all, Z_all
		origin_fullres = (0.0, 0.0, 0.0)

	spacing = (float(ds_meta), float(ds_meta), float(ds_meta))

	print(f"[fit_data] load_3d: zarr shape={shape_czyx}, reading "
		  f"z=[{z0v}:{z1v}] y=[{y0v}:{y1v}] x=[{x0v}:{x1v}], "
		  f"origin={origin_fullres}, spacing={spacing}", flush=True)

	def _read_ch(name: str) -> np.ndarray:
		return np.asarray(zsrc[ci[name], z0v:z1v, y0v:y1v, x0v:x1v])

	def _u8_to_t(a: np.ndarray) -> torch.Tensor:
		t = torch.from_numpy(a.astype(np.float32) / 255.0).to(device=device, dtype=torch.float32)
		return t.unsqueeze(0).unsqueeze(0)  # (1, 1, Z, Y, X)

	def _u8_to_t_scaled(a: np.ndarray, *, scale: float) -> torch.Tensor:
		t = torch.from_numpy(a.astype(np.float32) / scale).to(device=device, dtype=torch.float32)
		return t.unsqueeze(0).unsqueeze(0)

	def _u8_valid_to_t(a: np.ndarray) -> torch.Tensor:
		t = torch.from_numpy((a > 0).astype(np.float32)).to(device=device, dtype=torch.float32)
		return t.unsqueeze(0).unsqueeze(0)

	def _u8_raw_to_t(a: np.ndarray) -> torch.Tensor:
		t = torch.from_numpy(np.sqrt(a.astype(np.float32))).to(device=device, dtype=torch.float32)
		return t.unsqueeze(0).unsqueeze(0)

	cos_t = _u8_to_t(_read_ch("cos"))
	mag_t = _u8_to_t_scaled(_read_ch("grad_mag"), scale=gmag_enc)
	# dir0/dir1 are the z-axis direction channels
	dir0_z_t = _u8_to_t(_read_ch("dir0"))
	dir1_z_t = _u8_to_t(_read_ch("dir1"))
	valid_t = _u8_valid_to_t(_read_ch("valid")) if "valid" in ci else None
	dir0_y_t = _u8_to_t(_read_ch("dir0_y")) if "dir0_y" in ci else None
	dir1_y_t = _u8_to_t(_read_ch("dir1_y")) if "dir1_y" in ci else None
	dir0_x_t = _u8_to_t(_read_ch("dir0_x")) if "dir0_x" in ci else None
	dir1_x_t = _u8_to_t(_read_ch("dir1_x")) if "dir1_x" in ci else None
	pred_dt_t = _u8_raw_to_t(_read_ch("pred_dt")) if "pred_dt" in ci else None

	return FitData3D(
		cos=cos_t,
		grad_mag=mag_t,
		dir0_z=dir0_z_t,
		dir1_z=dir1_z_t,
		dir0_y=dir0_y_t,
		dir1_y=dir1_y_t,
		dir0_x=dir0_x_t,
		dir1_x=dir1_x_t,
		valid=valid_t,
		pred_dt=pred_dt_t,
		corr_points=None,
		origin_fullres=origin_fullres,
		spacing=spacing,
	)
