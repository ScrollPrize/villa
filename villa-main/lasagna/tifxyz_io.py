"""Load/save tifxyz surface directories (x.tif, y.tif, z.tif, meta.json)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import tifffile


def surface_step_stats(xyz: torch.Tensor, valid: torch.Tensor) -> tuple[float, float, float, float]:
	"""Measure tifxyz grid spacing from h/w neighbors and diagonals.

	Returns (h_mean, w_mean, diag_mean, combined_mean), all in voxel units.
	Diagonal distances are divided by sqrt(2) so they estimate one grid step.
	"""
	with torch.no_grad():
		xyz_t = xyz.detach()
		valid_t = valid.detach().bool()
		parts: list[torch.Tensor] = []

		def _mean_or_nan(values: torch.Tensor) -> float:
			if values.numel() == 0:
				return float("nan")
			return float(values.mean().detach().cpu())

		h_vals = torch.empty(0, device=xyz_t.device, dtype=xyz_t.dtype)
		w_vals = torch.empty(0, device=xyz_t.device, dtype=xyz_t.dtype)
		diag_vals = torch.empty(0, device=xyz_t.device, dtype=xyz_t.dtype)
		if int(xyz_t.shape[0]) > 1:
			ok_h = valid_t[1:, :] & valid_t[:-1, :]
			if bool(ok_h.any().detach().cpu()):
				h_vals = (xyz_t[1:, :] - xyz_t[:-1, :]).norm(dim=-1)[ok_h]
				parts.append(h_vals)
		if int(xyz_t.shape[1]) > 1:
			ok_w = valid_t[:, 1:] & valid_t[:, :-1]
			if bool(ok_w.any().detach().cpu()):
				w_vals = (xyz_t[:, 1:] - xyz_t[:, :-1]).norm(dim=-1)[ok_w]
				parts.append(w_vals)
		if int(xyz_t.shape[0]) > 1 and int(xyz_t.shape[1]) > 1:
			diag_parts = []
			ok_d0 = valid_t[1:, 1:] & valid_t[:-1, :-1]
			if bool(ok_d0.any().detach().cpu()):
				diag_parts.append((xyz_t[1:, 1:] - xyz_t[:-1, :-1]).norm(dim=-1)[ok_d0])
			ok_d1 = valid_t[1:, :-1] & valid_t[:-1, 1:]
			if bool(ok_d1.any().detach().cpu()):
				diag_parts.append((xyz_t[1:, :-1] - xyz_t[:-1, 1:]).norm(dim=-1)[ok_d1])
			if diag_parts:
				diag_vals = torch.cat([v.reshape(-1) for v in diag_parts]) / math.sqrt(2.0)
				parts.append(diag_vals)
		combined = (
			float(torch.cat([p.reshape(-1) for p in parts]).mean().detach().cpu())
			if parts else float("nan")
		)
		return _mean_or_nan(h_vals), _mean_or_nan(w_vals), _mean_or_nan(diag_vals), combined


def load_tifxyz(path: str | Path, *, device: torch.device | str = "cpu"
				) -> tuple[torch.Tensor, torch.Tensor, dict]:
	"""Load a tifxyz directory into a mesh tensor.

	Returns (xyz, valid, meta) where:
	  xyz: (H, W, 3) float32 tensor — invalid vertices zeroed out
	  valid: (H, W) bool tensor — True for valid vertices
	  meta: dict from meta.json (or empty if missing)
	"""
	p = Path(path)
	if not p.is_dir():
		raise ValueError(f"tifxyz path is not a directory: {p}")
	x = tifffile.imread(str(p / "x.tif")).astype(np.float32)
	y = tifffile.imread(str(p / "y.tif")).astype(np.float32)
	z = tifffile.imread(str(p / "z.tif")).astype(np.float32)
	if x.shape != y.shape or x.shape != z.shape:
		raise ValueError(f"tifxyz shape mismatch: x={x.shape} y={y.shape} z={z.shape}")
	xyz = np.stack([x, y, z], axis=-1)  # (H, W, 3)
	meta_path = p / "meta.json"
	meta: dict = {}
	if meta_path.exists():
		meta = json.loads(meta_path.read_text(encoding="utf-8"))

	xyz_t = torch.from_numpy(xyz)
	# VC3D uses (-1, -1, -1) as invalid sentinel
	valid = (xyz_t != -1.0).all(dim=-1)  # (H, W) bool
	# Zero out invalid vertices so they don't pollute pyramid construction
	xyz_t[~valid] = 0.0

	n_valid = int(valid.sum())
	n_total = valid.numel()
	step_h, step_w, step_diag, step_avg = surface_step_stats(xyz_t, valid)
	print(f"[tifxyz_io] loaded {p.name}: {x.shape[0]}x{x.shape[1]}, "
		  f"{n_valid}/{n_total} valid ({100*n_valid/max(1,n_total):.1f}%) "
		  f"step_h={step_h:.3f} step_w={step_w:.3f} step_diag={step_diag:.3f} "
		  f"step_avg={step_avg:.3f}", flush=True)

	return xyz_t.to(device=device), valid.to(device=device), meta
