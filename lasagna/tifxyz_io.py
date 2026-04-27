"""Load/save tifxyz surface directories (x.tif, y.tif, z.tif, meta.json)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import tifffile


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
	print(f"[tifxyz_io] loaded {p.name}: {x.shape[0]}x{x.shape[1]}, "
		  f"{n_valid}/{n_total} valid ({100*n_valid/max(1,n_total):.1f}%)", flush=True)

	return xyz_t.to(device=device), valid.to(device=device), meta
