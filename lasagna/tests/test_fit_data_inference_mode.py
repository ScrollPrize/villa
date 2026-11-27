"""FitData3D sampling after an inference-mode sample has cached its device constants."""
from __future__ import annotations

import numpy as np
import torch
from vc_reader.reader import ZarrWriter, open_zarr

import fit_data
from sparse_cache import SparseChunkGroupCache


def test_differentiable_sample_after_inference_mode_sample(tmp_path):
	values = np.random.default_rng(9).integers(0, 256, (48, 48, 48), dtype=np.uint8)
	ZarrWriter.create(tmp_path / "a.zarr", shape=values.shape, chunks=(16, 16, 16), dtype="u1")[...] = values
	cache = SparseChunkGroupCache(
		channels=["grad_mag"], reader=open_zarr(tmp_path / "a.zarr"), vol_shape_zyx=values.shape,
		channel_indices={"grad_mag": 0}, is_3d_zarr=True, device=torch.device("cpu"),
	)
	# a spacing and origin no other test uses, so the module cache has no entry for them yet
	origin, spacing = (0.5, 0.25, 0.75), (1.125, 1.125, 1.125)
	data = fit_data.FitData3D(
		cos=None, grad_mag=None, nx=None, ny=None, pred_dt=None, corr_points=None, winding_volume=None,
		origin_fullres=origin, spacing=spacing, sparse_caches={"g": cache}, _vol_size=values.shape,
	)
	xyz = torch.from_numpy(np.random.default_rng(10).uniform(4, 40, (2, 3, 4, 3)).astype(np.float32))
	cache.prefetch(xyz, origin, spacing)
	cache.sync()

	with torch.inference_mode():
		data.grid_sample_fullres(xyz, channels={"grad_mag"})

	g = xyz.clone().requires_grad_(True)
	out = data.grid_sample_fullres(g, diff=True, channels={"grad_mag"}).grad_mag
	out.sum().backward()
	assert g.grad is not None and float(g.grad.abs().max()) > 0.0
