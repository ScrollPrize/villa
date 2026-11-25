"""The slot-indexed sparse cache used off CUDA (CPU and MPS): sync, growth and sampling."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from vc_reader.reader import ZarrWriter, open_zarr

import u8_sample_torch
from sparse_cache import SparseChunkGroupCache

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.fixture(autouse=True)
def _allow_unloaded_samples(monkeypatch):
	# these tests sample unloaded chunks on purpose; importing fit_service anywhere in the session turns on the
	# debug guard that rejects such samples
	monkeypatch.setenv("LASAGNA_CHECK_SPARSE_CACHE", "0")


def _cache(tmp_path, values, device, channels=("cos",), channel_indices=None):
	is_3d = values.ndim == 3
	ZarrWriter.create(tmp_path / "a.zarr", shape=values.shape, chunks=(16,) * values.ndim, dtype="u1")[...] = values
	return SparseChunkGroupCache(
		channels=list(channels), reader=open_zarr(tmp_path / "a.zarr"), vol_shape_zyx=values.shape[-3:],
		channel_indices=channel_indices or {c: i for i, c in enumerate(channels)}, is_3d_zarr=is_3d,
		device=torch.device(device),
	)


@pytest.mark.parametrize("device", DEVICES)
def test_prefetched_sparse_samples_equal_dense_samples_inside_the_volume(tmp_path, device):
	values = np.random.default_rng(1).integers(0, 256, (3, 75, 70, 90), dtype=np.uint8)
	cache = _cache(tmp_path, values, device, channels=("nx", "ny", "grad_mag"),
				   channel_indices={"nx": 0, "ny": 1, "grad_mag": 2})
	assert cache.slot_mode and cache.chunk_table.dtype == torch.int32
	rng = np.random.default_rng(2)
	# spacing 2 in fullres units; keep points one voxel inside so dense and sparse agree on every corner
	pts = rng.uniform([2, 2, 2], [2 * 88, 2 * 68, 2 * 73], (6, 9, 11, 3)).astype(np.float32)
	xyz = torch.from_numpy(pts).to(device)
	origin, spacing = (0.0, 0.0, 0.0), (2.0, 2.0, 2.0)

	# two prefetch rounds so the pool has to grow and keep the first round's slots
	cache.prefetch(xyz[:3], origin, spacing)
	cache.sync()
	first = cache.loaded_chunks()
	cache.prefetch(xyz, origin, spacing)
	cache.sync()
	assert cache.loaded_chunks() >= first > 0
	assert cache.n_slots == cache.loaded_chunks() <= int(cache.pool.shape[0])

	off = torch.zeros(3, device=device)
	inv = torch.full((3,), 0.5, device=device)
	g = xyz.clone().requires_grad_(True)
	sparse = cache.grid_sample(g, off, inv, diff=True)
	dense_v = torch.from_numpy(values).to(device)
	gd = xyz.clone().requires_grad_(True)
	dense = u8_sample_torch.dense_sample(dense_v, gd, off, inv, diff=True)
	# two float32 paths that each stay within 2e-4 of float64 (test_u8_sample_parity) differ by up to twice that
	torch.testing.assert_close(sparse.cpu(), dense.cpu(), rtol=0, atol=4e-4)
	w = torch.randn_like(dense)
	sparse.backward(w)
	dense.backward(w)
	# the gradient bar of test_u8_sample_parity, scaled by the largest gradient component, doubled for two paths
	scale = float(gd.grad.abs().max())
	torch.testing.assert_close(g.grad.cpu(), gd.grad.cpu(), rtol=0, atol=2 * (1e-4 * scale + 1e-5))
	u8 = cache.grid_sample(xyz, off, inv, diff=False)
	assert u8.dtype == torch.uint8
	# float32 rounding may put a value just either side of a .5 tie, so allow a rare off-by-one
	diff = (u8.cpu().int() - u8_sample_torch.dense_sample(dense_v, xyz, off, inv, diff=False).cpu().int()).abs()
	assert int(diff.max()) <= 1 and int((diff > 0).sum()) <= 2


@pytest.mark.parametrize("device", DEVICES)
def test_unloaded_chunks_sample_as_zero(tmp_path, device):
	values = np.full((40, 40, 40), 200, np.uint8)
	cache = _cache(tmp_path, values, device)
	xyz = torch.tensor([[[[5.0, 5.0, 5.0]]]], device=device)
	off = torch.zeros(3, device=device)
	inv = torch.ones(3, device=device)
	assert float(cache.grid_sample(xyz, off, inv, diff=True).sum()) == 0.0
	cache.prefetch(xyz, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
	cache.sync()
	assert float(cache.grid_sample(xyz, off, inv, diff=True).sum()) == pytest.approx(200.0)


@pytest.mark.parametrize("device", DEVICES)
def test_sync_between_forward_and_backward(tmp_path, device):
	# anticipatory pulls prefetch and sync chunks after a differentiable sample and before its backward; the
	# backward must still run and give the gradient it would have without the extra sync
	values = np.random.default_rng(3).integers(0, 256, (70, 70, 70), dtype=np.uint8)
	pts = np.random.default_rng(4).uniform(2, 30, (2, 5, 7, 3)).astype(np.float32)
	far = np.random.default_rng(5).uniform(40, 66, (2, 5, 7, 3)).astype(np.float32)
	off = torch.zeros(3, device=device)
	inv = torch.ones(3, device=device)
	grads = []
	for extra_sync in (False, True):
		cache = _cache(tmp_path / str(extra_sync), values, device)
		xyz = torch.from_numpy(pts).to(device)
		cache.prefetch(xyz, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
		cache.sync()
		slots_before = cache.n_slots
		g = xyz.clone().requires_grad_(True)
		out = cache.grid_sample(g, off, inv, diff=True)
		if extra_sync:
			# new chunks: the table gains entries and the pool fills (or grows) in place
			cache.prefetch(torch.from_numpy(far).to(device), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
			cache.sync()
			assert cache.n_slots > slots_before
		out.backward(torch.ones_like(out))
		grads.append(g.grad.cpu())
	assert float(grads[0].abs().max()) > 0.0
	torch.testing.assert_close(grads[1], grads[0], rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("grow", [False, True])
def test_backward_ignores_chunks_loaded_after_the_forward(tmp_path, device, grow):
	# loss terms sync the cache mid-forward, so a chunk that sampled as zero can be loaded before the backward;
	# its gradient must stay zero, and a grown pool must not send the backward past the pool the forward read
	values = np.random.default_rng(6).integers(0, 256, (160, 160, 160), dtype=np.uint8)
	near = np.random.default_rng(7).uniform(2, 30, (2, 5, 7, 3)).astype(np.float32)
	far = np.random.default_rng(8).uniform(100, 126, (2, 5, 7, 3)).astype(np.float32)
	pts = np.concatenate([near, far], axis=0)
	off = torch.zeros(3, device=device)
	inv = torch.ones(3, device=device)
	ones = (1.0, 1.0, 1.0)
	grads = []
	for late_sync in (False, True):
		cache = _cache(tmp_path / str(late_sync), values, device)
		cache.prefetch(torch.from_numpy(near).to(device), (0.0, 0.0, 0.0), ones)
		cache.sync()
		capacity = int(cache.pool.shape[0])
		g = torch.from_numpy(pts).to(device).requires_grad_(True)
		out = cache.grid_sample(g, off, inv, diff=True)
		assert float(out[:, 2:].abs().max()) == 0.0
		if late_sync:
			# without growth the far chunks land in the old allocation past the forward's slots; with it, the
			# whole volume's chunks need a pool larger than the old allocation
			if grow:
				span = np.arange(0, 160, 16, dtype=np.float32)
				every = np.stack(np.meshgrid(span, span, span, indexing="ij"), -1)[None]
				cache.prefetch(torch.from_numpy(np.ascontiguousarray(every)).to(device), (0.0, 0.0, 0.0), ones)
			else:
				cache.prefetch(torch.from_numpy(far).to(device), (0.0, 0.0, 0.0), ones)
			cache.sync()
			assert (int(cache.pool.shape[0]) > capacity) == grow
			assert cache.n_slots > capacity if grow else cache.n_slots <= capacity
		out.backward(torch.ones_like(out))
		grads.append(g.grad.cpu())
	assert float(grads[0][:2].abs().max()) > 0.0
	assert float(grads[1][2:].abs().max()) == 0.0
	torch.testing.assert_close(grads[1], grads[0], rtol=0, atol=0)
