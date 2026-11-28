"""PooledChunkGroupCache (CPU and MPS): reads, prefetch coverage, sampling, growth, autograd."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from vc_reader.reader import ZarrWriter, open_zarr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import sparse_pool_cache  # noqa: E402
import sparse_pool_ops as ops  # noqa: E402
from sparse_pool_cache import PooledChunkGroupCache  # noqa: E402
from sparse_pool_cases import padded_brick  # noqa: E402

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


def _cache(path, values, device, *, channels=("cos",), channel_indices=None):
    is_3d = values.ndim == 3
    return PooledChunkGroupCache(
        channels=list(channels), reader=open_zarr(path), vol_shape_zyx=values.shape[-3:],
        channel_indices=channel_indices or {channels[0]: 0}, is_3d_zarr=is_3d, device=torch.device(device),
    )


def _zarr(tmp_path, values, chunks):
    path = tmp_path / "a.zarr"
    ZarrWriter.create(path, shape=values.shape, chunks=chunks, dtype="u1")[...] = values
    return path


def _brick(cache, coord):
    slot = int(cache.chunk_table[coord])
    assert slot >= 0
    return cache.pool[slot].cpu().numpy()


@pytest.mark.parametrize("device", DEVICES)
def test_sync_reads_padded_bricks_of_a_3d_array(tmp_path, device):
    values = np.random.default_rng(1).integers(1, 256, (40, 70, 33), dtype=np.uint8)
    cache = _cache(_zarr(tmp_path, values, (16, 16, 16)), values, device)
    coords = [(0, 0, 0), (1, 2, 1), (0, 1, 0)]
    cache._pending = dict.fromkeys(coords)
    cache.sync()
    assert cache.chunk_table.dtype == torch.int32 and cache.chunk_table.device.type == device
    for i, coord in enumerate(coords):
        assert int(cache.chunk_table[coord]) == i
        np.testing.assert_array_equal(_brick(cache, coord), padded_brick(values[None], coord))
    assert cache.loaded_chunks() == 3
    assert int((cache.chunk_table >= 0).sum()) == 3
    cache.sync()
    assert cache.loaded_chunks() == 3


@pytest.mark.parametrize("device", DEVICES)
def test_sync_reads_the_selected_channels_of_a_4d_array(tmp_path, device):
    values = np.random.default_rng(2).integers(1, 256, (3, 40, 40, 40), dtype=np.uint8)
    cache = _cache(_zarr(tmp_path, values, (3, 16, 16, 16)), values, device,
                   channels=("ny", "nx"), channel_indices={"nx": 0, "ny": 2})
    cache._pending = {(1, 0, 1): None}
    cache.sync()
    brick = _brick(cache, (1, 0, 1))
    np.testing.assert_array_equal(brick[0], padded_brick(values[2:3], (1, 0, 1))[0])
    np.testing.assert_array_equal(brick[1], padded_brick(values[0:1], (1, 0, 1))[0])


def test_non_uint8_array_is_refused(tmp_path):
    ZarrWriter.create(tmp_path / "a.zarr", shape=(4, 4, 4), chunks=(4, 4, 4), dtype="u2")
    with pytest.raises(ValueError, match="uint8"):
        PooledChunkGroupCache(
            channels=["cos"], reader=open_zarr(tmp_path / "a.zarr"), vol_shape_zyx=(4, 4, 4),
            channel_indices={"cos": 0}, is_3d_zarr=True, device=torch.device("cpu"),
        )


def _dense_expected(values, xyz, origin, spacing, chunk_table):
    """float64 trilinear of the zero-extended volume at the kernels' p, zero where its chunk is not loaded."""
    p = ((xyz.reshape(-1, 3) - torch.tensor(origin)) * (1.0 / torch.tensor(spacing))).double().numpy()
    vol = values.astype(np.float64)
    Z, Y, X = vol.shape
    table = chunk_table.cpu().numpy()
    out = np.zeros(len(p))
    for n, (x, y, z) in enumerate(p):
        if not np.isfinite([x, y, z]).all():
            continue
        c = np.floor(np.array([z, y, x]) / 32).astype(int)
        if (c < 0).any() or (c >= table.shape).any() or table[tuple(c)] < 0:
            continue
        x0, y0, z0 = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
        fx, fy, fz = x - x0, y - y0, z - z0
        acc = 0.0
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    xi, yi, zi = x0 + dx, y0 + dy, z0 + dz
                    if 0 <= xi < X and 0 <= yi < Y and 0 <= zi < Z:
                        w = (fx if dx else 1 - fx) * (fy if dy else 1 - fy) * (fz if dz else 1 - fz)
                        acc += w * vol[zi, yi, xi]
        out[n] = acc
    return out


@pytest.mark.parametrize("device", DEVICES)
def test_prefetch_sync_then_sample_matches_dense(tmp_path, device):
    rng = np.random.default_rng(3)
    values = rng.integers(0, 256, (45, 70, 66), dtype=np.uint8)
    cache = _cache(_zarr(tmp_path, values, (16, 16, 16)), values, device)
    origin = (10.0, -6.0, 4.0)
    spacing = (2.0, 2.0, 2.0)
    local = rng.random((400, 3)) * np.array([66 + 6, 70 + 6, 45 + 6]) - 3
    xyz = torch.from_numpy((local * 2.0 + np.array(origin)).astype(np.float32)).reshape(1, 20, 20, 3)
    xyz_dev = xyz.to(device)
    cache.prefetch(xyz_dev, origin, spacing)
    cache.sync()
    o = torch.tensor(origin, device=device)
    inv = torch.tensor([1.0 / s for s in spacing], device=device)
    f32 = cache.grid_sample(xyz_dev, o, inv, diff=True).cpu().reshape(-1).numpy()
    u8 = cache.grid_sample(xyz_dev, o, inv).cpu().reshape(-1).numpy()
    want = _dense_expected(values, xyz, origin, spacing, cache.chunk_table)
    np.testing.assert_allclose(f32, want, rtol=0, atol=2e-3)
    np.testing.assert_array_equal(u8, np.clip(np.floor(f32.astype(np.float64) + 0.5), 0, 255).astype(np.uint8))
    # Prefetch covered every sample inside the chunk grid: none reads as unloaded.
    p = local
    in_grid = ((p >= 0) & (p < np.array([96, 96, 64]))).all(axis=1)
    c = np.floor(p[in_grid] / 32).astype(int)
    assert (cache.chunk_table.cpu().numpy()[c[:, 2], c[:, 1], c[:, 0]] >= 0).all()


@pytest.mark.parametrize("device", DEVICES)
def test_pool_growth_keeps_slots_and_backward_reads_the_current_pool(tmp_path, device, monkeypatch):
    monkeypatch.setattr(sparse_pool_cache, "_MIN_SLOTS", 1)
    monkeypatch.setattr(sparse_pool_cache, "_HEADROOM_SLOTS", 0)
    rng = np.random.default_rng(4)
    values = rng.integers(0, 256, (100, 100, 100), dtype=np.uint8)
    cache = _cache(_zarr(tmp_path, values, (32, 32, 32)), values, device)
    origin, spacing = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    o = torch.zeros(3, device=device)
    inv = torch.ones(3, device=device)
    a = torch.tensor([[[[10.3, 11.6, 12.2], [40.5, 20.25, 15.75]]]], device=device)
    cache.prefetch(a, origin, spacing)
    cache.sync()
    first_capacity = int(cache.pool.shape[0])
    grid = a.clone().requires_grad_(True)
    out = cache.grid_sample(grid, o, inv, diff=True)
    # Load far-away chunks between forward and backward, growing the pool.
    b = torch.tensor([[[[90.0, 90.0, 90.0], [70.0, 5.0, 95.0]]]], device=device)
    cache.prefetch(b, origin, spacing)
    cache.sync()
    assert int(cache.pool.shape[0]) > first_capacity and cache._pool_grows >= 2
    go = torch.tensor([[[[1.5, -2.0]]]], device=device)
    (out * go).sum().backward()
    expected = ops.sample_grad(cache.pool, cache.chunk_table, a, o, inv, go)
    torch.testing.assert_close(grid.grad, expected, rtol=0, atol=0)
    assert float(grid.grad.abs().sum()) > 0
    # Earlier slots survived the copies.
    for coord in [(0, 0, 0), (0, 0, 1)]:
        np.testing.assert_array_equal(_brick(cache, coord), padded_brick(values[None], coord))
    np.testing.assert_allclose(
        cache.grid_sample(b, o, inv, diff=True).cpu().reshape(-1).numpy(),
        _dense_expected(values, b.cpu(), origin, spacing, cache.chunk_table), atol=2e-3)


@pytest.mark.parametrize("device", DEVICES)
def test_diff_sample_backpropagates_to_the_grid(tmp_path, device):
    rng = np.random.default_rng(5)
    values = rng.integers(0, 256, (3, 50, 40, 70), dtype=np.uint8)
    cache = _cache(_zarr(tmp_path, values, (3, 16, 16, 16)), values, device,
                   channels=("nx", "ny"), channel_indices={"nx": 0, "ny": 1})
    xyz = torch.from_numpy((rng.random((2, 9, 11, 3)) * [72, 42, 52] - 1).astype(np.float32)).to(device)
    cache.prefetch(xyz, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    cache.sync()
    o = torch.zeros(3, device=device)
    inv = torch.ones(3, device=device)
    grid = xyz.clone().requires_grad_(True)
    out = cache.grid_sample(grid, o, inv, diff=True)
    assert out.shape == (2, 2, 9, 11) and out.requires_grad
    go = torch.randn(out.shape, generator=torch.Generator().manual_seed(0)).to(device)
    (out * go).sum().backward()
    torch.testing.assert_close(grid.grad, ops.sample_grad(cache.pool, cache.chunk_table, xyz, o, inv, go),
                               rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_check_guard_reports_unloaded_chunks(tmp_path, device, monkeypatch):
    values = np.random.default_rng(6).integers(1, 256, (64, 64, 64), dtype=np.uint8)
    cache = _cache(_zarr(tmp_path, values, (32, 32, 32)), values, device)
    monkeypatch.setenv("LASAGNA_CHECK_SPARSE_CACHE", "1")
    xyz = torch.tensor([[[[5.0, 5.0, 5.0]]]], device=device)
    o, inv = torch.zeros(3, device=device), torch.ones(3, device=device)
    with pytest.raises(RuntimeError, match="sparse chunk cache miss"):
        cache.grid_sample(xyz, o, inv)
    cache.prefetch(xyz, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    cache.sync()
    assert int(cache.grid_sample(xyz, o, inv).reshape(-1)[0]) == int(values[5, 5, 5])
    with pytest.raises(RuntimeError, match="non-finite"):
        cache.grid_sample(torch.tensor([[[[float("nan"), 5.0, 5.0]]]], device=device), o, inv)


@pytest.mark.parametrize("raw,mode", [(None, "auto"), ("", "auto"), ("0", "off"), ("1", "required")])
def test_metal_mode_values(monkeypatch, raw, mode):
    if raw is None:
        monkeypatch.delenv("LASAGNA_METAL", raising=False)
    else:
        monkeypatch.setenv("LASAGNA_METAL", raw)
    assert sparse_pool_cache.metal_mode() == mode


@pytest.mark.parametrize("raw", ["yes", "true", "2", "off"])
def test_metal_mode_rejects_other_values(monkeypatch, raw):
    monkeypatch.setenv("LASAGNA_METAL", raw)
    with pytest.raises(ValueError, match="LASAGNA_METAL"):
        sparse_pool_cache.metal_mode()


def _no_metal(monkeypatch, reason="ModuleNotFoundError: no sparse_pool_metal here"):
    monkeypatch.setattr(sparse_pool_cache, "_metal_module", None)
    monkeypatch.setattr(sparse_pool_cache, "_metal_error", reason)


def _never_import_metal(monkeypatch):
    def _fail():
        raise AssertionError("the Metal module must not be imported")
    monkeypatch.setattr(sparse_pool_cache, "_metal", _fail)


def test_metal_off_uses_torch_ops_without_importing_metal(monkeypatch):
    monkeypatch.setenv("LASAGNA_METAL", "0")
    _never_import_metal(monkeypatch)
    for device in ("cpu", "mps"):
        assert sparse_pool_cache.resolve_backend(torch.device(device)) == (ops, "torch (LASAGNA_METAL=0)")


def test_cpu_uses_torch_ops_and_never_imports_metal(monkeypatch):
    monkeypatch.delenv("LASAGNA_METAL", raising=False)
    _never_import_metal(monkeypatch)
    assert sparse_pool_cache.resolve_backend(torch.device("cpu")) == (ops, "torch")


def test_auto_on_mps_falls_back_with_the_reason(monkeypatch):
    monkeypatch.delenv("LASAGNA_METAL", raising=False)
    _no_metal(monkeypatch, "RuntimeError: self-check failed")
    module, label = sparse_pool_cache.resolve_backend(torch.device("mps"))
    assert module is ops
    assert label == "torch (metal unavailable: RuntimeError: self-check failed)"


def test_required_raises_without_metal_or_off_mps(tmp_path, monkeypatch):
    monkeypatch.setenv("LASAGNA_METAL", "1")
    _no_metal(monkeypatch)
    with pytest.raises(RuntimeError, match="Metal kernels are unavailable: ModuleNotFoundError"):
        sparse_pool_cache.resolve_backend(torch.device("mps"))
    values = np.zeros((4, 4, 4), np.uint8)
    with pytest.raises(RuntimeError, match="needs an MPS device"):
        _cache(_zarr(tmp_path, values, (4, 4, 4)), values, "cpu")


def test_unknown_value_fails_at_cache_construction(tmp_path, monkeypatch):
    monkeypatch.setenv("LASAGNA_METAL", "yes")
    values = np.zeros((4, 4, 4), np.uint8)
    with pytest.raises(ValueError, match="LASAGNA_METAL"):
        _cache(_zarr(tmp_path, values, (4, 4, 4)), values, "cpu")


class _CountingBackend:
    """Stands in for sparse_pool_metal: delegates to the torch ops and counts calls."""

    def __init__(self):
        self.calls = {}

    def __getattr__(self, name):
        fn = getattr(ops, name)

        def wrapped(*args, **kwargs):
            self.calls[name] = self.calls.get(name, 0) + 1
            return fn(*args, **kwargs)
        return wrapped


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_cache_routes_every_call_through_the_metal_backend(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("LASAGNA_METAL", raising=False)
    fake = _CountingBackend()
    monkeypatch.setattr(sparse_pool_cache, "_metal_module", fake)
    monkeypatch.setattr(sparse_pool_cache, "_metal_error", None)
    values = np.random.default_rng(8).integers(0, 256, (40, 40, 40), dtype=np.uint8)
    cache = _cache(_zarr(tmp_path, values, (16, 16, 16)), values, "mps")
    assert "sampler=metal" in capsys.readouterr().out
    xyz = torch.tensor([[[[5.5, 6.25, 7.75], [33.0, 20.0, 39.0]]]], device="mps")
    cache.prefetch(xyz, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    cache.sync()
    o, inv = torch.zeros(3, device="mps"), torch.ones(3, device="mps")
    cache.grid_sample(xyz, o, inv)
    grid = xyz.clone().requires_grad_(True)
    cache.grid_sample(grid, o, inv, diff=True).sum().backward()
    assert fake.calls == {"missing_chunks": 1, "sample_u8": 1, "sample_f32": 1, "sample_grad": 1}


def _write_manifest(root: Path, values: np.ndarray) -> Path:
    ZarrWriter.create(root / "surface.zarr", shape=values.shape, chunks=(3, 16, 16, 16), dtype="u1")[...] = values
    path = root / "vol.lasagna.json"
    path.write_text(json.dumps({
        "version": 2, "source_to_base": 1.0, "base_shape_zyx": list(values.shape[1:]),
        "grad_mag_encode_scale": 1000.0, "grad_mag_factor": 1.0,
        "groups": {"surface": {"zarr": "surface.zarr", "scaledown": 0, "channels": ["grad_mag", "nx", "ny"]}},
    }) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("device", DEVICES)
def test_load_3d_streaming_uses_the_pooled_cache_off_cuda(tmp_path, device):
    import fit_data

    values = np.random.default_rng(7).integers(0, 256, (3, 40, 50, 60), dtype=np.uint8)
    manifest = _write_manifest(tmp_path, values)
    data = fit_data.load_3d_streaming(path=str(manifest), device=torch.device(device))
    (cache,) = data.sparse_caches.values()
    assert isinstance(cache, PooledChunkGroupCache)
    assert cache.channels == ["grad_mag", "nx", "ny"]
    xyz = torch.tensor([[[[12.25, 30.5, 20.75], [59.0, 49.0, 39.0]]]], device=device)
    cache.prefetch(xyz, data.origin_fullres, data._spacing_for("grad_mag"))
    cache.sync()
    sampled = data.grid_sample_fullres(xyz, diff=True)
    x, y, z = 12.25, 30.5, 20.75
    x0, y0, z0 = int(x), int(y), int(z)
    fx, fy, fz = x - x0, y - y0, z - z0
    raw = 0.0
    v = values[0].astype(np.float64)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                w = (fx if dx else 1 - fx) * (fy if dy else 1 - fy) * (fz if dz else 1 - fz)
                raw += w * v[z0 + dz, y0 + dy, x0 + dx]
    assert abs(float(sampled.grad_mag.reshape(-1)[0]) - raw / 1000.0) < 1e-5
    assert abs(float(sampled.nx.reshape(-1)[1]) - (float(values[1, 39, 49, 59]) - 128.0) / 127.0) < 1e-6


@pytest.mark.parametrize("device", DEVICES)
def test_sync_refuses_chunk_coordinates_outside_the_grid(tmp_path, device):
    values = np.random.default_rng(9).integers(1, 256, (40, 40, 40), dtype=np.uint8)
    cache = _cache(_zarr(tmp_path, values, (16, 16, 16)), values, device)
    cache._pending = {(0, 0, 0): None, (9621, 9621, 49): None}
    with pytest.raises(RuntimeError, match="outside the chunk grid"):
        cache.sync()
    assert cache.loaded_chunks() == 0 and cache._used == 0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_missing_chunks_on_mps_matches_cpu_on_a_large_table():
    # torch.nonzero on MPS (torch 2.11) intermittently returns extra garbage rows for masks this size;
    # missing_chunks compacts on the CPU, so every call matches.
    gen = torch.Generator().manual_seed(3)
    table = torch.full((265, 128, 128), -1, dtype=torch.int32)
    table.view(-1)[torch.randperm(table.numel(), generator=gen)[: table.numel() // 5]] = 0
    xyz = (torch.rand(100_000, 3, generator=gen) * torch.tensor([128.0, 128.0, 265.0]) * 32.0).float()
    origin, spacing = torch.zeros(3), torch.ones(3)
    want = ops.missing_chunks(xyz, table, origin, spacing)
    args = [t.to("mps") for t in (xyz, table, origin, spacing)]
    for _ in range(30):
        torch.testing.assert_close(ops.missing_chunks(*args).cpu(), want, rtol=0, atol=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_mps_torch_backend_prefetches_on_the_host_and_mirrors_the_table(tmp_path, monkeypatch):
    # Off the CPU device the torch-ops prefetch runs on a host copy of the table (data-dependent
    # shapes on MPS compile a new graph each time), and sync() copies the whole table over.
    monkeypatch.setenv("LASAGNA_METAL", "0")
    rng = np.random.default_rng(10)
    values = rng.integers(0, 256, (70, 70, 70), dtype=np.uint8)
    cache = _cache(_zarr(tmp_path, values, (32, 32, 32)), values, "mps")
    seen = []
    real = ops.missing_chunks

    def spy(xyz, table, origin, spacing):
        seen.append((xyz.device.type, table.device.type))
        return real(xyz, table, origin, spacing)
    monkeypatch.setattr(ops, "missing_chunks", spy)
    for k in range(3):
        xyz = torch.from_numpy((rng.random((1, 7, 9, 3)) * 69).astype(np.float32)).to("mps")
        cache.prefetch(xyz, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        cache.sync()
        assert torch.equal(cache.chunk_table.cpu(), cache._table_host)
    assert seen == [("cpu", "cpu")] * 3
    assert cache.loaded_chunks() == int((cache.chunk_table.cpu() >= 0).sum()) > 0
    o, inv = torch.zeros(3, device="mps"), torch.ones(3, device="mps")
    got = cache.grid_sample(xyz, o, inv, diff=True).cpu().reshape(-1).numpy()
    np.testing.assert_allclose(got, _dense_expected(values, xyz.cpu(), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0),
                                                    cache.chunk_table), atol=2e-3)


def test_pool_headroom_absorbs_a_trickle_without_regrowing(tmp_path):
    values = np.random.default_rng(11).integers(1, 256, (64, 64, 96), dtype=np.uint8)
    cache = _cache(_zarr(tmp_path, values, (32, 32, 32)), values, "cpu")
    cache._pending = {(0, 0, x): None for x in range(2)}
    cache.sync()
    first = int(cache.pool.shape[0])
    assert first >= 2 + sparse_pool_cache._HEADROOM_SLOTS and cache._pool_grows == 1
    cache._pending = {(1, y, x): None for y in range(2) for x in range(3)}
    cache.sync()
    assert int(cache.pool.shape[0]) == first and cache._pool_grows == 1  # no copy for a few more chunks
    for coord in [(0, 0, 0), (1, 1, 2)]:
        np.testing.assert_array_equal(_brick(cache, coord), padded_brick(values[None], coord))


def test_cpu_pool_lives_on_its_own_mapping_and_is_released_on_growth(tmp_path, monkeypatch):
    # A freed torch.empty block stays in a macOS process footprint; the pool's own mmap is unmapped instead.
    import gc
    import weakref
    monkeypatch.setattr(sparse_pool_cache, "_HEADROOM_SLOTS", 0)
    monkeypatch.setattr(sparse_pool_cache, "_MIN_SLOTS", 1)
    values = np.random.default_rng(12).integers(1, 256, (64, 64, 96), dtype=np.uint8)
    cache = _cache(_zarr(tmp_path, values, (32, 32, 32)), values, "cpu")
    cache._pending = {(0, 0, 0): None}
    cache.sync()
    first = cache.pool
    assert first.untyped_storage().nbytes() == first.numel()
    ref = weakref.ref(first.untyped_storage())
    del first
    cache._pending = {(0, y, x): None for y in range(2) for x in range(3)}
    cache.sync()
    gc.collect()
    assert ref() is None  # the first pool's storage (and its mapping) is gone
    for coord in [(0, 0, 0), (0, 1, 2)]:
        np.testing.assert_array_equal(_brick(cache, coord), padded_brick(values[None], coord))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_mps_pool_growth_empties_the_mps_cache(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: calls.append(1))
    values = np.random.default_rng(13).integers(1, 256, (40, 40, 40), dtype=np.uint8)
    cache = _cache(_zarr(tmp_path, values, (16, 16, 16)), values, "mps")
    cache._pending = {(0, 0, 0): None}
    cache.sync()
    assert calls == [1]


def test_load_3d_streaming_caps_the_reader_cache_off_cuda(tmp_path, monkeypatch):
    import fit_data

    values = np.random.default_rng(14).integers(0, 256, (3, 40, 50, 60), dtype=np.uint8)
    manifest = _write_manifest(tmp_path, values)
    budgets = []
    real = fit_data.open_zarr

    def spy(*args, **kwargs):
        budgets.append(kwargs.get("memory_cache"))
        return real(*args, **kwargs)
    monkeypatch.setattr(fit_data, "open_zarr", spy)
    fit_data.load_3d_streaming(path=str(manifest), device=torch.device("cpu"))
    assert budgets == [sparse_pool_cache.POOLED_READER_MEMORY_CACHE_BYTES]


def _surface_manifest(root: Path, values: np.ndarray, chunks) -> Path:
    ZarrWriter.create(root / "surface.zarr", shape=values.shape, chunks=chunks, dtype="u1")[...] = values
    path = root / "vol.lasagna.json"
    path.write_text(json.dumps({
        "version": 2, "source_to_base": 1.0, "base_shape_zyx": list(values.shape[1:]),
        "grad_mag_encode_scale": 1000.0, "grad_mag_factor": 1.0,
        "groups": {"surface": {"zarr": "surface.zarr", "scaledown": 0, "channels": ["grad_mag", "nx", "ny"]}},
    }) + "\n")
    return path


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(1, 1, 1), (1, 5, 40), (32, 32, 32), (33, 2, 3)])
def test_tiny_volume_streams_end_to_end(tmp_path, monkeypatch, device, shape):
    # One-chunk grids and one-voxel axes through the fit's loader: prefetch, sync, the checked
    # differentiable sample and its backward; exact corner voxels read back exactly.
    import fit_data
    rng = np.random.default_rng(0)
    values = rng.integers(0, 256, (3, *shape), dtype=np.uint8)
    data = fit_data.load_3d_streaming(path=str(_surface_manifest(tmp_path, values, (3, 16, 16, 16))),
                                      device=torch.device(device))
    (cache,) = data.sparse_caches.values()
    assert isinstance(cache, PooledChunkGroupCache)
    Z, Y, X = shape
    pts = torch.tensor(rng.random((50, 3)) * np.array([X + 1, Y + 1, Z + 1]) - 0.5, dtype=torch.float32)
    pts = torch.cat([pts, torch.tensor([[0.0, 0.0, 0.0], [X - 1.0, Y - 1.0, Z - 1.0]])]).to(device)
    xyz = pts.view(1, 1, -1, 3).clone().requires_grad_(True)
    cache.prefetch(xyz, data.origin_fullres, data._spacing_for(cache.channels[0]))
    cache.sync()
    monkeypatch.setenv("LASAGNA_CHECK_SPARSE_CACHE", "1")
    out = data.grid_sample_fullres(xyz, diff=True)
    (out.grad_mag.sum() + out.nx.sum() + out.ny.sum()).backward()
    assert torch.isfinite(xyz.grad).all()
    corners = cache.grid_sample(pts[-2:].view(1, 1, 2, 3), torch.tensor(data.origin_fullres, device=device),
                                torch.tensor([1.0 / v for v in data._spacing_for("grad_mag")], device=device))
    assert corners[:, 0, 0, 0].tolist() == values[:, 0, 0, 0].tolist()
    assert corners[:, 0, 0, 1].tolist() == values[:, Z - 1, Y - 1, X - 1].tolist()


def test_cuda_keeps_the_cuda_cache_and_its_reader_budget(tmp_path):
    # "CUDA and ROCm unchanged", checkable without a GPU: a cuda device still gets
    # sparse_cache.SparseChunkGroupCache and the 8 GiB reader cache; other devices get the pooled cache.
    from unittest import mock
    import fit_data
    import sparse_cache

    class _FakeCudaCache:
        def __init__(self, **kw):
            self.channels = kw["channels"]

    budgets = []
    real_open = fit_data.open_zarr

    def _spy(*args, **kwargs):
        budgets.append(kwargs.get("memory_cache"))
        return real_open(*args, **kwargs)

    manifest = _surface_manifest(tmp_path, np.zeros((3, 8, 8, 8), np.uint8), (3, 8, 8, 8))
    with mock.patch("sparse_cache.SparseChunkGroupCache", _FakeCudaCache), \
            mock.patch.object(fit_data, "open_zarr", _spy):
        cuda = fit_data.load_3d_streaming(path=str(manifest), device=torch.device("cuda"))
        cpu = fit_data.load_3d_streaming(path=str(manifest), device=torch.device("cpu"))
    assert [type(c) for c in cuda.sparse_caches.values()] == [_FakeCudaCache]
    assert [type(c) for c in cpu.sparse_caches.values()] == [PooledChunkGroupCache]
    assert budgets == [sparse_cache.READER_MEMORY_CACHE_BYTES, sparse_pool_cache.POOLED_READER_MEMORY_CACHE_BYTES]
    assert sparse_cache.READER_MEMORY_CACHE_BYTES == 8 * 1024**3


def test_metal_label_states_a_rejected_no_contraction_pragma(monkeypatch):
    fake = type("FakeMetal", (), {"CONTRACTION": "compiler default"})()
    monkeypatch.delenv("LASAGNA_METAL", raising=False)
    monkeypatch.setattr(sparse_pool_cache, "_metal", lambda: fake)
    assert sparse_pool_cache.resolve_backend(torch.device("mps")) == (fake, "metal (contraction: compiler default)")
    fake.CONTRACTION = "off"
    assert sparse_pool_cache.resolve_backend(torch.device("mps")) == (fake, "metal")


def test_cpu_pool_map_failure_is_a_memory_error_with_diagnostics(tmp_path, capsys, monkeypatch):
    import errno
    import mmap as _mmap

    values = np.random.default_rng(15).integers(0, 256, (40, 50, 60), dtype=np.uint8)
    cache = _cache(_zarr(tmp_path, values, (16, 16, 16)), values, "cpu")
    cache.prefetch(torch.tensor([[[[10.0, 10.0, 10.0]]]]), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

    def _enomem(*args, **kwargs):
        raise OSError(errno.ENOMEM, "Cannot allocate memory")
    monkeypatch.setattr(_mmap, "mmap", _enomem)
    with pytest.raises(MemoryError, match="CPU chunk pool"):
        cache.sync()
    assert "[sparse_cache_oom] sync:" in capsys.readouterr().out
    assert cache.loaded_chunks() == 0
