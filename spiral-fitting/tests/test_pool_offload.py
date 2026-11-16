"""pool_offload (FIT_SPIRAL_POOL_OFFLOAD): the host-RAM pool must return EXACTLY what the resident pool returns.
Builds a small synthetic sidecar with the packer the fit uses (no real scroll data), then gathers the same random and edge indices through ResidentBrickPool and the offloaded subclass.
Needs a CUDA card whose driver supports cuMemHostRegister(DEVICEMAP) and NVRTC; skipped otherwise (CPU-only CI)."""
import os
import sys

import numpy as np
import pytest
import torch
import zarr

pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("pool offload needs a CUDA device", allow_module_level=True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import brick_pool_offload                              # noqa: E402
import sparse_cuda_cache as scc                       # noqa: E402
from pack_resident_pools import pack_arrays   # noqa: E402


def _write(path, data):
    a = zarr.open(str(path), mode="w", shape=data.shape, chunks=(16, 16, 16), dtype="|u1", compressor=None, fill_value=0, zarr_format=2, dimension_separator="/")
    a[:] = data
    return str(path)


@pytest.fixture(scope="module")
def sidecar(tmp_path_factory):
    d = tmp_path_factory.mktemp("offload")
    z, y, x = np.indices((48, 64, 80))
    first = ((z * 17 + y * 5 + x) % 251 + 1).astype(np.uint8)
    second = ((z * 3 + y * 11 + x * 7) % 251 + 1).astype(np.uint8)
    first[:16] = 0                                     # an empty brick row: absent bricks must read zero in both pools
    paths = [_write(d / f"a{i}", a) for i, a in enumerate((first, second))]
    return pack_arrays(paths, paths[0] + ".pool", label="offload_test"), first, second


@pytest.fixture(scope="module")
def pools(sidecar):
    out = sidecar[0]
    Base = scc.ResidentBrickPool
    Off = brick_pool_offload.offload_class(scc)
    return Base(out, device="cuda", label="resident"), Off(out, device="cuda", label="offload")


def _indices(shape, n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.stack([torch.randint(0, shape[0], (n,), generator=g), torch.randint(0, shape[1], (n,), generator=g), torch.randint(0, shape[2], (n,), generator=g)], 1).cuda()


def test_offload_gather_is_identical_to_resident(pools, sidecar):
    resident, offload = pools
    shape = sidecar[1].shape
    for n, seed in ((1, 0), (1000, 1), (100_000, 2)):
        idx = _indices(shape, n, seed)
        a, b = resident.gather(idx), offload.gather(idx)
        torch.cuda.synchronize()
        assert torch.equal(a, b), f"gather differs at n={n}"
        assert int((a > 0).any(1).sum()) > 0           # the comparison saw real data, not only zeros


def test_offload_matches_the_dense_array_and_zero_fills_absent_bricks(pools, sidecar):
    _, offload = pools
    _, first, second = sidecar
    idx = torch.tensor([[0, 0, 0], [15, 63, 79], [16, 0, 0], [47, 63, 79], [30, 31, 40]]).cuda()
    out = offload.gather(idx).cpu().numpy()
    ii = idx.cpu().numpy().T
    expected = np.stack([first[tuple(ii)], second[tuple(ii)]], -1)
    np.testing.assert_array_equal(out, expected)       # includes the first (empty) brick row: zeros


def test_empty_gather_and_pool_lives_in_host_memory(pools):
    resident, offload = pools
    assert offload.gather(torch.zeros(0, 3, dtype=torch.long).cuda()).shape[0] == 0
    assert isinstance(offload, scc.ResidentBrickPool)
    # the offloaded pool must not hold the pool bytes on the card: its device footprint is at most the brick table plus bookkeeping
    free0, _ = torch.cuda.mem_get_info()
    assert getattr(offload, "pool_bytes", 0) > 0
    assert not (isinstance(getattr(offload, "pool", None), torch.Tensor) and offload.pool.is_cuda), "pool bytes are on the card"


def test_factory_default_is_resident_and_env_selects_offload(monkeypatch):
    monkeypatch.delenv("FIT_SPIRAL_POOL_OFFLOAD", raising=False)
    assert scc.resident_brick_pool_class() is scc.ResidentBrickPool
    monkeypatch.setenv("FIT_SPIRAL_POOL_OFFLOAD", "1")
    cls = scc.resident_brick_pool_class()
    assert cls is not scc.ResidentBrickPool and issubclass(cls, scc.ResidentBrickPool)


def _count_unregisters(monkeypatch):
    calls = []
    real = brick_pool_offload._unregister
    monkeypatch.setattr(brick_pool_offload, "_unregister", lambda arr: (calls.append(arr.ctypes.data), real(arr))[1])
    return calls


def test_a_failed_construction_unregisters_the_host_memory(sidecar, monkeypatch):
    """Review: a failure after cuMemHostRegister (kernel load, device allocation) must not leave the pool's pages registered."""
    calls = _count_unregisters(monkeypatch)
    monkeypatch.setattr(brick_pool_offload, "_module", lambda dev: (_ for _ in ()).throw(RuntimeError("kernel load failed")))
    Off = brick_pool_offload.offload_class(scc)
    with pytest.raises(RuntimeError, match="kernel load failed"):
        Off(sidecar[0], device="cuda", label="failing")
    assert len(calls) == 1, "the registered pool memory was not unregistered"


def test_close_unregisters_once_and_an_abandoned_pool_is_unregistered_by_its_finalizer(sidecar, monkeypatch):
    import gc
    calls = _count_unregisters(monkeypatch)
    Off = brick_pool_offload.offload_class(scc)
    closed = Off(sidecar[0], device="cuda", label="closed")
    closed.close(); closed.close()
    assert len(calls) == 1
    abandoned = Off(sidecar[0], device="cuda", label="abandoned")
    del abandoned; gc.collect()
    assert len(calls) == 2, "an abandoned pool kept its host memory registered"


def test_the_gather_stays_on_the_pools_card_when_another_device_becomes_current(sidecar):
    """Review: a pool built for cuda:1 must allocate, launch and return on cuda:1 whatever the current device is later."""
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two CUDA devices")
    Off = brick_pool_offload.offload_class(scc)
    torch.cuda.set_device(0)
    pool = Off(sidecar[0], device="cuda:1", label="card1")
    assert torch.cuda.current_device() == 0, "building an offloaded pool must not change the caller's current device"
    idx = torch.tensor([[0, 0, 0], [15, 63, 79], [16, 0, 0], [47, 63, 79], [30, 31, 40]], device="cuda:1")
    out = pool.gather(idx)
    torch.cuda.synchronize(1)
    assert out.device == torch.device("cuda:1")
    ii = idx.cpu().numpy().T
    np.testing.assert_array_equal(out.cpu().numpy(), np.stack([sidecar[1][tuple(ii)], sidecar[2][tuple(ii)]], -1))


def test_verify_mirror_is_released_on_close_and_the_exit_callback_does_not_hold_the_pool(sidecar, monkeypatch):
    """Review: with FIT_SPIRAL_POOL_OFFLOAD_VERIFY the card holds a full mirror of the pool; close() must free it, and the atexit callback must not keep the pool (and its registered host memory) alive."""
    import gc
    import weakref
    monkeypatch.setenv("FIT_SPIRAL_POOL_OFFLOAD_VERIFY", "5")
    Off = brick_pool_offload.offload_class(scc)
    pool = Off(sidecar[0], device="cuda", label="verify_release")
    assert pool._shadow is not None
    ref = weakref.ref(pool)
    pool.close()
    assert pool._shadow is None and pool._verify_left == 0
    del pool
    gc.collect()
    assert ref() is None, "something (the exit callback) still holds the closed pool"
