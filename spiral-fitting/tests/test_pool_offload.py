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
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "pool_offload"))
import rb_offload                                      # noqa: E402
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
    Off = rb_offload.install(scc)
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
