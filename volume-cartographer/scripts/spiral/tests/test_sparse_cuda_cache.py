from pathlib import Path

import numpy as np
import pytest
import tensorstore as ts
import torch

from sdt_losses import sample_sdt_trilinear
from sparse_cuda_cache import (
    CHUNK_VOXELS,
    BoundedSparseCudaCache,
    SparseScalarStore,
)


def write_array(path: Path, data: np.ndarray) -> str:
    array = ts.open(
        {
            "driver": "zarr",
            "kvstore": {"driver": "file", "path": str(path)},
            "metadata": {
                "dtype": "|u1",
                "shape": list(data.shape),
                "chunks": [16, 16, 16],
                "compressor": None,
                "order": "C",
                "fill_value": 0,
            },
        },
        create=True,
        open=True,
    ).result()
    array.write(data).result()
    return str(path)


def test_gather_matches_dense_across_lru_evictions(tmp_path):
    z, y, x = np.indices((40, 40, 70))
    first = ((z * 17 + y * 5 + x) % 251 + 1).astype(np.uint8)
    second = ((z * 3 + y * 11 + x * 7) % 251 + 1).astype(np.uint8)
    paths = [
        write_array(tmp_path / "first", first),
        write_array(tmp_path / "second", second),
    ]
    cache = BoundedSparseCudaCache(
        source_paths=paths,
        shape_zyx=first.shape,
        budget_bytes=2 * 2 * CHUNK_VOXELS,
        device="cpu",
        label="test normals",
        tensorstore_cache_bytes=1 << 20,
    )

    requests = [
        torch.tensor([[0, 0, 0], [1, 2, 33], [5, 3, 63]]),
        torch.tensor([[35, 2, 2], [35, 35, 35]]),
        torch.tensor([[0, 0, 0], [5, 3, 63]]),
    ]
    for indices in requests:
        actual = cache.gather(indices)
        expected = torch.from_numpy(np.stack([
            first[tuple(indices.numpy().T)], second[tuple(indices.numpy().T)]
        ], axis=-1))
        torch.testing.assert_close(actual, expected)
    cache.gather(requests[-1])

    stats = cache.stats()
    assert stats["resident_chunks"] <= 2
    assert stats["evictions"] > 0
    assert stats["hits"] > 0


def test_one_gather_must_fit_the_bounded_working_set(tmp_path):
    data = np.arange(40 * 40 * 70, dtype=np.uint32).reshape(40, 40, 70)
    data = (data % 255).astype(np.uint8)
    path = write_array(tmp_path / "scalar", data)
    cache = BoundedSparseCudaCache(
        source_paths=[path],
        shape_zyx=data.shape,
        budget_bytes=CHUNK_VOXELS,
        device="cpu",
        label="test scalar",
        tensorstore_cache_bytes=1 << 20,
    )
    with pytest.raises(RuntimeError, match="exceeding its 1-chunk LRU capacity"):
        cache.gather(torch.tensor([[0, 0, 0], [0, 0, 33]]))


def test_large_z_roi_uses_source_origin_and_eviction(tmp_path):
    data = np.broadcast_to(
        (np.arange(5200, dtype=np.uint16) % 251 + 1).astype(np.uint8)[:, None, None],
        (5200, 2, 2),
    ).copy()
    path = write_array(tmp_path / "large_z", data)
    cache = BoundedSparseCudaCache(
        source_paths=[path],
        shape_zyx=data.shape,
        origin_zyx=(100, 0, 0),
        budget_bytes=CHUNK_VOXELS,
        device="cpu",
        label="test 5000-slice ROI",
        tensorstore_cache_bytes=1 << 20,
    )

    first = cache.gather(torch.tensor([[0, 0, 0]]))
    last = cache.gather(torch.tensor([[4999, 1, 1]]))
    assert int(first[0, 0]) == int(data[100, 0, 0])
    assert int(last[0, 0]) == int(data[5099, 1, 1])
    assert cache.stats()["evictions"] == 1


def test_sparse_sdt_sampling_matches_dense(tmp_path):
    x = np.arange(70, dtype=np.float32)
    encoded = (
        np.clip(np.rint(np.abs(x - 35.0) - 2.0), -127, 127) + 128
    ).astype(np.uint8)
    data = np.broadcast_to(encoded, (6, 6, 70)).copy()
    path = write_array(tmp_path / "sdt", data)
    cache = BoundedSparseCudaCache(
        source_paths=[path],
        shape_zyx=data.shape,
        budget_bytes=3 * CHUNK_VOXELS,
        device="cpu",
        label="test sdt",
        tensorstore_cache_bytes=1 << 20,
    )
    dense = {
        "backend": "dense_test",
        "kind": "sdt",
        "volume": torch.from_numpy(data),
        "z_origin": 0,
        "scale_zyx": (1.0, 1.0, 1.0),
        "unit": 1.0,
        "offset": 128,
        "cap": 127.0,
        "shape": data.shape,
        "fingerprint": {},
    }
    sparse = {
        **dense,
        "backend": "sparse_cuda",
        "store": SparseScalarStore(cache),
    }
    sparse.pop("volume")
    points_dense = (
        torch.rand([256, 3]) * torch.tensor([4.0, 4.0, 68.0]) + 0.5
    ).requires_grad_(True)
    points_sparse = points_dense.detach().clone().requires_grad_(True)
    dense_value, dense_valid, dense_corners = sample_sdt_trilinear(
        dense, points_dense
    )
    sparse_value, sparse_valid, sparse_corners = sample_sdt_trilinear(
        sparse, points_sparse
    )
    torch.testing.assert_close(sparse_value, dense_value)
    torch.testing.assert_close(sparse_valid, dense_valid)
    torch.testing.assert_close(sparse_corners, dense_corners)

    dense_value.sum().backward()
    sparse_value.sum().backward()
    torch.testing.assert_close(points_sparse.grad, points_dense.grad)
