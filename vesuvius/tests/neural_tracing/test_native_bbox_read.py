"""Real native bbox API regression fixtures; no network or GPU required."""
import itertools
from types import SimpleNamespace

import numpy as np
import pytest
import zarr


def make_volume(root, dtype="uint8", fill=37, shape=(5, 6, 7)):
    from vc import Volume
    group = zarr.open_group(str(root), mode="w", zarr_format=2)
    array = group.create_array("0", shape=shape, chunks=(4, 4, 4), dtype=dtype,
                               fill_value=fill, compressor=None)
    group.attrs["multiscales"] = [{"axes": [{"name": n, "type": "space"} for n in "zyx"],
                                  "datasets": [{"path": "0"}]}]
    data = (np.arange(np.prod(shape)).reshape(shape) % 193 + 1).astype(dtype)
    if dtype == "uint16":
        data *= 257
    array[:] = data
    return Volume.open(str(root)), data, array


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
@pytest.mark.parametrize("mask,stats", list(itertools.product([False, True], repeat=2)))
def test_bbox_dtype_fill_mask_stats_and_xyz(tmp_path, dtype, mask, stats):
    fill = 37 if dtype == "uint8" else 40000
    volume, data, _ = make_volume(tmp_path / "volume.zarr", dtype, fill)
    offset, shape = (-1, 2, 3), (7, 6, 7)
    expected = np.full(shape, fill, dtype=dtype)
    expected_mask = np.zeros(shape, np.uint8)
    expected[1:6, :4, :4] = data[:, 2:, 3:]
    expected_mask[1:6, :4, :4] = 1
    result = volume.read_zyx(offset, shape, with_mask=mask, with_stats=stats)
    if mask or stats:
        values, valid, metadata = result
        if mask:
            np.testing.assert_array_equal(valid, expected_mask)
        else:
            assert valid is None
        if stats:
            assert metadata["covered_pixels"] == 80
            assert metadata["requested_level_only"] and metadata["fallback_levels"] == 0
        else:
            assert metadata is None
    else:
        values = result
    assert values.dtype == np.dtype(dtype) and values.flags.c_contiguous
    np.testing.assert_array_equal(values, expected)
    np.testing.assert_array_equal(volume.read_xyz(offset[::-1], shape[::-1]), expected)
    np.testing.assert_array_equal(volume[:], data)
    assert not hasattr(volume, "sample_zyx_block")


@pytest.mark.parametrize("dtype,fill", [("uint8", 37), ("uint16", 40000)])
def test_missing_and_all_fill_are_valid_declared_fill(tmp_path, dtype, fill):
    root = tmp_path / "volume.zarr"
    _, _, array = make_volume(root, dtype, fill, (8, 4, 4))
    array[:] = fill
    # Explicit all-fill payload plus absent sibling, independent of Zarr's
    # optimization that normally omits fill chunks.
    (root / "0/0.0.0").write_bytes(np.full((4, 4, 4), fill, dtype=dtype).tobytes())
    (root / "0/1.0.0").unlink(missing_ok=True)
    from vc import Volume
    volume = Volume.open(str(root))
    values, mask, stats = volume.read_zyx((0, 0, 0), (8, 4, 4), with_mask=True, with_stats=True)
    assert np.all(values == fill) and np.all(mask)
    assert stats["requested_chunks"] == 2


def test_exact_level_empty_outside_and_invalid_shape(tmp_path):
    volume, _, _ = make_volume(tmp_path / "volume.zarr")
    for shape in ((0, 3, 4), (2, 3, 4)):
        with pytest.raises((IndexError, RuntimeError), match="missing.*level"):
            volume.read_zyx((100, 100, 100), shape, level=1)
    with pytest.raises((IndexError, RuntimeError), match="non-negative"):
        volume.read_zyx((0, 0, 0), (1, 1, 1), level=-1)
    with pytest.raises((OverflowError, ValueError, TypeError)):
        volume.read_zyx((0, 0, 0), (2**31, 1, 1))
    values, mask, stats = volume.read_zyx((100, 100, 100), (2, 3, 4), with_mask=True, with_stats=True)
    assert np.all(values == 37) and not np.any(mask)
    assert stats["requested_chunks"] == stats["covered_pixels"] == 0
    assert volume.read_zyx((0, 0, 0), (0, 3, 4)).shape == (0, 3, 4)
    with pytest.raises(TypeError):
        volume.read_zyx((0, 0, 0), (1, 1, 1), missing_policy="all_fill")


@pytest.mark.parametrize("size", [3, 65])
def test_corrupt_chunk_is_not_silently_fill(tmp_path, size):
    root = tmp_path / "volume.zarr"
    make_volume(root)
    (root / "0/0.0.0").write_bytes(b"x" * size)
    from vc import Volume
    volume = Volume.open(str(root))
    with pytest.raises(RuntimeError):
        volume.read_zyx((0, 0, 0), (4, 4, 4))


@pytest.mark.parametrize("offset", [-(2**31), 2**31-1])
def test_extreme_offsets_do_not_overflow(tmp_path, offset):
    volume, _, _ = make_volume(tmp_path / "volume.zarr")
    values, mask, stats = volume.read_zyx((offset,) * 3, (2,) * 3,
                                         with_mask=True, with_stats=True)
    assert np.all(values == 37) and not np.any(mask)
    assert stats["requested_chunks"] == 0


def test_concurrent_reads_share_volume_and_preserve_exact_values(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    volume, data, _ = make_volume(tmp_path / "volume.zarr", dtype="uint16", fill=40000)
    def read(index):
        mask, stats = bool(index % 2), bool(index % 3)
        result = volume.read_zyx((0, 0, 0), data.shape, with_mask=mask, with_stats=stats)
        values = result[0] if mask or stats else result
        np.testing.assert_array_equal(values, data)
        return True
    with ThreadPoolExecutor(max_workers=16) as workers:
        assert all(workers.map(read, range(64)))


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_inference_calls_raw_full_bbox_without_aux_or_conversion(dtype):
    from lasagna.demand_predict3d import DemandGeometry
    from vesuvius.neural_tracing.fiber_trace_3d.live_evaluate import NativeTileReader
    geometry = DemandGeometry((8, 8, 8), 4, 2, 1, 1, 2)
    raw = np.full((4, 4, 4), 37, dtype=dtype)
    calls = []
    class Volume:
        def read_zyx(self, offset, shape, level):
            calls.append((offset, shape, level))
            return raw
    # No sample_block_zyx attribute: invoking the training adapter fails.
    sampler = SimpleNamespace(volume=Volume(), level=2)
    reader = NativeTileReader(sampler, geometry, dtype)
    assert reader((0, 0, 0)) is raw
    assert calls == [((-1, -1, -1), (4, 4, 4), 2)]


def test_inference_boundary_uses_native_fill_not_reflection(tmp_path):
    from lasagna.demand_predict3d import DemandGeometry
    from vesuvius.neural_tracing.fiber_trace_3d.live_evaluate import NativeTileReader
    volume, data, _ = make_volume(tmp_path / "volume.zarr", shape=(4, 4, 4))
    geometry = DemandGeometry(data.shape, 4, 2, 1, 1, 2)
    reader = NativeTileReader(SimpleNamespace(volume=volume, level=0), geometry, "uint8")
    expected = np.full((4, 4, 4), 37, np.uint8)
    expected[1:, 1:, 1:] = data[:3, :3, :3]
    np.testing.assert_array_equal(reader((0, 0, 0)), expected)
