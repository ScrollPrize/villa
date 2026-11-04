import numpy as np
import pytest
import torch
from vc_reader.reader import ZarrWriter, open_zarr

from sparse_cache import SparseChunkGroupCache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the cache holds chunks on a CUDA device")


def _padded(values: np.ndarray, coord: tuple[int, int, int]) -> np.ndarray:
	"""The 34^3 brick of `coord`, 1 voxel margin each side, zero outside the array."""
	out = np.zeros((34, 34, 34), np.uint8)
	src = tuple(slice(max(0, c * 32 - 1), min(n, c * 32 + 33)) for c, n in zip(coord, values.shape[-3:]))
	dst = tuple(slice(s.start - (c * 32 - 1), s.stop - (c * 32 - 1)) for s, c in zip(src, coord))
	out[dst] = values[(..., *src)]
	return out


def test_sync_reads_padded_bricks_of_a_3d_array(tmp_path):
	values = np.random.default_rng(1).integers(1, 256, (40, 70, 33), dtype=np.uint8)
	ZarrWriter.create(tmp_path / "a.zarr", shape=values.shape, chunks=(16, 16, 16), dtype="u1")[...] = values
	cache = SparseChunkGroupCache(
		channels=["cos"], reader=open_zarr(tmp_path / "a.zarr"), vol_shape_zyx=values.shape,
		channel_indices={"cos": 0}, is_3d_zarr=True, device=torch.device("cuda"),
	)
	coords = [(0, 0, 0), (1, 2, 1), (0, 1, 0)]
	cache._pending = dict.fromkeys(coords)
	cache.sync()

	batch = cache._batches[0].cpu().numpy()
	assert batch.shape == (3, 1, 34, 34, 34)
	for brick, coord in zip(batch, coords):
		np.testing.assert_array_equal(brick[0], _padded(values, coord))
		pointer = int(cache.chunk_table[coord])
		assert pointer == cache._batches[0].data_ptr() + coords.index(coord) * 34**3
	assert cache.loaded_chunks() == 3
	cache.sync()
	assert cache.loaded_chunks() == 3


def test_sync_reads_the_selected_channels_of_a_4d_array(tmp_path):
	values = np.random.default_rng(2).integers(1, 256, (3, 40, 40, 40), dtype=np.uint8)
	ZarrWriter.create(tmp_path / "a.zarr", shape=values.shape, chunks=(3, 16, 16, 16), dtype="u1")[...] = values
	cache = SparseChunkGroupCache(
		channels=["ny", "nx"], reader=open_zarr(tmp_path / "a.zarr"), vol_shape_zyx=values.shape[1:],
		channel_indices={"nx": 0, "ny": 2}, is_3d_zarr=False, device=torch.device("cuda"),
	)
	cache._pending = {(1, 0, 1): None}
	cache.sync()

	brick = cache._batches[0].cpu().numpy()[0]
	np.testing.assert_array_equal(brick[0], _padded(values[2], (1, 0, 1)))
	np.testing.assert_array_equal(brick[1], _padded(values[0], (1, 0, 1)))


def test_non_uint8_array_is_refused(tmp_path):
	ZarrWriter.create(tmp_path / "a.zarr", shape=(4, 4, 4), chunks=(4, 4, 4), dtype="u2")
	with pytest.raises(ValueError, match="uint8"):
		SparseChunkGroupCache(
			channels=["cos"], reader=open_zarr(tmp_path / "a.zarr"), vol_shape_zyx=(4, 4, 4),
			channel_indices={"cos": 0}, is_3d_zarr=True, device=torch.device("cuda"),
		)
