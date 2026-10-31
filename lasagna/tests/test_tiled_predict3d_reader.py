"""run_tiled_inference_3d reads its tiles through an opened ZarrReader."""
import numpy as np
import pytest
import torch
from vc_reader.reader import MemoryCache, ZarrWriter, open_zarr

from lasagna.tiled_predict3d import OutputProductSpec, _input_has_chunks, run_tiled_inference_3d
from lasagna.tests.predict3d_spawn_helpers import SpawnIdentityAdapter


class _Output:
	def __init__(self):
		self.chunks = {}

	def product_chunk_complete(self, product, *, chunk_origin_zyx):
		return False

	def write_product_chunk(self, product, *, chunk_origin_zyx, data):
		self.chunks[chunk_origin_zyx] = np.array(data["value"], copy=True)


@pytest.mark.parametrize("devices", [None, (torch.device("cpu"), torch.device("cpu"))], ids=("serial", "multi-device"))
def test_inference_from_a_reader_matches_the_array_it_holds(tmp_path, devices):
	product = OutputProductSpec(
		name="identity", level=0, scaledown=1, inference_scaledown=1, channels=("value",), chunk_size=4,
	)
	volume = (np.arange(8 * 8 * 8, dtype=np.uint32) % 256).astype(np.uint8).reshape(8, 8, 8)
	path = tmp_path / "input.zarr"
	ZarrWriter.create(path, shape=volume.shape, chunks=(4, 4, 4), dtype="u1", compressor="blosc", cname="zstd")[...] = volume
	reader = open_zarr(path, memory_cache=MemoryCache(1 << 20), threads=2)

	outputs = []
	for source in (reader, volume):
		output = _Output()
		adapter = SpawnIdentityAdapter(product)
		run_tiled_inference_3d(
			None if devices else adapter.load_model(device=torch.device("cpu")), source,
			crop_slices=(0, 8, 0, 8, 0, 8), device=torch.device("cpu"), model_adapter=adapter, output_adapter=output,
			products=(product,), output_regions_zyx={"identity": (0, 0, 0, 8, 8, 8)},
			full_output_shapes_zyx={"identity": (8, 8, 8)}, input_zarr_path=str(path), tile_size=4,
			overlap=0, border=0, tmp_dir=str(tmp_path), prefetch_tiles_per_gpu=2, devices=devices,
			**({"slots_per_gpu": 1} if devices else {}),
		)
		outputs.append(output.chunks)
	assert len(outputs[0]) == 8 and set(outputs[0]) == set(outputs[1])
	for origin, chunk in outputs[0].items():
		np.testing.assert_array_equal(chunk, outputs[1][origin])


def test_input_chunk_presence_is_what_the_array_stores(tmp_path):
	path = tmp_path / "input.zarr"
	array = ZarrWriter.create(path, shape=(8, 8, 8), chunks=(4, 4, 4), dtype="u1", separator="/")
	array[4:8, 0:4, 0:4] = 1
	assert _input_has_chunks(str(path), 4, 8, 0, 8, 0, 8)
	assert _input_has_chunks(str(path), 0, 5, 0, 1, 0, 1)
	assert not _input_has_chunks(str(path), 0, 4, 0, 8, 0, 8)
