from __future__ import annotations

from concurrent.futures import Future

import numpy as np
import torch

from lasagna import tiled_predict3d as tiled


class _InputLevelSpy:
    """Stands in for a remote InputLevel: every region is stored, reads resolve at once."""

    def __init__(self, shape, read_ahead_tiles: int = 8) -> None:
        self.shape = tuple(shape)
        self.remote = True
        self.read_ahead_tiles = read_ahead_tiles
        self.requests: list[tuple[int, ...]] = []

    def request_region(self, bounds):
        self.requests.append(tuple(bounds))
        future = Future()
        future.set_result(True)
        return future

    def region_has_chunks(self, _bounds) -> bool:
        return True

    def snapshot(self):
        return {"remote": True}


class _IdentityAdapter:
    def __init__(self, product, cache: _InputLevelSpy) -> None:
        self.product = product
        self.cache = cache
        self.request_counts_at_inference: list[int] = []

    def run_tile_inference(self, _model, tile, *, device):
        self.request_counts_at_inference.append(len(self.cache.requests))
        return tile

    def product_tensors_from_output(self, output):
        return {self.product.name: output}


class _Output:
    def __init__(self, *, complete: bool = False) -> None:
        self.complete = complete
        self.writes = 0

    def product_chunk_complete(self, _product, *, chunk_origin_zyx):
        return self.complete

    def write_product_chunk(self, _product, *, chunk_origin_zyx, data):
        self.writes += 1


def _run(tmp_path: Path, *, complete: bool):
    remote_input = _InputLevelSpy((8, 4, 4))
    product = tiled.OutputProductSpec(
        name="identity", level=0, scaledown=1, inference_scaledown=1,
        channels=("value",), chunk_size=4,
    )
    adapter = _IdentityAdapter(product, remote_input)
    output = _Output(complete=complete)
    data = np.ones((8, 4, 4), dtype=np.uint8)
    tiled.run_tiled_inference_3d(
        object(), data,
        crop_slices=(0, 8, 0, 4, 0, 4), device=torch.device("cpu"),
        model_adapter=adapter, output_adapter=output, products=(product,),
        output_regions_zyx={product.name: (0, 0, 0, 8, 4, 4)},
        full_output_shapes_zyx={product.name: (8, 4, 4)},
        output_scaledown_base={product.name: 1},
        tile_size=4, overlap=0, border=0, tmp_dir=str(tmp_path),
        flush_workers=0, accumulator_workers=0,
        input_level=remote_input,
    )
    return remote_input, adapter, output


def test_live_scheduler_materializes_lookahead_before_serial_gpu_work(tmp_path):
    cache, adapter, output = _run(tmp_path, complete=False)
    assert len(cache.requests) == 2
    assert adapter.request_counts_at_inference[0] == 2
    assert output.writes > 0


def test_live_scheduler_skips_completed_output_before_fetch(tmp_path):
    cache, adapter, output = _run(tmp_path, complete=True)
    assert cache.requests == []
    assert adapter.request_counts_at_inference == []
    assert output.writes == 0
