from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from lasagna import tiled_predict3d as tiled


class _LiveCacheSpy:
    def __init__(self, level_path: Path, lookahead_tiles: int = 8) -> None:
        self.source = SimpleNamespace(level_path=level_path.resolve())
        self.lookahead_tiles = lookahead_tiles
        self.requests: list[tuple[int, ...]] = []
        self.safe_boundaries: list[int] = []

    def request_region(self, bounds):
        self.requests.append(tuple(bounds))
        future = Future()
        future.set_result(True)
        return future

    def advance_safe_boundary(self, value: int) -> None:
        self.safe_boundaries.append(int(value))

    def region_has_remote_chunks(self, _bounds) -> bool:
        return True

    def snapshot(self):
        return {
            "resident_bytes": 0, "target_bytes": 1 << 30,
            "safe_plane_exclusive": len(self.safe_boundaries),
        }


class _IdentityAdapter:
    def __init__(self, product, cache: _LiveCacheSpy) -> None:
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


def _input_metadata(path: Path) -> None:
    path.mkdir(parents=True)
    (path / ".zarray").write_text(
        '{"zarr_format":2,"shape":[8,4,4],"chunks":[4,4,4],'
        '"dtype":"|u1","compressor":null,"fill_value":0,'
        '"order":"C","filters":null,"dimension_separator":"."}\n',
        encoding="utf-8",
    )
    (path / "0.0.0").write_bytes(b"present")
    (path / "1.0.0").write_bytes(b"present")


def _run(tmp_path: Path, *, complete: bool):
    input_path = tmp_path / "input.zarr" / "0"
    _input_metadata(input_path)
    live_cache = _LiveCacheSpy(input_path)
    product = tiled.OutputProductSpec(
        name="identity", level=0, scaledown=1, inference_scaledown=1,
        channels=("value",), chunk_size=4,
    )
    adapter = _IdentityAdapter(product, live_cache)
    output = _Output(complete=complete)
    data = np.ones((8, 4, 4), dtype=np.uint8)
    tiled.run_tiled_inference_3d(
        object(), data,
        crop_slices=(0, 8, 0, 4, 0, 4), device=torch.device("cpu"),
        model_adapter=adapter, output_adapter=output, products=(product,),
        output_regions_zyx={product.name: (0, 0, 0, 8, 4, 4)},
        full_output_shapes_zyx={product.name: (8, 4, 4)},
        input_zarr_path=str(input_path), output_scaledown_base={product.name: 1},
        tile_size=4, overlap=0, border=0, tmp_dir=str(tmp_path),
        input_reader="python-zarr", flush_workers=0, accumulator_workers=0,
        live_cache=live_cache,
    )
    return live_cache, adapter, output


def test_live_scheduler_materializes_lookahead_before_serial_gpu_work(tmp_path, monkeypatch):
    monkeypatch.setattr(tiled, "_input_has_chunks", lambda *_args: True)
    cache, adapter, output = _run(tmp_path, complete=False)
    assert len(cache.requests) == 2
    assert adapter.request_counts_at_inference[0] == 2
    assert cache.safe_boundaries == [4, 8]
    assert output.writes > 0


def test_live_scheduler_skips_completed_output_before_fetch(tmp_path, monkeypatch):
    monkeypatch.setattr(tiled, "_input_has_chunks", lambda *_args: True)
    cache, adapter, output = _run(tmp_path, complete=True)
    assert cache.requests == []
    assert adapter.request_counts_at_inference == []
    assert output.writes == 0
