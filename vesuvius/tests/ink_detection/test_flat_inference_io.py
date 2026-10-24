from __future__ import annotations

import numpy as np
import pytest
import tifffile
import torch
from torch.utils.data import DataLoader
import zarr

from vesuvius.ink_detection.config import InkConfig
from vesuvius.ink_detection.inference.infer import (
    FlatBlockDataset,
    FlatPatchReader,
    ThreadedBatchLoader,
    iter_blocks,
    main,
)
from vesuvius.ink_detection.models.model import make_model

from .test_model_foundation import _config_mapping


def _volume(path, shape=(6, 13, 17), chunks=(6, 4, 5)):
    values = np.random.default_rng(0).integers(1, 255, size=shape, dtype=np.uint8)
    array = zarr.open(path, mode="w", shape=shape, chunks=chunks, dtype="u1", zarr_format=2)
    array[:] = values
    return values


@pytest.mark.parametrize(
    "layers", ([1, 2, 3, 4], [4, 3, 2, 1], [0, 2, 3]), ids=("asc", "desc", "fancy")
)
def test_chunk_cached_reader_matches_direct_reads(tmp_path, layers):
    path = tmp_path / "surface.zarr"
    _volume(path)
    kwargs = dict(
        input_path=path,
        resolution="0",
        depth_axis_first=True,
        height=13,
        width=17,
        layer_indices=np.asarray(layers),
        output_depth=5,
        preprocessing="divide_255",
    )
    direct = FlatPatchReader(**kwargs)
    # A two-chunk cache forces evictions and re-decodes during the sweep.
    cached = FlatPatchReader(**kwargs, chunk_cache_size=2)
    for y0 in range(-3, 13, 3):
        for x0 in range(-4, 17, 3):
            np.testing.assert_array_equal(
                cached.read(y0, x0, 6, 7), direct.read(y0, x0, 6, 7)
            )
    assert len(cached._chunk_cache) <= 2
    assert len(cached._chunk_cache) > 0 or layers == [0, 2, 3]


def test_threaded_loader_matches_dataloader_order(tmp_path):
    path = tmp_path / "surface.zarr"
    _volume(path)
    reader = FlatPatchReader(
        input_path=path,
        resolution="0",
        depth_axis_first=True,
        height=13,
        width=17,
        layer_indices=np.arange(6),
        output_depth=6,
        preprocessing="divide_255",
        chunk_cache_size=8,
    )
    dataset = FlatBlockDataset(
        reader=reader,
        blocks=iter_blocks((13, 17), 8, 3),
        patch_size=8,
        preprocessing="divide_255",
    )
    expected = list(DataLoader(dataset, batch_size=3, num_workers=0))
    threaded = list(
        ThreadedBatchLoader(dataset, batch_size=3, num_threads=3, prefetch_batches=2)
    )
    assert len(threaded) == len(expected)
    for (images, metadata), (want_images, want_metadata) in zip(threaded, expected):
        torch.testing.assert_close(images, want_images, rtol=0, atol=0)
        torch.testing.assert_close(metadata, want_metadata, rtol=0, atol=0)


def test_ram_accumulation_and_threads_match_zarr_and_workers(tmp_path):
    torch.manual_seed(0)
    config_mapping = _config_mapping("vesuvius_unet_2p5d", depth=3, side=16)
    model = make_model(InkConfig.from_mapping(config_mapping))
    checkpoint = tmp_path / "model.pth"
    torch.save({"config": config_mapping, "model": model.state_dict()}, checkpoint)
    input_path = tmp_path / "surface.zarr"
    _volume(input_path, shape=(3, 40, 44), chunks=(3, 16, 16))
    common = [str(input_path), str(checkpoint), "--no-compile", "--overlap", "0.5"]

    reference = tmp_path / "reference.tif"
    assert main(
        [*common[:2], str(reference), *common[2:], "--workers", "0",
         "--chunk-cache-size", "0", "--max-ram-accumulation-gb", "0"]
    ) == 0
    fast = tmp_path / "fast.tif"
    assert main([*common[:2], str(fast), *common[2:], "--loader-threads", "2"]) == 0
    np.testing.assert_array_equal(tifffile.imread(fast), tifffile.imread(reference))
