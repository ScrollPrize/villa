from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import zarr

from vesuvius.data.vc_dataset import VCDataset
from vesuvius.models.run import inference
from vesuvius.models.run.inference import Inferer, build_parser

_ZARR_V3 = int(zarr.__version__.split(".", 1)[0]) >= 3
requires_zarr_v3 = pytest.mark.skipif(not _ZARR_V3, reason="the chunk cache needs zarr>=3")


def _required_args(tmp_path: Path) -> list[str]:
    return [
        "--model_path",
        str(tmp_path / "model.pth"),
        "--input_dir",
        "https://example.org/volume.zarr",
        "--output_dir",
        str(tmp_path / "out"),
    ]


def test_chunk_cache_is_off_by_default(tmp_path: Path) -> None:
    args = build_parser().parse_args(_required_args(tmp_path))
    assert args.chunk_cache_mb == 0


def test_main_passes_chunk_cache_mb(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured = {}
    logits_path = tmp_path / "logits_part_0.zarr"
    coords_path = tmp_path / "coordinates_part_0.zarr"
    logits_path.mkdir()
    coords_path.mkdir()

    class FakeInferer:
        skip_empty_patches = False
        dataset = None

        def __init__(self, **kwargs):
            captured.update(kwargs)

        def infer(self):
            return str(logits_path), str(coords_path)

    monkeypatch.setattr(inference, "Inferer", FakeInferer)
    monkeypatch.setattr(
        sys,
        "argv",
        ["vesuvius.predict", *_required_args(tmp_path), "--chunk-cache-mb", "512", "--device", "cpu"],
    )

    assert inference.main() == 0
    assert captured["chunk_cache_mb"] == 512


@pytest.mark.parametrize(("chunk_cache_mb", "expected_cache"), [(0, False), (512, True)])
def test_inferer_forwards_chunk_cache_to_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, chunk_cache_mb: int, expected_cache: bool
) -> None:
    captured = {}

    class StopAfterDataset(Exception):
        pass

    class FakeDataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            raise StopAfterDataset

    monkeypatch.setattr(inference, "VCDataset", FakeDataset)
    inferer = Inferer(
        model_path=str(tmp_path / "model.pth"),
        input_dir="https://example.org/volume.zarr",
        output_dir=str(tmp_path / "out"),
        device="cpu",
        chunk_cache_mb=chunk_cache_mb,
    )
    inferer.patch_size = (8, 8, 8)

    with pytest.raises(StopAfterDataset):
        inferer._create_dataset_and_loader()

    assert captured["cache"] is expected_cache
    assert captured["cache_size_mb"] == chunk_cache_mb


def test_inferer_rejects_negative_chunk_cache_mb(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="chunk_cache_mb"):
        Inferer(
            model_path=str(tmp_path / "model.pth"),
            input_dir="https://example.org/volume.zarr",
            output_dir=str(tmp_path / "out"),
            device="cpu",
            chunk_cache_mb=-1,
        )


@requires_zarr_v3
def test_vcdataset_chunk_cache_survives_pickling_and_reads_identically(tmp_path: Path) -> None:
    from zarr.experimental.cache_store import CacheStore

    path = str(tmp_path / "volume.zarr")
    array = zarr.create_array(store=path, shape=(32, 32, 32), chunks=(16, 16, 16), dtype="uint8", zarr_format=2)
    array[:] = np.random.default_rng(0).integers(0, 255, (32, 32, 32), dtype="uint8")

    common = dict(
        input_path=path,
        patch_size=(16, 16, 16),
        normalization_scheme="none",
        return_as_type="np.float32",
        skip_empty_patches=False,
    )
    plain = VCDataset(**common)
    cached = VCDataset(**common, cache=True, cache_size_mb=1)

    assert not isinstance(plain.volume.data.store, CacheStore)
    assert isinstance(cached.volume.data.store, CacheStore)

    worker_copy = pickle.loads(pickle.dumps(cached))
    assert len(worker_copy) == len(plain) > 1
    for index in range(len(plain)):
        assert torch.equal(worker_copy[index]["data"], plain[index]["data"])

    store = worker_copy.volume.data.store
    assert store.cache_stats()["hits"] > 0
    assert store.cache_info()["current_size"] <= 2**20
