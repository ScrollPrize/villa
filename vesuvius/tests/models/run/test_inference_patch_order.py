from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import zarr
from torch.utils.data import BatchSampler, Dataset, Subset

from vesuvius.data.patch_order import WorkerContiguousBatchSampler
from vesuvius.models.run import inference
from vesuvius.models.run.inference import Inferer, build_parser

_ZARR_V3 = int(zarr.__version__.split(".", 1)[0]) >= 3


def _required_args(tmp_path: Path) -> list[str]:
    return [
        "--model_path",
        str(tmp_path / "model.pth"),
        "--input_dir",
        "https://example.org/volume.zarr",
        "--output_dir",
        str(tmp_path / "out"),
    ]


def test_patch_order_defaults_to_auto(tmp_path: Path) -> None:
    assert build_parser().parse_args(_required_args(tmp_path)).patch_order == "auto"


def test_patch_order_rejects_unknown_values(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([*_required_args(tmp_path), "--patch_order", "hilbert"])


def test_main_passes_patch_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
        ["vesuvius.predict", *_required_args(tmp_path), "--patch-order", "zyx", "--device", "cpu"],
    )

    assert inference.main() == 0
    assert captured["patch_order"] == "zyx"


@pytest.mark.parametrize("patch_order", ["auto", "chunk", "zyx"])
def test_inferer_forwards_patch_order_to_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, patch_order: str
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
        patch_order=patch_order,
    )
    inferer.patch_size = (8, 8, 8)

    with pytest.raises(StopAfterDataset):
        inferer._create_dataset_and_loader()

    assert captured["patch_order"] == patch_order


def test_inferer_rejects_invalid_patch_order(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="patch_order"):
        Inferer(
            model_path=str(tmp_path / "model.pth"),
            input_dir="https://example.org/volume.zarr",
            output_dir=str(tmp_path / "out"),
            device="cpu",
            patch_order="hilbert",
        )


@pytest.mark.parametrize(
    ("patch_order", "chunk_cache_mb", "expected"),
    [("auto", 0, "zyx"), ("auto", 64, "chunk"), ("chunk", 0, "chunk"), ("zyx", 64, "zyx")],
)
def test_effective_patch_order_resolves_auto_by_chunk_cache(
    tmp_path: Path, patch_order: str, chunk_cache_mb: int, expected: str
) -> None:
    inferer = Inferer(
        model_path=str(tmp_path / "model.pth"),
        input_dir="https://example.org/volume.zarr",
        output_dir=str(tmp_path / "out"),
        device="cpu",
        patch_order=patch_order,
        chunk_cache_mb=chunk_cache_mb,
    )
    assert inferer.effective_patch_order == expected


class _TenPatches(Dataset):
    """Stands in for VCDataset: ten positions, nothing pre-filtered."""

    all_positions = [(i, 0, 0) for i in range(10)]
    non_empty_mask = None
    input_shape = (80, 8, 8)
    collate_fn = staticmethod(lambda batch: batch)

    def __init__(self, **kwargs):
        pass

    def active_view(self):
        return self

    def __len__(self):
        return len(self.all_positions)

    def __getitem__(self, idx):
        return {"data": torch.zeros((1, 8, 8, 8)), "pos": self.all_positions[idx], "index": idx, "is_empty": False}


def _loader_for(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, num_workers: int, **inferer_kwargs):
    monkeypatch.setattr(inference, "VCDataset", _TenPatches)
    inferer = Inferer(
        model_path=str(tmp_path / "model.pth"),
        input_dir="https://example.org/volume.zarr",
        output_dir=str(tmp_path / "out"),
        device="cpu",
        num_dataloader_workers=num_workers,
        batch_size=3,
        **inferer_kwargs,
    )
    inferer.patch_size = (8, 8, 8)
    inferer._create_dataset_and_loader()
    return inferer.dataloader


PLAIN_BATCHES = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]


@pytest.mark.parametrize(
    ("patch_order", "chunk_cache_mb", "num_workers"),
    [("chunk", 0, 2), ("chunk", 0, 4), ("auto", 64, 2), ("auto", 64, 4)],
)
def test_chunk_order_with_workers_dispatches_contiguous_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, patch_order: str, chunk_cache_mb: int, num_workers: int
) -> None:
    loader = _loader_for(monkeypatch, tmp_path, num_workers, patch_order=patch_order, chunk_cache_mb=chunk_cache_mb)
    sampler = loader.batch_sampler
    assert isinstance(sampler, WorkerContiguousBatchSampler)
    assert sampler.num_workers == num_workers and sampler.batch_size == 3
    assert sorted(list(sampler)) == PLAIN_BATCHES
    assert loader.num_workers == num_workers


@pytest.mark.parametrize(
    ("patch_order", "chunk_cache_mb", "num_workers"),
    [
        ("auto", 0, 2),  # the default without the cache: the loader main builds
        ("auto", 0, 4),
        ("zyx", 64, 4),  # explicit row-major keeps plain batching even with the cache
        ("chunk", 64, 1),  # nothing to dispatch with one worker or none
        ("chunk", 64, 0),
    ],
)
def test_plain_batching_otherwise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, patch_order: str, chunk_cache_mb: int, num_workers: int
) -> None:
    loader = _loader_for(monkeypatch, tmp_path, num_workers, patch_order=patch_order, chunk_cache_mb=chunk_cache_mb)
    assert isinstance(loader.batch_sampler, BatchSampler)
    assert not isinstance(loader.batch_sampler, WorkerContiguousBatchSampler)
    assert list(loader.batch_sampler) == PLAIN_BATCHES
    assert loader.collate_fn is _TenPatches.collate_fn


@pytest.fixture
def sparse_volume(tmp_path: Path) -> str:
    """A 64^3 volume (16^3 chunks) with data in its first 32^3 corner only.

    Written in zarr v2 format so the chunk occupancy index can list its chunk
    files; the all-zero chunks are never written, so skip_empty_patches=True
    pre-filters the patches that never touch the corner into a Subset.
    """
    path = str(tmp_path / "sparse.zarr")
    if _ZARR_V3:
        arr = zarr.create_array(store=path, shape=(64, 64, 64), chunks=(16, 16, 16), dtype="uint8", zarr_format=2)
    else:
        arr = zarr.open(path, mode="w", shape=(64, 64, 64), chunks=(16, 16, 16), dtype="uint8")
    arr[:32, :32, :32] = np.random.default_rng(11).integers(1, 255, (32, 32, 32), dtype=np.uint8)
    return path


def test_subset_view_with_contiguous_dispatch_yields_each_active_patch_once(tmp_path: Path, sparse_volume: str) -> None:
    # With skip_empty_patches the loader iterates a Subset over the non-empty
    # indices. The sampler indexes the Subset, so a batch's `index` field (the
    # zarr write index) must still be the original dataset index of its `pos`.
    inferer = Inferer(
        model_path=str(tmp_path / "model.pth"),
        input_dir=sparse_volume,
        output_dir=str(tmp_path / "out"),
        device="cpu",
        num_dataloader_workers=2,
        batch_size=3,
        skip_empty_patches=True,
        patch_order="chunk",
        normalization_scheme="none",
    )
    inferer.patch_size = (16, 16, 16)
    dataset, loader = inferer._create_dataset_and_loader()

    assert isinstance(loader.dataset, Subset)
    assert isinstance(loader.batch_sampler, WorkerContiguousBatchSampler)
    active = sorted(loader.dataset.indices)
    assert 0 < len(active) < len(dataset.all_positions)

    seen = []
    for batch in loader:
        for index, pos in zip(batch["index"], batch["pos"]):
            assert dataset.all_positions[index] == tuple(pos)
            seen.append(index)
    assert sorted(seen) == active
