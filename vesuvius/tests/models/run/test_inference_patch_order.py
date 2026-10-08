from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import BatchSampler, Dataset

from vesuvius.data.patch_order import WorkerContiguousBatchSampler
from vesuvius.models.run import inference
from vesuvius.models.run.inference import Inferer, build_parser


def _required_args(tmp_path: Path) -> list[str]:
    return [
        "--model_path",
        str(tmp_path / "model.pth"),
        "--input_dir",
        "https://example.org/volume.zarr",
        "--output_dir",
        str(tmp_path / "out"),
    ]


def test_patch_order_defaults_to_chunk(tmp_path: Path) -> None:
    assert build_parser().parse_args(_required_args(tmp_path)).patch_order == "chunk"


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


@pytest.mark.parametrize("patch_order", ["chunk", "zyx"])
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


def _loader_for(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, num_workers: int):
    monkeypatch.setattr(inference, "VCDataset", _TenPatches)
    inferer = Inferer(
        model_path=str(tmp_path / "model.pth"),
        input_dir="https://example.org/volume.zarr",
        output_dir=str(tmp_path / "out"),
        device="cpu",
        num_dataloader_workers=num_workers,
        batch_size=3,
    )
    inferer.patch_size = (8, 8, 8)
    inferer._create_dataset_and_loader()
    return inferer.dataloader


@pytest.mark.parametrize("num_workers", [2, 4])
def test_multi_worker_loader_dispatches_contiguous_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, num_workers: int) -> None:
    loader = _loader_for(monkeypatch, tmp_path, num_workers)
    sampler = loader.batch_sampler
    assert isinstance(sampler, WorkerContiguousBatchSampler)
    assert sampler.num_workers == num_workers and sampler.batch_size == 3
    batches = list(sampler)
    assert sorted(batches) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    assert loader.num_workers == num_workers


@pytest.mark.parametrize("num_workers", [0, 1])
def test_single_process_loader_keeps_plain_batching(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, num_workers: int) -> None:
    loader = _loader_for(monkeypatch, tmp_path, num_workers)
    assert isinstance(loader.batch_sampler, BatchSampler)
    assert not isinstance(loader.batch_sampler, WorkerContiguousBatchSampler)
    assert list(loader.batch_sampler) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    assert loader.collate_fn is _TenPatches.collate_fn
