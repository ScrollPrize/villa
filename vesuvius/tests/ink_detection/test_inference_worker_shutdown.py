from __future__ import annotations

import sys
import time
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from vesuvius.ink_detection.inference import infer_full3d_tifxyz
from vesuvius.ink_detection.inference.infer import keep_worker_descriptors_private


class _Items(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.zeros(4)


@pytest.mark.skipif(sys.platform != "darwin", reason="the join timeout is only hit on macOS")
def test_spawned_workers_are_joined_without_the_five_second_timeout() -> None:
    loader = DataLoader(
        _Items(),
        num_workers=2,
        persistent_workers=True,
        multiprocessing_context="spawn",
        worker_init_fn=keep_worker_descriptors_private,
    )
    batches = iter(loader)
    assert len(list(batches)) == 4
    started = time.perf_counter()
    del batches, loader
    # Each worker took its full 5 s join timeout before: 10 s for these two.
    assert time.perf_counter() - started < 4.0


def test_full3d_tifxyz_loader_is_built_with_the_same_worker_init_fn(monkeypatch) -> None:
    built = {}

    def record_and_stop(**kwargs):
        built.update(kwargs)
        raise RuntimeError("loader built")

    monkeypatch.setattr(infer_full3d_tifxyz, "DataLoader", record_and_stop)
    args = SimpleNamespace(batch_size=2, gpu_ids=[], num_workers=2, prefetch_factor=2)
    with pytest.raises(RuntimeError, match="loader built"):
        infer_full3d_tifxyz.run_native_inference(
            args=args,
            bundle=SimpleNamespace(device=torch.device("cpu")),
            dataset=_Items(),
            plan=None,
            output=None,
            importance_map=None,
        )
    assert built.get("worker_init_fn") is keep_worker_descriptors_private
