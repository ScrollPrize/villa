from __future__ import annotations

import sys
import time
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from vesuvius.models.training import train as train_module
from vesuvius.utils.dataloader_workers import keep_worker_descriptors_private


class _Items(Dataset):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.zeros(4)


@pytest.mark.skipif(sys.platform != "darwin", reason="the 5 s join timeout is only hit on macOS")
def test_spawned_workers_shut_down_without_waiting_for_the_join_timeout() -> None:
    loader = DataLoader(_Items(), num_workers=2, multiprocessing_context="spawn",
                        worker_init_fn=keep_worker_descriptors_private)
    batches = iter(loader)
    assert len([next(batches) for _ in range(8)]) == 8
    # The next call finds the pass finished and joins both workers before raising StopIteration.
    # Only that shutdown is timed, so a slow worker start cannot fail the test.
    started = time.perf_counter()
    with pytest.raises(StopIteration):
        next(batches)
    # Without the worker_init_fn each join waits its full 5 s timeout, about 10 s here.
    assert time.perf_counter() - started < 3.0


def test_base_trainer_loaders_use_the_worker_init_fn(monkeypatch) -> None:
    built = []
    monkeypatch.setattr(train_module, "DataLoader", lambda *args, **kwargs: built.append(kwargs) or kwargs)
    trainer = train_module.BaseTrainer.__new__(train_module.BaseTrainer)
    trainer.mgr = SimpleNamespace(seed=0, verbose=False, tr_val_split=0.75, train_batch_size=1,
                                  train_num_dataloader_workers=2)
    trainer.device, trainer.is_distributed, trainer.rank, trainer.world_size = torch.device("cpu"), False, 0, 1
    trainer._configure_dataloaders(_Items())
    assert len(built) == 2
    assert all(kwargs["worker_init_fn"] is keep_worker_descriptors_private for kwargs in built)
