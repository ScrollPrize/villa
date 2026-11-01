"""Numba thread pools in ink training DataLoader workers.

Each loader worker runs tifxyz's parallel Numba kernels with its own pool,
which Numba sizes to the whole machine. Under a container CPU quota the host
count is far above what the job may use, so the trainer's worker_init_fn gives
each worker its share of the quota instead.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numba
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from vesuvius.ink_detection.training import train as train_module


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_cgroup_v2_quota_reads_the_process_cgroup(tmp_path: Path):
    proc = tmp_path / "proc_cgroup"
    _write(proc, "0::/job\n")
    _write(tmp_path / "cg" / "cpu.max", "max 100000\n")
    _write(tmp_path / "cg" / "job" / "cpu.max", "800000 100000\n")

    assert train_module._cgroup_cpu_quota(proc, tmp_path / "cg") == 8.0


def test_cgroup_v1_quota_takes_the_tightest_parent(tmp_path: Path):
    proc = tmp_path / "proc_cgroup"
    _write(proc, "5:memory:/a/b\n4:cpu,cpuacct:/a/b\n")
    root = tmp_path / "cg"
    _write(root / "cpu" / "a" / "b" / "cpu.cfs_quota_us", "-1\n")
    _write(root / "cpu" / "a" / "b" / "cpu.cfs_period_us", "100000\n")
    _write(root / "cpu" / "a" / "cpu.cfs_quota_us", "50000\n")
    _write(root / "cpu" / "a" / "cpu.cfs_period_us", "100000\n")

    assert train_module._cgroup_cpu_quota(proc, root) == 0.5


def test_no_cgroup_quota(tmp_path: Path):
    proc = tmp_path / "proc_cgroup"
    _write(proc, "0::/\n")
    _write(tmp_path / "cg" / "cpu.max", "max 100000\n")

    assert train_module._cgroup_cpu_quota(proc, tmp_path / "cg") is None
    assert train_module._cgroup_cpu_quota(tmp_path / "missing", tmp_path) is None


@pytest.mark.parametrize(("quota", "expected"), [(None, 16), (2.5, 3), (0.5, 1)])
def test_usable_cpu_count_caps_affinity_by_quota(monkeypatch, quota, expected):
    monkeypatch.setattr(train_module.os, "sched_getaffinity", lambda pid: set(range(16)), raising=False)
    monkeypatch.setattr(train_module, "_cgroup_cpu_quota", lambda: quota)

    assert train_module.usable_cpu_count() == expected


@pytest.mark.parametrize(("host_cpus", "expected"), [(16, 16), (None, 1)])
def test_usable_cpu_count_falls_back_when_affinity_is_unavailable(monkeypatch, host_cpus, expected):
    def unavailable(pid):
        raise OSError("affinity is unavailable")

    monkeypatch.setattr(train_module.os, "sched_getaffinity", unavailable, raising=False)
    monkeypatch.setattr(train_module.os, "cpu_count", lambda: host_cpus)
    monkeypatch.setattr(train_module, "_cgroup_cpu_quota", lambda: None)

    assert train_module.usable_cpu_count() == expected


@pytest.mark.parametrize(("cpus", "workers", "expected"), [(8, 4, 2), (23, 8, 2), (2, 8, 1)])
def test_worker_init_sets_each_workers_share(monkeypatch, cpus, workers, expected):
    calls = []
    monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)
    monkeypatch.setattr(train_module, "usable_cpu_count", lambda: cpus)
    monkeypatch.setattr(
        torch.utils.data, "get_worker_info", lambda: SimpleNamespace(num_workers=workers)
    )
    monkeypatch.setattr(numba, "set_num_threads", calls.append)

    train_module.limit_loader_worker_threads(0)

    assert calls == [min(expected, numba.config.NUMBA_NUM_THREADS)]


def test_worker_init_keeps_an_explicit_numba_setting(monkeypatch):
    calls = []
    monkeypatch.setenv("NUMBA_NUM_THREADS", "3")
    monkeypatch.setattr(numba, "set_num_threads", calls.append)

    train_module.limit_loader_worker_threads(0)

    assert calls == []


class _ReportNumbaThreads(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        # A spawned interpreter initializes its own Numba configuration. In
        # particular, it may differ from a parent that imported Numba before
        # removing NUMBA_NUM_THREADS from the environment.
        return (
            numba.get_num_threads(),
            train_module.usable_cpu_count(),
            numba.config.NUMBA_NUM_THREADS,
        )


def test_spawned_loader_workers_run_with_their_share(monkeypatch):
    monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)
    workers = 2
    loader = DataLoader(
        _ReportNumbaThreads(),
        batch_size=1,
        num_workers=workers,
        multiprocessing_context="spawn",
        worker_init_fn=train_module.limit_loader_worker_threads,
    )

    for threads, cpus, configured_max in loader:
        expected = min(max(1, int(cpus) // workers), int(configured_max))
        assert int(threads) == expected
