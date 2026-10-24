"""zarr_tasks' default worker count follows the CPUs the job may use, not the host's cores."""
import os

import pytest

from vesuvius.image_proc.run.zarr_tasks import base


@pytest.fixture
def host(monkeypatch, tmp_path):
    cpu_max = tmp_path / "cpu.max"
    monkeypatch.setattr(base, "_CGROUP_CPU_MAX", cpu_max)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(96)), raising=False)
    return cpu_max


def test_no_quota_uses_affinity(host):
    assert base.available_cpus() == 96


def test_unlimited_quota_uses_affinity(host):
    host.write_text("max 100000\n")
    assert base.available_cpus() == 96


def test_quota_caps_workers(host):
    host.write_text("800000 100000\n")  # 8 CPUs
    assert base.available_cpus() == 8


def test_fractional_quota_rounds_up(host):
    host.write_text("150000 100000\n")  # 1.5 CPUs
    assert base.available_cpus() == 2


def test_affinity_smaller_than_quota_wins(host, monkeypatch):
    host.write_text("800000 100000\n")
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1}, raising=False)
    assert base.available_cpus() == 2
