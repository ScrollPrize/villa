"""tiled_predict3d sizes its worker pools from the CPUs the job may use, not the host's cores."""
import os

import pytest

from lasagna import tiled_predict3d as tp


@pytest.fixture
def host(monkeypatch, tmp_path):
    cpu_max = tmp_path / "cpu.max"
    monkeypatch.setattr(tp, "_CGROUP_CPU_MAX", str(cpu_max))
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(96)), raising=False)
    return cpu_max


def test_no_quota_uses_affinity(host):
    assert tp._available_cpus() == 96


def test_quota_caps_workers(host):
    host.write_text("800000 100000\n")  # 8 CPUs
    assert tp._available_cpus() == 8


def test_fractional_quota_rounds_up(host):
    host.write_text("150000 100000\n")
    assert tp._available_cpus() == 2
