"""Cgroup CPU quota parsing for the ink DataLoader workers' Numba pools."""

from __future__ import annotations

import pytest

from vesuvius.ink_detection.training.train import _cgroup_cpu_limit


@pytest.mark.parametrize(
    ("proc_line", "files", "expected"),
    [
        ("0::/job", {"job/cpu.max": "250000 100000"}, 2.5),
        ("0::/", {"cpu.max": "max 100000"}, None),
        # cgroup v1 without a cgroup namespace: the mount root is the container.
        ("4:cpu,cpuacct:/kubepods/pod1", {"cpu,cpuacct/cpu.cfs_quota_us": "400000",
                                          "cpu,cpuacct/cpu.cfs_period_us": "100000"}, 4.0),
        ("2:cpu:/", {"cpu/cpu.cfs_quota_us": "-1", "cpu/cpu.cfs_period_us": "100000"}, None),
    ],
)
def test_cgroup_cpu_limit_reads_v1_and_v2_quotas(tmp_path, proc_line, files, expected):
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text + "\n")
    (tmp_path / "proc_cgroup").write_text("7:memory:/x\n" + proc_line + "\n")
    assert _cgroup_cpu_limit(str(tmp_path / "proc_cgroup"), str(tmp_path)) == expected
