"""Scoped cleanup must include orphaned collectors without touching other runs."""
import subprocess
import sys

import psutil
import pytest

from vesuvius.neural_tracing.fiber_follow.train.stop_run import collector_matches, stop_run, alive, PREFIX


def collector_args(root):
    return ['-m', PREFIX+'regression.collect', '--checkpoint', str(root/'dagger/source.pt'),
            '--out', str(root/'dagger/afv/decisions.npz')]


def test_collector_scope_requires_module_and_both_paths(tmp_path):
    root = tmp_path/'run'
    assert collector_matches(collector_args(root), root)
    assert not collector_matches(collector_args(tmp_path/'run_other'), root)
    args = collector_args(root)
    args[-1] = str(tmp_path/'other/decisions.npz')
    assert not collector_matches(args, root)
    assert not collector_matches(['echo', str(root)], root)
    assert not collector_matches(['-m', PREFIX+'regression.collect', '--checkpoint'], root)


@pytest.mark.parametrize('with_trainer', [True, False])
def test_stop_cleans_orphan_collector_and_preserves_other_run(tmp_path, with_trainer):
    logs = tmp_path/'logs'
    logs.mkdir()
    pid_file = logs/'run.pid'
    spawn = lambda args: subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(90)', *args])
    processes = []
    try:
        trainer = spawn(['-m', PREFIX+'regression.train', '--name', 'run']) if with_trainer else None
        if trainer:
            processes.append(trainer)
            pid_file.write_text(str(trainer.pid))
        orphan = spawn(collector_args(tmp_path/'run'))
        other = spawn(collector_args(tmp_path/'other'))
        processes += [orphan, other]
        stopped = stop_run(pid_file, 'run')
        assert orphan.pid in stopped and alive(psutil.Process(other.pid))
        assert orphan.poll() is not None
        if trainer:
            assert trainer.pid in stopped and trainer.poll() is not None
        assert other.poll() is None
    finally:
        for p in processes:
            if p.poll() is None:
                p.kill()
            p.wait()
