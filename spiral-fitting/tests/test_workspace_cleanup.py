"""Disposable editing storage and crash reclamation."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from service_editing import EditingWorkspace
from test_service_editing import Resident
from workspace_storage import MARKER, owner_lock_path, reclaim_workspaces


def test_subprocess_crash_live_owner_concurrent_reclamation_and_legacy(tmp_path):
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    output = tmp_path / 'output'
    code = '''
import sys
from service_editing import EditingWorkspace
w = EditingWorkspace(sys.argv[1], sys.argv[2], {}, lambda: None)
w.claim('owner', 'claim')
print(w.root, flush=True)
sys.stdin.read()
'''
    process = subprocess.Popen([sys.executable, '-c', code, str(dataset), str(output)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        root = Path(process.stdout.readline().strip())
        assert root.is_dir()
        legacy = root.parent / 'legacy'
        legacy.mkdir()
        (legacy / 'data').write_text('keep')
        link = root.parent / 'link'
        link.symlink_to(legacy, target_is_directory=True)
        unknown = root.parent / 'unknown'
        unknown.mkdir()
        (unknown / MARKER).write_text('{"version": 999}')
        reclaim_workspaces(output)
        assert root.exists()
        process.kill()
        process.wait(timeout=5)
        command = [sys.executable, '-c',
                   'from workspace_storage import reclaim_workspaces; import sys; reclaim_workspaces(sys.argv[1])',
                   str(output)]
        processes = [subprocess.Popen(command) for _ in range(3)]
        assert all(p.wait(timeout=10) == 0 for p in processes)
        assert not root.exists()
        assert not owner_lock_path(root).exists()
        assert (legacy / 'data').read_text() == 'keep'
        assert link.is_symlink()
        assert unknown.exists()
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


def test_real_scroll_discard_and_reopen(tmp_path):
    source = Path(os.environ.get('SPIRAL_REAL_PCL', '/home/sean/Desktop/spiral_dataset/abs_winding.json'))
    if not source.is_file():
        pytest.skip('Set SPIRAL_REAL_PCL to a real scroll PCL input')
    before = source.read_bytes()
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    target = dataset / 'same_windings.json'
    target.write_bytes(before)
    sources = {'pcl_inputs': [{'path': str(target), 'role': 'same_winding'}]}
    resident = Resident()
    workspace = EditingWorkspace(dataset, tmp_path / 'output', sources, lambda: resident)
    workspace.claim('owner', 'claim')
    baseline = len(workspace.catalog.entries())
    assert baseline
    revision = workspace.catalog.desired()[0]
    workspace.change('owner', {'command_id': 'delete', 'changes': [
        {'id': revision.id, 'expected_revision': revision.number, 'deleted': True}]})
    workspace.close()
    assert not workspace.root.exists()
    fresh = EditingWorkspace(dataset, tmp_path / 'output', sources, lambda: resident)
    try:
        fresh.claim('owner', 'reopen')
        assert len(fresh.catalog.entries()) == baseline
        assert all(r.content is not None for r in fresh.catalog.desired())
        assert target.read_bytes() == before == source.read_bytes()
    finally:
        fresh.close()


