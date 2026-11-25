"""Reclamation of the editing workspaces older services left behind."""
from pathlib import Path
import subprocess
import sys

from workspace_storage import MARKER, owner_lock_path, reclaim_workspaces


def test_subprocess_crash_live_owner_concurrent_reclamation_and_legacy(tmp_path):
    output = tmp_path / 'output'
    # An owned workspace as an older service created it, held by a live process.
    code = '''
import sys
from pathlib import Path
from workspace_storage import create_workspace
root = Path(sys.argv[1]) / 'editing-workspaces' / 'live'
root.parent.mkdir(parents=True, exist_ok=True)
owner = create_workspace(root)
print(root, flush=True)
sys.stdin.read()
'''
    process = subprocess.Popen([sys.executable, '-c', code, str(output)],
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
