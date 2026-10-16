"""Teardown must release a workspace owner lock before unlinking its file.

``ExclusiveFileLock`` holds an open handle on its lock file while acquired,
and Windows refuses to unlink a file whose byte range is locked.
``EditingWorkspace.close()`` and ``reclaim_workspaces()`` used to unlink
``.<workspace>.owner.lock`` first, so every Windows teardown failed with
``WinError 32``: the service could not stop cleanly, and the failed close
left the lock file behind for the next start to trip over.
"""

from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import workspace_storage
from service_editing import EditingWorkspace
from workspace_storage import MARKER, owner_lock_path, reclaim_workspaces


class _Resident:
    def apply_input_changes(self, _command, _records):
        return {"applied": True}


def _editing_workspace(temporary):
    output = Path(temporary) / "output"
    output.mkdir(exist_ok=True)
    dataset = Path(temporary) / "dataset"
    dataset.mkdir(exist_ok=True)
    return EditingWorkspace(dataset, output, {}, _Resident())


def _simulate_owner_death(workspace):
    """Close the workspace's lock handles as a dead process would."""
    workspace._owner_lock.release()
    workspace.lease._file_lock.release()


class EditingWorkspaceCloseTests(unittest.TestCase):
    def test_close_releases_the_owner_lock_and_removes_the_workspace(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = _editing_workspace(temporary)
            workspace.claim("owner", "claim")
            (workspace.root / "draft.bin").write_bytes(b"content")
            lock = owner_lock_path(workspace.root)
            self.assertTrue(workspace.root.is_dir())
            self.assertTrue(lock.is_file())
            # Before the fix this raised PermissionError (WinError 32) on
            # Windows: close unlinked the lock file while the workspace
            # still held it open.
            workspace.close()
            self.assertFalse(workspace.root.exists())
            self.assertFalse(lock.exists())
            # A new workspace can be created, claimed and closed at once:
            # nothing of the old ownership survived teardown.
            fresh = _editing_workspace(temporary)
            try:
                fresh.claim("owner", "claim")
            finally:
                fresh.close()

    def test_close_without_a_claim_still_removes_the_workspace(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = _editing_workspace(temporary)
            (workspace.root / "draft.bin").write_bytes(b"content")
            lock = owner_lock_path(workspace.root)
            # Service teardown after a failed claim: the workspace was
            # never seeded but is fully owned and must still be removable.
            workspace.close()
            self.assertFalse(workspace.root.exists())
            self.assertFalse(lock.exists())


class ReclaimTests(unittest.TestCase):
    def test_reclaim_removes_a_dead_owners_workspace_and_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = _editing_workspace(temporary)
            (workspace.root / "draft.bin").write_bytes(b"content")
            root = workspace.root
            # A crashed service: its process is gone (handles closed, as
            # _simulate_owner_death does), but its files remain.
            _simulate_owner_death(workspace)
            reclaim_workspaces(Path(temporary) / "output")
            self.assertFalse(root.exists())
            self.assertFalse(owner_lock_path(root).exists())

    def test_reclaim_keeps_a_live_owners_workspace_untouched(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = _editing_workspace(temporary)
            workspace.claim("owner", "claim")
            root = workspace.root
            reclaim_workspaces(Path(temporary) / "output")
            self.assertTrue(root.exists())
            self.assertTrue(owner_lock_path(root).exists())
            workspace.close()
            reclaim_workspaces(Path(temporary) / "output")
            self.assertFalse(root.exists())
            self.assertFalse(owner_lock_path(root).exists())

    def test_reclaim_failed_remove_retains_marker_and_lock_for_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = _editing_workspace(temporary)
            root = workspace.root
            (root / "draft").mkdir()
            _simulate_owner_death(workspace)
            original_rmtree = shutil.rmtree

            def fail_rmtree(path, *args, **kwargs):
                if Path(path).name == "draft":
                    raise PermissionError("simulated removal failure")
                return original_rmtree(path, *args, **kwargs)

            with mock.patch.object(
                    workspace_storage.shutil, "rmtree", fail_rmtree):
                reclaim_workspaces(Path(temporary) / "output")
            self.assertTrue(root.exists())
            self.assertTrue((root / MARKER).is_file())
            self.assertTrue(owner_lock_path(root).is_file())
            # The retry, with the obstruction gone, reclaims both.
            reclaim_workspaces(Path(temporary) / "output")
            self.assertFalse(root.exists())
            self.assertFalse(owner_lock_path(root).exists())


if __name__ == "__main__":
    unittest.main()
