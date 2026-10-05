"""Owned disposable workspaces; legacy and live directories are never reclaimed."""
import json
import logging
from pathlib import Path
import shutil

from service_files import ExclusiveFileLock, FileLockUnavailable

MARKER = '.spiral-workspace.json'
METADATA = {'kind': 'spiral-editing-workspace', 'version': 1}


def storage_lock(parent):
    return ExclusiveFileLock(parent / '.reclamation.lock').acquire(timeout=15)


def owner_lock_path(root):
    # Keep the lock inode stable even if directory deletion fails at the last step.
    return root.parent / f'.{root.name}.owner.lock'


def remove_workspace(root):
    """Keep ownership metadata until all disposable contents have been removed."""
    try:
        for child in root.iterdir():
            if child.name == MARKER:
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
        # The caller holds the parent lock throughout deletion.
        (root / MARKER).unlink()
        try:
            root.rmdir()
        except OSError:
            (root / MARKER).write_text(json.dumps(METADATA))
            raise
    except OSError as exc:
        if isinstance(exc, FileNotFoundError) and not root.exists():
            return
        logging.exception('Failed to clean Spiral workspace %s', root)
        raise


def create_workspace(root):
    with storage_lock(root.parent):
        root.mkdir()
        owner = ExclusiveFileLock(owner_lock_path(root)).acquire()
        try:
            (root / MARKER).write_text(json.dumps(METADATA))
        except BaseException:
            owner.release()
            raise
    return owner


def reclaim_workspaces(output):
    parent = Path(output) / 'editing-workspaces'
    if parent.is_symlink():
        return
    with storage_lock(parent):
        for root in parent.iterdir():
            if root.is_symlink() or not root.is_dir():
                continue
            marker, lock = root / MARKER, owner_lock_path(root)
            if marker.is_symlink() or lock.is_symlink() or not lock.is_file():
                continue
            try:
                metadata = json.loads(marker.read_text())
                if metadata != METADATA or type(metadata['version']) is not int:
                    continue
            except (OSError, ValueError):
                continue
            try:
                owner = ExclusiveFileLock(lock).acquire()
            except FileLockUnavailable:
                continue
            try:
                remove_workspace(root)
                lock.unlink()
            except OSError:
                logging.exception('Failed to reclaim Spiral workspace %s (lock %s)', root, lock)
            finally:
                owner.release()
