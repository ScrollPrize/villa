"""Atomic file output, advisory file locks and JSON with numpy values."""
from __future__ import annotations
import errno
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
try:
    import fcntl
except ImportError:  # Windows has no fcntl
    fcntl = None
    import msvcrt


def atomic_write(path, data):
    """Write ``data`` (bytes) to ``path`` through a unique temporary file and an atomic rename, creating parent folders;
    concurrent readers see the old or the new file, never a partial one."""
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.partial-',delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        os.replace(temporary,path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def lock_exclusive(handle, *, blocking=True):
    """Take an exclusive advisory lock on the open file ``handle``, held until the handle closes: ``flock`` on
    Linux/macOS, a one-byte ``msvcrt`` lock on Windows (polled, since ``LK_LOCK`` gives up after ~10 s). Separate
    handles exclude each other in one process as across processes. With ``blocking=False`` a lock held through another
    handle raises ``BlockingIOError`` on every platform."""
    if fcntl is not None:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        return
    handle.seek(0)
    while True:
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return
        except OSError as error:
            # Only contention is retried: anything else (a bad descriptor) would spin here forever.
            if error.errno not in (errno.EACCES, errno.EDEADLK):
                raise
            if not blocking:
                raise BlockingIOError(errno.EAGAIN, 'file lock is held through another handle') from None
        time.sleep(.01)


def jsonable(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    raise TypeError(type(x).__name__)


def write_json(path, value):
    """Write ``value`` (numpy values allowed) as indented JSON, atomically, creating parent folders."""
    atomic_write(path, (json.dumps(value, default=jsonable, indent=1)+'\n').encode())


def atomic_torch_save(value, path):
    """``torch.save`` through a temporary sibling (``.partial.pt``) and an atomic rename, so a reader (a collector,
    an evaluation) never opens a partial checkpoint."""
    import torch
    path = Path(path)
    temporary = path.with_suffix('.partial.pt')
    torch.save(value, temporary)
    temporary.replace(path)
