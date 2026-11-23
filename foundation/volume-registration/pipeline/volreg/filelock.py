"""Blocking exclusive lock on an open file, on POSIX and on Windows.

The lock lives as long as the handle: closing it releases the lock, which is
what `fcntl.flock` did before this module existed.
"""

import errno
import time

try:
    import fcntl
except ImportError:  # Windows has no fcntl
    fcntl = None
    import msvcrt


def lock_exclusive(handle):
    if fcntl is not None:
        fcntl.flock(handle, fcntl.LOCK_EX)
        return
    # msvcrt locks one byte range and gives up after ~10 s (LK_LOCK); wait on
    # contention only, any other error is real and must surface.
    handle.seek(0)
    while True:
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            return
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EDEADLK):
                raise
            time.sleep(0.05)


def unlock(handle):
    if fcntl is not None:
        fcntl.flock(handle, fcntl.LOCK_UN)
        return
    handle.seek(0)
    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
