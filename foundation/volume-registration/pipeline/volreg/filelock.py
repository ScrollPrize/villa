"""Blocking exclusive lock on an open file, and the file steps that differ on Windows.

The lock lives as long as the handle: closing it releases the lock, which is
what `fcntl.flock` did before this module existed. `try_lock` is the same without
waiting; a job's lock file is held that way for the job's whole life, so whoever can
take it knows the job is gone. Whoever only asks takes it shared (`shared=True`): askers
do not stop each other, and a held exclusive lock still refuses them.

Windows also refuses to replace or delete a file while anyone has it open (Python
opens files without delete sharing), and `os.kill(pid, 0)` is not a liveness check
there. `replace` and `unlink` wait out a brief reader; `pid_alive` asks the process
itself. On POSIX all three are the plain calls.
"""

import errno
import os
import time
from pathlib import Path

try:
    import fcntl
except ImportError:  # Windows has no fcntl
    fcntl = None
    import msvcrt


# Where a job's ownership lock sits on Windows: a byte far past anything written to the file,
# since Windows byte locks are mandatory and would stop others reading the pid.
OWNER_OFFSET = 2 ** 30


def lock_exclusive(handle):
    if fcntl is not None:
        fcntl.flock(handle, fcntl.LOCK_EX)
        return
    # Wait on contention only; any other error is real and must surface.
    while True:
        if try_lock(handle):
            return
        time.sleep(0.05)


def try_lock(handle, offset=0, shared=False):
    """Take the exclusive lock if nobody holds it, without waiting; whether it was taken.
    `shared`: take it beside other shared holders, refused only while it is held exclusive.
    `offset` is the byte that is locked on Windows (POSIX locks the whole file)."""
    if fcntl is not None:
        try:
            fcntl.flock(handle, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
            return True
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            return False
    return _windows_lock(handle, offset, shared)


def unlock(handle, offset=0, shared=False):
    if fcntl is not None:
        fcntl.flock(handle, fcntl.LOCK_UN)
        return
    _windows_unlock(handle, offset)


# One byte locked with LockFileEx, the call under msvcrt.locking, which has no shared lock
# (and whose errno on a refusal is not reliable under Wine); the Win32 error is.
_LOCKFILE_FAIL_IMMEDIATELY = 0x1
_LOCKFILE_EXCLUSIVE_LOCK = 0x2
_ERROR_LOCK_VIOLATION = 33
_ERROR_IO_PENDING = 997


def _overlapped(offset):
    import ctypes
    from ctypes import wintypes

    class Overlapped(ctypes.Structure):
        _fields_ = [("Internal", ctypes.c_size_t), ("InternalHigh", ctypes.c_size_t),
                    ("Offset", wintypes.DWORD), ("OffsetHigh", wintypes.DWORD),
                    ("hEvent", wintypes.HANDLE)]
    return Overlapped(0, 0, offset & 0xFFFFFFFF, offset >> 32, None)


def _windows_lock(handle, offset, shared):
    import ctypes
    flags = _LOCKFILE_FAIL_IMMEDIATELY | (0 if shared else _LOCKFILE_EXCLUSIVE_LOCK)
    overlapped = _overlapped(offset)
    if _kernel32().LockFileEx(msvcrt.get_osfhandle(handle.fileno()), flags, 0, 1, 0,
                              ctypes.byref(overlapped)):
        return True
    code = ctypes.get_last_error()
    if code in (_ERROR_LOCK_VIOLATION, _ERROR_IO_PENDING):
        return False
    raise ctypes.WinError(code)


def _windows_unlock(handle, offset):
    import ctypes
    overlapped = _overlapped(offset)
    if not _kernel32().UnlockFileEx(msvcrt.get_osfhandle(handle.fileno()), 0, 1, 0,
                                    ctypes.byref(overlapped)):
        raise ctypes.WinError(ctypes.get_last_error())


# How long a replace or delete waits for a reader to let go. Villa's readers hold a file
# for the length of one `json.load`; this is generous, and a real permission problem still
# surfaces, just after the wait.
RETRY_SECONDS = 10.0


def _retry(action):
    if fcntl is not None:
        return action()
    deadline = time.monotonic() + RETRY_SECONDS
    delay = 0.005
    while True:
        try:
            return action()
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.1)


def replace(source, target):
    """`os.replace`, retried on Windows while a reader has `target` open."""
    _retry(lambda: os.replace(source, target))


def unlink(path, missing_ok=False):
    """`Path.unlink`, retried on Windows while a reader has the file open."""
    _retry(lambda: Path(path).unlink(missing_ok=missing_ok))


# ----------------------------------------------------------------------- process liveness

_STILL_ACTIVE = 259
_WAIT_TIMEOUT = 0x102
# Enough to query and to wait on a process we may not otherwise touch.
_QUERY_LIMITED_AND_SYNCHRONIZE = 0x1000 | 0x100000
_kernel = None


def _kernel32():
    global _kernel
    if _kernel is None:
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.LockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                      wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
        kernel.UnlockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                        wintypes.DWORD, ctypes.c_void_p]
        _kernel = kernel
    return _kernel


def _creation_time(kernel, handle):
    import ctypes
    from ctypes import wintypes
    times = [wintypes.FILETIME() for _ in range(4)]
    if not kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
        return None
    return (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime


def process_token(pid=None):
    """What tells this process from a later one that gets the same pid (its creation time,
    on Windows). None where pids are not reused that way, or if it cannot be read."""
    if fcntl is not None:
        return None
    kernel = _kernel32()
    if pid is None:
        return _creation_time(kernel, kernel.GetCurrentProcess())
    handle = kernel.OpenProcess(_QUERY_LIMITED_AND_SYNCHRONIZE, False, pid)
    if not handle:
        return None
    try:
        return _creation_time(kernel, handle)
    finally:
        kernel.CloseHandle(handle)


def pid_alive(pid, token=None):
    """Whether process `pid` is running, without signalling it. `token` (from
    `process_token` when the pid was recorded) must still match, so a reused pid is not
    taken for the old process."""
    if pid <= 0:
        return False
    if fcntl is not None:
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError, OverflowError):
            return False
    if pid >= 2 ** 32:
        return False
    import ctypes
    from ctypes import wintypes
    kernel = _kernel32()
    # No handle: no such process, or one we cannot even query, which a worker of ours is not.
    handle = kernel.OpenProcess(_QUERY_LIMITED_AND_SYNCHRONIZE, False, pid)
    if not handle:
        return False
    try:
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != _STILL_ACTIVE:
            return False
        # A process that exited with code 259 also reads as active; a signalled handle is exited.
        if kernel.WaitForSingleObject(handle, 0) != _WAIT_TIMEOUT:
            return False
        return token is None or _creation_time(kernel, handle) == token
    finally:
        kernel.CloseHandle(handle)
