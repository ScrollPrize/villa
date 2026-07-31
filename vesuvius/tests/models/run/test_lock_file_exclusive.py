"""_lock_file_exclusive must block on contention but surface real errors.

The Windows branch retries because msvcrt.locking gives up after ~10 s where
fcntl.flock waits indefinitely. That retry loop is only correct if it is
limited to lock contention: any other OSError would otherwise spin forever.
Both branches are exercised here regardless of host OS by swapping the module
globals, since CI runs on Linux only.
"""

from __future__ import annotations

import errno

import pytest

from vesuvius.models.run import inference


class _FakeMsvcrt:
    """Stand-in for msvcrt that fails a fixed number of times, then succeeds."""

    LK_LOCK = 1

    def __init__(self, failures: list[int]) -> None:
        self._failures = list(failures)
        self.calls = 0

    def locking(self, fd: int, mode: int, nbytes: int) -> None:
        self.calls += 1
        if self._failures:
            raise OSError(self._failures.pop(0), "locking failed")


@pytest.fixture()
def windows_branch(monkeypatch):
    """Force _lock_file_exclusive down the no-fcntl path with a fake msvcrt."""

    def _install(failures: list[int]) -> _FakeMsvcrt:
        fake = _FakeMsvcrt(failures)
        monkeypatch.setattr(inference, "fcntl", None, raising=False)
        monkeypatch.setattr(inference, "msvcrt", fake, raising=False)
        return fake

    return _install


def test_retries_until_contention_clears(windows_branch, tmp_path):
    fake = windows_branch([errno.EDEADLK, errno.EACCES])
    with open(tmp_path / "lock", "w") as fh:
        inference._lock_file_exclusive(fh)
    assert fake.calls == 3


@pytest.mark.parametrize("code", [errno.EBADF, errno.ENOSPC, errno.EINVAL])
def test_non_contention_errors_propagate(windows_branch, tmp_path, code):
    # Without this the loop would never terminate.
    windows_branch([code])
    with open(tmp_path / "lock", "w") as fh:
        with pytest.raises(OSError) as excinfo:
            inference._lock_file_exclusive(fh)
    assert excinfo.value.errno == code


def test_posix_branch_uses_flock(monkeypatch, tmp_path):
    calls = []

    class _FakeFcntl:
        LOCK_EX = 2

        def flock(self, fh, op):
            calls.append(op)

    monkeypatch.setattr(inference, "fcntl", _FakeFcntl(), raising=False)
    with open(tmp_path / "lock", "w") as fh:
        inference._lock_file_exclusive(fh)
    assert calls == [_FakeFcntl.LOCK_EX]
