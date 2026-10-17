"""close_volume_root closes the remote HTTP session and leaves everything else alone."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vesuvius.ink_detection.volume_io import close_volume_root


class _Session:
    def __init__(self) -> None:
        self.closed = False
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1
        self.closed = True


def _root_with_session(nested: bool):
    fs = SimpleNamespace(_session=_Session())
    remote_store = SimpleNamespace(fs=fs)
    store = SimpleNamespace(store=remote_store) if nested else remote_store
    return SimpleNamespace(store=store), fs


@pytest.mark.parametrize("nested", [False, True])
def test_open_session_is_closed_once_and_dropped(nested):
    root, fs = _root_with_session(nested)
    session = fs._session

    close_volume_root(root)

    assert session.closed is True
    assert session.close_calls == 1
    assert fs._session is None
    # idempotent: a second call finds nothing to close
    close_volume_root(root)
    assert session.close_calls == 1


def test_local_and_sessionless_roots_are_no_ops():
    close_volume_root(SimpleNamespace(store=SimpleNamespace()))
    close_volume_root(SimpleNamespace(store=SimpleNamespace(fs=SimpleNamespace(_session=None))))
    already = _Session()
    already.closed = True
    fs = SimpleNamespace(_session=already)
    close_volume_root(SimpleNamespace(store=SimpleNamespace(fs=fs)))
    assert already.close_calls == 0
    close_volume_root(None)
