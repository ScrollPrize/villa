"""The service's mutation coordinator, client lease and checkpoint uploads."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import threading
import zipfile
from pathlib import Path

import pytest

from service_http import ApiError
from service_mutations import ClientLease, MutationCoordinator
from service_uploads import UploadEnvironment, UploadManager


def test_parallel_duplicate_commands_execute_once_and_reject_different_payload():
    coordinator = MutationCoordinator()
    started, finish = threading.Event(), threading.Event()
    calls = []

    def operation(payload):
        calls.append(payload)
        started.set()
        assert finish.wait(5)
        return {"revision": 1}

    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(coordinator.execute, "one", "apply", {"v": 1}, operation)
        assert started.wait(5)
        assert coordinator.outcome("one")["state"] == "running"
        second = pool.submit(coordinator.execute, "one", "apply", {"v": 1}, operation)
        with pytest.raises(ApiError):
            coordinator.execute("one", "apply", {"v": 2}, operation)
        finish.set()
        assert first.result(5) == second.result(5) == {"revision": 1}
    assert calls == [{"v": 1}]
    for n in range(300):
        coordinator.execute(str(n), "apply", {}, lambda payload: {})
    assert coordinator.execute("one", "apply", {"v": 1}, operation) == {"revision": 1}
    assert len(calls) == 1  # No LRU expiration while the coordinator lives.


def test_recovery_rejects_queued_waiters_without_blocking_later_commands(monkeypatch):
    coordinator = MutationCoordinator()
    started, fail, queued = threading.Event(), threading.Event(), threading.Event()
    original_wait = coordinator._condition.wait

    def wait(timeout=None):
        queued.set()
        return original_wait(timeout)

    monkeypatch.setattr(coordinator._condition, 'wait', wait)

    def operation(payload):
        started.set()
        assert fail.wait(5)
        raise OSError('publication interrupted')

    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(coordinator.execute, 'first', 'commit', {}, operation,
                            recoverable=True)
        assert started.wait(5)
        second = pool.submit(coordinator.execute, 'second', 'apply', {}, lambda p: p)
        assert queued.wait(5)
        fail.set()
        with pytest.raises(OSError):
            first.result(5)
        with pytest.raises(ApiError, match='needs recovery'):
            second.result(5)
    assert coordinator.outcome('second')['state'] == 'rejected'
    assert coordinator._queue == []
    coordinator.execute('first', 'commit', {}, lambda p: p, recoverable=True)
    with pytest.raises(ApiError, match='needs recovery'):
        coordinator.execute('second', 'apply', {}, lambda p: p)
    assert coordinator.execute('reconnect', 'claim', {}, lambda p: 'claimed') == 'claimed'


def test_one_service_and_one_client_hold_a_lease(tmp_path):
    lease = ClientLease(tmp_path)
    lease.claim('owner')
    lease.claim('owner')  # Reconnecting with the same token.
    with pytest.raises(ApiError, match='does not own'):
        lease.claim('another-client')
    other = ClientLease(tmp_path)
    with pytest.raises(ApiError, match='Another service'):
        other.claim('another-client')
    with pytest.raises(ApiError) as refused:
        lease.claim('')
    assert refused.value.status == 400
    lease.release('owner')
    other.claim('another-client')
    other.close()


def _checkpoint_bytes():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('data.pkl', b'payload')
    return buffer.getvalue()


DATA = _checkpoint_bytes()


@pytest.fixture
def manager(tmp_path):
    return UploadManager(UploadEnvironment(
        lock=threading.RLock(), output_root=lambda: tmp_path,
        session_id=lambda: "session"))


def upload_manifest(kind='checkpoint'):
    return {"upload_id": "a" * 32, "kind": kind, "id": "resume.ckpt",
            "files": [{"name": "resume.ckpt", "size": len(DATA),
                       "sha256": hashlib.sha256(DATA).hexdigest()}]}


def test_only_checkpoints_are_uploaded(manager):
    for index, kind in enumerate(('patch', 'fiber', 'pcl')):
        with pytest.raises(ApiError, match='Only checkpoints'):
            manager.begin({**upload_manifest(kind), 'upload_id': f'{index:032x}'})
    assert manager.uploads == {}


def test_cancellation_retains_receipt_and_never_recreates_transfer(manager):
    upload_id = manager.begin(upload_manifest())["upload_id"]
    manager.cancel(upload_id)
    manager.cancel(upload_id)
    assert manager.begin(upload_manifest())["upload_id"] == upload_id
    assert manager.status(upload_id)["state"] == "cancelled"
    assert not manager.uploads[upload_id].staging_dir.exists()
    with pytest.raises(ApiError, match="cancelled"):
        manager.receive(upload_id, "resume.ckpt", io.BytesIO(DATA), len(DATA))
    with pytest.raises(ApiError, match="cancelled"):
        manager.finalize(upload_id)


@pytest.mark.parametrize('partial', [False, True])
def test_failed_transfer_cannot_finalize_and_retries_exact_bytes(manager, partial):
    upload_id = manager.begin(upload_manifest())['upload_id']
    with pytest.raises(ApiError):
        manager.receive(upload_id, 'resume.ckpt',
                        io.BytesIO(DATA[:8] if partial else b'x' * len(DATA)),
                        len(DATA), offset=0)
    with pytest.raises(ApiError, match='missing'):
        manager.finalize(upload_id)
    offset = manager.status(upload_id)['files'][0]['offset']
    manager.receive(upload_id, 'resume.ckpt', io.BytesIO(DATA[offset:]),
                    len(DATA) - offset, offset=offset)
    result = manager.finalize(upload_id)
    assert Path(result.record['path']).read_bytes() == DATA
