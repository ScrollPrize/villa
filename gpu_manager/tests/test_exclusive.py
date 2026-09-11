import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from villa_gpu_manager.client import Client
from villa_gpu_manager.exclusive import ExclusiveTask
from villa_gpu_manager.policy import empty
from villa_gpu_manager.server import Server


GRANT = {"cpu_cores": 4, "gpus": {"GPU-fake": 100}}


@pytest.fixture
def manager(tmp_path):
    path = tmp_path / "manager.sock"
    stop = threading.Event()
    with Server(path, GRANT, engine_options={"protected_processes": lambda: []}) as server:
        server.engine.inspect = lambda *args: "alive"
        server.engine.discover = lambda *args: []
        def serve():
            while not stop.is_set():
                server.step()
        thread = threading.Thread(target=serve)
        thread.start()
        try:
            yield path
        finally:
            stop.set()
            thread.join(timeout=10)
            assert not thread.is_alive()
            server.close(force=True)  # Fake inventory; failure tests retain reservations.


class Cohort:
    closed = False
    processes = []

    def __init__(self, lease):
        self.lease = lease
        self.authorized = False

    def authorize(self):
        assert self.lease.client.poll()["phase"] == "applied"
        self.authorized = True

    def abort(self):
        assert self.lease.client.poll()["granted"] == GRANT
        self.closed = True


@pytest.mark.parametrize("pausable", [True, False])
def test_exclusive_waits_for_release_and_joins_before_release(manager, pausable):
    with Client(manager) as old:
        request = ExclusiveTask(manager, GRANT, label="unused").request | {"exclusive": False}
        old.register("training", request, root={"pid": 99999999, "start_us": 1, "boot_us": 1},
                     pausable=pausable, adaptable=pausable)
        old.report("applied", GRANT)
        lease = ExclusiveTask(manager, GRANT, label="probe", priority=10)
        with ThreadPoolExecutor(1) as pool:
            pending = pool.submit(lease.__enter__)
            deadline = time.monotonic() + 5
            while len(old.status()["tasks"]) < 2:
                if pending.done():
                    pending.result()
                assert time.monotonic() < deadline
                time.sleep(.01)
            assert not pending.done()
            assert old.poll()["phase"] == ("release" if pausable else "applied")
            if not pausable:
                old.pause(True)
            old.report("released", empty())
            assert pending.result(timeout=5) is lease
        lease.cohort = cohort = Cohort(lease)
        lease.authorize()
        lease.check()
        assert cohort.authorized
        lease.close()
        assert cohort.closed
        if pausable:
            assert old.poll()["phase"] == "initialize"
            old.pause(True)
            old.report("released", empty())
        old.finish()


def test_exception_releases_after_abort(manager):
    lease = ExclusiveTask(manager, GRANT, label="probe")
    with pytest.raises(ValueError, match="probe failure"):
        with lease:
            lease.cohort = cohort = Cohort(lease)
            lease.authorize()
            raise ValueError("probe failure")
    assert cohort.closed
    with Client(manager) as client:
        task = next(t for t in client.status()["tasks"] if t["task_id"] == lease.client.task_id)
        assert task["granted"] == empty()


def test_failed_join_does_not_claim_release(manager):
    lease = ExclusiveTask(manager, GRANT, label="probe").__enter__()
    lease.cohort = cohort = Cohort(lease)
    lease.authorize()
    cohort.abort = lambda: None
    with pytest.raises(RuntimeError, match="before workers join"):
        lease.close()
    with Client(manager) as client:
        task = next(t for t in client.status()["tasks"] if t["task_id"] == lease.client.task_id)
        assert task["granted"] == GRANT


def test_priority_requires_integer():
    with pytest.raises(ValueError):
        ExclusiveTask("unused", GRANT, label="probe", priority=True)


def test_lost_allocation_aborts_before_release(manager, monkeypatch):
    lease = ExclusiveTask(manager, GRANT, label="probe")
    with pytest.raises(RuntimeError, match="allocation changed"):
        with lease:
            lease.cohort = cohort = Cohort(lease)
            lease.authorize()
            original = lease.client.poll
            # Simulate a changed generation response, without changing ownership.
            monkeypatch.setattr(lease.client, "poll", lambda: original() | {"generation": -1})
            lease.check()
    assert cohort.closed
