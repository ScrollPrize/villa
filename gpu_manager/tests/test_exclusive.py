"""Synthetic GPU promises and real CPU-only gated subprocesses; no CUDA."""
import os
from pathlib import Path
import sys
import threading

import pytest

from villa_gpu_manager.client import Client
from villa_gpu_manager.exclusive import run_exclusive
from villa_gpu_manager.policy import empty
from villa_gpu_manager.processes import process_state, process_identity
from test_manager import engine, call, register, report, task_call
from test_conformance import running_server


def memory_client(e, hook=None):
    class MemoryClient(Client):
        def connect(self):
            self.socket = True
            self.epoch = e.epoch
            return {"inventory": e.inventory}
        def close(self):
            self.socket = None
        def call(self, op, body, **kwargs):
            if hook:
                hook(op, body)
            return call(e, op, body)
    return MemoryClient


def marker_command(path):
    return [sys.executable, "-c", "import pathlib,sys,os; pathlib.Path(sys.argv[1]).write_text(os.environ['CUDA_VISIBLE_DEVICES'])", str(path)]


@pytest.mark.parametrize("cancel", [False, True])
def test_exact_wait_returns_partial_without_starting_workers(tmp_path, cancel):
    e = engine()
    def donor_identity(op, body):
        if op == "register":
            body["root"] = dict(pid=1, start_us=1, boot_us=1)
    donor = memory_client(e, donor_identity)("unused")
    donor.connect()
    donor.register("occupied", dict(max_cpu_cores=4, max_gpus=1,
        min_vram_bytes_per_gpu=80, max_vram_bytes_per_gpu=80,
        allowed_gpu_uuids=["GPU-b"], priority=0, exclusive=False),
        pausable=False, adaptable=False)
    donor.report("applied", donor.poll()["granted"])
    marker = tmp_path / "exact"
    stop = threading.Event()
    returned = []
    def hook(op, body):
        if op == "set_paused" and body["paused"] and not returned:
            assert not marker.exists()
            returned.append(body["task_id"])
            if cancel:
                stop.set()
            else:
                donor.pause(True)
                donor.report("released", empty())
                donor.finish()
    code = run_exclusive(marker_command(marker), socket="unused",
        gpu_uuids=["GPU-a", "GPU-b"], client_factory=memory_client(e, hook),
        exclusive=False, cpu_cores=8, vram_bytes_per_gpu=40,
        wait_for_exact=True, exact_retry_seconds=.01, poll_interval=.001,
        stop_event=stop)
    assert returned
    assert code == (130 if cancel else 0)
    assert marker.exists() == (not cancel)
    assert len(e.tasks) == (2 if cancel else 3)
    assert e.tasks[returned[0]]["terminal"]
    assert e.tasks[returned[0]]["granted"] == empty()


def test_fixed_nonexclusive_command_shares_inventory(tmp_path):
    e = engine()
    donor = register(e, max_cpu_cores=8, max_gpus=1,
        min_vram_bytes_per_gpu=40, max_vram_bytes_per_gpu=40,
        allowed_gpu_uuids=["GPU-b"], priority=0, exclusive=False)
    report(e, donor, "applied")
    marker = tmp_path / "shared"
    assert run_exclusive(marker_command(marker), socket="unused", gpu_uuids=["GPU-a"],
        client_factory=memory_client(e), poll_interval=.01, priority=0,
        exclusive=False, cpu_cores=4, vram_bytes_per_gpu=30) == 0
    assert marker.read_text() == "GPU-a"
    task = max(e.tasks.values(), key=lambda t: t["order"])
    assert not task["request"]["exclusive"] and not task["pausable"] and not task["adaptable"]
    assert task["request"]["max_vram_bytes_per_gpu"] == 30
    assert e.tasks[donor["task"]["task_id"]]["phase"] == "applied"


def test_nonexclusive_requires_explicit_budgets(tmp_path):
    with pytest.raises(ValueError, match="explicit CPU"):
        run_exclusive(marker_command(tmp_path / "no"), socket="unused", gpu_uuids=["GPU-a"],
            client_factory=memory_client(engine()), exclusive=False)


def test_exclusive_waits_for_donor_and_gates_before_applied(tmp_path):
    e = engine()
    donor = register(e)
    report(e, donor, "applied")
    marker = tmp_path / "executed"
    events = []
    def hook(op, body):
        events.append(op)
        if op == "poll" and e.tasks[donor["task"]["task_id"]]["phase"] == "release":
            assert not marker.exists()
            report(e, donor, "released", empty())
        if op == "report" and body["phase"] == "applied":
            assert not marker.exists()
            assert len(body["processes"]) == 2
            assert all(process_state(p, os.getuid()) == "alive" for p in body["processes"])
        if op == "report" and body["phase"] == "released":
            owner = e.tasks[body["task_id"]]
            assert all(process_state(p, os.getuid()) == "dead" for p in owner["processes"][1:])
    assert run_exclusive(marker_command(marker), socket="unused", gpu_uuids=["GPU-a", "GPU-b"],
                         client_factory=memory_client(e, hook), poll_interval=.01) == 0
    assert marker.read_text() == "GPU-a,GPU-b"
    high = max(e.tasks.values(), key=lambda t: t["order"])
    assert high["request"]["exclusive"] and high["request"]["priority"] == 100
    assert not high["pausable"] and not high["adaptable"]
    assert high["terminal"] and high["granted"] == empty()
    assert e.tasks[donor["task"]["task_id"]]["phase"] == "initialize"


def test_uncertain_applied_response_never_executes_and_releases_after_cleanup(tmp_path):
    e = engine()
    marker = tmp_path / "forbidden"
    base = memory_client(e)
    class LostReply(base):
        def report(self, phase, held, **kwargs):
            result = super().report(phase, held, **kwargs)
            if phase == "applied":
                self.close()
                raise ConnectionError("lost applied reply")
            return result
    with pytest.raises(ConnectionError, match="lost applied"):
        run_exclusive(marker_command(marker), socket="unused", gpu_uuids=["GPU-a"],
                      client_factory=LostReply, poll_interval=.01)
    assert not marker.exists()
    task, = e.tasks.values()
    assert task["terminal"] and task["granted"] == empty()


def test_nonpausable_blocker_does_not_get_evicted(tmp_path):
    e = engine()
    from test_scheduling import rigid
    blocker = rigid(e)
    report(e, blocker, "applied")
    stop = threading.Event()
    def hook(op, body):
        if op == "poll":
            stop.set()
    marker = tmp_path / "forbidden"
    assert run_exclusive(marker_command(marker), socket="unused", gpu_uuids=["GPU-a"],
                         client_factory=memory_client(e, hook), stop_event=stop, poll_interval=.01) == 130
    assert not marker.exists()
    assert e.tasks[blocker["task"]["task_id"]]["phase"] == "applied"


def test_unverifiable_cleanup_never_reports_release(monkeypatch, tmp_path):
    from villa_gpu_manager import exclusive
    e = engine()
    original = exclusive.GatedCommand.close
    def fail_after_actual_cleanup(self):
        original(self)
        raise RuntimeError("cleanup verification failed")
    monkeypatch.setattr(exclusive.GatedCommand, "close", fail_after_actual_cleanup)
    with pytest.raises(RuntimeError, match="cleanup verification"):
        run_exclusive(marker_command(tmp_path / "marker"), socket="unused", gpu_uuids=["GPU-a"],
                      client_factory=memory_client(e), poll_interval=.01)
    task, = e.tasks.values()
    assert not task["terminal"] and task["granted"] != empty()


def test_command_failure_cleans_surviving_group_child(tmp_path):
    e = engine()
    pidfile = tmp_path / "pid"
    code = "import subprocess,sys,pathlib; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); pathlib.Path(sys.argv[1]).write_text(str(p.pid)); raise SystemExit(7)"
    assert run_exclusive([sys.executable, "-c", code, str(pidfile)], socket="unused", gpu_uuids=["GPU-a"],
                         client_factory=memory_client(e), poll_interval=.01) == 7
    import psutil
    try:
        assert psutil.Process(int(pidfile.read_text())).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        pass
    task, = e.tasks.values()
    assert task["terminal"] and task["granted"] == empty()


def test_real_socket_exclusive_command(tmp_path):
    path = tmp_path / "private/socket"
    with running_server(path) as (server, errors):
        assert run_exclusive(marker_command(tmp_path / "marker"), socket=path,
                             gpu_uuids=["GPU-a", "GPU-b"], poll_interval=.01) == 0
        assert not errors
        task, = server.engine.tasks.values()
        assert task["terminal"] and task["granted"] == empty()


def test_running_disconnect_reconnects_without_duplicate_launch(tmp_path):
    e = engine()
    base = memory_client(e)
    class DisconnectOnce(base):
        lost = False
        def poll(self):
            if self.task["phase"] == "applied" and not self.lost:
                self.lost = True
                self.close()
                raise ConnectionError("injected")
            return super().poll()
    marker = tmp_path / "once"
    code = "import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); p.write_text(p.read_text()+'x' if p.exists() else 'x'); time.sleep(.08)"
    assert run_exclusive([sys.executable, "-c", code, str(marker)], socket="unused", gpu_uuids=["GPU-a"],
                         client_factory=DisconnectOnce, poll_interval=.01) == 0
    assert marker.read_text() == "x" and len(e.tasks) == 1
    assert next(iter(e.tasks.values()))["terminal"]


def test_cancellation_stops_active_group_before_release(tmp_path):
    e = engine()
    stop = threading.Event()
    def hook(op, body):
        if op == "poll" and any(t["phase"] == "applied" for t in e.tasks.values()):
            stop.set()
    assert run_exclusive([sys.executable, "-c", "import time; time.sleep(60)"],
                         socket="unused", gpu_uuids=["GPU-a"], poll_interval=.01,
                         stop_event=stop, client_factory=memory_client(e, hook)) == 130
    task, = e.tasks.values()
    assert task["terminal"] and task["granted"] == empty()


def test_partial_override_grant_is_rejected_before_command(tmp_path):
    e = engine()
    base = memory_client(e)
    class PartialGrant(base):
        def register(self, *args, **kwargs):
            task = super().register(*args, **kwargs)
            task["granted"]["gpus"].pop("GPU-b", None)
            return task
    marker = tmp_path / "forbidden"
    with pytest.raises(RuntimeError, match="only part"):
        run_exclusive(marker_command(marker), socket="unused", gpu_uuids=["GPU-a", "GPU-b"],
                      client_factory=PartialGrant, poll_interval=.01)
    assert not marker.exists()


def test_partial_cpu_grant_is_rejected_before_command(tmp_path):
    e = engine()
    base = memory_client(e)
    class PartialCPU(base):
        def register(self, *args, **kwargs):
            task = super().register(*args, **kwargs)
            task["granted"]["cpu_cores"] -= 1
            return task
    marker = tmp_path / "forbidden"
    with pytest.raises(RuntimeError, match="only part"):
        run_exclusive(marker_command(marker), socket="unused", gpu_uuids=["GPU-a", "GPU-b"],
                      client_factory=PartialCPU, poll_interval=.01)
    assert not marker.exists()


def test_heterogeneous_inventory_gets_exact_cpu_and_gpu_grant(tmp_path):
    e = engine()
    e.inventory["gpus"]["GPU-b"] //= 2
    assert run_exclusive(marker_command(tmp_path / "marker"), socket="unused",
                         gpu_uuids=["GPU-a", "GPU-b"], client_factory=memory_client(e),
                         poll_interval=.01) == 0
    task, = e.tasks.values()
    assert task["fixed_allocation"] == e.inventory


def test_prelaunch_guard_failure_does_not_launch_and_releases(tmp_path):
    e = engine()
    marker = tmp_path / "forbidden"
    def guard():
        raise RuntimeError("external GPU process")
    with pytest.raises(RuntimeError, match="external GPU"):
        run_exclusive(marker_command(marker), socket="unused", gpu_uuids=["GPU-a"],
                      client_factory=memory_client(e), poll_interval=.01, before_authorize=guard)
    assert not marker.exists()
    task, = e.tasks.values()
    assert task["terminal"] and task["granted"] == empty()


def test_sigterm_is_a_clean_cancellation(tmp_path):
    import signal
    e = engine()
    original = signal.getsignal(signal.SIGTERM)
    sent = False
    def hook(op, body):
        nonlocal sent
        if op == "poll" and not sent and any(t["phase"] == "applied" for t in e.tasks.values()):
            sent = True
            os.kill(os.getpid(), signal.SIGTERM)
    assert run_exclusive([sys.executable, "-c", "import time; time.sleep(60)"],
        socket="unused", gpu_uuids=["GPU-a"], poll_interval=.01, client_factory=memory_client(e, hook)) == 130
    assert signal.getsignal(signal.SIGTERM) == original
    task, = e.tasks.values()
    assert task["terminal"] and task["granted"] == empty()


def test_reported_watchdog_cleans_group_after_launcher_sigkill(tmp_path):
    import json
    import subprocess
    import time
    from villa_gpu_manager import exclusive
    marker = tmp_path / "identities"
    code = """import json,os,pathlib,sys,time
sys.path.insert(0, sys.argv[1])
from villa_gpu_manager.exclusive import GatedCommand
c = GatedCommand([sys.executable, '-c', 'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); time.sleep(60)'], dict(os.environ))
c.authorize()
pathlib.Path(sys.argv[2]).write_text(json.dumps(c.identities))
time.sleep(60)
"""
    launcher = subprocess.Popen([sys.executable, "-c", code,
        str(Path(exclusive.__file__).resolve().parents[1]), str(marker)])
    identities = []
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        identities = json.loads(marker.read_text())
        group = exclusive.OwnedProcessGroup(identities[0])
        while len(group.members()) < 2 and time.monotonic() < deadline:
            time.sleep(.01)
        assert len(group.members()) >= 2
        assert process_state(identities[1], os.getuid()) == "alive"
        launcher.kill()
        launcher.wait(timeout=5)
        deadline = time.monotonic() + 12
        while any(process_state(p, os.getuid()) != "dead" for p in identities) and time.monotonic() < deadline:
            time.sleep(.02)
        assert not group.members()
        assert all(process_state(p, os.getuid()) == "dead" for p in identities)
    finally:
        if launcher.poll() is None:
            launcher.kill()
            launcher.wait(timeout=5)
        if identities:
            exclusive.OwnedProcessGroup(identities[0]).close()

import threading
import time
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from villa_gpu_manager.client import Client
from villa_gpu_manager.exclusive import ExclusiveTask
from villa_gpu_manager.exclusive_command import ExclusiveCommand
from villa_gpu_manager.policy import empty
from villa_gpu_manager.server import Server


GRANT = {"cpu_cores": 4, "gpus": {"GPU-fake": 100}}


def test_command_starts_only_after_authorization_and_is_joined(manager, tmp_path):
    marker = tmp_path / "started"
    with ExclusiveTask(manager, GRANT, label="command-test") as lease:
        command = ExclusiveCommand([sys.executable, "-c",
            "import pathlib,time; pathlib.Path(" + repr(str(marker)) + ").touch(); time.sleep(30)"])
        lease.cohort = command
        assert command.process is None and not marker.exists()
        lease.authorize()
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()
        assert lease.client.poll()["phase"] == "applied"
    assert command.closed and command.process.poll() is not None


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
