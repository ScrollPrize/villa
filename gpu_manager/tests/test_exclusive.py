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
