"""Wire/recovery/policy regression gates. All GPU inventories are synthetic."""

from contextlib import contextmanager
import copy
import json
import multiprocessing as mp
import os
from pathlib import Path
import socket
import signal
import subprocess
import sys
import threading
import time
import uuid

import pytest
import psutil

from villa_gpu_manager.client import Client
from villa_gpu_manager.protocol import MAX_FRAME, ProtocolError, decode, encode
from villa_gpu_manager.server import Server
from villa_gpu_manager.schema import SCHEMA, response_schema
from villa_gpu_manager.policy import empty, fits
from villa_gpu_manager.processes import process_identity, process_state
from test_manager import POOL, engine, call, register, report, request, settle, task_call


@pytest.mark.parametrize("restart", [False, True])
def test_shared_socket_uses_directory_group(tmp_path, monkeypatch, restart):
    path = tmp_path / "shared" / "socket"
    path.parent.mkdir(mode=0o1770)
    path.parent.chmod(0o1770)
    if restart:
        with Server(path, POOL, shared_group=True):
            pass  # Existing journal must not replace the remembered directory GID.
    original = Path.lstat
    shared_gid = os.getgid() + 1

    def directory_stat(p, *args, **kwargs):
        info = original(p, *args, **kwargs)
        if p == path.parent:
            values = list(info)
            values[5] = shared_gid
            return os.stat_result(values)
        return info

    ownership = []
    monkeypatch.setattr(Path, "lstat", directory_stat)
    monkeypatch.setattr(os, "chown", lambda p, uid, gid: ownership.append((p, uid, gid)))
    with Server(path, POOL, shared_group=True):
        assert ownership == [(path, -1, shared_gid)]
        assert path.stat().st_mode & 0o777 == 0o660


def test_shared_socket_group_failure_aborts_startup(tmp_path, monkeypatch):
    path = tmp_path / "shared" / "socket"
    path.parent.mkdir(mode=0o1770)
    path.parent.chmod(0o1770)

    def denied(*args):
        raise PermissionError("cannot assign shared group")

    monkeypatch.setattr(os, "chown", denied)
    with pytest.raises(PermissionError, match="shared group"):
        with Server(path, POOL, shared_group=True):
            pytest.fail("must not start with wrong socket group")
    assert not path.exists()


@contextmanager
def running_server(path, **options):
    options["engine_options"] = {"protected_processes": lambda: [], **options.get("engine_options", {})}
    with Server(path, POOL, **options) as server:
        stop = threading.Event()
        errors = []

        def serve():
            try:
                while not stop.is_set():
                    server.step(0.005)
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=serve)
        thread.start()
        try:
            yield server, errors
        finally:
            stop.set()
            thread.join(5)
            assert not thread.is_alive()


def hello():
    return {"version": 1, "id": str(uuid.uuid4()), "op": "hello",
            "body": {"versions": [1], "required_capabilities": ["poll"]}}


def receive_rows(client, count):
    data = b""
    while data.count(b"\n") < count:
        part = client.recv(65536)
        assert part, "server closed before replying"
        data += part
    return [decode(row) for row in data.splitlines()]


def test_public_partial_coalesced_frames_and_flood_fairness(tmp_path):
    path = tmp_path / "private/socket"
    with running_server(path) as (server, errors):
        with socket.socket(socket.AF_UNIX) as peer:
            peer.settimeout(2)
            peer.connect(str(path))
            frames = [hello(), hello()]
            wire = b"".join(map(encode, frames))
            peer.sendall(wire[:7])
            time.sleep(0.01)
            peer.sendall(wire[7:])
            rows = receive_rows(peer, 2)
            assert [row["id"] for row in rows] == [row["id"] for row in frames]
            assert all(row["ok"] for row in rows)
            # Deliberately don't consume this peer's flood. Another peer must
            # still receive service regardless of the flood's close timing.
            try:
                peer.sendall(encode(hello()) * 100)
            except OSError:
                pass
            with Client(path) as other:
                assert other.status()["tasks"] == []
        assert not errors


def test_oversize_peer_and_connection_limit_leave_service_usable(tmp_path):
    path = tmp_path / "private/socket"
    with running_server(path, max_connections=1) as (server, errors):
        peer = socket.socket(socket.AF_UNIX)
        peer.settimeout(2)
        peer.connect(str(path))
        peer.sendall(encode(hello()))
        assert receive_rows(peer, 1)[0]["ok"]
        with socket.socket(socket.AF_UNIX) as excess:
            excess.settimeout(2)
            excess.connect(str(path))
            assert excess.recv(1) == b""
        try:
            peer.sendall(b"x" * (MAX_FRAME + 1))
            assert peer.recv(1) == b""
        except ConnectionResetError:
            pass
        finally:
            peer.close()
        with Client(path) as other:
            assert other.status()["tasks"] == []
        assert not errors


def test_authorization_pause_latches_and_protocol_rejection():
    e = engine()
    client = register(e)
    report(e, client, "applied")
    identity = {"task_id": client["task"]["task_id"], "token": client["token"]}
    with pytest.raises(ProtocolError, match="authentication"):
        call(e, "poll", {**identity, "token": "b" * 64, "last_generation": 0})
    for bad in ({**identity, "token": "b" * 64}, identity):
        with pytest.raises(ProtocolError, match="authentication"):
            call(e, "poll", {**bad, "last_generation": 0}, uid=os.getuid() + 1)
    with pytest.raises(ProtocolError, match="controller"):
        call(e, "clear_override", {"expected_policy_revision": 0}, uid=os.getuid() + 1)
    task_call(e, client, "set_paused", paused=True, scope="controller")
    task_call(e, client, "set_paused", paused=False, scope="self")
    assert e.tasks[identity["task_id"]]["paused"] == {"self": False, "controller": True}
    for mutation in ({"version": 2}, {"body": {"versions": [2], "required_capabilities": []}},
                     {"body": {"versions": [1], "required_capabilities": ["unknown-required"]}}):
        with pytest.raises(ProtocolError):
            e.dispatch(hello() | mutation, os.getuid())
    with pytest.raises(ProtocolError, match="restarted"):
        e.dispatch({"version": 1, "id": str(uuid.uuid4()), "op": "status", "body": {},
                    "server_epoch": str(uuid.uuid4())}, os.getuid())


def test_bound_override_expiry_and_status_pagination():
    e = engine()
    clients = [register(e) for _ in range(35)]
    a = clients[0]
    call(e, "set_override", {"expected_policy_revision": 0,
        "bound_task_id": a["task"]["task_id"], "allocations": {}})
    settle(e, clients)
    task_call(e, a, "finish", outcome="done")
    assert e.state["override"] is None
    seen, after = [], -1
    while True:
        page = call(e, "status", {"after_order": after, "limit": 3})
        assert len(page["tasks"]) <= 3
        seen.extend(t["task_id"] for t in page["tasks"])
        after = page["next_after_order"]
        if after is None:
            break
    assert len(seen) == len(set(seen)) == 35
    assert fits([t["granted"] for t in e.tasks.values()], POOL)


def test_missing_corrupt_journal_cannot_take_over(tmp_path):
    path = tmp_path / "private/socket"
    with Server(path, POOL):
        pass
    journal = Path(str(path) + ".json")
    original = journal.read_bytes()
    journal.write_text("{")
    with pytest.raises(json.JSONDecodeError):
        with Server(path, POOL):
            pass
    journal.unlink()
    with pytest.raises(RuntimeError, match="missing recovery"):
        with Server(path, POOL):
            pass
    journal.write_bytes(original)
    journal.chmod(0o600)
    with Server(path, POOL):
        pass


def test_all_operation_request_and_response_schemas():
    e = engine()
    fixtures = []

    def invoke(op, body):
        wire = {"version": 1, "id": str(uuid.uuid4()), "op": op, "body": body}
        if op != "hello":
            wire["server_epoch"] = e.epoch
        result = e.dispatch(wire, os.getuid())
        fixtures.extend([(SCHEMA, wire), (response_schema(op),
            {"version": 1, "id": wire["id"], "server_epoch": e.epoch, "ok": True, "body": result})])
        fixtures.append((response_schema(op), {"version": 1, "id": wire["id"], "server_epoch": e.epoch,
                         "ok": False, "error": {"code": "invalid_request", "message": "bad", "details": {}}}))
        return result

    invoke("hello", hello()["body"])
    registered = invoke("register", {"task_id": str(uuid.uuid4()), "token": "a" * 64, "label": "schema",
        "root": {"pid": 1, "start_us": 1, "boot_us": 1}, "reconfigure_mode": "full_release", "request": request()})
    identity = {"task_id": registered["task"]["task_id"], "token": registered["token"]}
    invoke("poll", {**identity, "last_generation": 0})
    invoke("report", {**identity, "generation": registered["task"]["generation"], "phase": "applied",
                      "held": registered["task"]["granted"], "processes": [], "detail": ""})
    invoke("set_request", {**identity, "request": request(), "expected_request_revision": 0})
    invoke("set_override", {"expected_policy_revision": 0, "allocations": {}, "bound_task_id": None})
    invoke("clear_override", {"expected_policy_revision": 1})
    invoke("set_paused", {**identity, "paused": True, "scope": "self"})
    task = e.tasks[identity["task_id"]]
    invoke("report", {**identity, "generation": task["generation"], "phase": "released", "held": empty(), "processes": [], "detail": ""})
    invoke("finish", {**identity, "outcome": "done"})
    invoke("status", {})
    # Optional test dependency may be supplied by a separate system Python.
    validator = os.environ.get("GPU_MANAGER_SCHEMA_PYTHON", sys.executable)
    subprocess.run([validator, "-c", "import json,sys,jsonschema; "
        "[(jsonschema.Draft7Validator(s, format_checker=jsonschema.FormatChecker()).validate(v)) for s,v in json.load(sys.stdin)]"],
        input=json.dumps(fixtures), text=True, check=True)


def test_real_root_death_retains_surviving_child_and_pid_reuse(monkeypatch):
    from villa_gpu_manager.engine import Engine
    root = subprocess.Popen([sys.executable, "-c",
        "import subprocess,sys,time; child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "print(child.pid,flush=True); time.sleep(60)"], stdout=subprocess.PIPE, text=True, start_new_session=True)
    try:
        child = int(root.stdout.readline())
        identity = process_identity(root.pid)
        assert process_state(identity | {"start_us": identity["start_us"] + 1}, os.getuid()) == "dead"
        e = Engine(POOL, controller_uids=[os.getuid()])
        registration = call(e, "register", {"task_id": str(uuid.uuid4()), "token": "a" * 64,
            "label": "real-tree", "root": identity, "request": request(), "reconfigure_mode": "full_release"})
        report(e, registration, "applied")
        root.terminate()
        root.wait(timeout=5)
        e.scan_processes()
        task = e.tasks[registration["task"]["task_id"]]
        assert not task["terminal"] and task["granted"] != empty()
        assert any(row["pid"] == child for row in task["processes"])
        os.kill(child, signal.SIGTERM)
        deadline = time.monotonic() + 5
        while not task["terminal"]:
            e.scan_processes()
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert task["granted"] == empty()
        def denied(_pid):
            raise psutil.AccessDenied()
        monkeypatch.setattr(psutil, "Process", denied)
        assert process_state(identity, os.getuid()) == "unknown"
    finally:
        try:
            os.killpg(root.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        root.wait(timeout=5)
        root.stdout.close()


def _server_process(path, ready):
    with Server(path, POOL) as server:
        ready.send(True)
        ready.close()
        while True:
            server.step(0.01)


def test_real_server_crash_releases_only_after_survivor_ack(tmp_path):
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    path = tmp_path / "private/socket"
    service = context.Process(target=_server_process, args=(path, child))
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    client = Client(path)
    service.start()
    child.close()
    try:
        assert parent.poll(10) and parent.recv()
        client.connect()
        task = client.register("survivor", request(), root=process_identity(sleeper.pid))
        client.report("applied", task["granted"])
        old_epoch = client.epoch
        service.terminate()
        service.join(5)
        assert not service.is_alive()
        client.close()
        with running_server(path) as (server, errors):
            restored = server.engine.tasks[client.task_id]
            assert restored["phase"] == "release" and restored["granted"] != empty()
            client.connect()
            assert client.epoch != old_epoch
            task = client.register("survivor", request(), root=process_identity(sleeper.pid))
            assert task["phase"] == "release"
            client.pause()
            client.report("released", empty())
            assert client.finish()["terminal"]
            assert not errors
    finally:
        client.close()
        if service.is_alive():
            service.terminate()
        service.join(5)
        parent.close()
        sleeper.terminate()
        sleeper.wait(timeout=5)


def test_journal_failure_does_not_acknowledge_grant(tmp_path, monkeypatch):
    import villa_gpu_manager.server as implementation
    path = tmp_path / "private/socket"
    with pytest.raises(OSError, match="simulated fsync failure"):
        with running_server(path) as (server, errors):
            def fail_save(*args):
                raise OSError("simulated fsync failure")
            monkeypatch.setattr(implementation, "save_journal", fail_save)
            with Client(path, timeout=0.2) as client:
                with pytest.raises(TimeoutError):
                    client.register("no-uncommitted-grant", request())
            assert not json.loads(Path(str(path) + ".json").read_text())["tasks"]
            assert errors
            raise errors[0]


def test_evicted_dedup_still_rejects_stale_mutation():
    e = engine()
    registered = register(e)
    identity = {"task_id": registered["task"]["task_id"], "token": registered["token"]}
    old = {"version": 1, "id": str(uuid.uuid4()), "server_epoch": e.epoch, "op": "set_request",
           "body": {**identity, "request": request(), "expected_request_revision": 0}}
    e.dispatch(old, os.getuid())
    for revision in range(1, 260):
        call(e, "set_request", {**identity, "request": request(), "expected_request_revision": revision})
    assert len(e.state["dedup"]) == 256
    with pytest.raises(ProtocolError, match="stale request revision"):
        e.dispatch(old, os.getuid())
