import copy
import os
import socket
import threading
import uuid
import random

import pytest

from villa_gpu_manager.client import Client
from villa_gpu_manager.engine import Engine
from villa_gpu_manager.policy import empty, fits
from villa_gpu_manager.protocol import ProtocolError, decode
from villa_gpu_manager.server import Server


POOL = {"cpu_cores": 16, "gpus": {"GPU-a": 100, "GPU-b": 100}}


def request(**kwargs):
    return dict(max_cpu_cores=16, max_gpus=2, min_vram_bytes_per_gpu=20,
                max_vram_bytes_per_gpu=100, allowed_gpu_uuids=[], priority=0, exclusive=False, **{}) | kwargs


def engine(state=None):
    return Engine(POOL, controller_uids=[os.getuid()], state=state,
                  inspect=lambda p, uid: "alive", discover=lambda p, uid: [])


def call(e, op, body, uid=None, request_id=None):
    return e.dispatch({"version": 1, "id": request_id or str(uuid.uuid4()), "op": op,
                       "body": body, "server_epoch": e.epoch}, os.getuid() if uid is None else uid)


def register(e, **kwargs):
    key = str(uuid.uuid4())
    return call(e, "register", {"task_id": key, "label": key,
                "root": {"pid": len(e.tasks) + 1, "start_us": 1, "boot_us": 1},
                "request": request(**kwargs), "reconfigure_mode": "full_release", "token": "a" * 64})


def task_call(e, r, op, **body):
    return call(e, op, {"task_id": r["task"]["task_id"], "token": r["token"], **body})


def report(e, r, phase, held=None):
    task = e.tasks[r["task"]["task_id"]]
    return task_call(e, r, "report", generation=task["generation"], phase=phase,
                     held=held if held is not None else task["granted"], processes=[], detail="")


@pytest.mark.parametrize("mutation", ["cpu_float", "cpu_bool", "vram_float", "extra"])
@pytest.mark.parametrize("already_applied", [False, True])
def test_report_resource_types_before_applied_retry(mutation, already_applied):
    e = engine()
    r = register(e, max_cpu_cores=1, max_gpus=1)
    if already_applied:
        report(e, r, "applied")
    held = copy.deepcopy(e.tasks[r["task"]["task_id"]]["granted"])
    if mutation == "cpu_float":
        held["cpu_cores"] = float(held["cpu_cores"])
    elif mutation == "cpu_bool":
        held["cpu_cores"] = True
    elif mutation == "vram_float":
        gpu = next(iter(held["gpus"]))
        held["gpus"][gpu] = float(held["gpus"][gpu])
    else:
        held["extra"] = 0
    before = copy.deepcopy(e.state)
    with pytest.raises(ProtocolError):
        report(e, r, "applied", held)
    assert e.state == before


def test_dedup_rejects_numerically_equal_but_differently_typed_request():
    e = engine()
    r = register(e, max_cpu_cores=1, max_gpus=1)
    body = {"task_id": r["task"]["task_id"], "token": r["token"],
            "generation": r["task"]["generation"], "phase": "applied",
            "held": copy.deepcopy(r["task"]["granted"]), "processes": [], "detail": ""}
    request_id = str(uuid.uuid4())
    first = call(e, "report", body, request_id=request_id)
    assert call(e, "report", body, request_id=request_id) == first
    body = copy.deepcopy(body)
    body["held"]["cpu_cores"] = True
    with pytest.raises(ProtocolError, match="request ID reused"):
        call(e, "report", body, request_id=request_id)


@pytest.mark.parametrize("field,value", [("label", 1), ("root", {"pid": True, "start_us": 1, "boot_us": 1}),
                                        ("request", request(max_gpus=1.0)), ("reconfigure_mode", "other")])
def test_registration_reconnect_validates_body(field, value):
    e = engine()
    r = register(e)
    body = {"task_id": r["task"]["task_id"], "token": r["token"],
            "label": r["task"]["label"], "root": r["task"]["root"],
            "request": request(), "reconfigure_mode": "full_release"}
    # Reconnect preserves the current request, rather than replaying stale settings.
    task_call(e, r, "set_request", request=request(priority=1), expected_request_revision=0)
    assert call(e, "register", body)["task"]["request"]["priority"] == 1
    body[field] = value
    with pytest.raises(ProtocolError):
        call(e, "register", body)


def settle(e, clients):
    for _ in range(20):
        changed = False
        for r in clients:
            t = e.tasks[r["task"]["task_id"]]
            if t["phase"] == "release":
                report(e, r, "released", empty())
                changed = True
            elif t["phase"] == "initialize":
                report(e, r, "applied")
                changed = True
        if not changed:
            return
    pytest.fail("handoff did not settle")


@pytest.mark.parametrize("data", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}', b'[]', b'\xff', b'{'])
def test_strict_json(data):
    with pytest.raises(ProtocolError):
        decode(data)


def test_donor_releases_before_grant_and_cpu_drain():
    e = engine()
    a = register(e)
    report(e, a, "applied")
    b = register(e, priority=10)
    ta, tb = e.tasks.values()
    assert ta["phase"] == "release" and tb["phase"] == "waiting"
    report(e, a, "released", {"cpu_cores": 16, "gpus": {}})
    assert tb["phase"] == "waiting"
    report(e, a, "released", empty())
    settle(e, [a, b])
    assert all(t["granted"]["gpus"] for t in e.tasks.values())
    assert fits([t["granted"] for t in e.tasks.values()], POOL)


def test_request_during_initialize_does_not_replace_grant():
    e = engine()
    a = register(e)
    original = copy.deepcopy(e.tasks[a["task"]["task_id"]]["granted"])
    task_call(e, a, "set_request", request=request(max_gpus=1), expected_request_revision=0)
    t = e.tasks[a["task"]["task_id"]]
    assert t["phase"] == "initialize" and t["granted"] == original
    report(e, a, "applied")
    assert t["phase"] == "release"
    settle(e, [a])
    assert len(t["granted"]["gpus"]) == 1


def test_override_swap_and_pause_invalidation():
    e = engine()
    a, b = register(e, max_gpus=1), register(e, max_gpus=1)
    settle(e, [a, b])
    ka, kb = [r["task"]["task_id"] for r in (a, b)]
    aa, ab = [copy.deepcopy(e.tasks[k]["granted"]) for k in (ka, kb)]
    call(e, "set_override", {"expected_policy_revision": 0, "allocations": {ka: ab, kb: aa}, "bound_task_id": ka})
    assert all(t["phase"] == "release" for t in e.tasks.values())
    settle(e, [a, b])
    assert e.tasks[ka]["granted"] == ab
    task_call(e, a, "set_paused", paused=True, scope="self")
    assert e.state["override"] is None
    settle(e, [a, b])
    assert e.tasks[ka]["granted"] == empty()


def test_exclusive_waits_for_entire_pool():
    e = engine()
    a = register(e, max_gpus=1)
    settle(e, [a])
    b = register(e, exclusive=True, priority=1)
    assert e.tasks[b["task"]["task_id"]]["phase"] == "waiting"
    settle(e, [a, b])
    assert e.tasks[a["task"]["task_id"]]["granted"] == empty()
    assert len(e.tasks[b["task"]["task_id"]]["granted"]["gpus"]) == 2


def test_partial_sharing_and_impossible_queue():
    e = engine()
    clients = [register(e, max_gpus=1, max_cpu_cores=1) for _ in range(11)]
    settle(e, clients)
    assert sum(t["granted"] != empty() for t in e.tasks.values()) == 10


def test_recovery_preserves_pause_requests_and_reservations():
    e = engine()
    a = register(e)
    report(e, a, "applied")
    task_call(e, a, "set_paused", paused=True, scope="controller")
    restored = engine(e.state)
    t = restored.tasks[a["task"]["task_id"]]
    assert t["phase"] == "release" and t["granted"] != empty()
    task_call(restored, a, "set_paused", paused=False, scope="self")
    assert t["paused"]["controller"]
    report(restored, a, "released", empty())
    assert t["phase"] == "waiting" and t["granted"] == empty()


def test_dead_root_not_enough_to_reclaim():
    e = engine()
    a = register(e)
    t = e.tasks[a["task"]["task_id"]]
    report(e, a, "applied")
    t["processes"].append({"pid": 123, "start_us": 1, "boot_us": 1})
    e.inspect = lambda p, uid: "alive" if p["pid"] == 123 else "dead"
    e.scan_processes()
    assert not t["terminal"] and t["granted"] != empty()
    e.inspect = lambda p, uid: "dead"
    e.scan_processes()
    assert t["terminal"] and t["granted"] == empty()


def test_retry_conflict_and_rollback():
    e = engine()
    a = register(e)
    body = {"task_id": a["task"]["task_id"], "token": a["token"], "paused": True, "scope": "self"}
    key = str(uuid.uuid4())
    first = call(e, "set_paused", body, request_id=key)
    assert call(e, "set_paused", body, request_id=key) == first
    with pytest.raises(ProtocolError):
        call(e, "set_paused", body | {"paused": False}, request_id=key)
    before = copy.deepcopy(e.state)
    with pytest.raises(ProtocolError):
        task_call(e, a, "set_request", request=request(max_gpus=True), expected_request_revision=0)
    assert before == e.state


def test_registration_response_loss_and_secret_redaction():
    e = engine()
    a = register(e)
    report(e, a, "applied")
    assert "a" * 64 not in str(call(e, "status", {}))
    restored = engine(e.state)
    task = restored.tasks[a["task"]["task_id"]]
    response = call(restored, "register", {"task_id": task["task_id"], "label": task["label"],
                    "root": task["root"], "request": task["request"], "reconfigure_mode": "full_release", "token": a["token"]})
    assert response["task"]["phase"] == "release"


def test_bad_override_type_is_protocol_error():
    e = engine()
    with pytest.raises(ProtocolError):
        call(e, "set_override", {"expected_policy_revision": 0, "allocations": {}, "bound_task_id": []})


def test_drain_and_unavailable_inventory_block_new_grants():
    e = engine()
    e.update_blocked_gpus(POOL["gpus"])
    a = register(e)
    assert e.tasks[a["task"]["task_id"]]["phase"] == "waiting"
    e.update_blocked_gpus([])
    settle(e, [a])
    e.drain()
    with pytest.raises(ProtocolError):
        register(e)
    settle(e, [a])
    assert e.tasks[a["task"]["task_id"]]["granted"] == empty()


def test_randomized_handoff_never_overcommits():
    rng = random.Random(43)
    e = engine()
    clients = [register(e, max_gpus=1, max_cpu_cores=4) for _ in range(8)]
    for _ in range(300):
        client = rng.choice(clients)
        task = e.tasks[client["task"]["task_id"]]
        op = rng.randrange(4)
        if op == 0:
            task_call(e, client, "set_paused", paused=bool(rng.randrange(2)), scope="self")
        elif op == 1:
            task_call(e, client, "set_request", request=request(max_cpu_cores=4, max_gpus=rng.randrange(1, 3), priority=rng.randrange(3), exclusive=rng.random() < 0.05),
                      expected_request_revision=task["request_revision"])
        elif task["phase"] == "initialize":
            report(e, client, "applied")
        elif task["phase"] == "release":
            report(e, client, "released", empty())
        assert fits([t["granted"] for t in e.tasks.values()], POOL)
    settle(e, clients)
    assert all(t["phase"] in ("waiting", "applied") for t in e.tasks.values())


def test_driver_monitor_external_memory():
    from villa_gpu_manager.processes import NvidiaMonitor
    monitor = NvidiaMonitor({"cpu_cores": 1, "gpus": {"GPU-a": 90 * 1024**2}}, 10 * 1024**2)
    xml = b'<nvidia_smi_log><gpu><uuid>GPU-a</uuid><fb_memory_usage><total>100 MiB</total><used>40 MiB</used></fb_memory_usage><processes><process_info><pid>42</pid><used_memory>35 MiB</used_memory></process_info></processes></gpu></nvidia_smi_log>'
    assert monitor.parse(xml, {42}) == set()
    assert monitor.parse(xml, set()) == {"GPU-a"}


def test_new_descendant_cannot_register_between_scans():
    e = engine()
    register(e)
    child = {"pid": 2, "start_us": 1, "boot_us": 1}
    e.discover = lambda p, uid: [child] if p["pid"] == 1 else []
    with pytest.raises(ProtocolError, match="overlapping"):
        register(e)


def test_shared_directory_requires_sticky_and_private_journal(tmp_path):
    directory = tmp_path / "shared"
    directory.mkdir(mode=0o770)
    with pytest.raises(ProtocolError, match="sticky"):
        with Server(directory / "manager.sock", POOL, shared_group=True):
            pass
    directory.chmod(0o1770)
    with Server(directory / "manager.sock", POOL, shared_group=True):
        pass
    (directory / "manager.sock.json").chmod(0o666)
    with pytest.raises(ProtocolError, match="journal ownership"):
        with Server(directory / "manager.sock", POOL, shared_group=True):
            pass


def test_real_socket_and_journal(tmp_path):
    directory = tmp_path / "private"
    with Server(directory / "manager.sock", POOL,
                engine_options={"protected_processes": lambda: []}) as server:
        stop = threading.Event()
        errors = []
        def serve():
            try:
                while not stop.is_set():
                    server.step(0.01)
            except BaseException as exc:
                errors.append(exc)
        thread = threading.Thread(target=serve)
        thread.start()
        try:
            with Client(directory / "manager.sock") as client:
                task = client.register("test", request())
                client.report("applied", task["granted"])
                assert client.poll()["phase"] == "applied"
                client.pause()
                client.report("released", empty())
                assert client.finish()["terminal"]
            assert (directory / "manager.sock.json").exists()
            with pytest.raises(BlockingIOError):
                with Server(directory / "manager.sock", POOL):
                    pass
            assert (directory / "manager.sock").exists()
        finally:
            stop.set()
            thread.join(timeout=5)
        assert not errors
    with Server(directory / "manager.sock", POOL) as restarted:
        assert all(t["terminal"] for t in restarted.engine.tasks.values())
