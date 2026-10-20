"""Capability/priority scheduling with synthetic resources; never initializes CUDA."""
import copy
import uuid

import pytest

from villa_gpu_manager.policy import empty
from villa_gpu_manager.protocol import ProtocolError
from test_manager import engine, call, request, register, report, settle, task_call


def rigid(e, *, pausable=False, **kw):
    key = str(uuid.uuid4())
    return call(e, "register", dict(task_id=key, token="a" * 64, label=key,
        root=dict(pid=len(e.tasks) + 1, start_us=1, boot_us=1),
        request=request(**kw), reconfigure_mode="full_release",
        pausable=pausable, adaptable=False))


def state(e, r):
    return e.tasks[r["task"]["task_id"]]


def finish(e, r):
    task_call(e, r, "set_paused", paused=True, scope="self")
    report(e, r, "released", empty())
    task_call(e, r, "finish", outcome="completed")


def test_exclusive_waits_for_rigid_then_drains_others_and_resumes():
    e = engine()
    a = rigid(e, max_gpus=1, max_cpu_cores=4)
    report(e, a, "applied")
    b = register(e, max_gpus=1, max_cpu_cores=4)
    settle(e, [a, b])
    before = copy.deepcopy(state(e, b)["granted"])
    x = rigid(e, exclusive=True, priority=10)
    assert state(e, x)["phase"] == "waiting"
    assert state(e, a)["task_id"] in state(e, x)["reason"]
    assert state(e, b)["granted"] == before
    assert state(e, b)["phase"] == "applied"
    finish(e, a)
    assert state(e, b)["phase"] == "release"
    assert state(e, x)["granted"] == empty()
    report(e, b, "released", empty())
    assert state(e, x)["phase"] == "initialize"
    report(e, x, "applied")
    finish(e, x)
    assert state(e, b)["phase"] == "initialize"


@pytest.mark.parametrize("exclusive_priority", [-1, 0])
def test_exclusive_has_no_priority_bonus(exclusive_priority):
    e = engine()
    a = register(e, priority=0)
    report(e, a, "applied")
    x = register(e, exclusive=True, priority=exclusive_priority)
    assert state(e, a)["phase"] == "applied"
    assert state(e, x)["granted"] == empty()
    assert state(e, x)["reason"].startswith("blocked_by_higher_priority_task:")


def test_higher_priority_regular_waits_for_outgoing_exclusive_release():
    e = engine()
    x = register(e, exclusive=True, max_gpus=1, max_cpu_cores=4)
    report(e, x, "applied")
    a = register(e, priority=10, max_gpus=1, max_cpu_cores=4)
    assert state(e, x)["phase"] == "release"
    assert state(e, a)["granted"] == empty()
    assert state(e, a)["reason"] == "waiting_for_release"
    report(e, x, "released", empty())
    assert state(e, a)["phase"] == "initialize"


def test_nonadaptable_restores_identical_allocation_after_pause():
    e = engine()
    a = rigid(e, pausable=True, max_gpus=1, max_cpu_cores=4)
    report(e, a, "applied")
    original = copy.deepcopy(state(e, a)["granted"])
    x = register(e, exclusive=True, priority=10)
    report(e, a, "released", empty())
    report(e, x, "applied")
    finish(e, x)
    assert state(e, a)["granted"] == original


def test_controller_cannot_pause_or_override_nonpausable():
    e = engine()
    a = rigid(e)
    report(e, a, "applied")
    for op, body in [
        ("set_paused", dict(task_id=state(e, a)["task_id"], scope="controller", paused=True)),
        ("set_override", dict(expected_policy_revision=0, allocations={}, bound_task_id=None)),
    ]:
        before = copy.deepcopy(e.state)
        with pytest.raises(ProtocolError):
            call(e, op, body)
        assert e.state == before


def test_rigid_restart_and_drain_keep_ownership():
    e = engine()
    a = rigid(e, exclusive=True)
    report(e, a, "applied")
    old = copy.deepcopy(state(e, a))
    e = engine(e.state)
    e.drain()
    assert state(e, a)["granted"] == old["granted"]
    assert state(e, a)["phase"] == "applied"
    finish(e, a)
    assert state(e, a)["terminal"]


def test_request_cannot_invalidate_fixed_allocation():
    e = engine()
    a = rigid(e)
    with pytest.raises(ProtocolError):
        task_call(e, a, "set_request", request=request(max_gpus=1), expected_request_revision=0)


def test_waiting_exclusive_prevents_new_rigid_blockers():
    e = engine()
    a = rigid(e, max_gpus=1, max_cpu_cores=4)
    report(e, a, "applied")
    x = rigid(e, exclusive=True, priority=10)
    b = rigid(e, max_gpus=1, max_cpu_cores=4)
    assert state(e, b)["granted"] == empty()
    assert state(e, b)["reason"] == "waiting_for_higher_priority_exclusive"
    assert state(e, x)["granted"] == empty()


@pytest.mark.parametrize("blocked_by", ["ordinary", "capacity"])
def test_every_pending_exclusive_blocks_new_rigid_admission(blocked_by):
    e = engine()
    if blocked_by == "ordinary":
        a = register(e, priority=20, max_gpus=1, max_cpu_cores=4)
        report(e, a, "applied")
        x = register(e, exclusive=True, priority=10)
    else:
        x = register(e, exclusive=True, priority=10,
                     min_vram_bytes_per_gpu=200, max_vram_bytes_per_gpu=200)
    b = rigid(e, priority=0, max_gpus=1, max_cpu_cores=4)
    assert state(e, b)["granted"] == empty()
    assert state(e, b)["reason"] == "waiting_for_higher_priority_exclusive"
    # Pausable work may still use spare capacity while the exclusive waits.
    c = register(e, priority=-1, max_gpus=1, max_cpu_cores=4)
    settle(e, [a, c] if blocked_by == "ordinary" else [c])
    assert state(e, c)["granted"] != empty()
    if blocked_by == "ordinary":
        finish(e, a)
        assert state(e, c)["phase"] == "release"
        report(e, c, "released", empty())
        assert state(e, x)["phase"] == "initialize"


@pytest.mark.parametrize("pausable,adaptable", [(False, True), (1, False), (False, 0)])
def test_invalid_capability_flags(pausable, adaptable):
    e = engine()
    with pytest.raises(ProtocolError):
        call(e, "register", dict(task_id=str(uuid.uuid4()), token="a" * 64, label="invalid",
             root=dict(pid=1, start_us=1, boot_us=1), request=request(),
             reconfigure_mode="full_release", pausable=pausable, adaptable=adaptable))
    assert not e.tasks


def test_rigid_exclusive_blocks_even_higher_priority_regular():
    e = engine()
    a = rigid(e, exclusive=True, max_gpus=1, max_cpu_cores=4)
    report(e, a, "applied")
    b = register(e, priority=100)
    assert state(e, a)["phase"] == "applied"
    assert state(e, b)["granted"] == empty()
    assert state(e, b)["reason"].startswith("blocked_by_nonpausable_task:")


def test_fixed_allocation_cannot_be_resized_by_override():
    e = engine()
    a = rigid(e, pausable=True)
    report(e, a, "applied")
    with pytest.raises(ProtocolError, match="non-adaptable"):
        call(e, "set_override", dict(expected_policy_revision=0, bound_task_id=None,
            allocations={state(e, a)["task_id"]: dict(cpu_cores=1, gpus={"GPU-a": 20})}))


def test_dead_nonpausable_blocker_releases_waiter():
    e = engine()
    a = rigid(e)
    report(e, a, "applied")
    b = rigid(e, exclusive=True, priority=10)
    pid = state(e, a)["root"]["pid"]
    e.inspect = lambda p, uid: "dead" if p["pid"] == pid else "alive"
    e.scan_processes()
    assert state(e, a)["terminal"]
    assert state(e, b)["phase"] == "initialize"


def test_initializing_nonpausable_grant_cannot_be_revoked():
    e = engine()
    a = rigid(e)
    b = rigid(e, exclusive=True, priority=10)
    assert state(e, a)["phase"] == "initialize"
    assert state(e, b)["granted"] == empty()
    report(e, a, "applied")
    assert state(e, a)["phase"] == "applied"


def test_reference_client_capabilities_survive_reconnect(tmp_path):
    from villa_gpu_manager.client import Client
    from test_conformance import running_server
    path = tmp_path / "manager/socket"
    with running_server(path):
        with Client(path) as client:
            task = client.register("eval", request(exclusive=True), pausable=False, adaptable=False)
            assert not task["pausable"] and not task["adaptable"]
            client.report("applied", task["granted"])
            client.connect()
            task = client.register("eval", request(exclusive=True))
            assert task["phase"] == "applied"
            client.pause()
            client.report("released", empty())
            client.finish()


def test_override_cannot_share_running_nonpausable_exclusive_job():
    e = engine()
    a = rigid(e, exclusive=True, max_gpus=1, max_cpu_cores=4)
    report(e, a, "applied")
    b = register(e, max_gpus=1, max_cpu_cores=4)
    with pytest.raises(ProtocolError, match="non-pausable exclusive"):
        call(e, "set_override", dict(expected_policy_revision=0, bound_task_id=None,
            allocations={state(e, a)["task_id"]: state(e, a)["granted"],
                         state(e, b)["task_id"]: dict(cpu_cores=4, gpus={"GPU-b": 100})}))


def test_rigid_restart_preserves_initialization_and_waiting_reason():
    e = engine()
    a = rigid(e)
    x = rigid(e, exclusive=True, priority=1)
    generation = state(e, a)["generation"]
    e = engine(e.state)
    assert state(e, a)["phase"] == "initialize"
    assert state(e, a)["generation"] == generation
    task_call(e, x, "poll", last_generation=0)
    assert state(e, x)["reason"].startswith("blocked_by_nonpausable_task:")
    report(e, a, "applied")
    finish(e, a)
    assert state(e, x)["phase"] == "initialize"
