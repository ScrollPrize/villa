"""A daemon bootstrapped below a client must not keep its parent task alive."""
import copy
import os

import pytest

from villa_gpu_manager.engine import Engine
from villa_gpu_manager.policy import empty
from villa_gpu_manager.protocol import ProtocolError
from test_manager import POOL, register, task_call, call, request


SERVICE = {"pid": 100, "start_us": 2, "boot_us": 1}
HELPER = {"pid": 101, "start_us": 2, "boot_us": 1}
WORKER = {"pid": 102, "start_us": 2, "boot_us": 1}


def make(state=None):
    return Engine(POOL, controller_uids=[os.getuid()], state=state,
                  inspect=lambda p, uid: "alive",
                  discover=lambda p, uid: [SERVICE, HELPER, WORKER],
                  protected_processes=lambda: [SERVICE, HELPER])


def test_service_filtered_but_real_worker_still_retains_reservation():
    e = make()
    r = register(e)
    task = e.tasks[r["task"]["task_id"]]
    assert task["processes"] == [task["root"], WORKER]
    e.inspect = lambda p, uid: "alive" if p in [SERVICE, HELPER, WORKER] else "dead"
    e.scan_processes()
    assert not task["terminal"]
    e.inspect = lambda p, uid: "alive" if p in [SERVICE, HELPER] else "dead"
    e.scan_processes()
    assert task["terminal"] and task["granted"] == empty()


def test_restored_legacy_membership_removes_service_without_journal_reset():
    e = make()
    r = register(e)
    task = e.tasks[r["task"]["task_id"]]
    task["processes"] += [SERVICE, HELPER]
    restored = make(copy.deepcopy(e.state))
    t = restored.tasks[task["task_id"]]
    assert t["processes"] == [task["root"], WORKER]
    assert t["token"] == task["token"]
    reused_pid = dict(SERVICE, start_us=999)
    t["processes"].append(reused_pid)
    restored.scan_processes()
    assert reused_pid in t["processes"]


def test_server_root_and_explicit_worker_report_rejected():
    e = make()
    r = register(e)
    with pytest.raises(ProtocolError, match="service cannot be reported"):
        task_call(e, r, "report", generation=r["task"]["generation"], phase="applied",
                  held=r["task"]["granted"], processes=[SERVICE], detail="")
    import uuid
    with pytest.raises(ProtocolError, match="service cannot register"):
        call(e, "register", {"task_id": str(uuid.uuid4()), "label": "bad", "root": SERVICE,
            "request": request(), "reconfigure_mode": "full_release", "token": "b"*64})
