"""Incremental ownership must never free unacknowledged memory."""
import copy
import uuid

import pytest

from test_manager import call, engine, register, report, request, task_call
from villa_gpu_manager.policy import empty, fits
from villa_gpu_manager.protocol import ProtocolError


def incremental(e, **kwargs):
    key = str(uuid.uuid4())
    return call(e, "register", dict(task_id=key, token="a" * 64, label=key,
                root=dict(pid=len(e.tasks) + 1, start_us=1, boot_us=1),
                request=request(**kwargs), reconfigure_mode="incremental_release"))


def override(e, allocations):
    call(e, "set_override", dict(expected_policy_revision=e.state["policy_revision"],
                                allocations=allocations, bound_task_id=None))


def test_shrink_retains_worker_and_only_acknowledgement_frees_memory():
    e = engine()
    a = incremental(e, max_cpu_cores=2)
    report(e, a, "applied")
    b = register(e, max_cpu_cores=2, max_gpus=1, allowed_gpu_uuids=["GPU-b"])
    aid, bid = a["task"]["task_id"], b["task"]["task_id"]
    keep = {"cpu_cores": 1, "gpus": {"GPU-a": 100}}
    other = {"cpu_cores": 1, "gpus": {"GPU-b": 100}}
    override(e, {aid: keep, bid: other})
    assert e.tasks[bid]["granted"] == empty()
    assert e.tasks[aid]["granted"]["gpus"]["GPU-b"] == 100
    report(e, a, "released", keep)
    assert e.tasks[aid]["granted"] == keep
    assert e.tasks[aid]["phase"] == "initialize"
    assert e.tasks[bid]["granted"] == other
    report(e, a, "applied", keep)
    assert fits([t["granted"] for t in e.tasks.values()], e.inventory)


def test_expansion_retains_old_grant_and_rejects_early_memory_claim():
    e = engine()
    a = incremental(e, max_cpu_cores=2, max_gpus=1, allowed_gpu_uuids=["GPU-a"])
    report(e, a, "applied")
    old = copy.deepcopy(e.tasks[a["task"]["task_id"]]["granted"])
    req = dict(a["task"]["request"], max_gpus=2, allowed_gpu_uuids=[])
    task_call(e, a, "set_request", request=req, expected_request_revision=0)
    with pytest.raises(ProtocolError, match="held exceeds grant"):
        report(e, a, "released", {"cpu_cores": 2, "gpus": {"GPU-a": 100, "GPU-b": 100}})
    report(e, a, "released", old)
    assert len(e.tasks[a["task"]["task_id"]]["granted"]["gpus"]) == 2


def test_legacy_partial_report_does_not_free_capacity():
    e = engine()
    a = register(e)
    report(e, a, "applied")
    old = copy.deepcopy(e.tasks[a["task"]["task_id"]]["granted"])
    task_call(e, a, "set_paused", paused=True, scope="self")
    report(e, a, "released", {"cpu_cores": 1, "gpus": {"GPU-a": 20}})
    assert e.tasks[a["task"]["task_id"]]["granted"] == old


def test_restart_still_requires_full_release():
    e = engine()
    a = incremental(e)
    report(e, a, "applied")
    e = engine(e.state)
    old = copy.deepcopy(e.tasks[a["task"]["task_id"]]["granted"])
    report(e, a, "released", {"cpu_cores": 1, "gpus": {"GPU-a": 20}})
    assert e.tasks[a["task"]["task_id"]]["granted"] == old


def test_minimum_prefers_current_gpu_even_when_shared():
    e = engine()
    a = register(e, max_gpus=1, max_cpu_cores=1, min_vram_bytes_per_gpu=40, max_vram_bytes_per_gpu=40)
    report(e, a, "applied")
    b = register(e, max_gpus=1, max_cpu_cores=1, min_vram_bytes_per_gpu=40, max_vram_bytes_per_gpu=40,
                 allowed_gpu_uuids=["GPU-a"], priority=10)
    assert e.tasks[a["task"]["task_id"]]["target"]["gpus"] == {"GPU-a": 40}


def test_registration_mode_cannot_change():
    e = engine()
    a = incremental(e)
    task = e.tasks[a["task"]["task_id"]]
    with pytest.raises(ProtocolError, match="immutable"):
        call(e, "register", dict(task_id=task["task_id"], token=a["token"], label=task["label"],
             root=task["root"], request=task["request"], reconfigure_mode="full_release"))


def test_exclusive_waits_for_complete_release_and_failure_keeps_reservation():
    e = engine()
    a = incremental(e, max_cpu_cores=2)
    report(e, a, "applied")
    b = register(e, priority=10, exclusive=True)
    aid, bid = a["task"]["task_id"], b["task"]["task_id"]
    report(e, a, "released", {"cpu_cores": 1, "gpus": {"GPU-a": 100}})
    assert e.tasks[bid]["granted"] == empty()
    assert e.tasks[aid]["phase"] == "release"
    report(e, a, "failed", {"cpu_cores": 1, "gpus": {"GPU-a": 50}})
    assert e.tasks[aid]["granted"]["gpus"] == {"GPU-a": 100}
    assert e.tasks[bid]["granted"] == empty()
    report(e, a, "released", empty())
    assert e.tasks[bid]["phase"] == "initialize"


def test_target_changes_while_retaining_cannot_expand_previous_grant():
    e = engine()
    a = incremental(e, max_cpu_cores=2)
    report(e, a, "applied")
    aid = a["task"]["task_id"]
    override(e, {aid: {"cpu_cores": 1, "gpus": {"GPU-a": 100}}})
    generation = e.tasks[aid]["generation"]
    override(e, {aid: {"cpu_cores": 1, "gpus": {"GPU-b": 100}}})
    # A stale target does not become permission to retain new resources.
    report(e, a, "released", {"cpu_cores": 1, "gpus": {"GPU-a": 100}})
    assert e.tasks[aid]["phase"] == "release"
    assert e.tasks[aid]["generation"] > generation
    with pytest.raises(ProtocolError, match="stale generation"):
        task_call(e, a, "report", generation=generation, phase="released", held=empty(), processes=[], detail="")
    report(e, a, "released", empty())
    assert e.tasks[aid]["granted"]["gpus"] == {"GPU-b": 100}
