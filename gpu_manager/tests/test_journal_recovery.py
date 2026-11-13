"""Retire obsolete ownership without discarding live jobs or their history."""
from copy import deepcopy
import json
import os

import psutil
import pytest

from villa_gpu_manager.constraints import Constraints
from villa_gpu_manager.policy import empty
from villa_gpu_manager.process_server import ProcessServer
from villa_gpu_manager.server import Server, save_journal
from villa_gpu_manager.protocol import ProtocolError
from test_manager import POOL, register


NEW_POOL = {"cpu_cores": 32, "gpus": {"GPU-new": 200}}


def journal_with_task(path):
    with Server(path, POOL, engine_options={"inspect": lambda p, u: "alive",
                                           "discover": lambda p, u: []}) as server:
        registration = register(server.engine)
        task_id = registration["task"]["task_id"]
        state = deepcopy(server.engine.state)
        save_journal(server.journal, state)
        server.close(force=True)
    return state, task_id


def test_dead_jobs_adopt_new_hardware_and_current_config(tmp_path):
    path = tmp_path / "m.sock"
    old, task_id = journal_with_task(path)
    config = tmp_path / "constraints.json"
    config.write_text('{"allowed_gpu_uuids": ["GPU-new"]}')
    with Server(path, NEW_POOL, engine_options={"inspect": lambda p, u: "dead",
                                               "discover": lambda p, u: []}) as server:
        Constraints(config).poll(server.engine, force=True)
        state = server.engine.state
        assert state["inventory"] == NEW_POOL
        assert server.engine.allowed_gpus == {"GPU-new"}
        task = state["tasks"][task_id]
        assert task["terminal"] and task["held"] == task["granted"] == empty()
        assert task["token"] == old["tasks"][task_id]["token"]
        assert task["reason"] == "process_tree_dead"


@pytest.mark.parametrize("survivor", ["root", "child"])
def test_live_root_or_orphan_child_blocks_inventory_change(tmp_path, survivor):
    path = tmp_path / "m.sock"
    state, task_id = journal_with_task(path)
    root = state["tasks"][task_id]["root"]
    child = dict(root, pid=root["pid"] + 10)
    state["tasks"][task_id]["processes"].append(child)
    journal = path.with_suffix(".sock.json")
    save_journal(journal, state)
    before = journal.read_bytes()
    alive = root if survivor == "root" else child
    with pytest.raises(ProtocolError, match="inventory differs.*live processes"):
        with Server(path, NEW_POOL,
                    engine_options={"inspect": lambda p, u: "alive" if p == alive else "dead",
                                    "discover": lambda p, u: []}):
            pass
    assert journal.read_bytes() == before


@pytest.mark.parametrize("inventory", [POOL, NEW_POOL])
def test_uncertain_matching_process_fails_loudly_and_preserves_journal(tmp_path, inventory):
    path = tmp_path / "m.sock"
    journal_with_task(path)
    journal = path.with_suffix(".sock.json")
    before = journal.read_bytes()
    with pytest.raises(ProtocolError, match=r"cannot verify journal ownership: pid=1.*journal="):
        with Server(path, inventory, engine_options={"inspect": lambda p, u: "unknown",
                                                    "discover": lambda p, u: []}):
            pass
    assert journal.read_bytes() == before


@pytest.mark.parametrize("legacy", [False, True])
def test_prior_boot_skips_pid_inspection(tmp_path, legacy):
    path = tmp_path / "m.sock"
    state, task_id = journal_with_task(path)
    if legacy:
        state.pop("host_boot_id")
    else:
        state["host_boot_id"] = "00000000-0000-0000-0000-000000000000"
    journal = path.with_suffix(".sock.json")
    save_journal(journal, state)
    if legacy:
        before_boot = psutil.boot_time() - 1
        os.utime(journal, (before_boot, before_boot))
    def forbidden(*args):
        pytest.fail("prior-boot PIDs cannot still own resources")
    with Server(path, NEW_POOL, engine_options={"inspect": forbidden, "discover": forbidden}) as server:
        assert server.engine.tasks[task_id]["terminal"]
        assert server.engine.inventory == NEW_POOL


def test_same_inventory_live_task_keeps_one_recovery_generation(tmp_path):
    path = tmp_path / "m.sock"
    state, task_id = journal_with_task(path)
    with Server(path, POOL, engine_options={"inspect": lambda p, u: "alive",
                                           "discover": lambda p, u: []}) as server:
        task = server.engine.tasks[task_id]
        assert task["generation"] == state["tasks"][task_id]["generation"] + 1
        assert task["granted"] == state["tasks"][task_id]["granted"]
        assert not task["terminal"]
        server.close(force=True)


def test_corrupt_boot_marker_cannot_retire_jobs(tmp_path):
    path = tmp_path / "m.sock"
    state, _ = journal_with_task(path)
    state["host_boot_id"] = "corrupt"
    journal = path.with_suffix(".sock.json")
    save_journal(journal, state)
    before = journal.read_bytes()
    with pytest.raises(ProtocolError, match="UUID"):
        with Server(path, NEW_POOL):
            pass
    assert journal.read_bytes() == before


@pytest.mark.parametrize("already_granted", [False, True])
def test_monitored_recovery_cannot_grant_waiting_jobs_before_first_observation(tmp_path, already_granted):
    path = tmp_path / "m.sock"
    with Server(path, POOL, engine_options={"inspect": lambda p, u: "alive",
                                           "discover": lambda p, u: []}) as server:
        if not already_granted:
            server.engine.update_blocked_gpus(POOL["gpus"])
        registration = register(server.engine)
        task_id = registration["task"]["task_id"]
        grant = deepcopy(server.engine.tasks[task_id]["granted"])
        save_journal(server.journal, server.engine.state)
        server.close(force=True)

    class UnavailableMonitor:
        def poll(self, owned):
            return set(POOL["gpus"])
        def close(self):
            pass

    with Server(path, POOL, monitor=UnavailableMonitor(),
                engine_options={"inspect": lambda p, u: "alive", "discover": lambda p, u: []}) as server:
        task = server.engine.tasks[task_id]
        assert task["granted"] == grant
        if already_granted:
            assert task["phase"] == "release"  # Retain real ownership until acknowledgement.
            server.close(force=True)
        else:
            assert task["phase"] == "waiting"
            assert task["reason"] == "inventory_unavailable"
            server._before_io()
            assert task["granted"] == empty()
            # Only an actual usable-driver observation may unblock a new grant.
            server.engine.update_blocked_gpus([])
            assert task["phase"] == "initialize"
            assert task["granted"]["gpus"]
            server.close(force=True)


def test_production_restart_uses_updated_constraints(tmp_path):
    path = tmp_path / "m.sock"
    with Server(path, POOL):
        pass
    (tmp_path / "constraints.json").write_text('{"allowed_gpu_uuids": ["GPU-new"]}')
    with ProcessServer(path, NEW_POOL):
        saved = json.loads(path.with_suffix(".sock.json").read_text())
        assert saved["inventory"] == NEW_POOL
        assert saved["allowed_gpu_uuids"] == ["GPU-new"]


def test_boot_mismatch_does_not_inspect_an_unrelated_pid(monkeypatch):
    from villa_gpu_manager import processes
    monkeypatch.setattr(processes.psutil, "boot_time", lambda: 20)
    monkeypatch.setattr(processes.psutil, "Process", lambda pid: pytest.fail("unrelated PID inspected"))
    assert processes.process_state(dict(pid=1, start_us=1, boot_us=10_000_000), 1234) == "dead"


def test_dead_pid_and_pid_reuse_do_not_query_uid(monkeypatch):
    from villa_gpu_manager import processes
    monkeypatch.setattr(processes.psutil, "boot_time", lambda: 20)
    identity = dict(pid=123, start_us=1, boot_us=20_000_000)
    class Process:
        pid = 123
        def create_time(self):
            return 2 / 1_000_000
        def uids(self):
            pytest.fail("UID of a different process incarnation queried")
    monkeypatch.setattr(processes.psutil, "Process", lambda pid: Process())
    assert processes.process_state(identity, 1234) == "dead"


def test_permission_failure_only_after_matching_pid_and_start_time(monkeypatch):
    from villa_gpu_manager import processes
    monkeypatch.setattr(processes.psutil, "boot_time", lambda: 20)
    identity = dict(pid=123, start_us=1, boot_us=20_000_000)
    class Process:
        def create_time(self):
            return 1 / 1_000_000
        def uids(self):
            raise psutil.AccessDenied(123)
    monkeypatch.setattr(processes.psutil, "Process", lambda pid: Process())
    assert processes.process_state(identity, 1234) == "unknown"
    def absent(pid):
        raise psutil.NoSuchProcess(pid)
    monkeypatch.setattr(processes.psutil, "Process", absent)
    assert processes.process_state(identity, 1234) == "dead"
