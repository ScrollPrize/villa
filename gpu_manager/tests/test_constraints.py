import json

import pytest

from test_manager import engine, register, report
from villa_gpu_manager.constraints import Constraints
from villa_gpu_manager.policy import empty


def test_live_reload_drains_and_preserves_accounting(tmp_path):
    e = engine()
    a = register(e)
    report(e, a, "applied")
    task = e.tasks[a["task"]["task_id"]]
    old = dict(task["granted"]["gpus"])
    path = tmp_path / "constraints.json"
    path.write_text(json.dumps({"allowed_gpu_uuids": ["GPU-a"]}))
    config = Constraints(path)
    config.poll(e, force=True)
    assert task["phase"] == "release"
    assert task["granted"]["gpus"] == old
    assert set(task["target"]["gpus"]) == {"GPU-a"}
    report(e, a, "released", empty())
    assert set(task["granted"]["gpus"]) == {"GPU-a"}
    path.write_text('{"allowed_gpu_uuids": ["GPU-unknown"]}')
    with pytest.raises(ValueError):
        config.poll(e, force=True)
    assert e.allowed_gpus == {"GPU-a"}
    path.unlink()
    config.poll(e, force=True)
    assert e.allowed_gpus == {"GPU-a"}
    # Missing file at restart also preserves the journaled constraint.
    Constraints(path).poll(e, force=True)
    assert e.allowed_gpus == {"GPU-a"}


def test_empty_and_expand(tmp_path):
    e = engine()
    path = tmp_path / "constraints.json"
    path.write_text('{"allowed_gpu_uuids": []}')
    config = Constraints(path)
    config.poll(e, force=True)
    a = register(e)
    task = e.tasks[a["task"]["task_id"]]
    assert task["granted"] == empty()
    path.write_text('{"allowed_gpu_uuids": ["GPU-b"]}')
    config.poll(e, force=True)
    assert set(task["granted"]["gpus"]) == {"GPU-b"}


def test_nonpausable_grant_is_not_revoked(tmp_path):
    from test_scheduling import rigid
    e = engine()
    a = rigid(e)
    report(e, a, "applied")
    task = e.tasks[a["task"]["task_id"]]
    old = task["granted"]
    e.set_allowed_gpus([])
    assert task["granted"] == old
    assert task["target"] == old
    assert task["phase"] == "applied"


def test_cli_writes_atomic_policy(tmp_path, capsys):
    from villa_gpu_manager.cli import main
    assert main(["constraints", "--gpu-uuids", "GPU-a"]) == 0
    assert json.loads((tmp_path / "constraints.json").read_text()) == {"allowed_gpu_uuids": ["GPU-a"]}
    assert main(["constraints"]) == 0


def test_override_cannot_bypass_constraints():
    from test_incremental import override
    from villa_gpu_manager.protocol import ProtocolError
    e = engine()
    a = register(e)
    aid = a["task"]["task_id"]
    override(e, {aid: {"cpu_cores": 1, "gpus": {"GPU-b": 20}}})
    e.set_allowed_gpus(["GPU-a"])
    assert e.state["override"] is None
    with pytest.raises(ProtocolError, match="administratively disabled"):
        override(e, {aid: {"cpu_cores": 1, "gpus": {"GPU-b": 20}}})


def test_production_service_reloads_without_restart(tmp_path):
    import time
    from test_process_service import service
    from villa_gpu_manager.client import Client
    path = tmp_path / "manager.sock"
    config = tmp_path / "constraints.json"
    with service(path):
        with Client(path) as client:
            before = client.status()["policy_revision"]
            config.write_text('{"allowed_gpu_uuids": []}')
            deadline = time.monotonic() + 5
            while client.status()["policy_revision"] == before:
                assert time.monotonic() < deadline
                time.sleep(0.05)
            revision = client.status()["policy_revision"]
            config.write_text('invalid')
            time.sleep(1.2)
            assert client.status()["policy_revision"] == revision
