import os

import pytest

from villa_gpu_manager.server import Server
from villa_gpu_manager.protocol import ProtocolError
from test_manager import POOL, register


def test_idle_grace_connections_and_live_disconnected_task(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("villa_gpu_manager.server.time.monotonic", lambda: clock[0])
    with Server(tmp_path / "m.sock", POOL,
                engine_options={"inspect": lambda p, u: "alive", "discover": lambda p, u: []}) as s:
        assert not s.idle_expired()
        clock[0] += 31
        assert s.idle_expired()
        s.connections["fake"] = {}
        assert not s.idle_expired()
        s.connections.clear()
        assert not s.idle_expired()
        r = register(s.engine)
        clock[0] += 100
        assert not s.idle_expired()  # EOF does not mean the GPU is free.
        s.engine.inspect = lambda p, u: "dead"
        s.engine.scan_processes()
        assert s.engine.tasks[r["task"]["task_id"]]["terminal"]
        assert not s.idle_expired()
        clock[0] += 31
        assert s.idle_expired()
    assert not (tmp_path / "m.sock").exists()
    assert (tmp_path / "m.sock.json").exists()


def test_retired_journal_accepts_new_cpu_inventory(tmp_path):
    path = tmp_path / "m.sock"
    with Server(path, POOL):
        pass
    with Server(path, POOL | {"cpu_cores": 224}) as s:
        assert s.engine.inventory["cpu_cores"] == 224


def test_live_journal_rejects_cpu_inventory_change(tmp_path):
    path = tmp_path / "m.sock"
    s = Server(path, POOL, engine_options={"inspect": lambda p, u: "alive"})
    with s:
        register(s.engine)
        s.drain()
        s.close(force=True)
    with pytest.raises(ProtocolError, match="inventory differs"):
        with Server(path, POOL | {"cpu_cores": 224}):
            pass
