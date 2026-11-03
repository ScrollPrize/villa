"""Actual OS-process backend, synthetic inventory; no GPU initialization."""

from contextlib import contextmanager
import multiprocessing as mp
import os
import signal
from pathlib import Path
import time

import pytest

from villa_gpu_manager.client import Client
from villa_gpu_manager.cohort import _Channel
from villa_gpu_manager.process_server import ProcessServer
from villa_gpu_manager.server import Server
from villa_gpu_manager.process_snapshot import ProcessSnapshot, collect_process_rows
from villa_gpu_manager.processes import process_identity
from villa_gpu_manager.policy import empty

POOL = {"cpu_cores": 2, "gpus": {"GPU-test": 100}}
REQUEST = dict(max_cpu_cores=2, max_gpus=1, min_vram_bytes_per_gpu=50,
               max_vram_bytes_per_gpu=100, allowed_gpu_uuids=[], priority=0, exclusive=False)


def slow_collector(sock):
    channel = _Channel(sock)
    try:
        while True:
            channel.receive()
            time.sleep(2)
            rows, started, finished = collect_process_rows()
            for offset in range(0, len(rows), 512):
                channel.send("rows", rows[offset:offset + 512])
                channel.flush()
            channel.send("complete", [started, finished])
            channel.flush()
    except (EOFError, BrokenPipeError, ConnectionResetError):
        pass
    finally:
        channel.close()


def service_main(path, ready, stop, slow):
    options = dict(scan_seconds=0.05)
    if slow:
        options["collector_target"] = slow_collector
    server = ProcessServer(path, POOL, **options)
    try:
        server.__enter__()
        ready.send(server.backend.pid)
        while not stop.is_set():
            server.step(0.005)
    finally:
        server.close(force=True)
        ready.close()


@contextmanager
def service(path, *, slow=False):
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    stop = context.Event()
    process = context.Process(target=service_main, args=(str(path), child, stop, slow))
    process.start()
    child.close()
    try:
        assert parent.poll(15), "process service did not start"
        assert parent.recv()
        yield process
    finally:
        stop.set()
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join(2)
        parent.close()
        assert process.exitcode == 0


def test_snapshot_identity_matches_reference():
    timings = []
    for _ in range(3):
        before = time.monotonic()
        rows, started, finished = collect_process_rows()
        timings.append(time.monotonic() - before)
    print(f"process_snapshot rows={len(rows)} collection_seconds={timings}")
    snapshot = ProcessSnapshot(rows, started, finished)
    identity = process_identity()
    assert snapshot.inspect(identity, os.getuid()) == "alive"
    assert snapshot.rows[os.getpid()][2:4] == [identity["start_us"], identity["boot_us"]]


def test_unknown_and_reused_snapshot_rows_do_not_prove_death():
    identity = process_identity()
    for rows in ([], [[identity["pid"], None, None, identity["boot_us"], None, "unknown"]],
                 [[identity["pid"], 1, 1, identity["boot_us"], os.getuid(), "alive"]]):
        snapshot = ProcessSnapshot(rows, time.time(), time.time())
        assert snapshot.inspect(identity, os.getuid()) == "alive"


def test_slow_collection_does_not_block_status_or_poll(tmp_path):
    path = tmp_path / "private/manager.sock"
    with service(path, slow=True):
        with Client(path, timeout=5) as client:
            task = client.register("test", REQUEST)
            client.report("applied", task["granted"])
            # Collector sleeps for2s every round. Status/poll must still return
            # within a tight timeout, not just accept a socket connection.
            client.socket.settimeout(0.75)
            timings = []
            for _ in range(30):
                start = time.monotonic()
                assert client.poll()["phase"] == "applied"
                assert client.status()["tasks"]
                timings.append(time.monotonic() - start)
                time.sleep(0.05)
            assert max(timings) < 0.75
            print(f"slow_collector status_plus_poll_seconds min={min(timings):.4f} "
                  f"median={sorted(timings)[len(timings)//2]:.4f} max={max(timings):.4f}")
            client.pause(True)
            client.report("released", empty())
            assert client.finish()["terminal"]


def test_process_service_journal_recovery_and_no_secret_status(tmp_path):
    path = tmp_path / "private/manager.sock"
    with service(path):
        with Client(path) as client:
            task = client.register("test", REQUEST)
            client.report("applied", task["granted"])
            key, token = client.task_id, client.token
            assert token not in str(client.status())
    assert Path(str(path) + ".json").exists()
    with service(path):
        with Client(path) as client:
            task = client.register("test", REQUEST, task_id=key, token=token)
            assert task["phase"] == "release" and task["granted"] == POOL
            client.pause(True)
            client.report("released", empty())
            assert client.finish()["terminal"]


@pytest.mark.skipif(not hasattr(signal, "SIGSTOP"), reason="POSIX process suspension")
def test_state_owner_retains_singleton_after_public_parent_death(tmp_path):
    path = tmp_path / "private/manager.sock"
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    stop = context.Event()
    process = context.Process(target=service_main, args=(str(path), child, stop, False))
    backend = None
    process.start()
    child.close()
    try:
        assert parent.poll(15)
        backend = parent.recv()
        os.kill(backend, signal.SIGSTOP)
        process.kill()
        process.join(5)
        # The journal writer still owns the same locked file description.
        contender = Server(path, POOL)
        with pytest.raises(BlockingIOError):
            contender.__enter__()
    finally:
        if backend is not None:
            os.kill(backend, signal.SIGCONT)
        if process.is_alive():
            process.kill()
            process.join(5)
        parent.close()
