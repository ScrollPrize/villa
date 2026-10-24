import os
import queue
import threading
import uuid
from unittest import mock

import numpy as np
import torch

from lasagna.cooperative_inference import IncrementalWorkers
from villa_gpu_manager.client import Client
from villa_gpu_manager.engine import Engine
from villa_gpu_manager.policy import empty
import pytest


class EngineClient(Client):
    def __init__(self, inventory=None):
        super().__init__("unused", reconfigure_mode="incremental_release",
                         registration_root=dict(pid=999, start_us=1, boot_us=1))
        self.engine = Engine(inventory or dict(cpu_cores=8, gpus={"GPU-a": 100, "GPU-b": 100}),
                             controller_uids=[os.getuid()], inspect=lambda p, uid: "alive",
                             discover=lambda p, uid: [])
        self.offline = False

    def connect(self):
        self.epoch = self.engine.epoch

    def close(self):
        pass

    def call(self, op, body, **kwargs):
        if self.offline:
            raise ConnectionError("test outage")
        return self.engine.dispatch(dict(version=1, id=str(uuid.uuid4()), server_epoch=self.epoch,
                                         op=op, body=body), os.getuid())

    def split(self, allocation):
        self.call("set_override", dict(expected_policy_revision=self.engine.state["policy_revision"],
                  allocations={self.task_id: allocation}, bound_task_id=None))


REQUEST = dict(max_cpu_cores=2, max_gpus=2, min_vram_bytes_per_gpu=100,
               max_vram_bytes_per_gpu=100, allowed_gpu_uuids=["GPU-a", "GPU-b"], priority=0, exclusive=False)
ONE = dict(cpu_cores=1, gpus={"GPU-a": 100})
TWO = dict(cpu_cores=2, gpus={"GPU-a": 100, "GPU-b": 100})


class Process:
    pid = os.getpid()
    def __init__(self):
        self.alive = True
        self.joined = False
    def is_alive(self):
        return self.alive
    def join(self):
        assert not self.alive
        self.joined = True
    def terminate(self):
        self.alive = False


def setup():
    client = EngineClient()
    clock = [0]
    workers = [None, None]
    queues = [queue.Queue(), queue.Queue()]
    controller = IncrementalWorkers(client, REQUEST, ["GPU-a", "GPU-b"], workers, queues,
                                    lambda i, u, b: Process(), clock=lambda: clock[0])
    def tick(outstanding=None):
        clock[0] += 6
        return controller.tick(outstanding or [{}, {}])
    return client, controller, tick


def test_retained_worker_identity_and_release_after_result_and_join():
    client, pool, tick = setup()
    assert tick() == {0, 1}
    kept, removed = pool.workers
    client.split(ONE)
    assert tick([{}, {7: "in flight"}]) == set()
    assert pool.queues[1].empty()  # Cannot stop before result consumption.
    assert client.task["granted"] == TWO
    tick()
    assert pool.queues[1].get_nowait() is None
    assert client.task["granted"] == TWO  # Sending stop is not release.
    removed.alive = False
    tick()
    assert removed.joined
    assert pool.workers[0] is kept
    assert client.task["granted"] == ONE
    assert tick() == {0}
    client.split(TWO)
    tick()
    assert tick() == {0, 1}
    assert pool.workers[0] is kept
    assert pool.workers[1] is not removed


def test_pause_zeroes_grant_only_after_join_and_resumes():
    client, pool, tick = setup()
    tick()
    old = list(pool.workers)
    client.pause(True)
    tick()
    for p in old:
        p.alive = False
    tick()
    assert client.task["granted"] == empty()
    assert all(p.joined for p in old)
    client.pause(False)
    assert tick() == {0, 1}
    assert all(p not in old for p in pool.workers)


def test_disconnect_drains_without_freeing_ledger_then_reconnects():
    client, pool, tick = setup()
    tick()
    old = list(pool.workers)
    client.offline = True
    assert tick() == set()
    tick()
    for p in old:
        p.alive = False
    tick()
    assert client.engine.tasks[client.task_id]["granted"] == TWO
    client.offline = False
    tick()
    assert tick() == {0, 1}
    assert all(p not in old for p in pool.workers)


def test_close_refuses_live_workers():
    import pytest
    client, pool, tick = setup()
    tick()
    with pytest.raises(RuntimeError, match="still alive"):
        pool.close()
    for p in pool.workers:
        p.alive = False
        p.join()
    pool.close()
    assert client.engine.tasks[client.task_id]["terminal"]


def test_reconnect_releases_retained_waiting_grant_before_blocked_expansion():
    client, pool, tick = setup()
    tick()
    other = EngineClient()
    other.engine = client.engine
    other.connect()
    other.register("blocker", dict(REQUEST, max_gpus=1, allowed_gpu_uuids=["GPU-b"]),
                   root=dict(pid=1000, start_us=1, boot_us=1))
    removed = pool.workers[1]
    tick()
    removed.alive = False
    tick()
    tick()
    other.report("applied", other.poll()["granted"])
    other.pause(True)  # B remains reserved until its release acknowledgement.
    tick()
    assert client.task["phase"] == "waiting"
    assert client.task["granted"] == ONE
    retained = pool.workers[0]
    client.offline = True
    tick()
    tick()
    assert client.engine.tasks[client.task_id]["granted"] == ONE
    retained.alive = False
    tick()
    assert retained.joined
    client.offline = False
    tick()
    assert pool.connected
    assert client.task["granted"] == empty()
    other.report("released", empty())
    assert tick() == {0, 1}


def test_cuda_admission_gate_waits_for_manager_process_report():
    client, pool, tick = setup()
    def make(i, u, b):
        worker = Process()
        worker._cooperative_gate = threading.Event()
        return worker
    pool.factory = make
    original = client.report
    def report(*args, **kwargs):
        assert all(not w._cooperative_gate.is_set() for w in pool.workers if w is not None)
        return original(*args, **kwargs)
    with mock.patch.object(client, "report", side_effect=report):
        assert tick() == {0, 1}
    assert all(w._cooperative_gate.is_set() for w in pool.workers)


def test_lost_initial_report_does_not_authorize_gpu():
    client, pool, tick = setup()
    def make(i, u, b):
        worker = Process()
        worker._cooperative_gate = threading.Event()
        return worker
    pool.factory = make
    with mock.patch.object(client, "report", side_effect=ConnectionError("lost")):
        assert tick() == set()
    old = list(pool.workers)
    assert all(not w._cooperative_gate.is_set() for w in old)
    tick()
    assert all(w.joined for w in old)


def test_tmux_runner_uses_saved_budget_and_visibility(monkeypatch):
    from lasagna.cooperative_inference import runner_environment
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,6")
    monkeypatch.setenv("LASAGNA_GPU_MEMORY_GIB", "80")
    metadata = dict(job_kind="inference", environment={"LASAGNA_GPU_MEMORY_GIB": "46"})
    result = runner_environment(metadata)
    assert "CUDA_VISIBLE_DEVICES" not in result
    assert result["LASAGNA_GPU_MEMORY_GIB"] == "46"
    assert result["LASAGNA_COOPERATIVE"] == "1"


def test_old_installed_manager_rejected_before_probing_or_starting(monkeypatch):
    import villa_gpu_manager.protocol as protocol
    monkeypatch.setattr(protocol, "CAPABILITIES", ["full_release"])
    with mock.patch("villa_gpu_manager.bootstrap.ensure_manager") as start:
        with pytest.raises(RuntimeError, match="upgrade the installed"):
            IncrementalWorkers.connect_local((torch.device("cuda:0"),), [None], [queue.Queue()], None, 1)
    start.assert_not_called()


@pytest.mark.parametrize("device_count", [1, 2])
@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_real_tile_pipeline_survives_pause_resize_without_reloading_input(tmp_path, device_count, backend):
    if backend == "cuda" and os.environ.get("FIBER_COOPERATIVE_CUDA_TEST") != "1":
        pytest.skip("requires approved manager-reserved CUDA test launcher")
    from lasagna.tiled_predict3d import OutputProductSpec, run_tiled_inference_3d, _read_tile_zarr
    # CPU workers exercise real multiprocessing queues/results and accumulation.
    from test_preprocess_cos_omezarr import SpawnIdentityAdapter
    product = OutputProductSpec(name="identity", level=0, scaledown=1, inference_scaledown=1,
                                channels=("value",), chunk_size=4)
    class Output:
        def __init__(self):
            self.chunks = {}
        def product_chunk_complete(self, product, *, chunk_origin_zyx):
            return False
        def write_product_chunk(self, product, *, chunk_origin_zyx, data):
            assert chunk_origin_zyx not in self.chunks
            self.chunks[chunk_origin_zyx] = np.array(data["value"], copy=True)
    uuids = ["GPU-a", "GPU-b"]
    budget = 100
    if backend == "cuda":
        uuids = os.environ["FIBER_COOPERATIVE_TEST_GPUS"].split(",")
        assert len(uuids) >= device_count
        budget = 4 << 30
    uuids = uuids[:device_count]
    request = dict(REQUEST, max_gpus=device_count, allowed_gpu_uuids=uuids,
                   min_vram_bytes_per_gpu=budget, max_vram_bytes_per_gpu=budget)
    one = dict(cpu_cores=1, gpus={uuids[0]: budget})
    two = dict(cpu_cores=2, gpus={u: budget for u in uuids})
    client = EngineClient(dict(cpu_cores=8, gpus={u: budget for u in uuids}))
    stages = []
    def factory(devices, workers, queues, start, cores):
        if device_count == 1:
            return IncrementalWorkers(client, request, uuids, workers, queues,
                                      lambda i, u, b: start(i, u if backend == "cuda" else None, b))
        class Cycling(IncrementalWorkers):
            def __init__(self):
                self.t = 0
                super().__init__(client, request, uuids, workers, queues,
                                 lambda i, u, b: start(i, u if backend == "cuda" else None, b), clock=lambda: self.t)
                self.stage = 0
                self.kept = None
            def tick(self, outstanding):
                self.t += 1
                active = super().tick(outstanding)
                if self.stage == 0 and any(outstanding):
                    self.kept = workers[0]
                    client.split(one)
                    self.stage = 1
                elif self.stage == 1 and active == {0}:
                    assert workers[0] is self.kept
                    client.pause(True)
                    self.stage = 2
                elif self.stage == 2 and client.task["granted"] == empty():
                    client.pause(False)
                    client.split(two)
                    self.stage = 3
                elif self.stage == 3 and active == {0, 1}:
                    assert workers[0] is not self.kept
                    self.stage = 4
                stages.append(self.stage)
                # Poll immediately after our controller mutation, so no tile
                # admission uses the prior phase in this deterministic test.
                if self.stage in (1, 2, 3):
                    self.active.clear()
                    return set()
                return active
        return Cycling()
    volume = np.full((128, 4, 4), 127, dtype=np.uint8)
    output = Output()
    with mock.patch("lasagna.tiled_predict3d._input_has_chunks", return_value=True), \
         mock.patch("lasagna.tiled_predict3d._read_tile_zarr", wraps=_read_tile_zarr) as reads:
        run_tiled_inference_3d(None, volume, crop_slices=(0, 128, 0, 4, 0, 4),
            device=torch.device(backend), devices=tuple(torch.device(f"cuda:{i}" if backend == "cuda" else "cpu") for i in range(device_count)),
            model_adapter=SpawnIdentityAdapter(product), output_adapter=output, products=(product,),
            output_regions_zyx={"identity": (0, 0, 0, 128, 4, 4)}, full_output_shapes_zyx={"identity": volume.shape},
            tile_size=4, border=0, overlap=0, tmp_dir=str(tmp_path), slots_per_gpu=1,
            cooperative_factory=factory)
    if device_count == 2:
        assert 4 in stages
    assert reads.call_count == 32
    assert len(output.chunks) == 32
    for chunk in output.chunks.values():
        np.testing.assert_array_equal(chunk, np.full((4, 4, 4), 127, dtype=np.uint8))
    assert client.engine.tasks[client.task_id]["terminal"]
