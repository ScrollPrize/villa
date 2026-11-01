"""Incremental GPU ownership for the persistent full-volume tile coordinator.

No readers/accumulators live here. Removing a worker drains its submitted tiles;
shared-memory results remain owned by the coordinator after the worker exits.
"""
from copy import deepcopy
import logging
import math
import os
import time
import signal
import threading
from functools import wraps

INFERENCE_ENV = ("CUDA_VISIBLE_DEVICES", "LASAGNA_GPU_MEMORY_GIB",
                 "LASAGNA_GPU_PRIORITY", "VILLA_GPU_MANAGER_SOCKET")


def runner_environment(metadata):
    environment = dict(os.environ)
    if metadata.get("job_kind", "inference") == "inference":
        # A detached tmux server need not share the launching shell's env.
        if "environment" in metadata:
            for key in INFERENCE_ENV:
                environment.pop(key, None)
                if key in metadata["environment"]:
                    environment[key] = metadata["environment"][key]
        environment["LASAGNA_COOPERATIVE"] = "1"
    return environment


def cooperative_signals(function):
    @wraps(function)
    def run(*args, **kwargs):
        install = os.environ.get("LASAGNA_COOPERATIVE") == "1" and threading.current_thread() is threading.main_thread()
        if install:
            def stop(signum, frame):
                raise KeyboardInterrupt("cooperative inference interrupted")
            previous = signal.signal(signal.SIGTERM, stop)
        try:
            return function(*args, **kwargs)
        finally:
            if install:
                signal.signal(signal.SIGTERM, previous)
    return run


def enabled(devices):
    return os.environ.get("LASAGNA_COOPERATIVE") == "1" and any(d.type == "cuda" for d in devices)


def gpu_worker(uuid, budget, args, gate, parent_pid):
    # Publish process identities to the manager before any CUDA allocation.
    while not gate.wait(0.1):
        if os.getppid() != parent_pid:
            return
    # The spawned child has not initialized CUDA. UUIDs avoid ordinal remapping
    # when the coordinator and another client have different visibility masks.
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    import torch
    from .tiled_predict3d import _multi_gpu_worker_main
    torch.cuda.set_device(0)
    total = torch.cuda.get_device_properties(0).total_memory
    # Leave room for the context/non-allocator libraries under a partial grant.
    torch.cuda.set_per_process_memory_fraction(max(0.01, min(1.0, (budget - (1 << 30)) / total)), 0)
    _multi_gpu_worker_main(args[0], "cuda:0", *args[2:])


class IncrementalWorkers:
    """Event-loop client. ``workers`` and queues have stable per-UUID indices.

    A successful release report means removed processes have been joined, not
    merely signalled. Retained workers are never reconstructed on resize.
    """
    def __init__(self, client, request, uuids, workers, queues, factory, *, clock=time.monotonic):
        self.client, self.request, self.uuids = client, request, tuple(uuids)
        self.workers, self.queues, self.factory = workers, queues, factory
        self.clock = clock
        self.next_poll = 0
        self.connected = False
        self.task = None
        self.budgets = {}
        self.stopping = set()
        self.active = set()
        self.last_status = None
        self.epoch = None
        self.recover_drain = False

    @classmethod
    def connect_local(cls, devices, workers, queues, factory, cpu_cores):
        from villa_gpu_manager.bootstrap import ensure_manager, visible_gpu_uuids
        from villa_gpu_manager.client import Client
        from villa_gpu_manager.protocol import CAPABILITIES
        if "incremental_release" not in CAPABILITIES:
            raise RuntimeError("upgrade the installed villa_gpu_manager package to a version with incremental_release before volume inference")
        visible = visible_gpu_uuids()
        uuids = [visible[d.index or 0] for d in devices]
        path, inventory = ensure_manager(os.environ.get("VILLA_GPU_MANAGER_SOCKET"))
        if not set(uuids) <= inventory["gpus"].keys():
            raise ValueError("requested inference GPUs are outside manager inventory")
        gib = os.environ.get("LASAGNA_GPU_MEMORY_GIB")
        if gib is None:
            # Conservative until the user supplies a measured one-tile budget.
            budget = max(inventory["gpus"][g] for g in uuids)
        else:
            value = float(gib)
            if not math.isfinite(value) or value <= 1:
                raise ValueError("LASAGNA_GPU_MEMORY_GIB must be finite and > 1 (includes CUDA reserve)")
            budget = math.ceil(value * (1 << 30))
        if not any(inventory["gpus"][g] >= budget for g in uuids):
            raise ValueError("inference VRAM budget exceeds every selected GPU")
        request = dict(max_cpu_cores=min(cpu_cores, inventory["cpu_cores"]), max_gpus=len(uuids),
                       min_vram_bytes_per_gpu=budget, max_vram_bytes_per_gpu=budget,
                       allowed_gpu_uuids=uuids, priority=int(os.environ.get("LASAGNA_GPU_PRIORITY", "0")),
                       exclusive=False)
        client = Client(path, timeout=0.5, reconfigure_mode="incremental_release")
        # Fail clearly on an old server; never silently run unmanaged.
        client.connect()
        client.close()
        return cls(client, request, uuids, workers, queues, factory)

    def _drain(self, keep, outstanding):
        for i, worker in enumerate(self.workers):
            if worker is None:
                continue
            if self.budgets[i] <= keep.get(self.uuids[i], 0) and i not in self.stopping:
                continue
            if outstanding[i]:
                continue
            if i not in self.stopping:
                gate = getattr(worker, "_cooperative_gate", None)
                if gate is not None and not gate.is_set():
                    worker.terminate()  # Still behind the pre-CUDA admission gate.
                else:
                    self.queues[i].put_nowait(None)
                self.stopping.add(i)
            if worker.is_alive():
                continue
            worker.join()
            self.workers[i] = None
            self.budgets.pop(i)
            self.stopping.remove(i)
        return all(w is None or (i not in self.stopping and self.budgets[i] <= keep.get(self.uuids[i], 0))
                   for i, w in enumerate(self.workers))

    def tick(self, outstanding):
        from villa_gpu_manager.protocol import ProtocolError
        now = self.clock()
        # Local draining progresses even between retries; no new work offline.
        if not self.connected:
            self.active.clear()
            self._drain({}, outstanding)
        if now < self.next_poll:
            return self.active
        self.next_poll = now + 1.0
        try:
            if not self.connected:
                self.client.connect()
                if self.epoch is not None and self.client.epoch != self.epoch:
                    self.recover_drain = True
                self.epoch = self.client.epoch
                self.task = self.client.register("Lasagna full-volume inference", self.request)
                self.connected = True
            else:
                self.task = self.client.poll()
            task = self.task
            status = (task["phase"], task["generation"], task["reason"], task["granted"])
            if status != self.last_status:
                print(f"[predict3d:manager] phase={task['phase']} generation={task['generation']} "
                      f"grant={task['granted']} reason={task['reason']}", flush=True)
                self.last_status = deepcopy(status)
            self.active.clear()
            if task["phase"] == "release" or self.recover_drain:
                keep = {} if self.recover_drain or task["reason"] == "server_restart" else task["target"]["gpus"]
                if self._drain(keep, outstanding):
                    held = {"cpu_cores": min(task["granted"]["cpu_cores"], task["target"]["cpu_cores"]) if self.budgets else 0,
                            "gpus": {self.uuids[i]: b for i, b in self.budgets.items()}}
                    if task["granted"] != {"cpu_cores": 0, "gpus": {}}:
                        self.task = self.client.report("released", held)
                    self.recover_drain = False
                    self.next_poll = 0
                return self.active
            if task["phase"] in ("initialize", "applied"):
                if self.stopping:
                    self.recover_drain = True
                    return self.active
                if task["phase"] == "applied" and set(self.budgets) != {i for i, g in enumerate(self.uuids) if g in task["granted"]["gpus"]}:
                    # A disconnect drain completed but its report was lost.
                    self.recover_drain = True
                    return self.active
                for i, uuid in enumerate(self.uuids):
                    budget = task["granted"]["gpus"].get(uuid)
                    if budget is not None and self.workers[i] is None:
                        self.workers[i] = self.factory(i, uuid, budget)
                        self.budgets[i] = budget
                if task["phase"] == "initialize":
                    from villa_gpu_manager.processes import process_identity
                    self.task = self.client.report("applied", task["granted"],
                        processes=[process_identity(w.pid) for w in self.workers if w is not None])
                if self.task["phase"] == "applied":
                    for worker in self.workers:
                        gate = getattr(worker, "_cooperative_gate", None)
                        if gate is not None:
                            gate.set()
                    self.active = set(self.budgets)
            return self.active
        except (OSError, ProtocolError) as exc:
            self.client.close()
            self.connected = False
            self.recover_drain = True
            self.active.clear()
            self.next_poll = now + 5
            logging.getLogger(__name__).warning("inference manager communication: %s: %s; draining, retry in 5s", type(exc).__name__, exc)
            return self.active

    def close(self):
        """Called only after all owned GPU processes have been joined."""
        if any(w is not None and w.is_alive() for w in self.workers):
            raise RuntimeError("cannot release inference reservation: GPU worker still alive")
        try:
            if self.client.task_id is not None:
                self.client.connect()
                self.client.register("Lasagna full-volume inference", self.request)
                task = self.client.pause(True)
                if task["granted"] != {"cpu_cores": 0, "gpus": {}}:
                    self.client.report("released", {"cpu_cores": 0, "gpus": {}})
                self.client.finish()
        except (OSError, ValueError) as exc:
            logging.getLogger(__name__).warning("inference workers joined; manager release pending process-death reconciliation: %s", exc)
        finally:
            self.client.close()
