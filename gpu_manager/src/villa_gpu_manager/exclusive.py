"""Exclusive non-preemptible work, with explicit gated-worker ownership."""

from copy import deepcopy
import time

from .client import Client
from .policy import empty


class ExclusiveTask:
    """One exact-budget task; only this task's workers may be aborted.

    Assign ``cohort`` immediately after constructing gated workers. ``authorize``
    reports their process identities and confirms the grant before they run.
    Closing always joins them before reporting release. No protocol extensions.
    """

    def __init__(self, path, grant, *, label, priority=0, poll_seconds=0.1):
        if type(priority) is not int:
            raise ValueError("priority must be an integer")
        amounts = set(grant["gpus"].values())
        if len(amounts) != 1 or grant["cpu_cores"] < 1:
            raise ValueError("exclusive task requires uniform per-GPU budgets and positive CPUs")
        size = next(iter(amounts))
        self.grant = deepcopy(grant)
        self.request = {"max_cpu_cores": grant["cpu_cores"], "max_gpus": len(grant["gpus"]),
            "min_vram_bytes_per_gpu": size, "max_vram_bytes_per_gpu": size,
            "allowed_gpu_uuids": sorted(grant["gpus"]), "priority": priority, "exclusive": True}
        self.client = Client(path)
        self.label, self.poll_seconds = label, poll_seconds
        self.cohort = None
        self.epoch = self.generation = None

    def __enter__(self):
        try:
            self.client.connect()
            inventory = self.client.status()["inventory"]
            if (self.grant["cpu_cores"] > inventory["cpu_cores"] or
                any(size > inventory["gpus"].get(gpu, 0) for gpu, size in self.grant["gpus"].items())):
                raise ValueError("exclusive probe budget exceeds managed inventory")
            task = self.client.register(self.label, self.request, pausable=False, adaptable=False)
            last = None
            while task["phase"] != "initialize":
                status = (task["phase"], task.get("reason"))
                if status != last:
                    print(f"calibration manager: task={self.client.task_id} phase={status[0]} reason={status[1]}", flush=True)
                    last = status
                if task.get("terminal"):
                    raise RuntimeError("calibration task ended while waiting")
                time.sleep(self.poll_seconds)
                task = self.client.poll()
            if task["granted"] != self.grant:
                raise RuntimeError("manager grant does not match exact calibration budget (check overrides)")
            self.epoch, self.generation = self.client.epoch, task["generation"]
            print(f"calibration manager: task={self.client.task_id} exclusive grant={self.grant}", flush=True)
            return self
        except BaseException:
            self.close("failed before GPU startup")
            raise

    def _check(self, task, phase):
        if (self.client.epoch != self.epoch or task["generation"] != self.generation
            or task["phase"] != phase or task["granted"] != self.grant):
            raise RuntimeError("calibration manager allocation changed or was lost")

    def authorize(self):
        if self.cohort is None:
            raise RuntimeError("no gated calibration cohort")
        self._check(self.client.poll(), "initialize")
        task = self.client.report("applied", self.grant, processes=self.cohort.processes)
        self._check(task, "applied")
        self.cohort.authorize()

    def check(self):
        self._check(self.client.poll(), "applied")

    def close(self, outcome="completed"):
        try:
            if self.cohort is not None and not self.cohort.closed:
                self.cohort.abort()
            if self.cohort is not None and not self.cohort.closed:
                raise RuntimeError("refusing calibration release before workers join")
            if self.client.task is not None:
                task = self.client.pause(True)
                if task["granted"] != empty():
                    self.client.report("released", empty(), detail="calibration workers joined")
                self.client.finish(outcome)
                print(f"calibration manager: task={self.client.task_id} released", flush=True)
        finally:
            self.client.close()

    def __exit__(self, exc_type, exc, tb):
        self.close("completed" if exc is None else f"failed: {type(exc).__name__}")
