"""Opt-in measurements of the normal training loop (no alternate training math)."""
from contextlib import contextmanager, nullcontext
import json
import time

import torch


def loader_profile_enabled(training):
    enabled = training.get("profile_loader", False)
    if not isinstance(enabled, bool):
        raise ValueError("training.profile_loader must be a JSON boolean")
    if enabled and training.get("profile_enabled") is not True:
        raise ValueError("training.profile_loader requires profile_enabled=true")
    return enabled


def compile_training_model(model, mode):
    if mode == "off":
        return model
    if mode not in ("default", "reduce-overhead"):
        raise ValueError("training.compile_mode must be off/default/reduce-overhead")
    return torch.compile(model, mode=mode)


class TrainingProfile:
    def __init__(self, training, run_dir, rank, device):
        self.enabled = bool(training.get("profile_enabled", False))
        self.device = device
        self.rank = rank
        self.run_dir = run_dir
        self.trace_start = int(training.get("profile_trace_start", 21))
        self.trace_steps = int(training.get("profile_trace_steps", 5))
        if self.enabled and (self.trace_start < 1 or self.trace_steps < 0):
            raise ValueError("profile_trace_start must be positive; trace_steps nonnegative")
        self.profiler = None
        self.file = None
        self.row = {}
        if self.enabled:
            self.file = (run_dir / f"timings_rank{rank}.jsonl").open("w")

    def begin(self, step):
        if not self.enabled:
            return
        # Export outside the measured iteration; compilation normally warms up
        # before this window. No profiler exists in the default training path.
        if self.profiler is not None and step == self.trace_start + self.trace_steps:
            self._finish_trace()
        if self.trace_steps and step == self.trace_start:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if self.device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self.profiler = torch.profiler.profile(activities=activities)
            self.profiler.start()
        self.started = time.perf_counter()
        self.row = {"step": step, "trace_active": self.profiler is not None}

    @contextmanager
    def stage(self, name):
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        scope = torch.profiler.record_function(name) if self.profiler else nullcontext()
        with scope:
            yield
        self.row[name + "_wall_ms"] = (time.perf_counter() - start) * 1000

    def end(self, **timings):
        if not self.enabled:
            return
        self.row.update(timings)
        self.row["iteration_wall_ms"] = (time.perf_counter() - self.started) * 1000
        if self.device.type == "cuda":
            self.row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(self.device)
        self.file.write(json.dumps(self.row) + "\n")
        self.file.flush()

    def _finish_trace(self):
        profiler, self.profiler = self.profiler, None
        profiler.stop()
        profiler.export_chrome_trace(str(self.run_dir / f"trace_rank{self.rank}.json"))
        with (self.run_dir / f"operators_rank{self.rank}.txt").open("w") as out:
            out.write(profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=50))
            if self.device.type == "cuda":
                out.write("\n" + profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=50))

    def close(self):
        try:
            if self.profiler is not None:
                self._finish_trace()
        finally:
            if self.file is not None:
                self.file.close()
