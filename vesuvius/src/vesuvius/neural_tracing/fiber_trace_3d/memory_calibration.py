"""Explicit cold-start memory probe using the shared distributed training loop.

Run only within explicitly permitted GPU budgets. Unlike the manager, this bootstrap
probe does not need an existing memory estimate. It never selects a GPU itself.
"""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import tempfile
import time

from .cooperative_resources import MemoryCalibration
from .training_state import config_fingerprint


def calibration_from_observations(fingerprint, observations, *, reserve_bytes):
    if type(reserve_bytes) is not int or reserve_bytes < 0 or not observations:
        raise ValueError("calibration needs observations and a nonnegative reserve")
    for row in observations:
        for field in ("microbatch_size", "initialization_reserved_bytes", "peak_reserved_bytes",
                      "peak_allocated_bytes", "driver_peak_bytes"):
            if type(row.get(field)) is not int or row[field] <= 0:
                raise ValueError(f"invalid measured {field}")
        if not row.get("gpu_name") or not row.get("torch_version") or row.get("config_fingerprint") != fingerprint:
            raise ValueError("observation identity/config does not match calibration")
        if type(row.get("world_size")) is not int or row["world_size"] < 1 or not row.get("cuda_version"):
            raise ValueError("observation must record distributed size and CUDA version")
    peaks = {}
    for row in observations:
        size = row["microbatch_size"]
        peaks[size] = max(peaks.get(size, 0), row["driver_peak_bytes"], row["peak_reserved_bytes"])
    fixed = max(row["initialization_reserved_bytes"] for row in observations)
    slope = max(max(0, peak - fixed + size - 1) // size for size, peak in peaks.items())
    # The measured peak already includes asynchronous prepared batches, targets,
    # all gradient banks and optimizer state. Do not count preparation twice.
    estimate = MemoryCalibration(fixed, slope, 0, 0, reserve_bytes, max(peaks), tuple(sorted(peaks.items())))
    return {"config_fingerprint": fingerprint, "estimate": asdict(estimate), "observations": observations}


@dataclass(frozen=True)
class ProbeFactory:
    config_path: str
    microbatch_size: int
    steps: int
    cpu_workers: int
    output_dir: str
    rendezvous: str
    reserve_bytes: int

    def __call__(self, rank, grant, state):
        import os
        os.environ.update(RANK=str(rank), WORLD_SIZE=str(len(grant["gpus"])), LOCAL_RANK="0")
        return ProbeWorker(self, rank, grant)


class ProbeWorker:
    def __init__(self, settings, rank, grant):
        import torch
        from .train import _load_raw_config, training_units
        self.torch = torch
        self.settings = settings
        self.rank, self.devices = rank, sorted(grant["gpus"])
        raw = _load_raw_config(settings.config_path)
        training = raw.get("training", {})
        if str(training.get("device", "auto")).split(":")[0] not in ("cuda", "auto"):
            raise ValueError("memory calibration requires a CUDA training config")
        self.fingerprint = config_fingerprint(raw)
        from .cooperative_resources import configure_cuda_memory_budget, initialize_worker_volume_cache
        configure_cuda_memory_budget(grant["gpus"][self.devices[rank]], settings.reserve_bytes)
        initialize_worker_volume_cache(raw)
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        size = settings.microbatch_size
        slots = max(size, int(training.get("sample_pipeline_slots", 512)))
        slots = ((slots + size - 1) // size) * size
        self.units = training_units(settings.config_path, physical_microbatch_sizes=(size,) * len(self.devices),
            distributed_init_method=settings.rendezvous,
            training_overrides={"bounded_gradient_accumulation": True,
                "device": "cuda:0",
                "sync_batchnorm": False, "ddp_broadcast_buffers": False,
                "max_steps": settings.steps, "resume": None,
                "run_path": settings.output_dir, "run_name": f"memory-b{size}",
                "sample_pipeline_workers": settings.cpu_workers,
                "sample_pipeline_load_affinity_workers": max(1, settings.cpu_workers // 2),
                "sample_pipeline_slots": slots * len(self.devices)})
        self.boundary = next(self.units)
        torch.cuda.synchronize(0)
        self.initialization = int(torch.cuda.memory_reserved(0))
        self.count = 0

    def run_unit(self):
        if not self.boundary.done:
            self.boundary = next(self.units)
        self.torch.cuda.synchronize(0)
        self.count += 1
        return {"done": self.boundary.done, "step": self.boundary.step,
                "phase": self.boundary.phase, "unit": self.count}

    def capture(self):
        torch = self.torch
        row = {"microbatch_size": self.settings.microbatch_size,
            "initialization_reserved_bytes": self.initialization,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "gpu_name": torch.cuda.get_device_name(0), "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda, "config_fingerprint": self.fingerprint,
            "optimizer_steps": self.settings.steps, "work_units": self.count,
            "world_size": len(self.devices), "gpu_uuid": self.devices[self.rank]}
        if len(self.devices) > 1:
            rows = [None] * len(self.devices)
            torch.distributed.all_gather_object(rows, row)
        else:
            rows = [row]
        return "calibration", rows

    def close(self):
        self.units.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--gpu-uuids", "--gpu-uuid", nargs="+", required=True)
    parser.add_argument("--microbatch-sizes", nargs="+", type=int, required=True)
    parser.add_argument("--cpu-workers", type=int, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--reserve-bytes", type=int, default=4 * 1024**3)
    parser.add_argument("--vram-limit-bytes", type=int, required=True,
                        help="total per-GPU probe allowance, including driver reserve")
    parser.add_argument("--allow-shared", action="store_true",
                        help="permit other tasks on the selected GPUs; measure only this probe")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.cpu_workers < 2 * len(args.gpu_uuids) or args.steps < 2 or min(args.microbatch_sizes) < 1:
        parser.error("need >=2 CPU workers per GPU, >=2 optimizer steps and positive microbatch sizes")
    if not 0 <= args.reserve_bytes < args.vram_limit_bytes:
        parser.error("VRAM limit must exceed the nonnegative reserve")
    from villa_gpu_manager.cohort import ProcessCohort
    from villa_gpu_manager.processes import NvidiaMonitor, nvidia_inventory
    from vesuvius.neural_tracing.ordered_sample_pipeline import native_thread_environment
    from .train import _load_raw_config
    config = args.config.resolve()
    raw = _load_raw_config(config)
    if raw.get("resume") or raw.get("training", {}).get("resume"):
        parser.error("cold-start calibration requires a config without resume")
    inventory = nvidia_inventory(args.cpu_workers, args.gpu_uuids, 0)
    inventory["gpus"] = {gpu: min(total, args.vram_limit_bytes) for gpu, total in inventory["gpus"].items()}
    observations = []
    def completed(rows):
        if any(row != rows[0] for row in rows[1:]):
            raise RuntimeError("calibration ranks disagree about work progress")
        print(f"calibration progress: {rows[0]}", flush=True)
        return rows[0]["done"]

    for size in sorted(set(args.microbatch_sizes)):
        monitor = NvidiaMonitor(inventory, 0, interval=0.05)
        rendezvous = tempfile.TemporaryDirectory(prefix="fiber-memory-probe-")
        cohort = ProcessCohort(ProbeFactory(str(config), size, args.steps, args.cpu_workers,
                                          str(args.output.parent / "memory-probes"),
                                          (Path(rendezvous.name) / "group").as_uri(), args.reserve_bytes), inventory,
            environment=native_thread_environment(), on_unit=completed,
            startup_timeout=3600, unit_timeout=3600)
        driver_peak = {gpu: 0 for gpu in args.gpu_uuids}
        known_pids = set()
        try:
            deadline = time.monotonic() + 15
            while monitor.last_snapshot is None:
                monitor.poll(set())
                if time.monotonic() > deadline:
                    raise RuntimeError("could not inspect calibration GPU before authorization")
                time.sleep(0.01)
            if not args.allow_shared and any(monitor.last_snapshot[gpu]["processes"] for gpu in args.gpu_uuids):
                raise RuntimeError("calibration GPU already has a compute process")
            if any(monitor.last_snapshot[gpu]["total"] - monitor.last_snapshot[gpu]["used"] < inventory["gpus"][gpu]
                   for gpu in args.gpu_uuids):
                raise RuntimeError("not enough currently free VRAM for the permitted calibration budget")
            cohort.authorize()
            while not cohort.closed:
                cohort.poll()
                known_pids.update(row["pid"] for row in cohort.processes)
                monitor.poll(known_pids)
                if monitor.last_snapshot is not None:
                    for gpu in args.gpu_uuids:
                        device = monitor.last_snapshot[gpu]
                        # Never calibrate over another task.
                        if not args.allow_shared and set(device["processes"]) - known_pids:
                            raise RuntimeError("calibration GPU has another compute process")
                        owned = sum(size for pid, size in device["processes"].items() if pid in known_pids)
                        driver_peak[gpu] = max(driver_peak[gpu], owned)
                        if owned > inventory["gpus"][gpu]:
                            raise RuntimeError(f"calibration exceeded its VRAM allowance on {gpu}")
                if time.monotonic() - monitor.last_snapshot_at > 20:
                    raise RuntimeError("calibration driver memory monitoring unavailable")
                if cohort.done:
                    cohort.drain()
                elif cohort.ready and cohort.idle:
                    cohort.start_unit()
                time.sleep(0.005)
            rows = cohort.state.load()
            for row in rows:
                row["driver_peak_bytes"] = driver_peak[row["gpu_uuid"]]
            observations.extend(rows)
            result = calibration_from_observations(config_fingerprint(raw), observations,
                                                    reserve_bytes=args.reserve_bytes)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(rows), flush=True)
        finally:
            if not cohort.closed:
                cohort.abort()
            elif cohort.state is not None:
                cohort.state.close()
            monitor.close()
            rendezvous.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
