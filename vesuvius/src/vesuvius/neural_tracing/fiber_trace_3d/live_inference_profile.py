"""Short, isolated profiles of the real shared demand-inference path."""
from __future__ import annotations

from contextlib import ExitStack
import cProfile
from datetime import datetime, timezone
import gc
import io
import json
import os
from collections import Counter
from pathlib import Path
import pstats
import subprocess
import threading
import time

import numpy as np
import torch

from lasagna import demand_predict3d as demand, tiled_predict3d as tiled
from lasagna.inference_provenance import atomic_write, json_digest, sha256_file
from . import live_evaluate as live
from ..fiber_trace_2d.fiber_json import load_fiber_file
from ..fiber_trace_2d.sampling import Vc3dCoordinateSampler


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0}
    return {"count": len(values), "sum": float(values.sum()), "mean": float(values.mean()),
            "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)),
            "min": float(values.min()), "max": float(values.max())}


def process_counters(pid, proc_root=Path("/proc")):
    """Optional Linux process counters; unavailable counters remain absent.

    read/write_bytes are kernel-accounted storage traffic, not device timings.
    schedstat and wchan concern the main thread (the accumulator task executor).
    """
    path = proc_root / str(pid)
    result = {}
    try:
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        ticks = os.sysconf("SC_CLK_TCK")
        result.update(minor_faults=int(fields[7]), major_faults=int(fields[9]),
                      user_seconds=int(fields[11]) / ticks, system_seconds=int(fields[12]) / ticks,
                      state=fields[0])
        if len(fields) > 39:
            result["block_io_delay_seconds"] = int(fields[39]) / ticks
    except (OSError, ValueError, IndexError):
        pass
    try:
        for line in (path / "io").read_text().splitlines():
            key, value = line.split(":", 1)
            result[key] = int(value)
    except (OSError, ValueError):
        pass
    try:
        runtime, wait, slices = map(int, (path / "schedstat").read_text().split())
        result.update(main_thread_run_seconds=runtime / 1e9,
                      main_thread_runnable_wait_seconds=wait / 1e9, main_thread_timeslices=slices)
    except (OSError, ValueError):
        pass
    return result


class ProcessMonitor:
    """Low-rate wait-state samples plus counter deltas, outside codec code."""
    def __init__(self, processes):
        self.pids = [p.pid for p in processes]
        self.before = {p: process_counters(p) for p in self.pids}
        self.waits = {p: Counter() for p in self.pids}
        self.states = {p: Counter() for p in self.pids}
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, name="accumulator-os-profile", daemon=True)
        self.thread.start()

    def _sample(self):
        while not self.stop_event.is_set():
            for pid in self.pids:
                path = Path("/proc") / str(pid)
                try:
                    self.waits[pid][(path / "wchan").read_text().strip()] += 1
                    state = (path / "stat").read_text().rsplit(")", 1)[1].split()[0]
                    self.states[pid][state] += 1
                except (OSError, IndexError):
                    pass
            self.stop_event.wait(.05)

    def close(self):
        self.stop_event.set()
        self.thread.join()

    def finish(self):
        self.close()
        rows = []
        for pid in self.pids:
            after = process_counters(pid)
            delta = {k: v - self.before[pid][k] for k, v in after.items()
                     if isinstance(v, (int, float)) and k in self.before[pid]}
            rows.append({"pid": pid, "delta": delta, "wait_samples": dict(self.waits[pid]),
                         "state_samples": dict(self.states[pid])})
        return {"sampling_interval_seconds": .05, "workers": rows,
                "note": "Counters absent if unsupported/denied; zero blkio delay may mean kernel delay accounting disabled. wchan=0 may mean running or hidden. Wait counts are samples, not exact durations."}


def profile_workload(workload):
    started = time.perf_counter()
    params = workload["profile"]
    output = Path(params["output_root"]) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    output.mkdir(parents=True, exist_ok=False)
    atomic_write(output / "workload.json", workload)
    print(f"profile output={output}", flush=True)
    report = {"status": "initializing", "devices": workload["devices"], "torch": str(torch.__version__),
              "cuda": torch.version.cuda, "workload_sha256": json_digest(workload),
              "timing_note": "warmup/setup/cleanup excluded; windows end after current synchronous pump; phase timings overlap"}
    try:
        checkpoint = Path(workload["checkpoint"])
        if sha256_file(checkpoint) != workload["checkpoint_sha256"]:
            raise ValueError("checkpoint identity changed")
        report["checkpoint_sha256"] = workload["checkpoint_sha256"]
        report["implementation"] = live.implementation_identity()
        devices = [torch.device(f"cuda:{int(i)}") for i in workload["devices"].split(",")]
        if not devices or not torch.cuda.is_available():
            raise ValueError("profiling requires CUDA")
        report["gpu_names"] = [torch.cuda.get_device_name(d) for d in devices]
        settings = dict(workload["evaluation"]["inference"])
        for option in ("accumulator_layout", "accumulator_prefault"):
            if option in params:
                settings[option] = params[option]
        effective, _ = live._checkpoint_config_for_inference(checkpoint, None)
        precision, _ = live._resolve_inference_precision(settings.get("precision", "auto"), checkpoint=checkpoint, devices=devices)
        effective = {**effective, "training": {**effective.get("training", {}), "mixed_precision": precision}}
        binding = next(s for s in workload["sources"] if s["id"] == params["source_id"])
        cache_root = subprocess.check_output([str(Path(workload["native_binary"]).with_name("vc_cache_root"))], text=True).strip()
        sampler = Vc3dCoordinateSampler(binding["input_volume"], level=binding["input_level"],
            cache_root=cache_root, cache_budget_bytes=4 << 30, io_threads=16, decode_threads=8, sample_threads=1)
        attrs = live.input_root_attributes(sampler.volume)
        live.validate_input_lattice(attrs, binding["input_level"])
        shape = tuple(sampler.volume.shape_at(binding["input_level"]))
        geometry = demand.DemandGeometry(shape, settings["tile_size"], settings["overlap"], settings["border"],
            1 << settings["scaledown_power"], settings["chunk_size"])
        source = {"locator": binding["input_volume"], "level": binding["input_level"], "shape": shape,
                  "base_shape": tuple(sampler.volume.shape_at(0)), "metadata": attrs}
        model_identity = {"checkpoint_sha256": workload["checkpoint_sha256"], "settings": settings,
                          "precision": precision, "purpose": "isolated bounded profile"}
        references = sorted((Path(workload["split_root"]) / "test" / binding["id"]).glob("*.json"))
        requested = set()
        for path in references:
            fiber, = load_fiber_file(path)
            requested.update(live.reference_chunks(fiber.line_points_xyz,
                (1 << binding["input_level"]) * geometry.scale, geometry.output_shape, geometry.chunk))
        report.update(source=source, settings=settings, reference_files=[str(p) for p in references],
                      reference_sha256={str(p): sha256_file(p) for p in references}, requested_chunks=len(requested))
        with ExitStack() as cleanup:
            store = cleanup.enter_context(live.create_demand_store(output / "demand", source, model_identity,
                geometry, effective, checkpoint, binding["input_level"], settings))
            reader = live.NativeTileReader(sampler, geometry, sampler.volume.dtype)
            for index in sorted(requested):
                store.request(index, background=True)
            tile = min(store.priorities)
            raw = reader(tile)
            adapter = store.model_adapter
            model = adapter.load_model(device=devices[0])
            gpu_results = []
            for side, batch in ((geometry.tile, 1), (128, 16)):
                # Same real tile, smaller size is explicitly a model-cost comparison.
                cropped = np.ascontiguousarray(raw[:side, :side, :side])
                tensor = tiled._input_tile_to_device(cropped, devices[0])
                if batch > 1:
                    tensor = tensor.repeat(batch, 1, 1, 1, 1)
                if hasattr(adapter, "preprocess_tile"):
                    tensor = adapter.preprocess_tile(tensor, torch.ones_like(tensor, dtype=torch.bool))
                with torch.inference_mode():
                    for _ in range(2):
                        result = adapter.run_tile_inference(model, tensor, device=devices[0])
                        torch.cuda.synchronize(devices[0])
                        del result
                    walls, cuda = [], []
                    begin = time.perf_counter()
                    while time.perf_counter() - begin < float(params["gpu_seconds_per_shape"]):
                        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        one = time.perf_counter()
                        a.record(torch.cuda.current_stream(devices[0]))
                        result = adapter.run_tile_inference(model, tensor, device=devices[0])
                        b.record(torch.cuda.current_stream(devices[0]))
                        torch.cuda.synchronize(devices[0])
                        walls.append(time.perf_counter() - one)
                        cuda.append(a.elapsed_time(b) / 1000)
                        del result
                wall = time.perf_counter() - begin
                row = {"side": side, "batch": batch, "iterations": len(walls), "wall_seconds": wall,
                       "wall_latency_s": distribution(walls), "cuda_latency_s": distribution(cuda),
                       "tiles_per_second": len(walls) * batch / wall,
                       "input_megavoxels_per_second": len(walls) * batch * side**3 / wall / 1e6,
                       "equivalent_128_tiles_per_second": len(walls) * batch * (side / 128)**3 / wall}
                gpu_results.append(row)
                print(f"gpu-only side={side} batch={batch} tiles/s={row['tiles_per_second']:.3f} Mvox/s={row['input_megavoxels_per_second']:.2f}", flush=True)
                del tensor
            del model, raw
            gc.collect()
            torch.cuda.empty_cache()
            report["gpu_only"] = gpu_results
            atomic_write(output / "report.json", report)

            phase_values = {}
            lock = threading.Lock()
            def instrument(name, method):
                def call(*args, **kwargs):
                    begin = time.perf_counter()
                    try:
                        return method(*args, **kwargs)
                    finally:
                        with lock:
                            phase_values.setdefault(name, []).append(time.perf_counter() - begin)
                return call
            for name in ("begin_complete", "contribution_done", "publication_done", "_publish", "_fingerprints", "_check_disk_budget"):
                setattr(store, name, instrument(name, getattr(store, name)))
            pool = cleanup.enter_context(demand.DemandWorkerPool({"source": store}, {"source": instrument("native_read", reader)},
                devices, slots_per_gpu=settings["slots_per_gpu"], input_workers=settings["input_workers"],
                result_slots=params.get("result_slots", settings.get("result_slots")),
                accumulator_workers=settings.get("accumulator_workers"),
                flush_workers=int(settings.get("flush_workers", 4)), profile_pipeline=True))
            warm_start = time.perf_counter()
            while len(pool.worker_ids) < len(devices) and pool.busy and time.perf_counter() - warm_start < params["warmup_max_seconds"]:
                pool.pump()
                time.sleep(.005)
            warm_seconds = time.perf_counter() - warm_start
            warmed_worker_ids = sorted(pool.worker_ids)
            print(f"pipeline warmup={warm_seconds:.2f}s warmed_workers={len(pool.worker_ids)}/{len(devices)}", flush=True)
            baseline = dict(pool.stats)
            def backlog():
                return {"gpu_active": len(pool.active), "loading": len(pool.loading),
                        "accumulating": len(pool.accumulating), "publishing": len(pool.publishing),
                        "publication_queued": sum(len(s.publications) for s in pool.stores.values()),
                        "free_results": len(pool.free), "free_inputs": len(pool.free_inputs)}
            initial_backlog = backlog()
            baseline_store = dict(store.stats)
            sample_start = len(pool.profile_samples)
            accumulator_start = len(pool.accumulator_samples)
            accumulator_counts_start = list(pool.accumulator_counts)
            accumulator_backends_start = dict(pool.accumulator_backends)
            with lock:
                phase_values.clear()
            profiler = cProfile.Profile()
            monitor = ProcessMonitor(pool.accumulator_processes) if params.get("accumulator_os_metrics", False) else None
            if monitor is not None:
                cleanup.callback(monitor.close)
            begin = time.perf_counter()
            last_report = begin
            queue_samples = [{"elapsed_seconds": 0.0, **pool.queue_snapshot()}]
            profiler.enable()
            while pool.busy and time.perf_counter() - begin < params["pipeline_seconds"]:
                pool.pump()
                time.sleep(.005)
                elapsed_now = time.perf_counter() - begin
                if elapsed_now - queue_samples[-1]["elapsed_seconds"] >= .25:
                    queue_samples.append({"elapsed_seconds": elapsed_now, **pool.queue_snapshot()})
                if time.perf_counter() - last_report > 5:
                    print(f"pipeline measured={time.perf_counter()-begin:.1f}s tiles={pool.stats['tiles']-baseline['tiles']} received={len(pool.profile_samples)-sample_start} published={store.stats['published_chunks']-baseline_store['published_chunks']} accumulating={len(pool.accumulating)} free_results={len(pool.free)}", flush=True)
                    print("queues " + demand.format_queue_snapshot(queue_samples[-1]), flush=True)
                    last_report = time.perf_counter()
            profiler.disable()
            elapsed = time.perf_counter() - begin
            queue_samples.append({"elapsed_seconds": elapsed, **pool.queue_snapshot()})
            with lock:
                measured_phases = {k: tuple(v) for k, v in phase_values.items()}
            os_metrics = monitor.finish() if monitor is not None else None
            profiler.dump_stats(str(output / "coordinator.prof"))
            stream = io.StringIO()
            pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(45)
            (output / "coordinator.txt").write_text(stream.getvalue())
            samples = pool.profile_samples[sample_start:]
            count = pool.stats["tiles"] - baseline["tiles"]
            published = store.stats["published_chunks"] - baseline_store["published_chunks"]
            report["pipeline"] = {"warmup_seconds": warm_seconds, "warmed_worker_ids": warmed_worker_ids,
                "queue_samples": queue_samples, "queue_sample_interval_seconds": .25,
                "queue_sample_gap_seconds": distribution(np.diff([r["elapsed_seconds"] for r in queue_samples])),
                "queue_distributions": {k: distribution([r[k] for r in queue_samples])
                                        for k in queue_samples[0] if k != "elapsed_seconds"},
                "queue_note": "Coordinator samples, not event-exact. Distributions are sample-weighted, not time-weighted. read_done holds a slot pending dispatch. GPU submitted includes queued/executing/completed-uncollected; not utilization. No native tracer RPCs in this profile, only reference chunk prefetch. Native underlying cache/HTTP requests are not counted.",
                "buffer_capacity": {k: baseline[k] for k in ("input_slots", "result_slots", "input_slot_bytes", "result_slot_bytes", "shared_slot_bytes")},
                "initial_backlog": initial_backlog, "final_backlog": backlog(),
                "received_tiles": len(samples), "received_tiles_per_second": len(samples) / elapsed,
                "all_workers_warm_before_measurement": len(warmed_worker_ids) == len(devices),
                "instrumented_with_cprofile": True,
                "requested_window_seconds": params["pipeline_seconds"], "wall_seconds": elapsed,
                "window_overrun_seconds": max(0, elapsed - params["pipeline_seconds"]), "completed_tiles": count,
                "published_chunks": published, "published_voxels_upper_bound": published * geometry.chunk**3,
                "tiles_per_second": count / elapsed, "input_megavoxels_per_second": count * geometry.tile**3 / elapsed / 1e6,
                "equivalent_128_tiles_per_second": count * (geometry.tile/128)**3 / elapsed,
                "pending_active": len(pool.active), "pending_loading": len(pool.loading),
                "pending_accumulating": len(pool.accumulating), "pending_publishing": len(pool.publishing),
                "accumulator_task_seconds": distribution(pool.accumulator_samples[accumulator_start:]),
                "accumulator_queue_plus_ack_seconds": distribution(pool.accumulator_queue_samples[accumulator_start:]),
                "accumulator_queue_timing_note": "Starts at worker-queue submission; excludes earlier coordinator pending-deque time. Includes acknowledgement delivery delay.",
                "accumulator_backends": {k: v - accumulator_backends_start.get(k, 0) for k, v in pool.accumulator_backends.items()},
                "accumulator_os_metrics": os_metrics,
                "accumulator_tasks_per_worker": [a - b for a, b in zip(pool.accumulator_counts, accumulator_counts_start)],
                "queued_tiles": sum(t not in store.tiles for t in store.priorities),
                "phase_seconds_inclusive": {k: distribution(v) for k,v in measured_phases.items()},
                "worker_profile": {k: distribution([s[k] for s in samples]) for k in samples[0] if k != "worker_index"} if samples else {},
                "worker_samples": samples}
            report["status"] = "profile_window_complete_not_full_inference"
            atomic_write(output / "report.json", report)
        report["total_wall_seconds_including_setup_cleanup"] = time.perf_counter() - started
        atomic_write(output / "report.json", report)
        print(f"profile complete: {output / 'report.json'}", flush=True)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", total_wall_seconds=time.perf_counter()-started)
        atomic_write(output / "report.json", report)
        raise
