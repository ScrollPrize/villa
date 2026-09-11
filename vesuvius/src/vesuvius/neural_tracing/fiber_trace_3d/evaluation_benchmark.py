"""Bounded evaluation-only profiling through the ordinary dense test implementation."""
import argparse
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import signal
import tempfile
import time


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation-only", action="store_true")
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backend", choices=("dataloader", "pipeline", "sync"), default="pipeline")
    p.add_argument("--devices", default="all")
    p.add_argument("--microbatch-size", type=int, default=None,
                   help="exercise managed/calibration test microbatching")
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--cpu-workers", type=int, default=32)
    p.add_argument("--timeout", type=float, default=300)
    return p


@dataclass
class Factory:
    options: dict
    rendezvous: str

    def __call__(self, rank, grant, state):
        return Worker(self, rank, grant)


class Worker:
    def __init__(self, factory, rank, grant):
        import torch
        from . import train
        from .loader import FiberTrace3DLoader, config_from_mapping
        from .cooperative_resources import initialize_worker_volume_cache, configure_cuda_memory_budget
        self.torch, self.train, self.options = torch, train, factory.options
        self.rank, self.world = rank, len(grant["gpus"])
        self.rows, self.iterator, self.source = [], None, None
        self.sample_queue = None
        started = time.monotonic()
        os.environ.update(RANK=str(rank), WORLD_SIZE=str(self.world), LOCAL_RANK="0")
        configure_cuda_memory_budget(grant["gpus"][sorted(grant["gpus"])[rank]], 2 * 1024**3)
        torch.cuda.set_device(0)
        torch.distributed.init_process_group("nccl", init_method=factory.rendezvous,
                                             rank=rank, world_size=self.world, device_id=torch.device("cuda:0"))
        raw = train._load_raw_config(self.options["reference"])
        initialize_worker_volume_cache(raw)
        training = raw["training"]
        test_raw = train._make_test_loader_raw_config(raw, training)
        config = config_from_mapping(test_raw, config_dir=Path(self.options["reference"]).parent)
        self.config = replace(config, batch_size=self.options["batch_size"])
        self.loader = FiberTrace3DLoader(self.config)
        _, self.start, self.mode = train._resolve_dense_test_selection(
            training, loader_sample_count=self.loader.sample_count, default_count=0)
        torch.manual_seed(int(training.get("initialization_seed", 0)))
        self.model = train.build_fiber_trace_3d_model(raw).to("cuda:0")
        self.precision = train._mixed_precision_config_from_training(training, torch.device("cuda:0"))
        self.raw = raw
        self.indices = tuple(range(rank, (self.options["samples"] + self.config.batch_size - 1)
                                   // self.config.batch_size, self.world))
        # Partition a fixed total CPU budget, matching training's per-rank pools.
        self.raw = dict(raw, training=dict(training, sample_pipeline_workers=self.options["cpu_workers"]))
        torch.cuda.synchronize()
        self.startup_s = time.monotonic() - started
        print(f"evaluation benchmark rank={rank} ready startup={self.startup_s:.3f}s backend={self.options['backend']} groups={len(self.indices)}", flush=True)

    def run_unit(self):
        train, torch = self.train, self.torch
        started = time.monotonic()
        if self.options["backend"] == "dataloader":
            if self.source is None:
                self.source = train._make_batch_dataloader(self.config, raw_config=self.raw,
                    start_batch_index=self.rank, batch_index_stride=self.world,
                    sample_index_offset=self.start, batch_count=len(self.indices), sample_mode=self.mode)
            self.iterator = iter(self.source) if self.source is not None else None
        elif self.options["backend"] == "pipeline":
            from .sample_pipeline import FiberTrace3DSampleQueue, SampleQueueSource
            from .queued_batches import QueuedBatchView, test_batches
            if self.sample_queue is None:
                self.sample_queue = FiberTrace3DSampleQueue(
                    sources={"test": SampleQueueSource(self.config, sample_mode=self.mode)},
                    workers=max(1, self.options["cpu_workers"] // self.world),
                    slot_count=max(self.config.batch_size, 512 // self.world), initial_in_flight=16)
            self.source = QueuedBatchView(self.sample_queue, "test", test_batches(self.indices,
                batch_size=self.config.batch_size, sample_count=self.options["samples"],
                start_sample_index=self.start))
            self.iterator = iter(self.source)
        times = {}
        rows = train._evaluate_dense_loss_rows(self.model, self.loader,
            device=torch.device("cuda:0"), start_sample_index=self.start,
            sample_count=self.options["samples"], sample_mode=self.mode,
            direction_weight=float(self.raw["training"].get("direction_weight", 10)),
            presence_weight=float(self.raw["training"].get("presence_weight", 1)),
            conditioned_loss_options=train._conditioned_loss_options_from_training(self.raw["training"]),
            mixed_precision=self.precision, global_batch_indices=self.indices,
            batch_iterator=self.iterator, profile_times=times,
            microbatch_size=self.options["microbatch_size"])
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        if self.options["backend"] == "pipeline":
            self.source.close()
            self.source = self.iterator = None
        sync_started = time.monotonic()
        torch.distributed.barrier()
        result = dict(rank=self.rank, pass_index=len(self.rows), startup_s=self.startup_s,
                      elapsed_s=elapsed, synchronization_s=time.monotonic()-sync_started,
                      profile_times=times, metrics=rows)
        self.rows.append(result)
        print(f"evaluation benchmark rank={self.rank} pass={len(self.rows)} elapsed={elapsed:.3f}s stages={times}", flush=True)
        return {"done": len(self.rows) >= self.options["passes"]}

    def capture(self):
        rows = [None] * self.world
        self.torch.distributed.all_gather_object(rows, self.rows)
        return "evaluation-benchmark", rows

    def close(self):
        if self.source is not None:
            if hasattr(self.source, "close"):
                self.source.close()
            iterator = getattr(self.source, "_iterator", None) or self.iterator
            if iterator is not None and hasattr(iterator, "_shutdown_workers"):
                iterator._shutdown_workers()
        if self.sample_queue is not None:
            self.sample_queue.close()
        self.torch.distributed.destroy_process_group()


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if min(args.samples, args.batch_size, args.passes, args.cpu_workers, args.timeout) <= 0:
        p.error("counts and timeout must be positive")
    if args.microbatch_size is not None and args.microbatch_size < 1:
        p.error("microbatch size must be positive")
    from villa_gpu_manager.bootstrap import ensure_manager, visible_gpu_uuids
    from villa_gpu_manager.exclusive import ExclusiveTask
    from villa_gpu_manager.cohort import ProcessCohort
    from villa_gpu_manager.processes import NvidiaMonitor
    from vesuvius.neural_tracing.ordered_sample_pipeline import native_thread_environment
    if args.devices == "all":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    devices = visible_gpu_uuids()
    if args.cpu_workers < 2 * len(devices):
        p.error("need at least two CPU workers per GPU")
    def timeout(*_):
        raise TimeoutError(f"evaluation benchmark exceeded {args.timeout}s budget")
    old = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, args.timeout)
    try:
        socket, inventory = ensure_manager()
        if set(devices) != set(inventory["gpus"]):
            p.error("evaluation profiling requires exclusive ownership of all managed GPUs; use --devices all")
        budget = min(20 * 1024**3, *(inventory["gpus"][gpu] for gpu in devices))
        grant = {"cpu_cores": args.cpu_workers, "gpus": {gpu: budget for gpu in devices}}
        options = vars(args) | {"reference": str(args.reference.resolve()), "output": str(args.output)}
        with ExclusiveTask(socket, grant, label="fiber-evaluation-profile", priority=100) as lease:
            with tempfile.TemporaryDirectory(prefix="fiber-eval-") as directory:
                cohort = None
                monitor = NvidiaMonitor(grant, 0, interval=.1)
                try:
                    until = time.monotonic() + 10
                    while monitor.last_snapshot is None:
                        monitor.poll(set())
                        if time.monotonic() > until:
                            raise RuntimeError("GPU inventory inspection timed out")
                        time.sleep(.02)
                    if any(row["processes"] for row in monitor.last_snapshot.values()):
                        raise RuntimeError("selected GPU has an unmanaged compute process")
                    lease.cohort = cohort = ProcessCohort(Factory(options, (Path(directory)/"group").as_uri()),
                        grant, environment=native_thread_environment(),
                        on_unit=lambda rows: all(row["done"] for row in rows),
                        startup_timeout=args.timeout, unit_timeout=args.timeout)
                    lease.authorize()
                    check_at = 0
                    owned_pids = {row["pid"] for row in cohort.processes}
                    while not cohort.closed:
                        if time.monotonic() >= check_at:
                            lease.check()
                            check_at = time.monotonic() + .25
                        cohort.poll()
                        monitor.poll(owned_pids)
                        if monitor.last_snapshot is not None:
                            for gpu, allowance in grant["gpus"].items():
                                used = sum(value for pid, value in monitor.last_snapshot[gpu]["processes"].items()
                                           if pid in owned_pids)
                                if used > allowance:
                                    raise RuntimeError(f"benchmark exceeded VRAM allowance on {gpu}: {used} > {allowance}")
                        if cohort.done:
                            cohort.drain()
                        elif cohort.ready and cohort.idle:
                            cohort.start_unit()
                        time.sleep(.01)
                    args.output.mkdir(parents=True, exist_ok=True)
                    path = args.output / f"evaluation-{args.backend}-{time.time_ns()}.json"
                    path.write_text(json.dumps({"options": options, "ranks": cohort.state.load()}, indent=2)+"\n")
                    print(f"evaluation benchmark report={path}", flush=True)
                finally:
                    if cohort is not None:
                        if not cohort.closed:
                            cohort.abort()
                        if cohort.state is not None:
                            cohort.state.close()
                    monitor.close()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
