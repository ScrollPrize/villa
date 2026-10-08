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
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Evaluate saved weights instead of random initialization")
    p.add_argument("--checkpoint-series", type=Path, nargs="+", default=None,
                   help="evaluate each snapshot once and export cross-section artifacts")
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
    p.add_argument("--cross-section", action="store_true",
                   help="accumulate Lasagna-normal-oriented presence cross sections")
    p.add_argument("--cross-section-width-base", type=int, default=160)
    p.add_argument("--cross-section-height-base", type=int, default=80)
    p.add_argument("--cross-section-histogram-bins", type=int, default=1024)
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
        self.refinement_context = None
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
        self.loader = None
        self.start = self.mode = None
        self.sample_count = None
        self.indices = None
        if not self.options["cross_section"]:
            self.loader = FiberTrace3DLoader(self.config)
            _, self.start, self.mode = train._resolve_dense_test_selection(
                training, loader_sample_count=self.loader.sample_count, default_count=0)
        torch.manual_seed(int(training.get("initialization_seed", 0)))
        self.model = train.build_fiber_trace_3d_model(raw).to("cuda:0")
        if self.options.get("checkpoint") and not self.options["cross_section"]:
            train._load_snapshot(self.options["checkpoint"], model=self.model, map_location="cpu")
        self.precision = train._mixed_precision_config_from_training(training, torch.device("cuda:0"))
        self.raw = raw
        if self.loader is not None:
            self.sample_count = (
                self.loader.sample_count if int(self.options["samples"]) == 0
                else min(int(self.options["samples"]), self.loader.sample_count)
            )
            self.indices = tuple(range(rank, (self.sample_count + self.config.batch_size - 1)
                                       // self.config.batch_size, self.world))
        # Partition a fixed total CPU budget, matching training's per-rank pools.
        self.raw = dict(raw, training=dict(training, sample_pipeline_workers=self.options["cpu_workers"]))
        torch.cuda.synchronize()
        self.startup_s = time.monotonic() - started
        groups = "deferred" if self.indices is None else len(self.indices)
        print(f"evaluation benchmark rank={rank} ready startup={self.startup_s:.3f}s backend={self.options['backend']} groups={groups}", flush=True)

    def run_unit(self):
        train, torch = self.train, self.torch
        started = time.monotonic()
        cross_section = None
        snapshot = None
        if self.options["cross_section"]:
            from .cross_section import CrossSectionAccumulator, CrossSectionConfig
            from .loader import FiberTrace3DLoader
            from .refinement import frozen_refinement_root
            snapshots = self.options["checkpoint_series"]
            snapshot = snapshots[len(self.rows)]
            self.refinement_context = frozen_refinement_root(
                self.options["refinement_roots"][snapshot]
            )
            self.refinement_context.__enter__()
            self.loader = FiberTrace3DLoader(self.config)
            _, self.start, self.mode = train._resolve_dense_test_selection(
                self.raw["training"], loader_sample_count=self.loader.sample_count, default_count=0
            )
            self.sample_count = (
                self.loader.sample_count if int(self.options["samples"]) == 0
                else min(int(self.options["samples"]), self.loader.sample_count)
            )
            self.indices = tuple(range(
                self.rank,
                (self.sample_count + self.config.batch_size - 1) // self.config.batch_size,
                self.world,
            ))
            train._load_snapshot(snapshot, model=self.model, map_location="cpu")
            cross_section = CrossSectionAccumulator(CrossSectionConfig(
                width_base=self.options["cross_section_width_base"],
                height_base=self.options["cross_section_height_base"],
                spacing_base=self.options["cross_section_spacing_base"],
                histogram_bins=self.options["cross_section_histogram_bins"],
            ))
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
                batch_size=self.config.batch_size, sample_count=self.sample_count,
                start_sample_index=self.start))
            self.iterator = iter(self.source)
        times = {}
        rows = train._evaluate_dense_loss_rows(self.model, self.loader,
            device=torch.device("cuda:0"), start_sample_index=self.start,
            sample_count=self.sample_count, sample_mode=self.mode,
            direction_weight=float(self.raw["training"].get("direction_weight", 10)),
            presence_weight=float(self.raw["training"].get("presence_weight", 1)),
            conditioned_loss_options=train._conditioned_loss_options_from_training(self.raw["training"]),
            mixed_precision=self.precision, global_batch_indices=self.indices,
            batch_iterator=self.iterator, profile_times=times,
            microbatch_size=self.options["microbatch_size"],
            output_upsample=self.raw["training"].get("test_output_upsample"),
            prediction_observer=cross_section)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        if self.options["backend"] == "pipeline":
            self.source.close()
            self.source = self.iterator = None
        sync_started = time.monotonic()
        torch.distributed.barrier()
        shard = None
        if cross_section is not None:
            label = self.options["checkpoint_labels"][len(self.rows)]
            shard = Path(self.options["output"]) / "shards" / f"{label}-rank{self.rank}.npz"
            shard.parent.mkdir(parents=True, exist_ok=True)
            cross_section.save_npz(shard)
        result = dict(rank=self.rank, pass_index=len(self.rows), startup_s=self.startup_s,
                      elapsed_s=elapsed, synchronization_s=time.monotonic()-sync_started,
                      profile_times=times, metrics=rows, snapshot=snapshot,
                      cross_section_shard=None if shard is None else str(shard))
        self.rows.append(result)
        print(f"evaluation benchmark rank={self.rank} pass={len(self.rows)} elapsed={elapsed:.3f}s stages={times}", flush=True)
        if cross_section is not None:
            if self.sample_queue is not None:
                self.sample_queue.close()
                self.sample_queue = None
            self.loader = None
            self.indices = None
            self.refinement_context.__exit__(None, None, None)
            self.refinement_context = None
        target_units = (len(self.options["checkpoint_series"])
                        if self.options["cross_section"] else self.options["passes"])
        return {"done": len(self.rows) >= target_units}

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
        if self.refinement_context is not None:
            self.refinement_context.__exit__(None, None, None)
            self.refinement_context = None
        self.torch.distributed.destroy_process_group()


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.samples < 0 or min(args.batch_size, args.passes, args.cpu_workers, args.timeout) <= 0:
        p.error("samples must be nonnegative; other counts and timeout must be positive")
    if args.microbatch_size is not None and args.microbatch_size < 1:
        p.error("microbatch size must be positive")
    if args.cross_section:
        if args.checkpoint is not None:
            p.error("use --checkpoint-series, not --checkpoint, with --cross-section")
        if not args.checkpoint_series:
            p.error("--cross-section requires --checkpoint-series")
        if args.passes != 1:
            p.error("--cross-section evaluates each snapshot once; use --passes 1")
    elif args.checkpoint_series is not None:
        p.error("--checkpoint-series requires --cross-section")
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
        options["checkpoint"] = str(args.checkpoint.resolve()) if args.checkpoint else None
        if args.cross_section:
            snapshots = [path.resolve() for path in args.checkpoint_series]
            if len(set(snapshots)) != len(snapshots):
                p.error("--checkpoint-series paths must be unique")
            labels = [path.stem for path in snapshots]
            if len(set(labels)) != len(labels):
                p.error("--checkpoint-series basenames must be unique")
            raw = json.loads(args.reference.read_text())
            if bool(raw.get("training", {}).get("test_augment_enabled", False)):
                p.error("cross-section base-voxel geometry requires test_augment_enabled=false")
            scales = {int(row.get("base_volume_scale", 0)) for row in raw.get("test_datasets", [])}
            if len(scales) != 1:
                p.error("cross-section evaluation requires one common test base_volume_scale")
            options["checkpoint_series"] = [str(path) for path in snapshots]
            options["checkpoint_labels"] = labels
            options["cross_section_spacing_base"] = float(1 << scales.pop())
        with ExclusiveTask(socket, grant, label="fiber-evaluation-profile", priority=100) as lease:
            with tempfile.TemporaryDirectory(prefix="fiber-eval-") as directory:
                cohort = None
                monitor = NvidiaMonitor(grant, 0, interval=.1)
                try:
                    if args.cross_section:
                        import torch
                        from .refinement import GeometryStore
                        roots = {}
                        restored = {}
                        for checkpoint in options["checkpoint_series"]:
                            payload = torch.load(
                                checkpoint, map_location="cpu", weights_only=False, mmap=True
                            )
                            saved = payload.get("refinement_state")
                            if payload.get("config", {}).get("refinement", {}).get("enabled") and not saved:
                                raise ValueError(f"refinement checkpoint lacks fiber state: {checkpoint}")
                            if saved:
                                digest = saved["sha256"]
                                if digest not in restored:
                                    root = Path(directory) / "refinement" / digest
                                    root.mkdir(parents=True)
                                    GeometryStore(root).restore(
                                        Path(checkpoint).parent / saved["file"], digest
                                    )
                                    restored[digest] = str(root)
                                roots[checkpoint] = restored[digest]
                            else:
                                root = Path(directory) / "refinement" / "empty"
                                root.mkdir(parents=True, exist_ok=True)
                                (root / "restored.json").write_text("{}")
                                roots[checkpoint] = str(root)
                        options["refinement_roots"] = roots
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
                    if args.cross_section:
                        from .cross_section import merge_shards, write_cross_section_artifacts
                        reports = []
                        for label, checkpoint in zip(
                            options["checkpoint_labels"], options["checkpoint_series"], strict=True
                        ):
                            shards = [
                                args.output / "shards" / f"{label}-rank{rank}.npz"
                                for rank in range(len(devices))
                            ]
                            merged = merge_shards(shards)
                            files = write_cross_section_artifacts(
                                merged,
                                args.output / label,
                                metadata={
                                    "checkpoint": checkpoint,
                                    "base_volume_scale": int(round(
                                        __import__("math").log2(options["cross_section_spacing_base"])
                                    )),
                                    "samples_requested": options["samples"],
                                },
                            )
                            reports.append({
                                "label": label,
                                "checkpoint": checkpoint,
                                "sections": merged.sections,
                                "skipped": merged.skipped,
                                "files": files,
                            })
                            print(
                                f"cross-section snapshot={label} sections={merged.sections} "
                                f"skipped={merged.skipped} output={args.output / label}",
                                flush=True,
                            )
                        (args.output / "cross_section_series.json").write_text(
                            json.dumps(reports, indent=2) + "\n"
                        )
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
