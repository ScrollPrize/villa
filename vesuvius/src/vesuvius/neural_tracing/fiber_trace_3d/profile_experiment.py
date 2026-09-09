"""Launch an isolated two-GPU reference experiment using this interpreter."""
import argparse
import hashlib
import importlib.metadata
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from vesuvius.neural_tracing.fiber_trace_3d.analyze_training import distribution


def summarize_experiment(experiment):
    experiment = Path(experiment)
    ranks = {}
    for path in sorted(experiment.glob("s1a*/timings_rank*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        # Exclude startup, trace and evaluation steps, not loader stalls.
        selected = [r for r in rows if r["step"] > 30 and not r["trace_active"]
                    and r["step"] % 100 != 0]
        ranks[path.stem] = {
            "rows": len(rows),
            "steady": {key: distribution(r[key] for r in selected if key in r)
                       for key in ("iteration_wall_ms", "wait_ms", "to_device_ms",
                                   "value_augmentation_cuda_ms",
                                   "target_ms", "fw_bw_step_ms", "forward_loss_wall_ms",
                                   "backward_wall_ms", "optimizer_wall_ms", "scalar_reduce_sync_wall_ms")},
            "step1": rows[0] if rows else None,
            "peak_allocated_bytes": max((r.get("peak_allocated_bytes", 0) for r in rows), default=0),
        }
    steady_rows_by_rank = {}
    for path in sorted(experiment.glob("s1a*/timings_rank*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        steady_rows_by_rank[path.stem] = {
            int(row["step"]): row
            for row in rows
            if row["step"] > 30 and not row["trace_active"] and row["step"] % 100 != 0
        }
    common_steps = sorted(
        set.intersection(*(set(rows) for rows in steady_rows_by_rank.values()))
    ) if steady_rows_by_rank else []
    critical_path = {
        key: distribution(
            max(float(rows[step][key]) for rows in steady_rows_by_rank.values())
            for step in common_steps
            if all(key in rows[step] for rows in steady_rows_by_rank.values())
        )
        for key in ("iteration_wall_ms", "wait_ms", "to_device_ms",
                    "value_augmentation_cuda_ms", "target_ms", "fw_bw_step_ms")
    }
    result = {"experiment": str(experiment), "ranks": ranks,
              "critical_path": critical_path,
              "selection": "steps >30 excluding trace and multiples of100; loader stalls retained"}
    (experiment / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True,
                        help="Saved baseline JSON containing the authoritative config")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-mode", choices=["off", "default", "reduce-overhead"], default="off")
    parser.add_argument(
        "--normalization",
        choices=["sync-batch", "batch", "group"],
        default="group",
    )
    parser.add_argument("--output-stride", type=int, default=1)
    parser.add_argument(
        "--batch-size-per-gpu",
        type=int,
        default=8,
        help="benchmark-only local batch size for each of the two GPU ranks",
    )
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--trace-steps", type=int, default=5)
    parser.add_argument("--profile-loader", action="store_true")
    parser.add_argument(
        "--sample-pipeline",
        action="store_true",
        help="use the bounded shared per-sample CPU pipeline in training",
    )
    parser.add_argument(
        "--async-batch-preparation",
        action="store_true",
        help="prepare two future pinned/CUDA batches outside the training thread",
    )
    parser.add_argument(
        "--gpu-value-augmentation",
        choices=("blur", "all"),
        default=None,
        help="defer the selected value-augmentation suffix to CUDA preparation",
    )
    parser.add_argument(
        "--replay-batch",
        action="store_true",
        help="capture one prepared GPU batch per rank, stop loaders, and replay it",
    )
    parser.add_argument(
        "--replay-cpu-batch",
        action="store_true",
        help="capture one augmented CPU batch per rank and repeat normal H2D/target preparation",
    )
    parser.add_argument(
        "--throughput-only",
        action="store_true",
        help="disable evaluation, visualization, snapshots, TensorBoard, and tracing",
    )
    args = parser.parse_args()
    if args.replay_batch and args.replay_cpu_batch:
        parser.error("--replay-batch and --replay-cpu-batch are mutually exclusive")
    if args.async_batch_preparation and not args.sample_pipeline:
        parser.error("--async-batch-preparation requires --sample-pipeline")
    if args.gpu_value_augmentation and not args.async_batch_preparation:
        parser.error("--gpu-value-augmentation requires --async-batch-preparation")
    if args.async_batch_preparation and (args.replay_batch or args.replay_cpu_batch):
        parser.error("--async-batch-preparation is incompatible with replay modes")
    if args.steps < 30 or args.trace_steps < 0:
        parser.error("steps must be >=30 and trace-steps nonnegative")
    if args.output_stride <= 0:
        parser.error("output-stride must be > 0")
    if args.batch_size_per_gpu <= 0:
        parser.error("batch-size-per-gpu must be > 0")
    if args.timeout_seconds <= 0.0:
        parser.error("timeout-seconds must be > 0")
    config = json.loads(args.reference.read_text())["config"]
    if config.get("batch_size") != 8:
        parser.error("this reference experiment requires batch_size=8 per rank")
    if config.get("resume") or config.get("training", {}).get("resume"):
        parser.error("reference experiment must start fresh, not resume")
    config.pop("_config_dir", None)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    suffix = (
        "_replay"
        if args.replay_batch
        else "_cpu_replay"
        if args.replay_cpu_batch
        else "_loader"
        if args.profile_loader
        else ""
    )
    suffix += f"_{args.normalization.replace('-', '_')}"
    suffix += f"_os{args.output_stride}"
    suffix += f"_b{args.batch_size_per_gpu}"
    if args.sample_pipeline:
        suffix += "_sample_pipeline"
    if args.async_batch_preparation:
        suffix += "_async_batch"
    if args.gpu_value_augmentation:
        suffix += f"_gpu_{args.gpu_value_augmentation.replace('-', '_')}"
    experiment = args.output.resolve() / f"profile_2gpu_{args.compile_mode}{suffix}_{stamp}"
    experiment.mkdir(parents=True, exist_ok=False)
    training = config["training"]
    config["batch_size"] = int(args.batch_size_per_gpu)
    model_config = config.setdefault("model_3d", {})
    model_config["normalization"] = "group" if args.normalization == "group" else "batch"
    model_config["output_stride"] = int(args.output_stride)
    training["sync_batchnorm"] = args.normalization == "sync-batch"
    training["ddp_broadcast_buffers"] = False
    training.update(run_path=str(experiment), run_name="s1a_128_sd1_2x8",
                    max_steps=args.steps, compile_mode=args.compile_mode,
                    initialization_seed=0, profile_enabled=True,
                    profile_loader=args.profile_loader,
                    profile_replay_batch=args.replay_batch,
                    profile_replay_cpu_batch=args.replay_cpu_batch,
                    profile_trace_start=21, profile_trace_steps=args.trace_steps,
                    scalar_log_interval=1)
    if args.sample_pipeline:
        slots_per_rank = (
            (256 + int(args.batch_size_per_gpu) - 1)
            // int(args.batch_size_per_gpu)
            * int(args.batch_size_per_gpu)
        )
        training.update(
            sample_pipeline_workers=128,
            sample_pipeline_load_affinity_workers=(
                87
            ),
            sample_pipeline_slots=2 * slots_per_rank,
            sample_pipeline_initial_in_flight=2 * int(args.batch_size_per_gpu),
        )
    if args.async_batch_preparation:
        training["sample_pipeline_cuda_prefetch_depth"] = 2
    if args.gpu_value_augmentation:
        training["sample_pipeline_cuda_value_augmentation"] = str(
            args.gpu_value_augmentation
        )
    throughput_only = bool(
        args.throughput_only or args.replay_batch or args.replay_cpu_batch
    )
    if throughput_only:
        training.update(
            profile_throughput_start_step=31,
            profile_trace_steps=0,
            scalar_log_interval=args.steps + 1,
            tensorboard_enabled=False,
            sample_vis_interval=0,
            test_trace2cp_enabled=False,
            test_interval=args.steps + 1,
            checkpoint_interval=args.steps + 1,
            kept_snapshot_interval=0,
        )
        config.pop("test_datasets", None)
    config_path = experiment / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="0,1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", AGENTS_AGENT_MODE="1", PYTHONUNBUFFERED="1",
               TORCHINDUCTOR_COMPILE_THREADS="4",
               TORCHINDUCTOR_CACHE_DIR=str(args.output.resolve() / "inductor_cache"),
               TRITON_CACHE_DIR=str(args.output.resolve() / "triton_cache"))
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               "--nproc_per_node=2", "-m", "vesuvius.neural_tracing.fiber_trace_3d.train",
               str(config_path)]
    metadata = {"command": command, "start_unix": time.time(),
                "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
                "reference": str(args.reference.resolve()),
                "batch_size_per_gpu": int(args.batch_size_per_gpu),
                "global_batch": 2 * int(args.batch_size_per_gpu),
                "sample_pipeline_slots_per_rank": (
                    slots_per_rank if args.sample_pipeline else None
                ),
                "compile_mode": args.compile_mode, "trace_steps": args.trace_steps,
                "normalization": args.normalization,
                "output_stride": int(args.output_stride),
                "profile_loader": args.profile_loader,
                "sample_pipeline": args.sample_pipeline,
                "async_batch_preparation": args.async_batch_preparation,
                "gpu_value_augmentation": args.gpu_value_augmentation,
                "replay_batch": args.replay_batch,
                "replay_cpu_batch": args.replay_cpu_batch,
                "throughput_only": throughput_only,
                "timeout_seconds": float(args.timeout_seconds)}
    metadata["python"] = sys.version
    metadata["packages"] = {name: importlib.metadata.version(name)
                            for name in ("torch", "numpy", "tensorboard", "tensorstore")}
    metadata["source_sha256"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("train.py", "training_profile.py", "profile_experiment.py")
    }
    (experiment / "launch.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"experiment={experiment}", flush=True)
    timed_out = False
    with (experiment / "run.log").open("w") as log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=float(args.timeout_seconds))
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                returncode = process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait()
    log_text = (experiment / "run.log").read_text(errors="replace")
    failure_kind = (
        "timeout"
        if timed_out
        else "cuda_oom"
        if "CUDA out of memory" in log_text or "torch.OutOfMemoryError" in log_text
        else None if returncode == 0 else "error"
    )
    metadata.update(
        end_unix=time.time(),
        returncode=returncode,
        failure_kind=failure_kind,
    )
    (experiment / "launch.json").write_text(json.dumps(metadata, indent=2) + "\n")
    summarize_experiment(experiment)
    print(f"completed returncode={returncode} failure_kind={failure_kind} elapsed={metadata['end_unix']-metadata['start_unix']:.1f}s", flush=True)
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
