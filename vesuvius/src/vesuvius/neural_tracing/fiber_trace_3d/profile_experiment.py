"""Launch an isolated two-GPU reference experiment using this interpreter."""
import argparse
import hashlib
import importlib.metadata
from datetime import datetime, timezone
import json
import os
from pathlib import Path
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
            "steady": {key: distribution(r[key] for r in selected)
                       for key in ("iteration_wall_ms", "wait_ms", "to_device_ms",
                                   "target_ms", "fw_bw_step_ms", "forward_loss_wall_ms",
                                   "backward_wall_ms", "optimizer_wall_ms", "scalar_reduce_sync_wall_ms")},
            "step1": rows[0] if rows else None,
            "peak_allocated_bytes": max((r.get("peak_allocated_bytes", 0) for r in rows), default=0),
        }
    result = {"experiment": str(experiment), "ranks": ranks,
              "selection": "steps >30 excluding trace and multiples of100; loader stalls retained"}
    (experiment / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True,
                        help="Saved baseline JSON containing the authoritative config")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-mode", choices=["off", "default", "reduce-overhead"], default="off")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--trace-steps", type=int, default=5)
    parser.add_argument("--profile-loader", action="store_true")
    args = parser.parse_args()
    if args.steps < 30 or args.trace_steps < 0:
        parser.error("steps must be >=30 and trace-steps nonnegative")
    config = json.loads(args.reference.read_text())["config"]
    if config.get("batch_size") != 8:
        parser.error("this reference experiment requires batch_size=8 per rank")
    if config.get("resume") or config.get("training", {}).get("resume"):
        parser.error("reference experiment must start fresh, not resume")
    config.pop("_config_dir", None)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    suffix = "_loader" if args.profile_loader else ""
    experiment = args.output.resolve() / f"profile_2gpu_{args.compile_mode}{suffix}_{stamp}"
    experiment.mkdir(parents=True, exist_ok=False)
    training = config["training"]
    training.update(run_path=str(experiment), run_name="s1a_128_sd1_2x8",
                    max_steps=args.steps, compile_mode=args.compile_mode,
                    initialization_seed=0, profile_enabled=True,
                    profile_loader=args.profile_loader,
                    profile_trace_start=21, profile_trace_steps=args.trace_steps,
                    scalar_log_interval=1)
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
                "reference": str(args.reference.resolve()), "global_batch": 16,
                "compile_mode": args.compile_mode, "trace_steps": args.trace_steps,
                "profile_loader": args.profile_loader}
    metadata["python"] = sys.version
    metadata["packages"] = {name: importlib.metadata.version(name)
                            for name in ("torch", "numpy", "tensorboard", "tensorstore")}
    metadata["source_sha256"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("train.py", "training_profile.py", "profile_experiment.py")
    }
    (experiment / "launch.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"experiment={experiment}", flush=True)
    with (experiment / "run.log").open("w") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    metadata.update(end_unix=time.time(), returncode=result.returncode)
    (experiment / "launch.json").write_text(json.dumps(metadata, indent=2) + "\n")
    summarize_experiment(experiment)
    print(f"completed returncode={result.returncode} elapsed={metadata['end_unix']-metadata['start_unix']:.1f}s", flush=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
