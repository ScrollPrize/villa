#!/usr/bin/env python3
"""Compare two vc_extend_edge binaries on real input; retain reports and verify XYZ bytes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("before", "after", "input", "output", "config-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--volume", required=True)
    parser.add_argument("--integrator", choices=("midpoint", "heun", "rk4"), default="rk4")
    parser.add_argument("--cells", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--normal-step", type=float, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--build-type", default="RelWithDebInfo")
    args = parser.parse_args()
    if min(args.iterations, args.cells, args.threads) < 1:
        parser.error("iterations, cells, and threads must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, VC3D_CONFIG_DIR=str(args.config_dir.resolve()),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", AGENTS_AGENT_MODE="1")
    common = ["-i", str(args.input.resolve()), "-n", args.volume, "--sides", "right",
              "--cells", str(args.cells), "--rounds", "1", "--threads", str(args.threads),
              "--normal-step", str(args.normal_step), "--walk-mode", "streamline",
              "--integrator", args.integrator]
    times = {label: [] for label in ("before", "after")}
    commands = {}
    # First pair warms the disk chunk cache; each process starts with empty normal caches.
    for iteration in range(-1, args.iterations):
        hashes = {}
        order = ("before", "after") if iteration % 2 == 0 else ("after", "before")
        for label in order:
            output = args.output / f"{label}-{iteration}"
            report = output.with_suffix(".json")
            command = [str(getattr(args, label).resolve()), *common,
                       "-o", str(output.resolve()), "--report", str(report.resolve())]
            commands[label] = command
            result = subprocess.run(command, env=env, text=True, capture_output=True)
            if result.returncode:
                raise RuntimeError(f"{command!r}\n{result.stdout}\n{result.stderr}")
            seconds = json.loads(report.read_text())["seconds"]
            hashes[label] = [hashlib.sha256((output / f"{axis}.tif").read_bytes()).hexdigest()
                             for axis in "xyz"]
            if iteration >= 0:
                times[label].append(seconds)
            print(label, iteration, seconds, flush=True)
        if hashes["before"] != hashes["after"]:
            raise RuntimeError(f"XYZ TIFF bytes differ at iteration {iteration}")
    summary = {"commands": commands, "build_type": args.build_type,
               "iterations": args.iterations, "xyz_byte_identical": True,
               "input_sha256": {axis: hashlib.sha256((args.input / f"{axis}.tif").read_bytes()).hexdigest()
                                for axis in "xyz"},
               "environment": {key: env[key] for key in ("VC3D_CONFIG_DIR", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}}
    for label, values in times.items():
        summary[label] = dict(seconds=values, mean=statistics.mean(values),
                              minimum=min(values), median=statistics.median(values), maximum=max(values))
    summary["speedup"] = summary["before"]["mean"] / summary["after"]["mean"]
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
