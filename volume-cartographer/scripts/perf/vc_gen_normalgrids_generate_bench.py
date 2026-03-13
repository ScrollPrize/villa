#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark vc_gen_normalgrids generate")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--threads", default="1,8,16,32")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--sparse-volume", type=int, default=4)
    parser.add_argument("--preview-every", type=int, default=0)
    parser.add_argument("--verify-grid-save", action="store_true")
    parser.add_argument("--chunk-budget-mib", type=int, default=512)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--profiler", choices=["none", "strace", "pidstat"], default="none")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def parse_time_v(stderr_text: str):
    result = {}
    for line in stderr_text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "Percent of CPU this job got":
            result["cpu_percent"] = value
        elif key == "Maximum resident set size (kbytes)":
            result["max_rss_kb"] = int(value)
        elif key == "File system inputs":
            result["fs_inputs"] = int(value)
        elif key == "File system outputs":
            result["fs_outputs"] = int(value)
    return result


def summarize(values):
    return {
        "mean": statistics.mean(values),
        "p50": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def run_command(cmd, env, profiler):
    start = time.perf_counter()
    if profiler == "none":
        proc = subprocess.run(
            ["/usr/bin/time", "-v", *cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            env=env,
        )
        return {"stdout": proc.stdout, "stderr": proc.stderr, "elapsed_seconds": time.perf_counter() - start, "profiler": None}

    if profiler == "strace":
        proc = subprocess.run(
            ["/usr/bin/time", "-v", "strace", "-c", *cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            env=env,
        )
        return {"stdout": proc.stdout, "stderr": proc.stderr, "elapsed_seconds": time.perf_counter() - start, "profiler": "strace"}

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    pidstat = subprocess.Popen(
        ["pidstat", "-u", "-r", "-d", "-h", "1", "-p", str(proc.pid)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    stdout, stderr = proc.communicate()
    pidstat_stdout, pidstat_stderr = pidstat.communicate()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=stdout, stderr=stderr)
    return {
        "stdout": stdout,
        "stderr": stderr,
        "elapsed_seconds": time.perf_counter() - start,
        "profiler": "pidstat",
        "pidstat_stdout": pidstat_stdout,
        "pidstat_stderr": pidstat_stderr,
    }


def main():
    args = parse_args()
    build_dir = Path(args.build_dir)
    binary = build_dir / "bin" / "vc_gen_normalgrids"
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    thread_values = [int(x) for x in args.threads.split(",") if x]
    results = {
        "dataset": args.dataset,
        "build_dir": str(build_dir),
        "binary": str(binary),
        "iterations": args.iterations,
        "warmup": args.warmup,
        "sparse_volume": args.sparse_volume,
        "preview_every": args.preview_every,
        "verify_grid_save": args.verify_grid_save,
        "chunk_budget_mib": args.chunk_budget_mib,
        "level": args.level,
        "profiler": args.profiler,
        "threads": {},
    }

    for threads in thread_values:
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(threads)
        metrics_runs = []
        wall_runs = []
        rss_runs = []
        cpu_runs = []
        fs_inputs_runs = []
        fs_outputs_runs = []
        profiler_output = None

        total_runs = args.warmup + args.iterations
        for run_idx in range(total_runs):
            out_dir = Path(tempfile.mkdtemp(prefix=f"vc_gen_normalgrids_t{threads}_", dir=output_root))
            metrics_path = out_dir / "metrics.json"
            cmd = [
                str(binary),
                "generate",
                "-i", args.dataset,
                "-o", str(out_dir / "out"),
                "--sparse-volume", str(args.sparse_volume),
                "--preview-every", str(args.preview_every),
                "--chunk-budget-mib", str(args.chunk_budget_mib),
                "--level", str(args.level),
                "--metrics-json", str(metrics_path),
            ]
            if args.verify_grid_save:
                cmd.append("--verify-grid-save")

            current_profiler = args.profiler if run_idx == args.warmup else "none"
            run = run_command(cmd, env, current_profiler)
            time_stats = parse_time_v(run["stderr"])

            if current_profiler != "none":
                profiler_output = run

            if run_idx < args.warmup:
                shutil.rmtree(out_dir, ignore_errors=True)
                continue

            metrics_runs.append(json.loads(metrics_path.read_text()))
            wall_runs.append(run["elapsed_seconds"])
            rss_runs.append(time_stats.get("max_rss_kb", 0))
            cpu_runs.append(time_stats.get("cpu_percent", ""))
            fs_inputs_runs.append(time_stats.get("fs_inputs", 0))
            fs_outputs_runs.append(time_stats.get("fs_outputs", 0))
            shutil.rmtree(out_dir, ignore_errors=True)

        results["threads"][str(threads)] = {
            "wall_seconds": summarize(wall_runs),
            "max_rss_kb": summarize(rss_runs),
            "fs_inputs": summarize(fs_inputs_runs),
            "fs_outputs": summarize(fs_outputs_runs),
            "cpu_percent_samples": cpu_runs,
            "metrics_runs": metrics_runs,
            "profiler_output": profiler_output,
        }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
