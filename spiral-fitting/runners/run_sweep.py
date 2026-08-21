#!/usr/bin/env python3
"""Run a folder of Spiral configurations concurrently on fixed GPU groups."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Sequence, TextIO

# Direct execution puts runners/ rather than the Spiral directory on sys.path.
SPIRAL_DIR = Path(__file__).resolve().parent.parent
if str(SPIRAL_DIR) not in sys.path:
    sys.path.insert(0, str(SPIRAL_DIR))

from runners import run_single  # noqa: E402
from config import Config  # noqa: E402


_STEM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_LIVE_OUTPUT_RE = re.compile(r"(?:PROGRESS |step \d+: loss = )")
DEFAULT_NUM_THREADS = 6


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    run_single.add_common_arguments(parser, include_gpus=False)
    parser.set_defaults(num_threads=DEFAULT_NUM_THREADS)
    parser.add_argument("--config-folder", required=True, type=Path)
    parser.add_argument("--sweep-config", required=True, type=Path)
    parser.add_argument(
        "--gpus-per-run", type=run_single.positive_int, default=1,
        help="number of GPUs assigned to each run (default: 1)")
    return parser


def discover_configs(folder: Path, sweep_config: Path) -> list[Path]:
    if not folder.is_dir():
        raise ValueError(f"config folder is not a directory: {folder}")
    excluded = sweep_config.resolve()
    configs = sorted(
        (path for path in folder.glob("*.json")
         if path.is_file() and path.resolve() != excluded),
        key=lambda path: path.name,
    )
    if not configs:
        raise ValueError(f"no run configuration JSON files found in {folder}")
    stems = set()
    for path in configs:
        stem = path.stem
        if not _STEM_RE.fullmatch(stem):
            raise ValueError(f"configuration filename has unsafe stem: {path.name}")
        if stem in stems:
            raise ValueError(f"duplicate configuration stem: {stem}")
        stems.add(stem)
    return configs


def _validate_effective_config(value: dict, source: Path) -> None:
    candidate = dict(value)
    for key in run_single.WANDB_CONFIG_KEYS:
        setting = candidate.pop(key, None)
        if not isinstance(setting, str) or not setting.strip():
            raise ValueError(f"{source}: {key} must be a non-empty string")
    try:
        Config(candidate)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: invalid run config: {exc}") from exc


def prepare_configs(
    configs: list[Path], sweep_config: Path, materialized_dir: Path
) -> tuple[list[tuple[str, Path]], dict[str, str]]:
    defaults = run_single._load_json_object(
        run_single.DEFAULT_RUN_CONFIG, description="default run config")
    shared = run_single._load_json_object(
        sweep_config, description="sweep config")
    base = {**defaults, **shared}
    valid = []
    failures = {}
    materialized_dir.mkdir(parents=True, exist_ok=True)
    for path in configs:
        try:
            individual = run_single._load_json_object(path, description="run config")
            effective = {**base, **individual}
            _validate_effective_config(effective, path)
            destination = materialized_dir / f"{path.stem}.json"
            destination.write_text(
                json.dumps(effective, indent=2, sort_keys=True) + "\n")
            valid.append((path.stem, destination))
        except (OSError, ValueError) as exc:
            failures[path.stem] = str(exc)
    return valid, failures


def visible_gpu_ids() -> tuple[int, ...]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        try:
            return run_single.parse_gpu_ids(visible)
        except argparse.ArgumentTypeError as exc:
            raise ValueError(f"invalid CUDA_VISIBLE_DEVICES: {exc}") from exc
    try:
        import torch
        count = torch.cuda.device_count()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(f"could not discover CUDA devices: {exc}") from exc
    return tuple(range(count))


def partition_gpus(
    gpu_ids: tuple[int, ...], gpus_per_run: int
) -> tuple[list[tuple[int, ...]], tuple[int, ...]]:
    complete = len(gpu_ids) // gpus_per_run
    groups = [
        gpu_ids[index * gpus_per_run:(index + 1) * gpus_per_run]
        for index in range(complete)
    ]
    idle = gpu_ids[complete * gpus_per_run:]
    if not groups:
        raise ValueError(
            f"need {gpus_per_run} GPUs per run, but only {len(gpu_ids)} are visible")
    return groups, idle


def child_command(
    args: argparse.Namespace, stem: str, config: Path, group: tuple[int, ...]
) -> list[str]:
    command = [
        sys.executable, str(Path(__file__).with_name("run_single.py")),
        "--dataset", str(args.dataset), "--ink-volume", str(args.ink_volume),
        "--output", str(args.output.resolve() / stem),
        "--config", str(config), "--resume", "--gpus",
        ",".join(map(str, group)), "--vc-render-bin", str(args.vc_render_bin),
    ]
    if args.seeds is not None:
        command.extend(["--seeds", ",".join(map(str, args.seeds))])
    if args.run_id is not None:
        command.extend(["--run-id", f"{args.run_id}_{stem}"])
    wandb_group = getattr(args, "wandb_group", None)
    if wandb_group is not None:
        command.extend(["--wandb-group", wandb_group])
    if args.num_threads is not None:
        command.extend(["--num-threads", str(args.num_threads)])
    if args.no_wandb:
        command.append("--no-wandb")
    return command


def _relay_child_output(
    stream: TextIO,
    log: TextIO,
    stem: str,
    console: TextIO,
    console_lock: threading.Lock,
) -> None:
    """Copy a child's complete output to its log and selected lines live."""
    try:
        for line in stream:
            log.write(line)
            log.flush()
            if _LIVE_OUTPUT_RE.match(line):
                with console_lock:
                    console.write(f"[{stem}] {line}")
                    console.flush()
    finally:
        stream.close()


def _start_child(
    command: list[str],
    log: TextIO,
    stem: str,
    console: TextIO,
    console_lock: threading.Lock,
) -> tuple[subprocess.Popen[str], threading.Thread]:
    env = os.environ.copy()
    # run_single passes its environment to fit_spiral, so this also makes the
    # fitter's stdout loss records observable immediately through the pipe.
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    if process.stdout is None:  # pragma: no cover - guaranteed by PIPE
        raise RuntimeError("child stdout pipe was not created")
    relay = threading.Thread(
        target=_relay_child_output,
        args=(process.stdout, log, stem, console, console_lock),
        name=f"sweep-output-{stem}",
        daemon=True,
    )
    relay.start()
    return process, relay


def execute(args: argparse.Namespace) -> int:
    configs = discover_configs(args.config_folder, args.sweep_config)
    output = args.output.resolve()
    if getattr(args, "overwrite", False):
        run_single.overwrite_output(
            args.output,
            protected_paths=(
                args.dataset,
                args.ink_volume,
                args.vc_render_bin,
                args.config_folder,
                args.sweep_config,
            ),
        )
    sweep_dir = output / ".sweep"
    valid, failures = prepare_configs(configs, args.sweep_config, sweep_dir)
    groups, idle = partition_gpus(visible_gpu_ids(), args.gpus_per_run)
    if idle:
        print("Idle GPUs (incomplete group): " + ",".join(map(str, idle)), flush=True)

    logs_dir = sweep_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    pending = list(valid)
    free_groups = list(groups)
    console = sys.stdout
    console_lock = threading.Lock()
    active: dict[
        subprocess.Popen[str],
        tuple[str, tuple[int, ...], TextIO, threading.Thread],
    ] = {}
    results: dict[str, str] = {stem: f"FAILED: {message}"
                               for stem, message in failures.items()}
    attempts: dict[str, int] = {}
    try:
        while pending or active:
            while pending and free_groups:
                stem, config = pending.pop(0)
                group = free_groups.pop(0)
                attempts[stem] = attempts.get(stem, 0) + 1
                log_path = logs_dir / f"{stem}.log"
                log = log_path.open("a")
                log.write(f"\n=== attempt {attempts[stem]} GPUs {','.join(map(str, group))} ===\n")
                log.flush()
                command = child_command(args, stem, config, group)
                process, relay = _start_child(
                    command, log, stem, console, console_lock)
                active[process] = (stem, group, log, relay)
                print(f"LAUNCH {stem} GPUs={','.join(map(str, group))}", flush=True)

            completed = [process for process in active if process.poll() is not None]
            if not completed:
                time.sleep(0.05)
                continue
            for process in completed:
                stem, group, log, relay = active.pop(process)
                relay.join()
                log.close()
                free_groups.append(group)
                free_groups.sort(key=lambda item: groups.index(item))
                if process.returncode == 0:
                    results[stem] = "SUCCESS"
                    print(f"SUCCESS {stem}", flush=True)
                else:
                    results[stem] = f"FAILED: exit {process.returncode}"
                    print(f"FAILED {stem} exit={process.returncode}", flush=True)
    except KeyboardInterrupt:
        print("Interrupted; terminating active runs", file=sys.stderr, flush=True)
        for process in active:
            process.terminate()
        for process, (stem, _group, log, relay) in list(active.items()):
            process.wait()
            relay.join()
            log.close()
            results[stem] = "FAILED: interrupted"
        raise

    print("Sweep summary:")
    for path in configs:
        print(f"  {path.stem}: {results[path.stem]}")
    return 1 if any(value != "SUCCESS" for value in results.values()) else 0


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        status = execute(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    raise SystemExit(status)


if __name__ == "__main__":
    main()
