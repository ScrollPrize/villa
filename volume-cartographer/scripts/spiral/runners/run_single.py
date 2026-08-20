#!/usr/bin/env python3
"""Fit, render, and score one Spiral dataset sequentially."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
from numbers import Real
from statistics import fmean, pstdev
import subprocess
import sys
from typing import Sequence
import uuid


SPIRAL_DIR = Path(__file__).resolve().parent.parent
VC_ROOT = SPIRAL_DIR.parent.parent
DEFAULT_VC_RENDER_BIN = VC_ROOT / "build" / "bin" / "vc_render_tifxyz"
DEFAULT_OUTPUT = SPIRAL_DIR / "out"
DEFAULT_RUN_CONFIG = Path(__file__).with_name("default_run_config.json")
WANDB_CONFIG_KEYS = ("wandb_project", "wandb_entity")
TRAINING_HISTORY_FILENAME = "training_metrics.jsonl"
AGGREGATE_METRICS_FILENAME = "aggregate_metrics.json"
_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")

# Executing this file directly puts only runners/ on sys.path.
if str(SPIRAL_DIR) not in sys.path:
    sys.path.insert(0, str(SPIRAL_DIR))

from config import Config  # noqa: E402


def positive_int(value: str) -> int:
    """Argparse type for a strictly positive integer."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_gpu_ids(value: str) -> tuple[int, ...]:
    """Parse a comma-separated list of distinct physical CUDA device ids."""
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise argparse.ArgumentTypeError(
            "must be a comma-separated list such as 0 or 0,1,2,3")
    try:
        gpu_ids = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "entries must be non-negative integer device indices") from exc
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise argparse.ArgumentTypeError(
            "entries must be non-negative integer device indices")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise argparse.ArgumentTypeError("cannot contain duplicate devices")
    return gpu_ids


def parse_seeds(value: str) -> list[int]:
    """Parse a comma-separated list of distinct, non-negative seeds."""
    parts = value.split(",")
    if not parts or any(not part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            "must be a comma-separated list of non-negative integers")
    seeds = []
    for part in parts:
        token = part.strip()
        if not token.isdecimal():
            raise argparse.ArgumentTypeError(
                "must be a comma-separated list of non-negative integers")
        seed = int(token)
        if seed in seeds:
            raise argparse.ArgumentTypeError(f"duplicate seed: {seed}")
        seeds.append(seed)
    return seeds


def run_id(value: str) -> str:
    """Argparse type for path-safe W&B run and group identifiers."""
    if not _RUN_ID_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "must start with an alphanumeric character and contain only "
            "letters, digits, '.', '_', or '-'")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"fit output root (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--ink-volume", required=True, type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON object overlaid on the default run configuration",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="disable Weights & Biases logging (enabled by default)",
    )
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        help="comma-separated distinct non-negative optimizer seeds",
    )
    parser.add_argument(
        "--run-id",
        type=run_id,
        help="path-safe W&B batch ID (generated when --seeds is supplied)",
    )
    parser.add_argument(
        "--num-threads",
        type=positive_int,
        help="positive per-run CPU thread budget",
    )
    parser.add_argument(
        "--gpus",
        type=parse_gpu_ids,
        metavar="DEVICE[,DEVICE...]",
        help="physical CUDA devices to use for the entire pipeline; multiple "
             "devices launch one distributed fit rank per device",
    )
    parser.add_argument(
        "--vc-render-bin",
        type=Path,
        default=DEFAULT_VC_RENDER_BIN,
        help=f"vc_render_tifxyz binary (default: {DEFAULT_VC_RENDER_BIN})",
    )
    return parser


def _load_json_object(path: Path, *, description: str) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {description} from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return value


def load_run_config(path: Path | None) -> tuple[dict, str, str]:
    """Load defaults, overlay an optional user config, and validate all keys."""
    merged = _load_json_object(
        DEFAULT_RUN_CONFIG, description="default run config")
    if path is not None:
        merged.update(_load_json_object(path, description="run config"))

    wandb_values = {}
    for key in WANDB_CONFIG_KEYS:
        value = merged.pop(key, None)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a non-empty string")
        wandb_values[key] = value

    # Constructing Config rejects unknown keys, wrong types, and invalid
    # values. Keep only the supplied overrides rather than expanding defaults;
    # fit_spiral owns resolution of the complete fit configuration.
    Config(merged)
    return merged, wandb_values["wandb_project"], wandb_values["wandb_entity"]


def require_empty_output(output: Path) -> None:
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"output exists and is not a directory: {output}")
        if any(output.iterdir()):
            raise ValueError(f"output directory must be empty: {output}")


def _native_thread_env(num_threads: int) -> dict[str, str]:
    value = str(num_threads)
    return {
        "OMP_NUM_THREADS": value,
        "OPENBLAS_NUM_THREADS": value,
        "MKL_NUM_THREADS": value,
        "NUMEXPR_NUM_THREADS": value,
    }


def _set_gpu_visibility(
    env: dict[str, str], gpu_ids: tuple[int, ...] | None
) -> None:
    if gpu_ids is not None:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)


def fit_environment(
    overrides: dict,
    output: Path,
    num_threads: int | None,
    *,
    wandb_project: str,
    wandb_entity: str,
    wandb_enabled: bool,
    wandb_run_id: str | None = None,
    wandb_run_name: str | None = None,
    wandb_group: str | None = None,
    metrics_history: Path | None = None,
    gpu_ids: tuple[int, ...] | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    _set_gpu_visibility(env, gpu_ids)
    # The runner owns these controls; do not inherit an unrelated outer fit.
    env.pop("FIT_SPIRAL_CONFIG_OVERRIDES", None)
    env["FIT_SPIRAL_OUT_DIR"] = str(output)
    env["WANDB_MODE"] = "online" if wandb_enabled else "disabled"
    env["WANDB_PROJECT"] = wandb_project
    env["WANDB_ENTITY"] = wandb_entity
    env.pop("FIT_SPIRAL_BATCH_RUN", None)
    env.pop("FIT_SPIRAL_METRICS_HISTORY", None)
    if wandb_run_id is not None:
        env["FIT_SPIRAL_BATCH_RUN"] = "1"
        env["WANDB_RUN_ID"] = wandb_run_id
        if wandb_run_name is not None:
            env["WANDB_NAME"] = wandb_run_name
        if wandb_group is not None:
            env["WANDB_RUN_GROUP"] = wandb_group
    if metrics_history is not None:
        env["FIT_SPIRAL_METRICS_HISTORY"] = str(metrics_history)
    if overrides:
        env["FIT_SPIRAL_CONFIG_OVERRIDES"] = json.dumps(overrides)
    if num_threads is not None:
        env.update(_native_thread_env(num_threads))
        env["FIT_SPIRAL_NUM_THREADS"] = str(num_threads)
        env["FIT_SPIRAL_PATCH_LOAD_WORKERS"] = str(num_threads)
        env["FIT_SPIRAL_PATCH_LOAD_IO_THREADS"] = "1"
    return env


def downstream_environment(
    num_threads: int | None,
    *,
    metrics: bool = False,
    gpu_ids: tuple[int, ...] | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    _set_gpu_visibility(env, gpu_ids)
    if num_threads is not None:
        env.update(_native_thread_env(1 if metrics else num_threads))
    return env


def fit_command(dataset: Path, gpu_ids: tuple[int, ...] | None) -> list[str]:
    script_args = [
        str(SPIRAL_DIR / "fit_spiral.py"),
        "--dataset",
        str(dataset),
    ]
    if gpu_ids is not None and len(gpu_ids) > 1:
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={len(gpu_ids)}",
            *script_args,
        ]
    return [sys.executable, *script_args]


def find_fit_outputs(output: Path) -> tuple[Path, Path]:
    run_dirs = (
        sorted(path for path in output.iterdir() if path.is_dir())
        if output.is_dir()
        else []
    )
    if len(run_dirs) != 1:
        raise RuntimeError(
            f"expected exactly one generated run directory in {output}, found {len(run_dirs)}"
        )
    run_dir = run_dirs[0]
    meshes_root = run_dir / "meshes"
    fitted_dirs = sorted(
        path for path in meshes_root.glob("fitted*") if path.is_dir()
    ) if meshes_root.is_dir() else []
    if len(fitted_dirs) != 1:
        raise RuntimeError(
            f"expected exactly one meshes/fitted* directory in {run_dir}, "
            f"found {len(fitted_dirs)}"
        )
    return run_dir, fitted_dirs[0]


def _load_training_history(path: Path) -> list[dict]:
    records = []
    try:
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if (not isinstance(record, dict)
                        or not isinstance(record.get("iteration"), int)
                        or not isinstance(record.get("metrics"), dict)):
                    raise ValueError(f"invalid record on line {line_number}")
                records.append(record)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"could not read training metrics from {path}: {exc}") from exc
    return records


def _load_final_summary(fitted_dir: Path) -> dict:
    metrics_path = fitted_dir / "ink_metric" / "metrics.json"
    metrics = _load_json_object(metrics_path, description="ink metrics")
    summary = metrics.get("summary")
    if not isinstance(summary, dict):
        raise RuntimeError(f"ink metrics summary must be a JSON object: {metrics_path}")
    return summary


def _numeric(value) -> bool:
    return (isinstance(value, Real) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _stats(values: list[Real]) -> dict[str, float | int]:
    floats = [float(value) for value in values]
    return {
        "mean": fmean(floats),
        "stddev": pstdev(floats),
        "count": len(floats),
    }


def aggregate_metrics(
    histories: list[list[dict]], final_summaries: list[dict]
) -> tuple[list[dict], dict[str, dict]]:
    """Aggregate numeric metrics, aligning training records by iteration."""
    by_seed = []
    iterations = set()
    for history in histories:
        indexed = {record["iteration"]: record["metrics"] for record in history}
        by_seed.append(indexed)
        iterations.update(indexed)

    training = []
    for iteration in sorted(iterations):
        keys = set()
        for indexed in by_seed:
            keys.update(indexed.get(iteration, {}))
        metrics = {}
        for key in sorted(keys):
            values = [indexed[iteration][key] for indexed in by_seed
                      if iteration in indexed
                      and key in indexed[iteration]
                      and _numeric(indexed[iteration][key])]
            if values:
                metrics[key] = _stats(values)
        if metrics:
            training.append({"iteration": iteration, "metrics": metrics})

    final = {}
    final_keys = set().union(*(summary.keys() for summary in final_summaries))
    for key in sorted(final_keys):
        values = [summary[key] for summary in final_summaries
                  if key in summary and _numeric(summary[key])]
        if values:
            final[key] = _stats(values)
    return training, final


def _wandb_init(*, project: str, entity: str, run_id: str, name: str,
                group: str, resume: str):
    import wandb
    return wandb.init(
        project=project,
        entity=entity,
        id=run_id,
        name=name,
        group=group,
        resume=resume,
    )


def log_seed_final_metrics(
    summary: dict, *, project: str, entity: str, seed_run_id: str, group: str
) -> None:
    run = _wandb_init(
        project=project, entity=entity, run_id=seed_run_id,
        name=seed_run_id, group=group, resume="must")
    try:
        run.log({f"final/{key}": value for key, value in summary.items()
                 if _numeric(value)})
    finally:
        run.finish()


def log_aggregate_metrics(
    training: list[dict], final: dict[str, dict], *, seed_count: int,
    project: str, entity: str, aggregate_run_id: str, group: str
) -> None:
    run = _wandb_init(
        project=project, entity=entity, run_id=aggregate_run_id,
        name=aggregate_run_id, group=group, resume="never")
    try:
        for record in training:
            complete = {
                key: stats["mean"] for key, stats in record["metrics"].items()
                if stats["count"] == seed_count
            }
            if complete:
                run.log(complete, step=record["iteration"])
        complete_final = {
            f"final/{key}": stats["mean"] for key, stats in final.items()
            if stats["count"] == seed_count
        }
        if complete_final:
            run.log(complete_final)
    finally:
        run.finish()


def run_pipeline(
    args: argparse.Namespace, *, output: Path, overrides: dict,
    wandb_project: str, wandb_entity: str,
    seed_run_id: str | None = None, wandb_group: str | None = None,
) -> tuple[list[dict], dict]:
    """Run one fit/render/score pipeline and return seeded-run metrics."""
    seeded = seed_run_id is not None
    history_path = output / TRAINING_HISTORY_FILENAME if seeded else None
    if seeded:
        output.mkdir(parents=True, exist_ok=False)

    gpu_ids = getattr(args, "gpus", None)
    fit_cmd = fit_command(args.dataset, gpu_ids)
    subprocess.run(
        fit_cmd,
        check=True,
        env=fit_environment(
            overrides,
            output,
            args.num_threads,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_enabled=not args.no_wandb,
            wandb_run_id=seed_run_id,
            wandb_run_name=seed_run_id,
            wandb_group=wandb_group,
            metrics_history=history_path,
            gpu_ids=gpu_ids,
        ),
    )

    _run_dir, fitted_dir = find_fit_outputs(output)
    render_cmd = [
        sys.executable,
        str(SPIRAL_DIR / "render_ink.py"),
        str(fitted_dir),
        "--volume",
        str(args.ink_volume),
        "--vc-render-bin",
        str(args.vc_render_bin),
    ]
    if args.num_threads is not None:
        render_cmd.extend([
            "--flatboi-threads", str(args.num_threads),
            "--num-processes", "1",
        ])
    subprocess.run(
        render_cmd,
        check=True,
        env=downstream_environment(args.num_threads, gpu_ids=gpu_ids),
    )

    metrics_cmd = [
        sys.executable,
        str(SPIRAL_DIR / "get_ink_metrics.py"),
        str(fitted_dir / "ink"),
    ]
    if args.num_threads is not None:
        metrics_cmd.extend(["--procs", str(max(1, args.num_threads // 3))])
    subprocess.run(
        metrics_cmd,
        check=True,
        env=downstream_environment(
            args.num_threads, metrics=True, gpu_ids=gpu_ids),
    )
    if not seeded:
        return [], {}
    return _load_training_history(history_path), _load_final_summary(fitted_dir)


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    require_empty_output(output)
    overrides, wandb_project, wandb_entity = load_run_config(args.config)
    seeds = getattr(args, "seeds", None)
    caller_run_id = getattr(args, "run_id", None)

    if seeds is None:
        if caller_run_id is not None:
            raise ValueError("--run-id requires --seeds")
        run_pipeline(
            args, output=output, overrides=overrides,
            wandb_project=wandb_project, wandb_entity=wandb_entity)
        return

    batch_id = caller_run_id or uuid.uuid4().hex[:8]
    histories = []
    summaries = []
    for seed in seeds:
        seed_overrides = dict(overrides)
        seed_overrides["optimizer_random_seed"] = seed
        seed_id = f"{batch_id}_seed_{seed}"
        history, summary = run_pipeline(
            args,
            output=output / f"seed-{seed}",
            overrides=seed_overrides,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            seed_run_id=seed_id,
            wandb_group=batch_id,
        )
        histories.append(history)
        summaries.append(summary)
        if not args.no_wandb:
            log_seed_final_metrics(
                summary, project=wandb_project, entity=wandb_entity,
                seed_run_id=seed_id, group=batch_id)

    if len(seeds) < 2:
        return
    training, final = aggregate_metrics(histories, summaries)
    aggregate = {
        "run_id": batch_id,
        "seeds": seeds,
        "training": training,
        "final": final,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / AGGREGATE_METRICS_FILENAME).write_text(
        json.dumps(aggregate, indent=2) + "\n")
    if not args.no_wandb:
        aggregate_id = f"{batch_id}_aggregate"
        log_aggregate_metrics(
            training, final, seed_count=len(seeds),
            project=wandb_project, entity=wandb_entity,
            aggregate_run_id=aggregate_id, group=batch_id)


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
