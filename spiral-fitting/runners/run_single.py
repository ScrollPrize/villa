#!/usr/bin/env python3
"""Fit, render, and score one Spiral dataset sequentially."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
from numbers import Real
from statistics import fmean, pstdev
import subprocess
import sys
import time
from typing import Sequence
import uuid
import zipfile


SPIRAL_DIR = Path(__file__).resolve().parent.parent
VC_ROOT = SPIRAL_DIR.parent / "volume-cartographer"
DEFAULT_VC_RENDER_BIN = VC_ROOT / "build" / "bin" / "vc_render_tifxyz"
DEFAULT_OUTPUT = SPIRAL_DIR / "out"
DEFAULT_RUN_CONFIG = Path(__file__).with_name("default_run_config.json")
WANDB_CONFIG_KEYS = ("wandb_project", "wandb_entity")
TRAINING_HISTORY_FILENAME = "training_metrics.jsonl"
AGGREGATE_METRICS_FILENAME = "aggregate_metrics.json"
STATE_FILENAME = ".run_single_state.json"
STATE_VERSION = 1
_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_INK_STRIP_SUFFIX = ".jpg"
_WANDB_IN_USE_RETRY_DELAYS = (2.0, 5.0, 10.0, 20.0, 30.0)

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


def add_common_arguments(
    parser: argparse.ArgumentParser, *, include_gpus: bool = True
) -> None:
    """Register operational arguments shared by the single and sweep runners."""
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"fit output root (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="delete the existing output directory before starting",
    )
    parser.add_argument("--ink-volume", required=True, type=Path)
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="disable Weights & Biases logging (enabled by default)",
    )
    parser.add_argument(
        "--wandb-group",
        type=run_id,
        help="optional Weights & Biases run group",
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
    if include_gpus:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON object overlaid on the default run configuration",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume completed stages and interrupted fit checkpoints from a "
             "compatible saved run state",
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


def overwrite_output(output: Path, *, protected_paths: Sequence[Path] = ()) -> None:
    """Delete an explicitly selected output directory after safety checks."""
    if output.is_symlink():
        raise ValueError(f"refusing to overwrite symlinked output directory: {output}")
    output = output.resolve()
    if not output.exists():
        return
    if not output.is_dir():
        raise ValueError(f"output exists and is not a directory: {output}")

    forbidden = {
        Path(output.anchor),
        Path.home().resolve(),
        SPIRAL_DIR.resolve(),
        SPIRAL_DIR.parent.resolve(),
    }
    if output in forbidden:
        raise ValueError(f"refusing to overwrite unsafe output directory: {output}")

    for path in protected_paths:
        protected = path.resolve()
        if protected == output or output in protected.parents:
            raise ValueError(
                f"refusing to overwrite output directory containing an input: "
                f"{protected}")

    print(f"Removing existing output directory: {output}", flush=True)
    shutil.rmtree(output)


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
    resume_checkpoint: Path | None = None,
    run_directory: Path | None = None,
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
    env.pop("FIT_SPIRAL_RESUME_PATH", None)
    env.pop("FIT_SPIRAL_RUN_DIR", None)
    env.pop("FIT_SPIRAL_WANDB_RESUME", None)
    if wandb_run_id is not None:
        env["FIT_SPIRAL_BATCH_RUN"] = "1"
        env["WANDB_RUN_ID"] = wandb_run_id
        if wandb_run_name is not None:
            env["WANDB_NAME"] = wandb_run_name
    if wandb_group is not None:
        env["WANDB_RUN_GROUP"] = wandb_group
    if metrics_history is not None:
        env["FIT_SPIRAL_METRICS_HISTORY"] = str(metrics_history)
    if resume_checkpoint is not None:
        env["FIT_SPIRAL_RESUME_PATH"] = str(resume_checkpoint)
    if run_directory is not None:
        env["FIT_SPIRAL_RUN_DIR"] = str(run_directory)
    if resume_checkpoint is not None or run_directory is not None:
        # The run may already exist remotely even when interruption happened
        # before the first durable checkpoint was written.
        env["FIT_SPIRAL_WANDB_RESUME"] = "1"
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
                group: str | None = None, resume: str):
    import wandb
    return wandb.init(
        project=project,
        entity=entity,
        id=run_id,
        name=name,
        group=group,
        resume=resume,
    )


def _wandb_init_after_release(**kwargs):
    """Wait briefly for a just-finished training process to release its run."""
    run_id = kwargs["run_id"]
    for attempt in range(len(_WANDB_IN_USE_RETRY_DELAYS) + 1):
        try:
            return _wandb_init(**kwargs)
        except Exception as exc:
            in_use = "run ID " in str(exc) and " is in use" in str(exc)
            if not in_use or attempt == len(_WANDB_IN_USE_RETRY_DELAYS):
                raise
            delay = _WANDB_IN_USE_RETRY_DELAYS[attempt]
            print(
                f"W&B run {run_id} is still closing; retrying final metrics "
                f"in {delay:g}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)


def log_seed_final_metrics(
    summary: dict, *, project: str, entity: str, seed_run_id: str,
    group: str | None = None
) -> bool:
    try:
        run = _wandb_init_after_release(
            project=project, entity=entity, run_id=seed_run_id,
            name=seed_run_id, group=group, resume="must")
        try:
            run.log({f"final/{key}": value for key, value in summary.items()
                     if _numeric(value)})
        finally:
            run.finish()
    except Exception as exc:
        # Experiment tracking is an optional sink. The fit, render, and local
        # metrics are already durable, so an upload failure must not invalidate
        # the pipeline or prevent later seeds from running.
        print(
            f"WARNING: could not upload final W&B metrics for {seed_run_id}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return False
    return True


def _log_seed_final_metrics_once(
    summary: dict, *, metrics_stage: dict, state: dict, state_path: Path,
    project: str, entity: str, seed_run_id: str, group: str | None = None,
) -> None:
    if metrics_stage.get("wandb_final_logged"):
        return
    if log_seed_final_metrics(
        summary, project=project, entity=entity,
        seed_run_id=seed_run_id, group=group,
    ):
        metrics_stage["wandb_final_logged"] = True
        _atomic_write_json(state_path, state)


def log_aggregate_metrics(
    training: list[dict], final: dict[str, dict], *, seed_count: int,
    project: str, entity: str, aggregate_run_id: str, group: str | None = None
) -> bool:
    try:
        run = _wandb_init_after_release(
            project=project, entity=entity, run_id=aggregate_run_id,
            name=aggregate_run_id, group=group, resume="allow")
        try:
            for record in training:
                complete = {
                    key: stats["mean"]
                    for key, stats in record["metrics"].items()
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
    except Exception as exc:
        print(
            f"WARNING: could not upload aggregate W&B metrics for "
            f"{aggregate_run_id}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return False
    return True


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


def _atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _resume_invocation(
    args: argparse.Namespace, overrides: dict, project: str, entity: str
) -> dict:
    gpu_ids = getattr(args, "gpus", None)
    return {
        "effective_config": {
            **overrides,
            "wandb_project": project,
            "wandb_entity": entity,
        },
        "inputs": {
            "dataset": str(args.dataset.resolve()),
            "ink_volume": str(args.ink_volume.resolve()),
        },
        "operational_args": {
            "num_threads": args.num_threads,
            "no_wandb": args.no_wandb,
            "wandb_group": getattr(args, "wandb_group", None),
            "vc_render_bin": str(args.vc_render_bin.resolve()),
        },
        # Physical IDs may change between attempts; resource shape may not.
        "gpu_count": None if gpu_ids is None else len(gpu_ids),
        "seeds": getattr(args, "seeds", None),
        "requested_run_id": getattr(args, "run_id", None),
    }


def _new_resume_state(invocation: dict) -> dict:
    seeds = invocation["seeds"]
    keys = ["single"] if seeds is None else [str(seed) for seed in seeds]
    batch_id = invocation["requested_run_id"] or uuid.uuid4().hex[:8]
    return {
        "version": STATE_VERSION,
        "invocation": invocation,
        "effective_config": invocation["effective_config"],
        "resolved_inputs": invocation["inputs"],
        "operational_args": invocation["operational_args"],
        "gpu_count": invocation["gpu_count"],
        "seeds": seeds,
        "run_id": batch_id,
        "batch_id": batch_id if seeds is not None else None,
        "runs": {
            key: {
                "fit": {"status": "pending"},
                "render": {"status": "pending"},
                "metrics": {"status": "pending"},
            }
            for key in keys
        },
    }


def _state_artifact(output: Path, stage: dict, key: str) -> Path:
    value = stage.get(key)
    if not isinstance(value, str):
        raise RuntimeError(f"completed stage has no recorded {key}")
    candidate = (output / value).resolve()
    try:
        candidate.relative_to(output)
    except ValueError as exc:
        raise RuntimeError(f"state contains an unsafe artifact path: {value}") from exc
    return candidate


def _require_ink_output(ink: Path) -> None:
    if not ink.is_dir():
        raise RuntimeError(f"render completed without expected artifact: {ink}")
    if not any(
        path.is_file() and path.suffix.lower() == _INK_STRIP_SUFFIX
        for path in ink.iterdir()
    ):
        raise RuntimeError(f"render completed without ink strip images: {ink}")


def _validate_completed_stages(output: Path, state: dict) -> None:
    for label, stages in state["runs"].items():
        fit = stages["fit"]
        if fit.get("status") == "complete":
            fitted = _state_artifact(output, fit, "fitted_output")
            if not fitted.is_dir():
                raise RuntimeError(
                    f"completed fit artifact is missing for {label}: {fitted}")
            history_value = fit.get("training_history")
            if history_value is not None:
                history = _state_artifact(output, fit, "training_history")
                _load_training_history(history)
        render = stages["render"]
        if render.get("status") == "complete":
            ink = _state_artifact(output, render, "ink_output")
            try:
                _require_ink_output(ink)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"completed render artifact is invalid for {label}: {ink}"
                ) from exc
        metrics = stages["metrics"]
        if metrics.get("status") == "complete":
            metrics_path = _state_artifact(output, metrics, "metrics_output")
            _load_final_summary(metrics_path.parent.parent)


def _recover_interrupted_fit(output: Path) -> tuple[Path | None, Path | None]:
    """Locate the one durable checkpoint and its original output directory.

    Headless fits atomically refresh ``checkpoint_fitted.ckpt`` every 1,000
    completed iterations.  Reusing its parent directory avoids creating a
    second dated directory when recovery happens after midnight UTC.  A fit
    interrupted before its first autosave can still restart in its original
    directory, but no training progress is recoverable in that case.
    """
    run_dirs = (
        sorted(path for path in output.iterdir() if path.is_dir())
        if output.is_dir()
        else []
    )
    checkpoints = [
        run_dir / "checkpoint_fitted.ckpt"
        for run_dir in run_dirs
        if (run_dir / "checkpoint_fitted.ckpt").is_file()
    ]
    if len(checkpoints) > 1:
        raise RuntimeError(
            f"cannot choose a fit checkpoint in {output}: found "
            f"{len(checkpoints)} checkpoint_fitted.ckpt files")
    if checkpoints:
        checkpoint = checkpoints[0]
        if not zipfile.is_zipfile(checkpoint):
            raise RuntimeError(
                f"interrupted fit checkpoint is incomplete or corrupt: {checkpoint}")
        return checkpoint.resolve(), checkpoint.parent.resolve()
    if len(run_dirs) > 1:
        raise RuntimeError(
            f"cannot choose an interrupted fit directory in {output}: found "
            f"{len(run_dirs)} directories and no durable checkpoint")
    return None, run_dirs[0].resolve() if run_dirs else None


def _gpu_count_only_invocation_change(saved: dict, current: dict) -> bool:
    """Return whether two invocations differ only by explicit GPU count."""
    saved_gpu_count = saved.get("gpu_count")
    current_gpu_count = current.get("gpu_count")
    if (not isinstance(saved_gpu_count, int)
            or not isinstance(current_gpu_count, int)
            or saved_gpu_count == current_gpu_count):
        return False
    saved_without_gpus = dict(saved)
    current_without_gpus = dict(current)
    saved_without_gpus.pop("gpu_count", None)
    current_without_gpus.pop("gpu_count", None)
    return saved_without_gpus == current_without_gpus


def _load_or_create_state(output: Path, invocation: dict) -> tuple[dict, Path]:
    state_path = output / STATE_FILENAME
    if state_path.exists():
        try:
            state = _load_json_object(state_path, description="run state")
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        if state.get("version") != STATE_VERSION or not isinstance(state.get("runs"), dict):
            raise RuntimeError(f"unsupported or malformed run state: {state_path}")
        saved_invocation = state.get("invocation")
        if not isinstance(saved_invocation, dict):
            raise RuntimeError(f"run state has no valid invocation: {state_path}")
        gpu_count_change = (
            saved_invocation != invocation
            and _gpu_count_only_invocation_change(saved_invocation, invocation)
        )
        if saved_invocation != invocation and not gpu_count_change:
            raise RuntimeError("saved run state does not match the current invocation")
        seeds = invocation["seeds"]
        expected_labels = {"single"} if seeds is None else {
            str(seed) for seed in seeds}
        if set(state["runs"]) != expected_labels:
            raise RuntimeError(f"run state has unexpected pipeline entries: {state_path}")
        if not isinstance(state.get("run_id"), str):
            raise RuntimeError(f"run state has no valid run ID: {state_path}")
        if seeds is not None and state.get("batch_id") != state["run_id"]:
            raise RuntimeError(f"run state has no valid batch ID: {state_path}")
        for label, stages in state["runs"].items():
            if (not isinstance(stages, dict)
                    or any(not isinstance(stages.get(name), dict)
                           for name in ("fit", "render", "metrics"))):
                raise RuntimeError(f"malformed stage state for {label}: {state_path}")
            if any(stages[name].get("status") not in {
                    "pending", "running", "complete", "failed", "interrupted"}
                   for name in ("fit", "render", "metrics")):
                raise RuntimeError(f"invalid stage status for {label}: {state_path}")
            fit_status = stages.get("fit", {}).get("status")
            render_status = stages["render"]["status"]
            metrics_status = stages["metrics"]["status"]
            if ((render_status != "pending" and fit_status != "complete")
                    or (metrics_status != "pending" and render_status != "complete")):
                raise RuntimeError(
                    f"inconsistent stage dependencies for {label}: {state_path}")
            if fit_status == "failed":
                raise RuntimeError(
                    f"cannot automatically recover {fit_status} fit for {label}")
            if gpu_count_change and fit_status not in {"pending", "complete"}:
                raise RuntimeError(
                    f"cannot change GPU count while fit {label} is {fit_status}; "
                    "resume it with the saved GPU count")
        _validate_completed_stages(output, state)
        if gpu_count_change:
            old_gpu_count = saved_invocation["gpu_count"]
            new_gpu_count = invocation["gpu_count"]
            state["invocation"] = invocation
            state["gpu_count"] = new_gpu_count
            _atomic_write_json(state_path, state)
            print(
                f"Resume resource update: {old_gpu_count} -> {new_gpu_count} "
                "GPUs per fit",
                flush=True,
            )
        return state, state_path

    require_empty_output(output)
    output.mkdir(parents=True, exist_ok=True)
    state = _new_resume_state(invocation)
    _atomic_write_json(state_path, state)
    return state, state_path


def _run_saved_stage(state: dict, state_path: Path, stage: dict, action) -> None:
    stage["status"] = "running"
    stage.pop("error", None)
    _atomic_write_json(state_path, state)
    try:
        action()
    except KeyboardInterrupt:
        stage["status"] = "interrupted"
        _atomic_write_json(state_path, state)
        raise
    except BaseException as exc:
        stage["status"] = "failed"
        stage["error"] = f"{type(exc).__name__}: {exc}"
        _atomic_write_json(state_path, state)
        raise
    stage["status"] = "complete"
    _atomic_write_json(state_path, state)


def _run_resumable_pipeline(
    args: argparse.Namespace, *, root_output: Path, output: Path,
    overrides: dict, project: str, entity: str, state: dict,
    state_path: Path, label: str, seed_run_id: str | None,
    wandb_group: str | None = None,
) -> tuple[list[dict], dict]:
    stages = state["runs"][label]
    seeded = label != "single"
    history_path = output / TRAINING_HISTORY_FILENAME if seeded else None
    gpu_ids = getattr(args, "gpus", None)

    if stages["fit"]["status"] != "complete":
        output.mkdir(parents=True, exist_ok=True)
        recovering = stages["fit"]["status"] in {"running", "interrupted"}
        resume_checkpoint = None
        run_directory = None
        if recovering:
            resume_checkpoint, run_directory = _recover_interrupted_fit(output)
            if resume_checkpoint is not None:
                print(
                    f"RESUME fit {label} from {resume_checkpoint}", flush=True)
            else:
                print(
                    f"RESUME fit {label}: no checkpoint was written; restarting "
                    "from iteration 0",
                    flush=True,
                )

        def fit() -> None:
            subprocess.run(
                fit_command(args.dataset, gpu_ids), check=True,
                env=fit_environment(
                    overrides, output, args.num_threads, wandb_project=project,
                    wandb_entity=entity, wandb_enabled=not args.no_wandb,
                    wandb_run_id=seed_run_id, wandb_run_name=seed_run_id,
                    wandb_group=wandb_group,
                    metrics_history=history_path,
                    gpu_ids=gpu_ids,
                    resume_checkpoint=resume_checkpoint,
                    run_directory=run_directory))
            _run_dir, fitted = find_fit_outputs(output)
            if seeded:
                _load_training_history(history_path)
            stages["fit"]["fitted_output"] = str(fitted.resolve().relative_to(root_output))
            if seeded:
                stages["fit"]["training_history"] = str(
                    history_path.resolve().relative_to(root_output))

        _run_saved_stage(state, state_path, stages["fit"], fit)

    fitted_dir = _state_artifact(root_output, stages["fit"], "fitted_output")
    if stages["render"]["status"] != "complete":
        def render() -> None:
            command = [
                sys.executable, str(SPIRAL_DIR / "render_ink.py"), str(fitted_dir),
                "--volume", str(args.ink_volume), "--vc-render-bin",
                str(args.vc_render_bin),
            ]
            if args.num_threads is not None:
                command.extend(["--flatboi-threads", str(args.num_threads),
                                "--num-processes", "1"])
            subprocess.run(command, check=True,
                           env=downstream_environment(args.num_threads, gpu_ids=gpu_ids))
            ink = fitted_dir / "ink"
            _require_ink_output(ink)
            stages["render"]["ink_output"] = str(ink.resolve().relative_to(root_output))

        _run_saved_stage(state, state_path, stages["render"], render)

    if stages["metrics"]["status"] != "complete":
        def metrics() -> None:
            command = [sys.executable, str(SPIRAL_DIR / "get_ink_metrics.py"),
                       str(fitted_dir / "ink")]
            if args.num_threads is not None:
                command.extend(["--procs", str(max(1, args.num_threads // 3))])
            subprocess.run(command, check=True, env=downstream_environment(
                args.num_threads, metrics=True, gpu_ids=gpu_ids))
            metrics_path = fitted_dir / "ink_metric" / "metrics.json"
            _load_json_object(metrics_path, description="ink metrics")
            stages["metrics"]["metrics_output"] = str(
                metrics_path.resolve().relative_to(root_output))

        _run_saved_stage(state, state_path, stages["metrics"], metrics)

    if not seeded:
        return [], {}
    return _load_training_history(history_path), _load_final_summary(fitted_dir)


def run_resumable(
    args: argparse.Namespace, overrides: dict, project: str, entity: str
) -> None:
    output = args.output.resolve()
    invocation = _resume_invocation(args, overrides, project, entity)
    state, state_path = _load_or_create_state(output, invocation)
    seeds = getattr(args, "seeds", None)
    wandb_group = getattr(args, "wandb_group", None)

    if seeds is None:
        _run_resumable_pipeline(
            args, root_output=output, output=output, overrides=overrides,
            project=project, entity=entity, state=state, state_path=state_path,
            label="single", seed_run_id=state["run_id"],
            wandb_group=wandb_group)
        return

    histories = []
    summaries = []
    batch_id = state["batch_id"]
    for seed in seeds:
        seed_overrides = dict(overrides)
        seed_overrides["optimizer_random_seed"] = seed
        seed_id = f"{batch_id}_seed_{seed}"
        history, summary = _run_resumable_pipeline(
            args, root_output=output, output=output / f"seed-{seed}",
            overrides=seed_overrides, project=project, entity=entity,
            state=state, state_path=state_path, label=str(seed),
            seed_run_id=seed_id, wandb_group=wandb_group)
        histories.append(history)
        summaries.append(summary)
        if not args.no_wandb:
            _log_seed_final_metrics_once(
                summary, metrics_stage=state["runs"][str(seed)]["metrics"],
                state=state, state_path=state_path,
                project=project, entity=entity,
                seed_run_id=seed_id, group=wandb_group)

    if len(seeds) < 2:
        return
    training, final = aggregate_metrics(histories, summaries)
    aggregate = {"run_id": batch_id, "seeds": seeds,
                 "training": training, "final": final}
    aggregate_path = output / AGGREGATE_METRICS_FILENAME
    _atomic_write_json(aggregate_path, aggregate)
    if not args.no_wandb and not state.get("wandb_aggregate_logged"):
        if log_aggregate_metrics(
            training, final, seed_count=len(seeds), project=project,
            entity=entity, aggregate_run_id=f"{batch_id}_aggregate",
            group=wandb_group):
            state["wandb_aggregate_logged"] = True
            _atomic_write_json(state_path, state)


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    overrides, wandb_project, wandb_entity = load_run_config(args.config)
    overwrite = getattr(args, "overwrite", False)
    if overwrite and getattr(args, "resume", False):
        raise ValueError("--overwrite cannot be combined with --resume")
    if overwrite:
        protected_paths = [args.dataset, args.ink_volume, args.vc_render_bin]
        if args.config is not None:
            protected_paths.append(args.config)
        overwrite_output(args.output, protected_paths=protected_paths)
    if getattr(args, "resume", False):
        run_resumable(args, overrides, wandb_project, wandb_entity)
        return
    require_empty_output(output)
    seeds = getattr(args, "seeds", None)
    caller_run_id = getattr(args, "run_id", None)
    wandb_group = getattr(args, "wandb_group", None)

    if seeds is None:
        if caller_run_id is not None:
            raise ValueError("--run-id requires --seeds")
        run_pipeline(
            args, output=output, overrides=overrides,
            wandb_project=wandb_project, wandb_entity=wandb_entity,
            wandb_group=wandb_group)
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
            wandb_group=wandb_group,
        )
        histories.append(history)
        summaries.append(summary)
        if not args.no_wandb:
            log_seed_final_metrics(
                summary, project=wandb_project, entity=wandb_entity,
                seed_run_id=seed_id, group=wandb_group)

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
            aggregate_run_id=aggregate_id, group=wandb_group)


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
