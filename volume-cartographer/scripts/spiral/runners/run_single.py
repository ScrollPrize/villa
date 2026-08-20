#!/usr/bin/env python3
"""Fit, render, and score one Spiral dataset sequentially."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence


SPIRAL_DIR = Path(__file__).resolve().parent.parent
VC_ROOT = SPIRAL_DIR.parent.parent
DEFAULT_VC_RENDER_BIN = VC_ROOT / "build" / "bin" / "vc_render_tifxyz"
DEFAULT_OUTPUT = SPIRAL_DIR / "out"
DEFAULT_RUN_CONFIG = Path(__file__).with_name("default_run_config.json")
WANDB_CONFIG_KEYS = ("wandb_project", "wandb_entity")

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
        "--num-threads",
        type=positive_int,
        help="positive per-run CPU thread budget",
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


def fit_environment(
    overrides: dict,
    output: Path,
    num_threads: int | None,
    *,
    wandb_project: str,
    wandb_entity: str,
    wandb_enabled: bool,
) -> dict[str, str]:
    env = os.environ.copy()
    # The runner owns these controls; do not inherit an unrelated outer fit.
    env.pop("FIT_SPIRAL_CONFIG_OVERRIDES", None)
    env["FIT_SPIRAL_OUT_DIR"] = str(output)
    env["WANDB_MODE"] = "online" if wandb_enabled else "disabled"
    env["WANDB_PROJECT"] = wandb_project
    env["WANDB_ENTITY"] = wandb_entity
    if overrides:
        env["FIT_SPIRAL_CONFIG_OVERRIDES"] = json.dumps(overrides)
    if num_threads is not None:
        env.update(_native_thread_env(num_threads))
        env["FIT_SPIRAL_NUM_THREADS"] = str(num_threads)
        env["FIT_SPIRAL_PATCH_LOAD_WORKERS"] = str(num_threads)
        env["FIT_SPIRAL_PATCH_LOAD_IO_THREADS"] = "1"
    return env


def downstream_environment(num_threads: int | None, *, metrics: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    if num_threads is not None:
        env.update(_native_thread_env(1 if metrics else num_threads))
    return env


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


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    require_empty_output(output)
    overrides, wandb_project, wandb_entity = load_run_config(args.config)

    fit_cmd = [
        sys.executable,
        str(SPIRAL_DIR / "fit_spiral.py"),
        "--dataset",
        str(args.dataset),
    ]
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
        env=downstream_environment(args.num_threads),
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
        env=downstream_environment(args.num_threads, metrics=True),
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
