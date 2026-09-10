"""Launch an isolated multi-GPU Fiber3D reference experiment."""
import argparse
import hashlib
import importlib.metadata
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from vesuvius.neural_tracing.fiber_trace_3d.analyze_training import distribution


_MODULE_NAME = "vesuvius.neural_tracing.fiber_trace_3d.profile_experiment"
_TEST_METRIC_PATTERN = re.compile(
    r"^test (?:samples=\d+ update=|step=)(?P<step>\d+) "
    r"loss_total=(?P<loss_total>[-+0-9.eE]+) "
    r"loss_direction=(?P<loss_direction>[-+0-9.eE]+) "
    r"loss_presence=(?P<loss_presence>[-+0-9.eE]+) "
    r"(?:loss_embedding=[-+0-9.eE]+ embedding_negative_pairs=[-+0-9.eE]+ )?"
    r"angle_mean_deg=(?P<angle_mean_deg>[-+0-9.eE]+)"
)


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
                                   "loss_stat_sync_wall_ms", "backward_wall_ms",
                                   "manual_gradient_sync_wall_ms", "optimizer_wall_ms",
                                   "scalar_reduce_sync_wall_ms")},
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
                    "value_augmentation_cuda_ms", "target_ms", "fw_bw_step_ms",
                    "loss_stat_sync_wall_ms", "manual_gradient_sync_wall_ms",
                    "scalar_reduce_sync_wall_ms")
    }
    result = {"experiment": str(experiment), "ranks": ranks,
              "critical_path": critical_path,
              "selection": "steps >30 excluding trace and multiples of100; loader stalls retained"}
    (experiment / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _percentile(values, fraction):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = float(fraction) * float(len(ordered) - 1)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    weight = position - float(lo)
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def analyze_lr_range(experiment, *, beta=0.98, divergence_factor=4.0):
    """Analyze rank-0 globally averaged metrics from an LR range experiment."""
    experiment = Path(experiment)
    paths = sorted(experiment.glob("s1a*/timings_rank0.jsonl"))
    if len(paths) != 1:
        raise RuntimeError("LR range analysis requires exactly one rank-0 timing file")
    rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
    range_rows = [
        row
        for row in rows
        if int(row.get("lr_range_index", -1)) >= 0
        and "loss_total" in row
        and "learning_rate" in row
    ]
    smoothed = []
    ema = 0.0
    minimum = math.inf
    minimum_index = -1
    divergence_index = None
    for index, row in enumerate(range_rows):
        loss = float(row["loss_total"])
        if not math.isfinite(loss):
            divergence_index = index
            break
        ema = float(beta) * ema + (1.0 - float(beta)) * loss
        corrected = ema / (1.0 - float(beta) ** float(index + 1))
        smoothed.append(corrected)
        if corrected < minimum:
            minimum = corrected
            minimum_index = index
        elif (
            index >= 20
            and minimum_index >= 0
            and corrected > float(divergence_factor) * minimum
        ):
            divergence_index = index
            break
    usable_count = (
        len(range_rows) if divergence_index is None else int(divergence_index)
    )
    stable_rows = range_rows[:usable_count]
    grad_norms = [
        float(row["grad_norm_l2"])
        for row in stable_rows
        if math.isfinite(float(row.get("grad_norm_l2", math.nan)))
    ]
    divergence_lr = (
        float(range_rows[divergence_index]["learning_rate"])
        if divergence_index is not None and divergence_index < len(range_rows)
        else float(range_rows[-1]["learning_rate"])
        if range_rows
        else None
    )
    result = {
        "beta": float(beta),
        "divergence_factor": float(divergence_factor),
        "range_rows": len(range_rows),
        "usable_rows": usable_count,
        "minimum_smoothed_loss": None if not math.isfinite(minimum) else minimum,
        "minimum_loss_lr": (
            float(range_rows[minimum_index]["learning_rate"])
            if minimum_index >= 0
            else None
        ),
        "divergence_lr": divergence_lr,
        "lr_candidates": (
            None
            if divergence_lr is None
            else {
                "divergence_div_10": divergence_lr / 10.0,
                "divergence_div_6": divergence_lr / 6.0,
                "divergence_div_3": divergence_lr / 3.0,
            }
        ),
        "grad_norm_l2": {
            "count": len(grad_norms),
            "median": _percentile(grad_norms, 0.5),
            "p95": _percentile(grad_norms, 0.95),
            "p99": _percentile(grad_norms, 0.99),
            "max": max(grad_norms) if grad_norms else None,
        },
        "curve": [
            {
                "step": int(row["step"]),
                "range_index": int(row["lr_range_index"]),
                "learning_rate": float(row["learning_rate"]),
                "loss_total": float(row["loss_total"]),
                "smoothed_loss": (
                    smoothed[index] if index < len(smoothed) else None
                ),
                "loss_direction": row.get("loss_direction"),
                "loss_presence": row.get("loss_presence"),
                "grad_norm_l2": row.get("grad_norm_l2"),
            }
            for index, row in enumerate(range_rows)
        ],
    }
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        result["plot"] = None
    else:
        figure, loss_axis = plt.subplots(figsize=(9, 5))
        learning_rates = [item["learning_rate"] for item in result["curve"]]
        losses = [item["loss_total"] for item in result["curve"]]
        smooth_losses = [item["smoothed_loss"] for item in result["curve"]]
        loss_axis.semilogx(learning_rates, losses, alpha=0.25, label="loss")
        loss_axis.semilogx(learning_rates, smooth_losses, label="EMA loss")
        loss_axis.set_xlabel("learning rate")
        loss_axis.set_ylabel("globally averaged training loss")
        loss_axis.grid(True, which="both", alpha=0.25)
        loss_axis.legend(loc="upper left")
        grad_values = [item["grad_norm_l2"] for item in result["curve"]]
        if any(value is not None for value in grad_values):
            grad_axis = loss_axis.twinx()
            grad_axis.semilogx(
                learning_rates,
                [math.nan if value is None else value for value in grad_values],
                color="tab:orange",
                alpha=0.35,
                label="gradient L2 norm",
            )
            grad_axis.set_ylabel("pre-clip global L2 gradient norm")
        figure.tight_layout()
        plot_path = experiment / "lr_range.png"
        figure.savefig(plot_path, dpi=160)
        plt.close(figure)
        result["plot"] = str(plot_path)
    (experiment / "lr_range_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def _extract_test_metrics(experiment):
    """Return ordered held-out test rows printed by one experiment."""
    rows = []
    for line in (Path(experiment) / "run.log").read_text(errors="replace").splitlines():
        match = _TEST_METRIC_PATTERN.match(line)
        if match is None:
            continue
        row = {"step": int(match.group("step"))}
        row.update(
            (key, float(match.group(key)))
            for key in (
                "loss_total",
                "loss_direction",
                "loss_presence",
                "angle_mean_deg",
            )
        )
        rows.append(row)
    return rows


def _round_threshold_up(value):
    """Round a positive threshold upward to two significant decimal places."""
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("clip threshold source must be finite and positive")
    scale = 10.0 ** (1 - math.floor(math.log10(value)))
    return math.ceil(value * scale) / scale


def _tuning_lr_candidates(range_analysis, *, baseline_lr, explicit=None):
    """Return a small deterministic set of fixed-peak verification LRs."""
    if explicit:
        raw = [float(value) for value in str(explicit).split(",")]
    else:
        minimum_lr = float(range_analysis["minimum_loss_lr"])
        divergence_lr = float(range_analysis["divergence_lr"])
        raw = [
            float(baseline_lr),
            minimum_lr / 3.0,
            minimum_lr,
            divergence_lr / 10.0,
        ]
    result = []
    for value in sorted(raw):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("tuning LR candidates must be finite and positive")
        if not result or not math.isclose(value, result[-1], rel_tol=1.0e-9):
            result.append(value)
    return result


def _suite_common_child_args(args, suite_dir):
    child = [
        "--reference", str(args.reference.resolve()),
        "--output", str(Path(suite_dir).resolve()),
        "--devices", str(args.devices),
        "--compile-mode", str(args.compile_mode),
        "--normalization", str(args.normalization),
        "--output-stride", str(args.output_stride),
        "--batch-size-per-gpu", str(args.batch_size_per_gpu),
        "--timeout-seconds", str(args.timeout_seconds),
        "--trace-steps", str(args.trace_steps),
        "--gradient-norm-log-interval", str(args.gradient_norm_log_interval),
    ]
    if args.optimizer_global_batch_size is not None:
        child.extend(
            ["--optimizer-global-batch-size", str(args.optimizer_global_batch_size)]
        )
    for enabled, flag in (
        (args.profile_loader, "--profile-loader"),
        (args.sample_pipeline, "--sample-pipeline"),
        (args.async_batch_preparation, "--async-batch-preparation"),
        (args.throughput_only, "--throughput-only"),
        (args.force_scalar_reduce_every_step, "--force-scalar-reduce-every-step"),
    ):
        if enabled:
            child.append(flag)
    if args.gpu_value_augmentation:
        child.extend(["--gpu-value-augmentation", str(args.gpu_value_augmentation)])
    return child


def _run_suite_child(arguments):
    command = [sys.executable, "-m", _MODULE_NAME, *arguments]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    experiment = None
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        if line.startswith("experiment="):
            experiment = Path(line.removeprefix("experiment=").strip())
    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"tuning child failed with return code {returncode}")
    if experiment is None:
        raise RuntimeError("tuning child did not report its experiment directory")
    return experiment


def _write_tuning_suite_report(suite_dir, report):
    suite_dir = Path(suite_dir)
    (suite_dir / "tuning_report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Fiber3D LR and gradient-clipping tuning report",
        "",
        "No final setting is selected automatically; inspect the matched held-out results.",
        "",
        f"- Range experiment: `{report['range_experiment']}`",
        f"- Range EMA minimum LR: {report['range_analysis']['minimum_loss_lr']:.8g}",
        f"- Range divergence LR: {report['range_analysis']['divergence_lr']:.8g}",
        f"- Derived clip threshold: {report['derived_clip_threshold']:.8g}",
        "",
        "| Peak LR | Clip L2 | Final test step | Final test loss | Experiment |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
    for trial in report["fixed_trials"]:
        metric = trial["final_test"]
        clip = "disabled" if trial["clip_max_norm"] is None else f"{trial['clip_max_norm']:.8g}"
        lines.append(
            f"| {trial['learning_rate']:.8g} | {clip} | {metric['step']} | "
            f"{metric['loss_total']:.6f} | `{trial['experiment']}` |"
        )
    lines.extend(
        [
            "",
            "The clipped trial uses the best unclipped candidate by final held-out loss "
            "only to choose which LR receives a matched clipping comparison.",
        ]
    )
    (suite_dir / "tuning_report.md").write_text("\n".join(lines) + "\n")


def run_tuning_suite(args):
    """Run range discovery and matched fixed trials from fresh initialization."""
    if args.lr_range_start is None or args.lr_range_end is None:
        raise ValueError("--tuning-suite requires --lr-range-start and --lr-range-end")
    if args.throughput_only or args.replay_batch or args.replay_cpu_batch:
        raise ValueError("--tuning-suite requires real training with held-out evaluation")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    suite_dir = args.output.resolve() / f"tuning_suite_{stamp}"
    suite_dir.mkdir(parents=True, exist_ok=False)
    common = _suite_common_child_args(args, suite_dir)
    range_experiment = _run_suite_child(
        [
            *common,
            "--lr-range-start", str(args.lr_range_start),
            "--lr-range-end", str(args.lr_range_end),
            "--lr-range-warmup-steps", str(args.lr_range_warmup_steps),
            "--lr-range-steps", str(args.lr_range_steps),
        ]
    )
    range_analysis = json.loads(
        (range_experiment / "lr_range_summary.json").read_text()
    )
    baseline = json.loads(args.reference.read_text())["config"]["training"]
    baseline_lr = float(baseline["learning_rate"])
    candidates = _tuning_lr_candidates(
        range_analysis,
        baseline_lr=baseline_lr,
        explicit=args.tuning_lr_candidates,
    )
    clip_source = range_analysis["grad_norm_l2"][args.tuning_clip_percentile]
    clip_threshold = _round_threshold_up(clip_source)
    trials = []
    for learning_rate in candidates:
        experiment = _run_suite_child(
            [
                *common,
                "--steps", str(args.tuning_fixed_steps),
                "--learning-rate", str(learning_rate),
                "--learning-rate-warmup-steps", str(args.tuning_fixed_warmup_steps),
            ]
        )
        test_rows = _extract_test_metrics(experiment)
        if not test_rows:
            raise RuntimeError(f"fixed trial produced no held-out test metrics: {experiment}")
        trials.append(
            {
                "learning_rate": learning_rate,
                "clip_max_norm": None,
                "experiment": str(experiment),
                "test_metrics": test_rows,
                "final_test": test_rows[-1],
            }
        )
    best_unclipped = min(trials, key=lambda trial: trial["final_test"]["loss_total"])
    clipped_experiment = _run_suite_child(
        [
            *common,
            "--steps", str(args.tuning_fixed_steps),
            "--learning-rate", str(best_unclipped["learning_rate"]),
            "--learning-rate-warmup-steps", str(args.tuning_fixed_warmup_steps),
            "--gradient-clip-max-norm", str(clip_threshold),
        ]
    )
    clipped_rows = _extract_test_metrics(clipped_experiment)
    if not clipped_rows:
        raise RuntimeError("clipped trial produced no held-out test metrics")
    trials.append(
        {
            "learning_rate": best_unclipped["learning_rate"],
            "clip_max_norm": clip_threshold,
            "experiment": str(clipped_experiment),
            "test_metrics": clipped_rows,
            "final_test": clipped_rows[-1],
        }
    )
    report = {
        "suite": str(suite_dir),
        "reference": str(args.reference.resolve()),
        "devices": str(args.devices),
        "optimizer_global_batch_size": args.optimizer_global_batch_size,
        "range_experiment": str(range_experiment),
        "range_analysis": {
            key: range_analysis[key]
            for key in (
                "minimum_smoothed_loss",
                "minimum_loss_lr",
                "divergence_lr",
                "grad_norm_l2",
            )
        },
        "derived_lr_candidates": candidates,
        "clip_percentile": args.tuning_clip_percentile,
        "derived_clip_threshold": clip_threshold,
        "best_unclipped_by_final_test_loss": best_unclipped["learning_rate"],
        "fixed_trials": trials,
    }
    _write_tuning_suite_report(suite_dir, report)
    print(f"tuning_report={suite_dir / 'tuning_report.md'}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True,
                        help="Saved baseline JSON containing the authoritative config")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--devices",
        default="0,1",
        help="comma-separated physical CUDA device indices",
    )
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
        help="benchmark-only maximum local microbatch size for each GPU rank",
    )
    parser.add_argument(
        "--optimizer-global-batch-size",
        type=int,
        default=None,
        help="exact sample count represented by one optimizer update",
    )
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--bounded-accumulation", action="store_true",
                        help="use memory-bounded component VJPs instead of retaining microbatch graphs")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--learning-rate-warmup-steps", type=int, default=0)
    parser.add_argument("--gradient-clip-max-norm", type=float, default=None)
    parser.add_argument("--gradient-norm-log-interval", type=int, default=0)
    parser.add_argument("--lr-range-start", type=float, default=None)
    parser.add_argument("--lr-range-end", type=float, default=None)
    parser.add_argument("--lr-range-warmup-steps", type=int, default=100)
    parser.add_argument("--lr-range-steps", type=int, default=1000)
    parser.add_argument(
        "--tuning-suite",
        action="store_true",
        help="run LR range, derived fixed-LR trials, and one matched clipping trial",
    )
    parser.add_argument("--tuning-fixed-steps", type=int, default=200)
    parser.add_argument("--tuning-fixed-warmup-steps", type=int, default=100)
    parser.add_argument(
        "--tuning-lr-candidates",
        default=None,
        help="optional comma-separated fixed LRs instead of range-derived candidates",
    )
    parser.add_argument(
        "--tuning-clip-percentile",
        choices=("p95", "p99"),
        default="p99",
    )
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
    parser.add_argument(
        "--force-scalar-reduce-every-step",
        action="store_true",
        help="benchmark control: retain every-step metric reduction without logging",
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
    if args.optimizer_global_batch_size is not None and args.optimizer_global_batch_size <= 0:
        parser.error("optimizer-global-batch-size must be > 0")
    if args.timeout_seconds <= 0.0:
        parser.error("timeout-seconds must be > 0")
    try:
        device_ids = tuple(int(value) for value in str(args.devices).split(","))
    except ValueError:
        parser.error("--devices must be a comma-separated list of integers")
    if not device_ids or len(set(device_ids)) != len(device_ids) or min(device_ids) < 0:
        parser.error("--devices must contain unique nonnegative indices")
    world_size = len(device_ids)
    if args.gradient_norm_log_interval < 0:
        parser.error("gradient-norm-log-interval must be nonnegative")
    if args.learning_rate_warmup_steps < 0:
        parser.error("learning-rate-warmup-steps must be nonnegative")
    lr_range_enabled = args.lr_range_start is not None or args.lr_range_end is not None
    if lr_range_enabled and (args.lr_range_start is None or args.lr_range_end is None):
        parser.error("lr-range-start and lr-range-end must be supplied together")
    if lr_range_enabled and (
        args.lr_range_warmup_steps < 0 or args.lr_range_steps < 2
    ):
        parser.error("LR range requires nonnegative warmup and at least two range steps")
    if lr_range_enabled and args.learning_rate_warmup_steps:
        parser.error("fixed-LR warmup cannot be combined with an LR range")
    if args.tuning_suite and not lr_range_enabled:
        parser.error("--tuning-suite requires an LR range")
    if args.tuning_fixed_steps < 30:
        parser.error("tuning-fixed-steps must be at least 30")
    if not 0 <= args.tuning_fixed_warmup_steps <= args.tuning_fixed_steps:
        parser.error("tuning fixed warmup must be between zero and fixed steps")
    if args.tuning_lr_candidates:
        try:
            explicit_candidates = [
                float(value) for value in args.tuning_lr_candidates.split(",")
            ]
        except ValueError:
            parser.error("tuning-lr-candidates must be comma-separated numbers")
        if not explicit_candidates or any(
            not math.isfinite(value) or value <= 0.0
            for value in explicit_candidates
        ):
            parser.error("tuning LR candidates must be finite and positive")
    config = json.loads(args.reference.read_text())["config"]
    if config.get("batch_size") != 8:
        parser.error("this reference experiment requires batch_size=8 per rank")
    if config.get("resume") or config.get("training", {}).get("resume"):
        parser.error("reference experiment must start fresh, not resume")
    if args.tuning_suite:
        run_tuning_suite(args)
        return
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
    if lr_range_enabled:
        suffix += "_lr_range"
    if args.force_scalar_reduce_every_step:
        suffix += "_scalar_sync_control"
    experiment = args.output.resolve() / f"profile_{world_size}gpu_{args.compile_mode}{suffix}_{stamp}"
    experiment.mkdir(parents=True, exist_ok=False)
    training = config["training"]
    config["batch_size"] = int(args.batch_size_per_gpu)
    model_config = config.setdefault("model_3d", {})
    model_config["normalization"] = "group" if args.normalization == "group" else "batch"
    model_config["output_stride"] = int(args.output_stride)
    training["sync_batchnorm"] = args.normalization == "sync-batch"
    training["ddp_broadcast_buffers"] = False
    optimizer_global_batch_size = (
        world_size * int(args.batch_size_per_gpu)
        if args.optimizer_global_batch_size is None
        else int(args.optimizer_global_batch_size)
    )
    training["optimizer_global_batch_size"] = optimizer_global_batch_size
    training["bounded_gradient_accumulation"] = bool(args.bounded_accumulation)
    effective_steps = (
        int(args.lr_range_warmup_steps) + int(args.lr_range_steps)
        if lr_range_enabled
        else int(args.steps)
    )
    training.update(run_path=str(experiment),
                    run_name=f"s1a_128_sd1_{world_size}x{args.batch_size_per_gpu}",
                    max_steps=effective_steps, compile_mode=args.compile_mode,
                    initialization_seed=0, profile_enabled=True,
                    profile_loader=args.profile_loader,
                    profile_replay_batch=args.replay_batch,
                    profile_replay_cpu_batch=args.replay_cpu_batch,
                    profile_trace_start=21, profile_trace_steps=args.trace_steps,
                    scalar_log_interval=1)
    if args.learning_rate is not None:
        training["learning_rate"] = float(args.learning_rate)
    training["learning_rate_warmup_steps"] = int(
        args.learning_rate_warmup_steps
    )
    if args.gradient_clip_max_norm is not None:
        training["gradient_clip_max_norm"] = float(args.gradient_clip_max_norm)
    training["gradient_norm_log_interval"] = int(args.gradient_norm_log_interval)
    if lr_range_enabled:
        training["lr_range_test"] = {
            "warmup_steps": int(args.lr_range_warmup_steps),
            "range_steps": int(args.lr_range_steps),
            "start_lr": float(args.lr_range_start),
            "end_lr": float(args.lr_range_end),
        }
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
            sample_pipeline_slots=world_size * slots_per_rank,
            sample_pipeline_initial_in_flight=(
                world_size * int(args.batch_size_per_gpu)
            ),
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
    if throughput_only or lr_range_enabled:
        training.update(
            profile_throughput_start_step=(
                int(args.lr_range_warmup_steps) + 1
                if lr_range_enabled
                else 31
            ),
            profile_trace_steps=0,
            scalar_log_interval=(1 if lr_range_enabled else effective_steps + 1),
            initial_scalar_log_steps=0,
            profile_force_scalar_reduce_every_step=bool(
                args.force_scalar_reduce_every_step
            ),
            tensorboard_enabled=bool(lr_range_enabled),
            sample_vis_interval=0,
            test_trace2cp_enabled=False,
            test_interval=effective_steps + 1,
            checkpoint_interval=effective_steps + 1,
            kept_snapshot_interval=0,
        )
        config.pop("test_datasets", None)
    config_path = experiment / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    env = os.environ.copy()
    visible_devices = ",".join(str(value) for value in device_ids)
    env.update(CUDA_VISIBLE_DEVICES=visible_devices, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", AGENTS_AGENT_MODE="1", PYTHONUNBUFFERED="1",
               TORCHINDUCTOR_COMPILE_THREADS="4",
               TORCHINDUCTOR_CACHE_DIR=str(args.output.resolve() / "inductor_cache"),
               TRITON_CACHE_DIR=str(args.output.resolve() / "triton_cache"))
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               f"--nproc_per_node={world_size}", "-m", "vesuvius.neural_tracing.fiber_trace_3d.train",
               str(config_path), "--unmanaged"]
    metadata = {"command": command, "start_unix": time.time(),
                "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
                "world_size": world_size,
                "reference": str(args.reference.resolve()),
                "batch_size_per_gpu": int(args.batch_size_per_gpu),
                "microbatch_global_capacity": world_size * int(args.batch_size_per_gpu),
                "global_batch": optimizer_global_batch_size,
                "optimizer_global_batch_size": optimizer_global_batch_size,
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
                "force_scalar_reduce_every_step": bool(
                    args.force_scalar_reduce_every_step
                ),
                "learning_rate": args.learning_rate,
                "learning_rate_warmup_steps": int(
                    args.learning_rate_warmup_steps
                ),
                "gradient_clip_max_norm": args.gradient_clip_max_norm,
                "gradient_norm_log_interval": int(args.gradient_norm_log_interval),
                "lr_range_test": training.get("lr_range_test"),
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
    if lr_range_enabled and returncode == 0:
        analyze_lr_range(experiment)
    print(f"completed returncode={returncode} failure_kind={failure_kind} elapsed={metadata['end_unix']-metadata['start_unix']:.1f}s", flush=True)
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
