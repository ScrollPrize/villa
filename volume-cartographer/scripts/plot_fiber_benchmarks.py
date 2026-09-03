#!/usr/bin/env python3
"""Render deterministic progress plots from recorded fiber benchmarks."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import date
from html import escape
from pathlib import Path
from typing import Any


REPLAY_METRIC = "distance_per_failure_percent"
CROP_METRIC = "negative_problematic_per_retained_fulfilled_percent"
REVISION_RE = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class PlotPoint:
    algorithm: str
    plot_label: str
    algorithm_date: date
    score_percent: float | None
    measured: bool


def _require_revision(value: Any, field: str) -> str:
    if not isinstance(value, str) or REVISION_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a full lowercase Git revision")
    return value


def _require_date(value: Any, field: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO date") from error


def _score_point(
    project_root: Path, metric: str, raw: dict[str, Any]
) -> PlotPoint:
    algorithm = raw.get("algorithm")
    if not isinstance(algorithm, str) or not algorithm.strip():
        raise ValueError("every point requires a nonempty algorithm")
    plot_label = raw.get("plot_label", algorithm)
    if not isinstance(plot_label, str) or not plot_label.strip():
        raise ValueError(f"{algorithm}: plot_label must be a nonempty string")
    algorithm_date = _require_date(raw.get("algorithm_date"), "algorithm_date")
    _require_revision(raw.get("algorithm_revision"), "algorithm_revision")
    status = raw.get("measurement_status")
    if status == "measured":
        _require_date(raw.get("measurement_date"), "measurement_date")
        _require_revision(raw.get("measurement_revision"), "measurement_revision")
        record = raw.get("run_record")
        if not isinstance(record, str) or not (project_root / record).is_file():
            raise ValueError(f"{algorithm}: measured point has no run record")
        if metric == REPLAY_METRIC:
            tested_length = float(raw.get("tested_length_mm", 0.0))
            failures = int(raw.get("failures", -1))
            if not math.isfinite(tested_length) or tested_length <= 0.0:
                raise ValueError(f"{algorithm}: tested length must be positive")
            if failures < 0:
                raise ValueError(f"{algorithm}: failures must be nonnegative")
            score = 100.0 / max(failures, 1)
        elif metric == CROP_METRIC:
            problematic = int(raw.get("problematic_unique_constraints", -1))
            fulfilled = int(
                raw.get("retained_fulfilled_unique_constraints", -1)
            )
            if problematic < 0 or fulfilled <= 0:
                raise ValueError(f"{algorithm}: invalid unique-constraint counts")
            score = -100.0 * problematic / fulfilled
        else:
            raise ValueError(f"unsupported metric: {metric}")
        measured = True
    elif status == "assumed_floor":
        if not isinstance(raw.get("assumption"), str) or not raw["assumption"]:
            raise ValueError(f"{algorithm}: assumed floor requires a rationale")
        if (
            "measurement_revision" in raw
            or "run_record" in raw
            or "assumed_score_percent" in raw
        ):
            raise ValueError(
                f"{algorithm}: assumed floor must not claim a metric or provenance"
            )
        score = None
        measured = False
    else:
        raise ValueError(f"{algorithm}: unsupported measurement status {status!r}")
    if score is not None:
        if not math.isfinite(score):
            raise ValueError(f"{algorithm}: score must be finite")
        if metric == REPLAY_METRIC and not 0.0 <= score <= 100.0:
            raise ValueError(f"{algorithm}: replay score must be in [0, 100]")
        if metric == CROP_METRIC and score > 0.0:
            raise ValueError(f"{algorithm}: crop error score must not exceed zero")
    return PlotPoint(algorithm, plot_label, algorithm_date, score, measured)


def _best_so_far(scores: list[float]) -> list[float]:
    best = -math.inf
    result = []
    for score in scores:
        best = max(best, score)
        result.append(best)
    return result


def load_plot_data(path: Path) -> tuple[dict[str, Any], dict[str, list[PlotPoint]]]:
    project_root = path.resolve().parents[1]
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("version") != 1:
        raise ValueError("unsupported fiber benchmark plot-data version")
    cohort = document.get("cohort", {})
    if not isinstance(cohort.get("name"), str):
        raise ValueError("plot data requires a named cohort")
    for field in ("reference_inventory_sha256", "normal_manifest_sha256"):
        value = cohort.get(field)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"cohort {field} must be a SHA-256 digest")

    raw_benchmarks = document.get("benchmarks", {})
    expected = {
        "reference_replay": REPLAY_METRIC,
        "crop_pruning": CROP_METRIC,
    }
    parsed: dict[str, list[PlotPoint]] = {}
    for name, metric in expected.items():
        benchmark = raw_benchmarks.get(name, {})
        if benchmark.get("metric") != metric:
            raise ValueError(f"{name}: unexpected metric")
        points = [
            _score_point(project_root, metric, point)
            for point in benchmark.get("points", [])
        ]
        if not points:
            raise ValueError(f"{name}: no plot points")
        parsed[name] = sorted(
            points, key=lambda point: (point.algorithm_date, point.algorithm)
        )
    return document, parsed


def _add_svg_accessibility(path: Path, title: str, description: str) -> None:
    text = path.read_text(encoding="utf-8")
    svg_end = text.find(">", text.find("<svg"))
    if svg_end < 0:
        raise ValueError(f"generated file is not SVG: {path}")
    accessible = (
        f'<title id="fiber-plot-title">{escape(title)}</title>'
        f'<desc id="fiber-plot-description">{escape(description)}</desc>'
    )
    text = (
        text[:svg_end]
        + ' role="img" aria-labelledby="fiber-plot-title fiber-plot-description"'
        + text[svg_end : svg_end + 1]
        + accessible
        + text[svg_end + 1 :]
    )
    path.write_text(text, encoding="utf-8")


def render_plot(
    output: Path,
    title: str,
    y_label: str,
    points: list[PlotPoint],
    description: str,
    annotation_rotation_degrees: float = 0.0,
) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "vc-matplotlib-cache")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.hashsalt": "vc-fiber-benchmark-progress-v1",
        }
    )
    measured = [point for point in points if point.measured]
    assumed = [point for point in points if not point.measured]
    dates = [point.algorithm_date for point in points]
    measured_scores = [
        point.score_percent
        for point in measured
        if point.score_percent is not None
    ]
    if not measured_scores:
        raise ValueError("a plot requires at least one measured point")
    assumed_floor = 0.0
    if assumed:
        assumed_floor = 25.0 * math.floor(min(measured_scores) / 25.0)
        if assumed_floor >= min(measured_scores):
            assumed_floor -= 25.0
    scores = [
        point.score_percent if point.score_percent is not None else assumed_floor
        for point in points
    ]
    progress_scores = _best_so_far(scores)

    figure, axis = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    line_color = "#777777" if assumed else "#16786f"
    line_style = "--" if assumed else "-"
    axis.step(
        dates,
        progress_scores,
        where="post",
        color=line_color,
        linestyle=line_style,
        linewidth=2.0 if assumed else 2.2,
    )
    if measured:
        axis.scatter(
            [point.algorithm_date for point in measured],
            [point.score_percent for point in measured],
            s=58,
            marker="o",
            color="#16786f",
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
            label="Measured",
        )
    if assumed:
        axis.scatter(
            [point.algorithm_date for point in assumed],
            [assumed_floor for point in assumed],
            s=66,
            marker="X",
            facecolor="white",
            edgecolor="#555555",
            linewidth=1.5,
            zorder=4,
            label="Assumed floor (not measured)",
        )
    zero_label = 0
    measured_label = 0
    for index, point in enumerate(points):
        display_score = (
            point.score_percent
            if point.score_percent is not None
            else assumed_floor
        )
        if point.score_percent is None:
            vertical = 12 + zero_label * 17
            zero_label += 1
        elif annotation_rotation_degrees:
            vertical = 10 + measured_label * 18
            measured_label += 1
        else:
            vertical = 10 if index % 2 == 0 else -18
        value_label = (
            f"{point.score_percent:.2f}%"
            if point.score_percent is not None
            else "assumed floor"
        )
        axis.annotate(
            f"{point.plot_label}\n{value_label}",
            (point.algorithm_date, display_score),
            xytext=(5, vertical),
            textcoords="offset points",
            ha="left",
            va="bottom" if vertical >= 0 else "top",
            fontsize=9,
            rotation=annotation_rotation_degrees,
            rotation_mode="anchor",
        )

    axis.set_title(title, loc="left", fontsize=15, fontweight="bold")
    axis.set_ylabel(y_label)
    axis.set_xlabel("Algorithm completion date")
    if min(scores) < 0.0:
        axis.set_ylim(bottom=min(scores) - 15.0, top=8.0)
        axis.axhline(0.0, color="#16786f", linewidth=1.2, linestyle=":")
        axis.text(
            dates[0], 1.5, "ideal target: 0%", color="#16786f", fontsize=9
        )
    else:
        axis.set_ylim(bottom=0.0, top=max(20.0, max(scores) * 1.32 + 2.0))
    axis.grid(axis="y", color="#d8dddc", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=7))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    figure.autofmt_xdate(rotation=0, ha="center")
    if assumed:
        axis.legend(loc="upper right", frameon=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        format="svg",
        metadata={"Creator": "plot_fiber_benchmarks.py", "Date": None},
    )
    plt.close(figure)
    _add_svg_accessibility(output, title, description)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=project_root / "docs" / "fiber_benchmark_plot_data.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "docs" / "fiber_benchmarks" / "imgs",
    )
    parser.add_argument(
        "--check", action="store_true", help="validate data without rendering"
    )
    args = parser.parse_args()

    document, points = load_plot_data(args.data)
    if args.check:
        for name, values in points.items():
            rendered = ", ".join(
                f"{point.algorithm}={point.score_percent:.6f}%"
                if point.score_percent is not None
                else f"{point.algorithm}=assumed_floor"
                for point in values
            )
            print(f"{name}: {rendered}")
        return 0

    benchmarks = document["benchmarks"]
    outputs = {
        "reference_replay": args.output_dir / "fiber_reference_replay_progress.svg",
        "crop_pruning": args.output_dir / "fiber_crop_pruning_progress.svg",
    }
    descriptions = {
        "reference_replay": (
            "Historical method milestones measured by directed reference distance "
            "per failure divided by total tested length. Higher is better."
        ),
        "crop_pruning": (
            "Historical method milestones measured by the negative ratio of "
            "problematic to retained fulfilled unique constraints after supervised "
            "oracle pruning. Zero is ideal and higher is better. Direct controls "
            "are unmeasured assumed floor points."
        ),
    }
    for name, output in outputs.items():
        annotation_rotation = float(
            benchmarks[name].get("annotation_rotation_degrees", 0.0)
        )
        if not math.isfinite(annotation_rotation):
            raise ValueError(f"{name}: annotation rotation must be finite")
        render_plot(
            output,
            benchmarks[name]["title"],
            benchmarks[name]["y_label"],
            points[name],
            descriptions[name],
            annotation_rotation,
        )
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
