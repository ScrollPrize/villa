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


REPLAY_METRIC = "mean_segment_length_percent"
CROP_METRIC = "negative_problematic_per_retained_fulfilled_percent"
REFERENCE_EXACT_METRIC = "pre_pruning_exact_reference_percent"
REVISION_RE = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class PlotPoint:
    algorithm: str
    method_id: str
    method_label: str
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
    project_root: Path,
    metric: str,
    raw: dict[str, Any],
    label_suffix: str,
) -> PlotPoint:
    algorithm = raw.get("algorithm")
    if not isinstance(algorithm, str) or not algorithm.strip():
        raise ValueError("every point requires a nonempty algorithm")
    method_id = raw.get("method_id")
    if not isinstance(method_id, str) or not method_id.strip():
        raise ValueError(f"{algorithm}: method_id must be a nonempty string")
    method_label = raw.get("method_label")
    if not isinstance(method_label, str) or not method_label.strip():
        raise ValueError(f"{algorithm}: method_label must be a nonempty string")
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
            score = _mean_segment_length_percent(failures)
        elif metric == CROP_METRIC:
            problematic = int(raw.get("problematic_unique_constraints", -1))
            fulfilled = int(
                raw.get("retained_fulfilled_unique_constraints", -1)
            )
            if problematic < 0 or fulfilled <= 0:
                raise ValueError(f"{algorithm}: invalid unique-constraint counts")
            score = -100.0 * problematic / fulfilled
        elif metric == REFERENCE_EXACT_METRIC:
            exact = int(raw.get("exact_references", -1))
            wrong = int(raw.get("wrong_references", -1))
            missing = int(raw.get("missing_references", -1))
            if exact < 0 or wrong < 0 or missing < 0 or exact + wrong + missing <= 0:
                raise ValueError(f"{algorithm}: invalid reference-result counts")
            score = 100.0 * exact / (exact + wrong + missing)
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
        if (
            metric in (REPLAY_METRIC, REFERENCE_EXACT_METRIC)
            and not 0.0 <= score <= 100.0
        ):
            raise ValueError(f"{algorithm}: percentage score must be in [0, 100]")
        if metric == CROP_METRIC and score > 0.0:
            raise ValueError(f"{algorithm}: crop error score must not exceed zero")
    plot_label = method_label + (label_suffix if measured else "")
    return PlotPoint(
        algorithm,
        method_id,
        method_label,
        plot_label,
        algorithm_date,
        score,
        measured,
    )


def _mean_segment_length_percent(failures: int) -> float:
    if failures < 0:
        raise ValueError("failures must be nonnegative")
    return 100.0 / (failures + 1)


def _frontier_indices(points: list[PlotPoint]) -> list[int]:
    """Return strict measured best-so-far points in experiment order."""
    best = -math.inf
    result = []
    for index, point in enumerate(points):
        score = point.score_percent
        if point.measured and score is not None and score > best:
            best = score
            result.append(index)
    return result


def _experiment_tick_labels(points: list[PlotPoint]) -> list[str]:
    date_groups: list[tuple[date, list[int]]] = []
    for index, point in enumerate(points):
        if not date_groups or date_groups[-1][0] != point.algorithm_date:
            date_groups.append((point.algorithm_date, [index]))
        else:
            date_groups[-1][1].append(index)

    dated_indices = {date_groups[0][1][0], date_groups[-1][1][-1]}
    if len(points) >= 6 and len(date_groups) > 2:
        _, middle_indices = date_groups[len(date_groups) // 2]
        dated_indices.add(middle_indices[len(middle_indices) // 2])
    dates_by_index = {
        index: point.algorithm_date.isoformat()
        for index, point in enumerate(points)
        if index in dated_indices
    }
    return [
        str(index + 1)
        + (f"\n{dates_by_index[index]}" if index in dates_by_index else "")
        for index in range(len(points))
    ]


def _sort_points_by_date(points: list[PlotPoint]) -> list[PlotPoint]:
    return sorted(points, key=lambda point: point.algorithm_date)


def _validate_method_labels(benchmarks: dict[str, list[PlotPoint]]) -> None:
    labels_by_method: dict[str, str] = {}
    for points in benchmarks.values():
        for point in points:
            previous = labels_by_method.setdefault(
                point.method_id, point.method_label
            )
            if previous != point.method_label:
                raise ValueError(
                    f"{point.method_id}: inconsistent method labels "
                    f"{previous!r} and {point.method_label!r}"
                )


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
        "pre_pruning_reference": REFERENCE_EXACT_METRIC,
    }
    parsed: dict[str, list[PlotPoint]] = {}
    for name, metric in expected.items():
        benchmark = raw_benchmarks.get(name, {})
        if benchmark.get("metric") != metric:
            raise ValueError(f"{name}: unexpected metric")
        label_suffix = benchmark.get("label_suffix", "")
        if not isinstance(label_suffix, str):
            raise ValueError(f"{name}: label_suffix must be a string")
        points = [
            _score_point(project_root, metric, point, label_suffix)
            for point in benchmark.get("points", [])
        ]
        if not points:
            raise ValueError(f"{name}: no plot points")
        # Python's sort is stable, so same-day experiments retain the explicit
        # sequence recorded in the benchmark data.
        parsed[name] = _sort_points_by_date(points)
    _validate_method_labels(parsed)
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
) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "vc-matplotlib-cache")
    )
    import matplotlib

    matplotlib.use("Agg")
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
    steps = list(range(1, len(points) + 1))
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
    display_scores = [
        point.score_percent if point.score_percent is not None else assumed_floor
        for point in points
    ]
    frontier_indices = _frontier_indices(points)
    frontier_set = set(frontier_indices)
    progress_scores = []
    best = math.nan
    for point in points:
        if point.measured and point.score_percent is not None:
            best = (
                point.score_percent
                if math.isnan(best)
                else max(best, point.score_percent)
            )
        progress_scores.append(best)

    frontier = [points[index] for index in frontier_indices]
    non_frontier_indices = [
        index for index in range(len(points)) if index not in frontier_set
    ]
    marker_styles = [
        ("D", "#d4771f"),
        ("s", "#7656a8"),
        ("^", "#3676b8"),
        ("v", "#c4516c"),
        ("P", "#579445"),
        ("*", "#a86726"),
        ("h", "#397d7a"),
        ("<", "#87506d"),
        (">", "#607c32"),
    ]

    figure, axis = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    axis.step(
        steps,
        progress_scores,
        where="post",
        color="#16786f",
        linewidth=2.2,
        label="_nolegend_",
    )
    if frontier:
        axis.scatter(
            [index + 1 for index in frontier_indices],
            [point.score_percent for point in frontier],
            s=62,
            marker="o",
            color="#16786f",
            edgecolor="white",
            linewidth=0.9,
            zorder=4,
            label="Pareto frontier",
        )
    for style_index, index in enumerate(non_frontier_indices):
        point = points[index]
        marker, color = marker_styles[style_index % len(marker_styles)]
        if point.measured:
            assert point.score_percent is not None
            score = point.score_percent
            facecolor = color
            edgecolor = "white"
            linewidth = 0.9
        else:
            score = assumed_floor
            facecolor = "white"
            edgecolor = color
            linewidth = 1.5
        axis.scatter(
            [index + 1],
            [score],
            s=58,
            marker=marker,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=3,
            label=(
                point.plot_label
                if point.measured
                else f"{point.plot_label} (assumed floor)"
            ),
        )
    for index in frontier_indices:
        point = points[index]
        assert point.score_percent is not None
        axis.annotate(
            f"{point.plot_label}\n{point.score_percent:.2f}%",
            (index + 1, point.score_percent),
            xytext=(-7, 7),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=9,
            rotation=-30.0,
            rotation_mode="anchor",
        )

    axis.set_title(title, loc="left", fontsize=15, fontweight="bold")
    axis.set_ylabel(y_label)
    axis.set_xlabel("Experiment step (sorted by completion date)")
    axis.set_xlim(0.0, len(points) + 0.5)
    if min(display_scores) < 0.0:
        axis.set_ylim(bottom=min(display_scores) - 15.0, top=8.0)
        axis.axhline(0.0, color="#16786f", linewidth=1.2, linestyle=":")
        axis.text(
            steps[0], 1.5, "ideal target: 0%", color="#16786f", fontsize=9
        )
    else:
        axis.set_ylim(
            bottom=0.0, top=max(20.0, max(display_scores) * 1.50 + 2.0)
        )
    axis.grid(axis="y", color="#d8dddc", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_xticks(steps, _experiment_tick_labels(points))
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=min(2, len(labels)),
        frameon=False,
    )

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
        "pre_pruning_reference": args.output_dir / "fiber_crop_reference_accuracy.svg",
    }
    descriptions = {
        "reference_replay": (
            "Historical method milestones measured by mean segment length, with "
            "failures splitting the tested reference corpus into failure count plus "
            "one segments, divided by total tested length. Higher is better."
        ),
        "crop_pruning": (
            "Historical method milestones measured by the negative ratio of "
            "problematic to retained fulfilled unique constraints after supervised "
            "oracle pruning. Zero is ideal and higher is better. Direct controls "
            "are unmeasured assumed floor points."
        ),
        "pre_pruning_reference": (
            "Historical Fiberlet crop milestones measured before supervised "
            "oracle pruning by exact reference estimates divided by exact plus "
            "wrong estimates. Missing references are excluded. Higher is better."
        ),
    }
    for name, output in outputs.items():
        render_plot(
            output,
            benchmarks[name]["title"],
            benchmarks[name]["y_label"],
            points[name],
            descriptions[name],
        )
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
