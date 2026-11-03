"""Standalone, offline HTML/SVG inspection of saved fiber evaluation results.

No native imports, model inference, annotation reload or metric recomputation.
Spatial projections are not CT strips or estimates of threshold membership.
"""
from __future__ import annotations

import argparse
from html import escape
import json
import math
import os
from pathlib import Path
import tempfile


def atomic_write_text(path, text):
    """Replace a completed artifact using a sibling temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("visualization requires finite numeric coordinates/metrics")
    return float(value)


def _points(values):
    result = []
    for point in values:
        if len(point) != 3:
            raise ValueError("points must be base XYZ triples")
        result.append(tuple(_number(v) for v in point))
    return result


def _projection(reference, segments, failures, axes):
    geometry = reference + [p for s in segments for p in s] + failures
    a, b = axes
    low = [min(p[i] for p in geometry) for i in axes]
    high = [max(p[i] for p in geometry) for i in axes]
    scale = 340 / max(1., high[0]-low[0], high[1]-low[1])
    def position(p):
        return (30 + (p[a]-low[0])*scale, 370 - (p[b]-low[1])*scale)
    def line(points, style):
        pairs = " ".join(f"{x:.3f},{y:.3f}" for x, y in map(position, points))
        return f'<polyline class="{style}" points="{pairs}"/>'
    label = "XYZ"[a] + " / " + "XYZ"[b]
    svg = [f'<svg viewBox="0 0 400 410" role="img" aria-label="{label} base voxel projection">',
           f'<text x="20" y="18">{label} · base voxels</text>', line(reference, "reference")]
    # Separate elements preserve restart/filter gaps; never join segments.
    svg.extend(line(points, "trace") for points in segments if points)
    for point in failures:
        x, y = position(point)
        svg.append(f'<circle class="failure" cx="{x:.3f}" cy="{y:.3f}" r="4"/>')
    svg.append(f'<text x="20" y="400">origin {low[0]:.3f}, {low[1]:.3f}; '
               f'{1/scale:.3f} base vx / display unit</text></svg>')
    return "".join(svg)


def render_html(report):
    """Render the complete ordered report, preserving reported units and counts."""
    if report.get("format") not in ("vc_chunk_evaluation", "vc_beam_evaluation") or report.get("version") != 1:
        raise ValueError("unsupported evaluation report")
    references = {}
    for row in report.get("references", []):
        if row["id"] in references:
            raise ValueError("duplicate reference ID")
        references[row["id"]] = _points(row["points_base_xyz"])
    sections, seen = [], set()
    for index, case in enumerate(report["cases"]):
        identity = case["id"]
        if identity in seen:
            raise ValueError("duplicate case ID")
        seen.add(identity)
        reference = references.get(identity)
        if not reference or len(reference) < 2:
            raise ValueError(f"missing saved reference geometry for case {identity!r}; do not reload annotations")
        segments = [_points(s["points_base_xyz"]) for s in case["segments"]]
        failures = _points([f["reference_point_base"] for f in case["failures"]])
        panels = "".join(_projection(reference, segments, failures, axes) for axes in ((0, 1), (0, 2), (1, 2)))
        rows = []
        for failure in case["failures"]:
            measurement = failure.get("threshold_measurement")
            ratio = "n/a" if measurement is None else f'{_number(measurement["threshold_error_ratio"]):.4g}'
            rows.append(f'<tr><td>{escape(str(failure["index"]))}</td>'
                        f'<td>{escape(str(failure["reason"]))}</td>'
                        f'<td>{_number(failure["reference_arc_base"]):.3f}</td><td>{ratio}</td></tr>')
        interval = f'{_number(case["reference_begin_arc_base"]):.3f}–{_number(case["reference_end_arc_base"]):.3f}'
        sections.append(f'<section id="case-{index}"><h2>{escape(str(identity))}</h2>'
                        f'<p>Failures: {len(failures)}; filtered stops: {len(case["filtered_stops"])}; '
                        f'reverse checks: {int(case["reverse_checks"])}; '
                        f'retained trace: {_number(case["retained_trace_length_base"]):.3f} base voxels. '
                        f'Evaluated reference arc: {interval}; completed through '
                        f'{_number(case["completed_reference_arc_base"]):.3f}.</p>'
                        f'<div class="panels">{panels}</div>'
                        '<table><tr><th>Failure</th><th>Reason</th><th>Reference arc (base vx)</th>'
                        '<th>Threshold error ratio</th></tr>' + "".join(rows) + '</table></section>')
    return ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Fiber evaluation</title><style>'
            'body{font:15px sans-serif;margin:24px;color:#222;background:white}'
            'section{border-top:1px solid #bbb;margin-top:28px}.panels{display:flex;flex-wrap:wrap}'
            'svg{width:400px;max-width:100%}svg text{font-size:11px}'
            'polyline{fill:none;stroke-width:1.7}.reference{stroke:#888}.trace{stroke:#0077bb}'
            '.failure{fill:#d22;stroke:white}td,th{padding:6px;text-align:left}'
            '</style><h1>Fiber evaluation</h1>'
            f'<p>{len(sections)} directed cases; failures: {int(report["failure_count"])}; '
            f'filtered stops: {int(report["filtered_stop_count"])}; retained trace: '
            f'{_number(report["retained_trace_length_base"]):.3f} base voxels.</p>'
            '<p>Gray: full submitted reference. Blue: retained trace segments. Red: failure locations '
            'on the reference. Projections use equal axis scales within each panel; depth is collapsed. '
            'These are geometry views, not CT/presence strips or acceptance ellipses. '
            'Filtered stops are not failures; two directions count separately.</p>'
            + "".join(sections) + '</html>')


def write_gallery(report, output):
    atomic_write_text(output, render_html(report))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    write_gallery(json.loads(args.report.read_text()), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
