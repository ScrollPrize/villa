"""Per-fiber complete-partition metrics and links into inline strip output."""
from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from urllib.parse import quote

from .replay_metrics import reference_segments, segment_statistics, complete_report


def write_trace_progress_report(evaluation_root, report_root):
    """Rank saved complete traces even when rendering/evaluation is unfinished."""
    import json
    from lasagna.inference_provenance import atomic_write
    evaluation_root, report_root = Path(evaluation_root), Path(report_root)
    grouped = {}
    for artifact in sorted((evaluation_root / "cases").glob("*/trace.json")):
        document = json.loads(artifact.read_text())
        p, replay = document["provenance"], document["replay"]
        begin, end = replay["reference_begin_arc_base"], replay["reference_end_arc_base"]
        if replay["completed_reference_arc_base"] < end - 1e-7:
            continue
        segments = reference_segments({"reference_length_base_voxels": end - begin,
            "evaluated_through_base_voxels": replay["completed_reference_arc_base"] - begin,
            "evaluation_complete": True, "failures": [
                {"directional_reference_arc_base_voxels": f["reference_arc_base"] - begin}
                for f in replay["failures"]]}, p["base_voxel_size_um"] / 1000)
        key = (p["source_id"], p["fiber_path"])
        row = grouped.setdefault(key, {"collection": key[0], "fiber": key[1],
            "reference_length_mm": 0,
            "directions": {}, "segments": []})
        direction = "reverse" if p["reverse"] else "forward"
        run = p.get("reference_run", {}).get("run_index", 0)
        value = row["directions"].setdefault(direction, {"runs": {}, "images": []})
        if run in value["runs"]:
            raise ValueError("duplicate directional trace")
        stats = segment_statistics(segments, 1)
        row["segments"].extend(segments)
        value["runs"][run] = {"case_index": p["case_index"],
            "restarts": stats["failures"], "mean_mm": stats["mean_segment_length_mm"],
            "length_mm": math.fsum(segments),
            "trace_json": str(artifact.relative_to(report_root)),
            "images": [str(path.relative_to(report_root)) for path in sorted(artifact.parent.glob("rendered*/strips_*.jpg"))]}
        selected = next((f for f in p.get("reference_selection", []) if Path(f["fiber_path"]).resolve() == Path(p["fiber_path"]).resolve()), None)
        row["expected_runs"] = len(selected["runs"]) if selected is not None else 1
    rows = list(grouped.values())
    for row in rows:
        directed_runs = sum(len(v["runs"]) for v in row["directions"].values())
        stats = segment_statistics(row.pop("segments"), directed_runs)
        row.update(restarts=stats["failures"], mean_mm=stats["mean_segment_length_mm"])
        for value in row["directions"].values():
            runs = list(value["runs"].values())
            value.update(case_index=runs[0]["case_index"], trace_json=runs[0]["trace_json"],
                images=[p for run in runs for p in run["images"]],
                restarts=sum(run["restarts"] for run in runs))
            length = math.fsum(run["length_mm"] for run in runs)
            value["mean_mm"] = length / (value["restarts"] + 1)
            row["reference_length_mm"] = max(row["reference_length_mm"], length)
        row["paired_complete"] = len(row["directions"]) == 2 and all(len(v["runs"]) == row["expected_runs"] for v in row["directions"].values())
    paired = [r for r in rows if r["paired_complete"]]
    average = math.fsum(r["mean_mm"] for r in paired) / len(paired) if paired else 0
    rows.sort(key=lambda r: (r["reference_length_mm"], r["mean_mm"]))
    report = {"status": "saved traces; rendering may be incomplete", "fibers": rows,
        "paired_fibers": len(paired), "arithmetic_mean_per_fiber_mm": average,
        "short_below_average": [r for r in rows if r["paired_complete"] and r["mean_mm"] < average]}
    atomic_write(report_root / "trace_progress.json", report)
    lines = ["# Saved trace metrics (rendering may be incomplete)", "",
        "Mean = combined directed eligible length / (restarts + 1). Excluded regions add no failures or distance.",
        f"Below-average criterion: arithmetic mean of paired per-fiber means = {average:.4f} mm.", "",
        "Sorted by fiber length. Trace JSON contains indexed failures and reference_arc_base (directional base voxels).", "",
        "| Collection / fiber | Directions | Length mm | Restarts | Mean mm | Below average | Cases / images |",
        "|---|---:|---:|---:|---:|---|---|"]
    for row in rows:
        links = []
        for direction, value in row["directions"].items():
            for run in value["runs"].values():
                links.append(f"[{direction} case {run['case_index']}]({quote(run['trace_json'])})")
            links += [f"[image]({quote(path)})" for path in value["images"]]
        lines.append(f"| {row['collection']} / {Path(row['fiber']).stem} | {len(row['directions'])} | {row['reference_length_mm']:.4f} | {row['restarts']} | {row['mean_mm']:.4f} | {'yes' if len(row['directions']) == 2 and row['mean_mm'] < average else ''} | {' · '.join(links)} |")
    (report_root / "trace_progress.md").write_text("\n".join(lines) + "\n")
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["collection", "fiber", "length_mm", "restarts", "mean_mm", "forward_case", "reverse_case"])
    for row in rows:
        writer.writerow([row["collection"], row["fiber"], row["reference_length_mm"], row["restarts"], row["mean_mm"],
            ",".join(str(v["case_index"]) for v in row["directions"].get("forward", {}).get("runs", {}).values()),
            ",".join(str(v["case_index"]) for v in row["directions"].get("reverse", {}).get("runs", {}).values())])
    (report_root / "trace_progress.csv").write_text(stream.getvalue())
    return report


def fiber_rows(summary):
    complete_report(summary)
    rows = []
    for source in summary["sources"]:
        binding = source["backend_config"]
        scale = source["summary"]["base_voxel_size_um"] / 1000
        grouped = {}
        for case in source["cases"]:
            index, direction, run = case["source_index"], case["direction"], case.get("run_index", 0)
            if direction not in ("forward", "reverse"):
                raise ValueError("unknown trace direction")
            runs = grouped.setdefault(index, {"forward": {}, "reverse": {}})[direction]
            if run in runs:
                raise ValueError("duplicate fiber direction/run")
            runs[run] = reference_segments(case, scale)
        selection = binding.get("reference_selection")
        for index, fiber in enumerate(binding["fibers"]):
            pair = grouped.get(index, {"forward": {}, "reverse": {}})
            expected = set(range(len(selection[index]["runs"]))) if selection is not None else {0}
            if any(set(pair[k]) != expected for k in pair):
                raise ValueError("fiber run membership differs from evaluated binding")
            if any(not math.isclose(math.fsum(pair["forward"][r]), math.fsum(pair["reverse"][r]), rel_tol=1e-9) for r in expected):
                raise ValueError("forward/reverse reference lengths differ")
            directions = {}
            all_segments = []
            for direction, runs in pair.items():
                values = [v for run in runs.values() for v in run]
                stats = segment_statistics(values, len(runs))
                all_segments.extend(values)
                directions[direction] = {
                    "restarts": stats["failures"], "mean_distance_mm": stats["mean_segment_length_mm"] if runs else None,
                    "run_count": len(runs), "runs": {str(r): {"restarts": len(v)-1,
                        "length_mm": math.fsum(v)} for r, v in runs.items()}}
            stats = segment_statistics(all_segments, 2 * len(expected))
            eligible_length = math.fsum(all_segments) / 2
            original_length = selection[index]["original_length_base_voxels"] * scale if selection is not None else eligible_length
            rows.append({"collection": source["source_id"],
                         "scroll": binding["coordinate_space"].split("/")[0], "fiber": fiber,
                         "status": "evaluated" if expected else "excluded",
                         "reference_length_mm": eligible_length, "original_length_mm": original_length,
                         "excluded_length_mm": original_length - eligible_length, "eligible_runs": len(expected),
                         "restarts": stats["failures"],
                         "mean_distance_mm": stats["mean_segment_length_mm"] if expected else None,
                         "directions": directions})
        if set(grouped) - set(range(len(binding["fibers"]))):
            raise ValueError("unknown fiber membership")
    return sorted(rows, key=lambda r: (r["mean_distance_mm"] if r["mean_distance_mm"] is not None else math.inf,
                                       -r["restarts"], r["collection"], r["fiber"]))


def write_visualization_report(evaluation_root, report_root, *, provenance=None):
    from lasagna.inference_provenance import atomic_write
    evaluation_root, report_root = Path(evaluation_root), Path(report_root)
    if any((evaluation_root / n).exists() for n in ("incomplete.json", "inference_incomplete.json")):
        raise ValueError("cannot publish a completed visualization index for incomplete evaluation")
    summary = json.loads((evaluation_root / "summary.json").read_text())
    rows = fiber_rows(summary)
    lookup = {(r["collection"], str(Path(r["fiber"]).resolve())): r for r in rows}
    seen = set()
    for artifact in (evaluation_root / "cases").glob("*/trace.json"):
        document = json.loads(artifact.read_text())
        p = document["provenance"]
        key = (p["source_id"], str(Path(p["fiber_path"]).resolve()))
        direction = "reverse" if p["reverse"] else "forward"
        run = p.get("reference_run", {}).get("run_index", 0)
        if (key, direction, run) in seen:
            raise ValueError("duplicate rendered trace artifact")
        seen.add((key, direction, run))
        target = lookup[key]["directions"][direction]
        replay = document["replay"]
        begin, end = replay["reference_begin_arc_base"], replay["reference_end_arc_base"]
        segments = reference_segments({"reference_length_base_voxels": end-begin,
            "evaluated_through_base_voxels": replay["completed_reference_arc_base"]-begin,
            "evaluation_complete": True, "failures": [{"directional_reference_arc_base_voxels": f["reference_arc_base"]-begin}
                for f in replay["failures"]]})
        if len(segments)-1 != target["runs"][str(run)]["restarts"]:
            raise ValueError("artifact and summary restart counts differ")
        target["case_index"] = p["case_index"]
        target.setdefault("case_indices", []).append(p["case_index"])
        run_target = target["runs"][str(run)]
        run_target.update(case_index=p["case_index"], trace_json=str(artifact.relative_to(report_root)))
        combined = json.loads((artifact.parent / "rendered/render.json").read_text()).get("format") == "vc_fiber_inspection"
        for modality, folder in (("combined", "rendered"),) if combined else (("ct", "rendered"), ("presence", "rendered_presence")):
            root = artifact.parent / folder
            metadata = json.loads((root / "render.json").read_text())
            paths = [root / f"strips_{i}.jpg" for i in range(metadata["pages"])]
            if not paths or not all(path.is_file() for path in paths):
                raise ValueError("missing rendered pages")
            target.setdefault(modality, []).extend(str(path.relative_to(report_root)) for path in paths)
            run_target[modality] = [str(path.relative_to(report_root)) for path in paths]
    if len(seen) != sum(2 * row["eligible_runs"] for row in rows):
        raise ValueError("missing directional render artifacts")
    report = {"metric_definition": "complete-reference-partition-v1",
              "mean_formula": "2 * eligible_length_mm / (forward_restarts + reverse_restarts + 1)",
              "reference_selection_policy": summary.get("reference_selection_policy", "whole_explicit_fibers_no_crop"),
              "provenance": provenance or {}, "fibers": rows}
    atomic_write(report_root / "per_fiber.json", report)
    fields = ["scroll", "collection", "fiber", "reference_length_mm", "restarts", "mean_distance_mm",
              "forward_restarts", "forward_mean_mm", "reverse_restarts", "reverse_mean_mm",
              "original_length_mm", "excluded_length_mm", "eligible_runs", "status"]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        value = {key: row[key] for key in fields[:6]}
        for direction in ("forward", "reverse"):
            value[direction + "_restarts"] = row["directions"][direction]["restarts"]
            value[direction + "_mean_mm"] = row["directions"][direction]["mean_distance_mm"]
        for key in ("original_length_mm", "excluded_length_mm", "eligible_runs", "status"):
            value[key] = row[key]
        writer.writerow(value)
    (report_root / "per_fiber.csv").write_text(stream.getvalue())
    lines = ["# Fiber trace strip inspection", "",
             "All fibers, both directions. Sorted by shortest mean first. Terminal segments are included.",
             "Merged mean = 2 × eligible length / (restarts + 1). Excluded gaps add no failures or distance.", "",
             "| Collection / fiber | Eligible / excluded mm | Runs | Restarts | Mean mm | Forward restarts / mean mm | Reverse restarts / mean mm | Images |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    fmt = lambda value: "—" if value is None else f"{value:.4f}"
    for row in rows:
        f, r = row["directions"]["forward"], row["directions"]["reverse"]
        links = []
        for direction, short in (("forward", "F"), ("reverse", "R")):
            for modality in (("combined",) if "combined" in row["directions"][direction] else ("ct", "presence")):
                for i, path in enumerate(row["directions"][direction].get(modality, [])):
                    links.append(f"[{short} {modality} {i+1}]({quote(path)})")
        lines.append(f"| {row['collection']} / {Path(row['fiber']).stem} | {row['reference_length_mm']:.4f} / {row['excluded_length_mm']:.4f} | {row['eligible_runs']} | {row['restarts']} | {fmt(row['mean_distance_mm'])} | {f['restarts']} / {fmt(f['mean_distance_mm'])} | {r['restarts']} / {fmt(r['mean_distance_mm'])} | {' · '.join(links) or row['status']} |")
    lines += ["", "## Most restarts", "", "| Collection / fiber | Restarts | Mean mm |", "|---|---:|---:|"]
    for row in sorted(rows, key=lambda r: (-r["restarts"], r["mean_distance_mm"] if r["mean_distance_mm"] is not None else math.inf)):
        lines.append(f"| {row['collection']} / {Path(row['fiber']).stem} | {row['restarts']} | {fmt(row['mean_distance_mm'])} |")
    (report_root / "README.md").write_text("\n".join(lines) + "\n")
    return report
