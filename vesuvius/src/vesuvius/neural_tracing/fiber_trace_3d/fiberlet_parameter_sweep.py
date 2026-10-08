"""Resumable graph-replay parameter matrices for the fixed short benchmarks."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import threading

from lasagna.inference_provenance import atomic_write, sha256_file


_PARAMETERS = {"beam_width", "lookahead_base_voxels", "beam_step_base_voxels"}
_NAME = re.compile(r"[a-z0-9][a-z0-9_-]*")
_PATH_DEFAULTS = {
    "cell_radius": 4,
    "neighborhood_margin_cells": 0.5,
    "longitudinal_step_prediction": 2.0,
    "transverse_step_prediction": 0.5,
    "maximum_endpoint_angle_degrees": 45.0,
    "maximum_prediction_deviation_degrees": 25.0,
    "corridor_radius_prediction": 0.0,
    "invalid_prediction_cost_per_voxel": 4.0,
    "smoothness_weight": 2.0,
    "smoothness_normal_weight": 0.1,
    "smoothness_tangent_weight": 10.0,
    "smoothness_free_angle_degrees": 0.0,
}


def ordered_variants(workload):
    settings = workload.get("fiberlet_sweep")
    if not isinstance(settings, dict) or set(settings) != {"variants"}:
        raise ValueError("fiberlet_sweep must contain only variants")
    variants = settings["variants"]
    if not isinstance(variants, list) or not variants:
        raise ValueError("fiberlet_sweep.variants must be a nonempty list")
    base = workload.get("fiberlet")
    if not isinstance(base, dict) or "anchor_fit" in base:
        raise ValueError("fiberlet parameter sweeps require legacy production anchors")
    result, names, warmup_settings, measured_settings = [], set(), set(), set()
    for entry in variants:
        if not isinstance(entry, dict) or set(entry) - {"name", "parameters", "warmup"}:
            raise ValueError("each fiberlet sweep variant requires name/parameters and optional warmup")
        name, parameters = entry.get("name"), entry.get("parameters")
        if not isinstance(name, str) or _NAME.fullmatch(name) is None or name in names:
            raise ValueError("fiberlet sweep variant names must be unique safe identifiers")
        if not isinstance(parameters, dict) or set(parameters) != _PARAMETERS:
            raise ValueError(f"fiberlet sweep parameters must be exactly {sorted(_PARAMETERS)}")
        width = parameters["beam_width"]
        step = parameters["beam_step_base_voxels"]
        lookahead = parameters["lookahead_base_voxels"]
        if type(width) is not int or width < 1:
            raise ValueError("fiberlet beam_width must be a positive integer")
        if any(type(value) not in (int, float) or value <= 0 or value % 8 != 0
               for value in (step, lookahead)) or lookahead < step:
            raise ValueError("fiberlet step/lookahead must be positive 8-base lattice values with lookahead >= step")
        identity = (width, float(step), float(lookahead))
        warmup = entry.get("warmup", False)
        if type(warmup) is not bool:
            raise ValueError("fiberlet sweep warmup must be boolean")
        if warmup:
            if identity in warmup_settings or identity in measured_settings:
                raise ValueError("fiberlet warmup settings must precede and may occur only once")
            warmup_settings.add(identity)
        elif identity in measured_settings:
            raise ValueError("measured fiberlet settings must be unique")
        else:
            measured_settings.add(identity)
        names.add(name)
        result.append({"name": name, "parameters": deepcopy(parameters),
                       "warmup": warmup, "identity": identity})
    return result


def ordered_path_variants(workload):
    settings = workload.get("fiberlet_path_sweep")
    if not isinstance(settings, dict) or set(settings) != {"variants"}:
        raise ValueError("fiberlet_path_sweep must contain only variants")
    variants = settings["variants"]
    if not isinstance(variants, list) or not variants:
        raise ValueError("fiberlet_path_sweep.variants must be a nonempty list")
    base = workload.get("fiberlet")
    if not isinstance(base, dict) or "anchor_fit" in base:
        raise ValueError("fiberlet path sweeps require legacy production anchors")
    configured = base.get("path", {})
    if not isinstance(configured, dict) or set(configured) - set(_PATH_DEFAULTS):
        raise ValueError("fiberlet.path contains an unknown semantic parameter")
    defaults = {**_PATH_DEFAULTS, **configured}
    result, names, warmup_settings, measured_settings = [], set(), set(), set()
    for entry in variants:
        if not isinstance(entry, dict) or set(entry) - {"name", "parameters", "warmup"}:
            raise ValueError("each fiberlet path variant requires name/parameters and optional warmup")
        name, parameters = entry.get("name"), entry.get("parameters")
        if not isinstance(name, str) or _NAME.fullmatch(name) is None or name in names:
            raise ValueError("fiberlet path variant names must be unique safe identifiers")
        if not isinstance(parameters, dict) or set(parameters) - set(_PATH_DEFAULTS):
            raise ValueError("fiberlet path parameters must be semantic FiberletPathConfig fields")
        resolved = {**defaults, **parameters}
        if type(resolved["cell_radius"]) is not int or not 1 <= resolved["cell_radius"] <= 64:
            raise ValueError("fiberlet path cell_radius must be an integer in [1, 64]")
        positive = ("neighborhood_margin_cells", "longitudinal_step_prediction",
                    "transverse_step_prediction")
        bounded = ("maximum_endpoint_angle_degrees", "maximum_prediction_deviation_degrees")
        nonnegative = ("corridor_radius_prediction", "invalid_prediction_cost_per_voxel",
                       "smoothness_weight", "smoothness_normal_weight",
                       "smoothness_tangent_weight", "smoothness_free_angle_degrees")
        if any(type(resolved[key]) not in (int, float) or resolved[key] <= 0 for key in positive):
            raise ValueError("fiberlet path lattice and margin values must be positive")
        if any(type(resolved[key]) not in (int, float) or not 0 <= resolved[key] <= 90 for key in bounded):
            raise ValueError("fiberlet path angle values must be in [0, 90]")
        if any(type(resolved[key]) not in (int, float) or resolved[key] < 0 for key in nonnegative):
            raise ValueError("fiberlet path cost and corridor values must be nonnegative")
        # Native radius zero means one anchor cell (currently four prediction
        # voxels). Explicit four is behaviorally identical but hashes differently.
        if parameters.get("corridor_radius_prediction") == 4:
            raise ValueError("explicit corridor radius 4 duplicates the native radius-zero default")
        identity = tuple((key, resolved[key]) for key in _PATH_DEFAULTS)
        warmup = entry.get("warmup", False)
        if type(warmup) is not bool:
            raise ValueError("fiberlet path sweep warmup must be boolean")
        if warmup:
            if identity in warmup_settings or identity in measured_settings:
                raise ValueError("fiberlet path warmup settings must precede and may occur only once")
            warmup_settings.add(identity)
        elif identity in measured_settings:
            raise ValueError("measured fiberlet path settings must be unique")
        else:
            measured_settings.add(identity)
        names.add(name)
        result.append({"name": name, "parameters": deepcopy(parameters),
                       "resolved_path": resolved, "warmup": warmup, "identity": identity})
    return result


def case_diagnostics(output_root):
    root = Path(output_root)
    paths = sorted(root.glob("full/cases/*/trace.json"))
    paths.extend(sorted(root.glob("scrolls/*/cases/*/trace.json")))
    rows = []
    for path in paths:
        trace = json.loads(path.read_text())
        replay = trace["replay"]
        failures = replay["failures"]
        reference_end = float(replay["reference_end_arc_base"])
        first = reference_end if not failures else float(failures[0]["reference_arc_base"])
        rows.append({
            "fiber_path": trace["provenance"]["fiber_path"],
            "coordinate_space": trace["provenance"]["coordinate_space"],
            "case_artifact": str(path),
            "failures": len(failures),
            "first_failure_or_end_base": first,
            "reference_end_base": reference_end,
            "first_failure_fraction": first / reference_end,
            "first_failure_reason": None if not failures else failures[0]["reason"],
        })
    if not rows:
        raise ValueError("fiberlet sweep candidate has no trace artifacts")
    rows.sort(key=lambda row: (row["fiber_path"], row["case_artifact"]))
    return rows


def coverage_percent(report):
    traced = sum(float(row["traced_length_mm"]) for row in report["coverage"].values())
    reference = float(report["global_result"]["directed_reference_mm"])
    return 100.0 * traced / reference


def ranking_key(row):
    if row["status"] != "completed" or row["warmup"]:
        return (1, float("inf"), float("inf"), float("inf"), row["name"])
    result = row["global_result"]
    return (0, int(result["failures"]), -float(row["coverage_percent"]),
            -float(row["first_failure_total_base"]), row["name"])


def _matrix_markdown(rows):
    out = ["# Fiberlet graph parameter sweep", "",
           "Production anchors and one frozen prediction/model/benchmark identity. "
           "Warmup rows are excluded from ranking. Wall time includes runner and cache effects.", "",
           "| Rank | Variant | Beam | Step base | Lookahead base | Failures | Coverage % | Mean mm | Median mm | First-failure total base | Wall s | Status |",
           "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    ranked = {row["name"]: index + 1 for index, row in enumerate(
        sorted((row for row in rows if row["status"] == "completed" and not row["warmup"]), key=ranking_key))}
    for row in rows:
        p = row["parameters"]
        if row["status"] == "completed":
            g = row["global_result"]
            values = (g["failures"], f'{row["coverage_percent"]:.6f}',
                      f'{g["mean_segment_length_mm"]:.6f}', f'{g["median_segment_length_mm"]:.6f}',
                      f'{row["first_failure_total_base"]:.3f}', f'{row["wall_seconds"]:.3f}')
        else:
            values = ("—", "—", "—", "—", "—", "—")
        rank = "warmup" if row["warmup"] else ranked.get(row["name"], "—")
        out.append(f"| {rank} | {row['name']} | {p['beam_width']} | "
                   f"{p['beam_step_base_voxels']} | {p['lookahead_base_voxels']} | "
                   f"{values[0]} | {values[1]} | {values[2]} | {values[3]} | "
                   f"{values[4]} | {values[5]} | {row['status']} |")
    return "\n".join(out) + "\n"


def _path_matrix_markdown(rows):
    out = ["# Fiberlet construction-DP parameter sweep", "",
           "Production anchors and fixed beam16/step48/lookahead384 graph replay. "
           "Every resolved path identity has a separate construction root, including duplicated anchors.", "",
           "| Rank | Variant | Longitudinal | Transverse | Other overrides | Failures | Coverage % | Mean mm | Median mm | First-failure total base | Wall s | Status |",
           "|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|"]
    ranked = {row["name"]: index + 1 for index, row in enumerate(
        sorted((row for row in rows if row["status"] == "completed" and not row["warmup"]), key=ranking_key))}
    for row in rows:
        p = row.get("resolved_path", row["parameters"])
        overrides = {key: value for key, value in row["parameters"].items()
                     if key not in {"longitudinal_step_prediction", "transverse_step_prediction"}}
        if row["status"] == "completed":
            g = row["global_result"]
            values = (g["failures"], f'{row["coverage_percent"]:.6f}',
                      f'{g["mean_segment_length_mm"]:.6f}', f'{g["median_segment_length_mm"]:.6f}',
                      f'{row["first_failure_total_base"]:.3f}', f'{row["wall_seconds"]:.3f}')
        else:
            values = ("—", "—", "—", "—", "—", "—")
        rank = "warmup" if row["warmup"] else ranked.get(row["name"], "—")
        out.append(f"| {rank} | {row['name']} | {p['longitudinal_step_prediction']} | "
                   f"{p['transverse_step_prediction']} | `{json.dumps(overrides, sort_keys=True)}` | "
                   f"{values[0]} | {values[1]} | {values[2]} | {values[3]} | "
                   f"{values[4]} | {values[5]} | {row['status']} |")
    return "\n".join(out) + "\n"


def _validate_completed(pointer, variant, identity):
    saved = json.loads(pointer.read_text())
    expected_variant = {key: value for key, value in variant.items() if key != "identity"}
    if saved["variant"] != expected_variant or saved["matrix_identity"] != identity:
        raise ValueError("completed fiberlet sweep candidate provenance changed")
    if saved["row"]["status"] == "failed":
        if saved["metrics_sha256"] is not None:
            raise ValueError("failed fiberlet sweep candidate unexpectedly has metrics")
        return saved["row"]
    metrics = Path(saved["attempt_root"]) / "metrics.json"
    if not metrics.is_file() or sha256_file(metrics) != saved["metrics_sha256"]:
        raise ValueError("completed fiberlet sweep candidate metrics changed")
    return saved["row"]


def next_attempt_root(candidate_root):
    candidate_root = Path(candidate_root)
    indices = [int(path.name.removeprefix("attempt_"))
               for path in candidate_root.glob("attempt_[0-9][0-9][0-9]")]
    return candidate_root / f"attempt_{max(indices, default=0) + 1:03d}"


def _run_matrix(workload, devices, gpu_manager, *, variants, sweep_key, markdown):
    from .short_benchmark_evaluate import run
    root = Path(workload["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        "workload": workload,
        "manifest_sha256": sha256_file(workload["benchmark_manifest"]),
        "checkpoint_sha256": sha256_file(workload["checkpoint"]),
        "native_sha256": sha256_file(workload["native_binary"]),
        "ordered_variants": [{key: value for key, value in row.items() if key != "identity"}
                             for row in variants],
    }
    identity["sha256"] = hashlib.sha256(json.dumps(identity, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    provenance = root / "sweep_provenance.json"
    if provenance.exists():
        if json.loads(provenance.read_text()) != identity:
            raise ValueError("fiberlet sweep identity changed; use a new output root")
    else:
        atomic_write(provenance, identity)
    rows = []
    timeout = float(workload.get("candidate_timeout_seconds", 180))
    if timeout <= 0:
        raise ValueError("candidate_timeout_seconds must be positive")
    for variant in variants:
        candidate_root = root / variant["name"]
        candidate_root.mkdir(exist_ok=True)
        pointer = candidate_root / "result.json"
        if pointer.exists():
            row = _validate_completed(pointer, variant, identity)
            print(f"fiberlet-sweep skip {variant['name']}: {row['status']}", flush=True)
            rows.append(row)
            continue
        attempt_root = next_attempt_root(candidate_root)
        current = deepcopy(workload)
        current.pop(sweep_key)
        current["output_root"] = str(attempt_root)
        current["quiet_report"] = True
        current["log_construction_chunks"] = False
        if sweep_key == "fiberlet_sweep":
            current["fiberlet"] = {**current["fiberlet"], **variant["parameters"]}
        else:
            current["fiberlet"] = {**current["fiberlet"], "path": variant["resolved_path"]}
        cancel = threading.Event()
        timer = threading.Timer(timeout, cancel.set)
        print(f"fiberlet-sweep start {variant['name']} attempt={attempt_root.name} "
              f"parameters={variant['parameters']}", flush=True)
        timer.start()
        try:
            report = run(current, devices, gpu_manager=gpu_manager, cancel_event=cancel)
            diagnostics = case_diagnostics(attempt_root)
            row = {
                "name": variant["name"], "parameters": variant["parameters"],
                "warmup": variant["warmup"], "status": "completed",
                "attempt_root": str(attempt_root), "wall_seconds": report["wall_seconds"],
                "global_result": report["global_result"], "scrolls": report["scrolls"],
                "coverage": report["coverage"], "coverage_percent": coverage_percent(report),
                "case_diagnostics": diagnostics,
                "first_failure_total_base": sum(row["first_failure_or_end_base"] for row in diagnostics),
            }
            if "resolved_path" in variant:
                row["resolved_path"] = variant["resolved_path"]
        except Exception as error:
            row = {
                "name": variant["name"], "parameters": variant["parameters"],
                "warmup": variant["warmup"], "status": "failed",
                "attempt_root": str(attempt_root), "error": f"{type(error).__name__}: {error}",
            }
            if "resolved_path" in variant:
                row["resolved_path"] = variant["resolved_path"]
            atomic_write(candidate_root / "failure.json", row)
            if variant["warmup"] or variant["name"].startswith("baseline"):
                raise
        finally:
            timer.cancel()
        saved = {"variant": {key: value for key, value in variant.items() if key != "identity"},
                 "matrix_identity": identity, "attempt_root": str(attempt_root),
                 "metrics_sha256": (sha256_file(attempt_root / "metrics.json")
                                    if row["status"] == "completed" else None), "row": row}
        atomic_write(pointer, saved)
        rows.append(row)
        atomic_write(root / "matrix.json", rows)
        (root / "matrix.md").write_text(markdown(rows))
        print(f"fiberlet-sweep done {variant['name']}: status={row['status']}", flush=True)
    ranked = sorted(rows, key=ranking_key)
    atomic_write(root / "matrix.json", rows)
    (root / "matrix.md").write_text(markdown(rows))
    atomic_write(root / "ranking.json", ranked)
    atomic_write(root / "completed.json", {"completed": True, "variants": len(rows),
                 "matrix_identity_sha256": identity["sha256"]})
    return rows


def run_sweep(workload, devices, gpu_manager=None):
    return _run_matrix(workload, devices, gpu_manager,
        variants=ordered_variants(workload), sweep_key="fiberlet_sweep",
        markdown=_matrix_markdown)


def run_path_sweep(workload, devices, gpu_manager=None):
    return _run_matrix(workload, devices, gpu_manager,
        variants=ordered_path_variants(workload), sweep_key="fiberlet_path_sweep",
        markdown=_path_matrix_markdown)
