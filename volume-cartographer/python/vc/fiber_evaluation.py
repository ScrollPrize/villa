"""Callable chunk/direct-beam reference evaluation; CLI is only config/result I/O.

No GPU inference, subprocesses, cohort selection or annotation rewriting.
References are explicit base-XYZ polylines, one entry per tracing direction.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from . import fiber_trace as ft


def _keys(values, allowed, name):
    if not isinstance(values, dict):
        raise ValueError(f"{name} must be an object")
    unknown = values.keys() - set(allowed)
    if unknown:
        raise ValueError(f"unknown {name} keys: {sorted(unknown)}")


def _settings(kind, values, allowed):
    _keys(values, allowed, kind.__name__)
    result = kind()
    for key, value in values.items():
        setattr(result, key, value)
    return result


CHUNK_KEYS = (
    "core_base", "halo_base", "grid_base", "seed_presence", "nms_base",
    "tube_half_base", "tube_step_base", "tube_presence", "tube_turn_degrees",
    "tube_total_degrees", "rim_drop", "rim_normal_base", "rim_tangent_base",
    "join_distance_base", "join_angle_degrees", "join_minimum_base",
    "join_width_weight", "pin_sampling_region", "max_grid_samples",
)
REPLAY_KEYS = ("error_threshold_base", "excursion_grace_base",
               "tangential_threshold_factor", "match_refine_steps")


def reference_cases(annotation, identity, *, direction=None):
    """Select CP-bounded non-lasagna runs with the shared native selector.

    Never bridge excluded spans or extrapolate. Input coordinates must already
    be base XYZ. Direction defaults to the annotation's evaluation_direction,
    or both. Return cases plus the full selection/exclusion provenance.
    """
    direction = annotation.get("evaluation_direction", "both") if direction is None else direction
    if direction not in ("both", "forward", "reverse"):
        raise ValueError("reference direction must be both, forward or reverse")
    plan = json.loads(ft.reference_plan_json(json.dumps(annotation, allow_nan=False)))
    cases = []
    for run in plan["runs"]:
        for reverse in (False, True):
            name = "reverse" if reverse else "forward"
            if direction not in ("both", name):
                continue
            points = run["points_base_xyz"]
            cases.append(dict(id=f"{identity}/run{run['run_index']}/{name}",
                              points_base_xyz=points[::-1] if reverse else points,
                              reference=dict(direction=name, **{k: v for k, v in run.items()
                                                                if k != "points_base_xyz"})))
    return cases, plan


def replay_dict(result):
    """Serialize native outcomes without reinterpreting failures as filters."""
    def measurement(value):
        if value is None:
            return None
        return {key: getattr(value, key) for key in (
            "euclidean_error_base", "normal_error_base", "tangential_error_base",
            "threshold_error_base", "threshold_error_ratio", "local_normal_valid", "tangential_factor")}

    segments = []
    retained = 0.
    for segment in result.segments:
        points = segment.points_base
        retained += float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        segments.append(dict(
            start_reference_arc_base=segment.start_reference_arc_base,
            end_reference_arc_base=segment.end_reference_arc_base,
            termination_reason=segment.termination_reason,
            points_base_xyz=points.tolist(),
            cumulative_losses=segment.cumulative_losses,
            filtered_distance_failure=segment.filtered_distance_failure,
            matches=[dict(trace_point_index=m.trace_point_index,
                          predicted_reference_arc_base=m.predicted_reference_arc_base,
                          matched_reference_arc_base=m.matched_reference_arc_base,
                          matched_reference_point_base=m.matched_reference_point_base,
                          search_begin_arc_base=m.search_begin_arc_base,
                          search_end_arc_base=m.search_end_arc_base,
                          threshold_measurement=measurement(m.threshold_measurement)) for m in segment.matches],
        ))
    failures = [dict(index=f.index, segment_index=f.segment_index, reason=f.reason,
                     reference_arc_base=f.reference_arc_base,
                     reference_arc_fraction=f.reference_arc_fraction,
                     reference_point_base=f.reference_point_base,
                     evaluator_point_base=f.evaluator_point_base,
                     segment_point_index=f.segment_point_index,
                     threshold_measurement=measurement(f.threshold_measurement)) for f in result.failures]
    return dict(segments=segments, failures=failures, filtered_stops=result.filtered_stops,
                reverse_checks=result.reverse_checks,
                tangential_threshold_factor=result.tangential_threshold_factor,
                reference_begin_arc_base=result.reference_begin_arc_base,
                reference_end_arc_base=result.reference_end_arc_base,
                completed_reference_arc_base=result.completed_reference_arc_base,
                seed_search_skipped_base=result.seed_search_skipped_base,
                retained_trace_length_base=retained)


def evaluate(config, *, base_directory=".", control=None, progress=None):
    """Run one source's cases with bounded native parallelism.

    ``progress(status)`` runs on the calling Python thread, never a suspended
    native stack. A callback exception or KeyboardInterrupt cancels and joins
    the native work before propagating. A supplied control is single-use and
    can also be polled/cancelled by the application. No config is reloaded.
    """
    cfg = copy.deepcopy(config)
    _keys(cfg, ("version", "prediction_manifest", "normal_manifest", "evaluation_normal_manifest",
                "source_identity", "chunk_cache", "scaledown_power", "cache_bytes", "enable_width",
                "remote_cache_root", "trace", "chunks", "replay", "cases", "workers",
                "contexts_per_worker", "reverse_interval_base", "live_endpoint", "backend"), "evaluation")
    if cfg.get("version") != 1:
        raise ValueError("evaluation version must be 1")
    backend = cfg.get("backend", "chunk")
    if backend not in ("chunk", "beam"):
        raise ValueError("backend must be chunk or beam")
    if backend == "beam" and ("chunks" in cfg or "chunk_cache" in cfg):
        raise ValueError("beam evaluation does not use chunks or chunk_cache")
    base = Path(base_directory).resolve()

    def location(value):
        text = str(value)
        return text if "://" in text else str((base / text).resolve())

    selected, selection = [], []
    input_ids = set()
    for item in cfg["cases"]:
        _keys(item, ("id", "points_base_xyz", "begin_arc_base", "end_arc_base", "fiber_path", "direction"), "case")
        identity = item["id"]
        if not isinstance(identity, str) or not identity or identity in input_ids:
            raise ValueError("case IDs must be nonempty unique strings")
        input_ids.add(identity)
        if "fiber_path" in item:
            if any(k in item for k in ("points_base_xyz", "begin_arc_base", "end_arc_base")):
                raise ValueError("fiber_path cannot be combined with explicit points/arc limits")
            path = Path(location(item["fiber_path"]))
            data = path.read_bytes()
            rows, plan = reference_cases(json.loads(data), identity, direction=item.get("direction"))
            selected.extend(rows)
            selection.append(dict(id=identity, fiber_path=str(path), sha256=hashlib.sha256(data).hexdigest(), **plan))
        else:
            if "direction" in item:
                raise ValueError("explicit points already specify direction")
            selected.append(item)
    identities = set()
    cases = []
    for item in selected:
        identity = item["id"]
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ValueError("case IDs must be nonempty unique strings")
        identities.add(identity)
        points = np.ascontiguousarray(item["points_base_xyz"], dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
            raise ValueError("case points must be finite (N,3) base XYZ, N >= 2")
        cases.append(ft.EvaluationCase(points, item.get("begin_arc_base", 0.), item.get("end_arc_base")))
    source_identity = cfg["source_identity"]
    if not isinstance(source_identity, str) or not source_identity:
        raise ValueError("source_identity must identify immutable prediction/normal data")
    cache_bytes = cfg.get("cache_bytes", 512 << 20)
    remote = location(cfg["remote_cache_root"]) if cfg.get("remote_cache_root") else None
    field = ft.open_prediction_field(location(cfg["prediction_manifest"]), cache_bytes=cache_bytes,
        scaledown_power=cfg.get("scaledown_power", 0), remote_cache_root=remote,
        enable_width=cfg.get("enable_width", False))
    normals = ft.open_normal_sampler(location(cfg["normal_manifest"]), field.trace_to_base_scale,
        cache_bytes=cache_bytes, remote_cache_root=remote)
    evaluation_normals = normals
    if cfg.get("evaluation_normal_manifest"):
        evaluation_normals = ft.open_normal_sampler(location(cfg["evaluation_normal_manifest"]),
            field.trace_to_base_scale, cache_bytes=cache_bytes, remote_cache_root=remote)
    trace = dict(parallel_threads=1, step_voxels=16 / field.trace_to_base_scale,
                 trace_to_base_scale=field.trace_to_base_scale)
    trace.update(cfg.get("trace", {}))
    trace = ft.TraceConfig.from_dict(trace)
    replay = _settings(ft.ChunkReplayConfig, cfg.get("replay", {}), REPLAY_KEYS)
    if backend == "chunk":
        chunks = _settings(ft.ChunkConfig, cfg.get("chunks", {}), CHUNK_KEYS)
        cache = ft.ChunkCache(field, normals, trace, chunks, location(cfg["chunk_cache"]), source_identity)
    control = control if control is not None else ft.EvaluationControl()
    bridge = None
    try:
        if "live_endpoint" in cfg:
            endpoint = cfg["live_endpoint"]
            _keys(endpoint, ("port", "token", "source", "timeout_seconds", "heartbeat_wait"), "live_endpoint")
            bridge = ft.LiveBridge(endpoint["port"], endpoint["token"],
                endpoint.get("timeout_seconds", 600), endpoint.get("heartbeat_wait", False))
            field.set_live_bridge(bridge, endpoint["source"])
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="fiber-evaluation") as pool:
            budgets = (cfg.get("workers", 4), cfg.get("contexts_per_worker", 8),
                       cfg.get("reverse_interval_base", 0.))
            if backend == "chunk":
                future = pool.submit(cache.replay_many, cases, evaluation_normals, replay, control, *budgets)
            else:
                future = pool.submit(ft.beam_replay_many, field, normals, evaluation_normals,
                                     trace, cases, replay, control, *budgets)
            try:
                while True:
                    try:
                        results = future.result(timeout=.25)
                        break
                    except TimeoutError:
                        if progress is not None:
                            progress(control.status())
                if progress is not None:
                    progress(control.status())
            except BaseException:
                control.cancel("evaluation caller interrupted")
                raise
        rows = [dict(id=item["id"], **({"reference": item["reference"]} if "reference" in item else {}),
                     **replay_dict(result)) for item, result in zip(selected, results, strict=True)]
        public_config = copy.deepcopy(cfg)
        if "live_endpoint" in public_config:
            public_config["live_endpoint"].pop("token", None)
        return dict(format="vc_chunk_evaluation" if backend == "chunk" else "vc_beam_evaluation",
                    version=1, config=public_config, reference_selection=selection,
                    cases=rows, failure_count=sum(len(row["failures"]) for row in rows),
                    filtered_stop_count=sum(len(row["filtered_stops"]) for row in rows),
                    retained_trace_length_base=sum(row["retained_trace_length_base"] for row in rows))
    finally:
        if bridge is not None:
            bridge.cancel("evaluation session finished")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    # All input is read once, before starting native work.
    config = json.loads(args.config.read_text())
    previous = None

    def progress(status):
        nonlocal previous
        if status != previous:
            print(json.dumps(status), flush=True)
            previous = status

    result = evaluate(config, base_directory=args.config.resolve().parent, progress=progress)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent,
                                         prefix=args.output.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(result, stream, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, args.output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
