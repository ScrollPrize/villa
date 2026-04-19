from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import numpy as np

from vesuvius.neural_tracing.autoreg_mesh.extend_tifxyz import extend_tifxyz_mesh
from vesuvius.tifxyz import Tifxyz, list_tifxyz, read_tifxyz, write_tifxyz


DEFAULT_SOURCE_S3_URI = None
DEFAULT_OUTPUT_S3_URI = None
DEFAULT_VOLUME_URI = "s3://vesuvius-challenge-open-data/PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr/"
DEFAULT_LOCAL_SOURCE_ROOT = Path("/ephemeral/title_tracer_inputs")
DEFAULT_LOCAL_OUTPUT_ROOT = Path("/ephemeral/title_tracer_outputs")
DEFAULT_STATE_ROOT = Path("/ephemeral/title_tracer_state")
DEFAULT_DRY_RUN_SURFACE_COUNT = 2
DEFAULT_COORDINATE_SCALE_FACTOR = 4.0
DEFAULT_MAX_EXTENSION_ITERS_PER_CALL = 16
DEFAULT_WINDOW_BATCH_SIZE = 256
DEFAULT_DRY_RUN_MAX_CALLS_PER_DIRECTION = 2
TITLE_HUNT_PROMPT_STRIPS = 6
TITLE_HUNT_PREDICT_STRIPS_PER_ITER = 1
TITLE_HUNT_WINDOW_STRIP_LENGTH = 16
TITLE_HUNT_WINDOW_OVERLAP = 8
TITLE_HUNT_DRY_RUN_MIN_FRONTIER_COVERAGE = 0.70
TITLE_HUNT_DRY_RUN_MAX_GAP = 32
REQUIRED_AWS_ENV = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")
SYNC_EXCLUDE_PATTERNS = (
    "layers",
    "layers/*",
    "*/layers",
    "*/layers/*",
    "*.zarr",
    "*.zarr/*",
    "*/*.zarr",
    "*/*.zarr/*",
)


@dataclass
class SurfacePaths:
    relative_surface_id: str
    prefix_name: str
    surface_dir_name: str
    timestamp_suffix: str
    state_path: Path
    output_dir: Path
    scaled_input_dir: Path
    final_tifxyz_dir: Path
    manifest_path: Path
    summaries_dir: Path
    s3_surface_prefix: str
    s3_final_tifxyz_uri: str
    s3_manifest_uri: str


@dataclass
class DirectionRunResult:
    final_tifxyz_path: Path
    stage_summary: dict[str, Any]


def _require_aws_env() -> None:
    missing = [key for key in REQUIRED_AWS_ENV if not os.environ.get(key)]
    if missing:
        raise RuntimeError(f"Missing AWS environment variables: {', '.join(missing)}")


def _timestamp_suffix(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return now.strftime("%Y%m%dT%H%M%SZ")


def _safe_state_name(relative_surface_id: str) -> str:
    return relative_surface_id.replace("/", "__")


def _surface_paths(
    *,
    relative_surface_id: str,
    timestamp_suffix: str,
    local_output_root: Path,
    state_root: Path,
    output_s3_uri: str,
) -> SurfacePaths:
    relative_path = Path(relative_surface_id)
    prefix_name = relative_path.parts[0]
    surface_dir_name = relative_path.name
    base_name = f"{surface_dir_name}__{timestamp_suffix}"
    output_dir = local_output_root / prefix_name / base_name
    s3_surface_prefix = _join_s3(output_s3_uri, prefix_name, base_name)
    return SurfacePaths(
        relative_surface_id=relative_surface_id,
        prefix_name=prefix_name,
        surface_dir_name=surface_dir_name,
        timestamp_suffix=timestamp_suffix,
        state_path=state_root / f"{_safe_state_name(relative_surface_id)}.json",
        output_dir=output_dir,
        scaled_input_dir=output_dir / "scaled_input_tifxyz",
        final_tifxyz_dir=output_dir / "final_tifxyz",
        manifest_path=output_dir / "surface_manifest.json",
        summaries_dir=output_dir / "summaries",
        s3_surface_prefix=s3_surface_prefix,
        s3_final_tifxyz_uri=_join_s3(s3_surface_prefix, "final_tifxyz"),
        s3_manifest_uri=_join_s3(s3_surface_prefix, "surface_manifest.json"),
    )


def _join_s3(base_uri: str, *parts: str) -> str:
    if base_uri is None:
        raise ValueError("base_uri must be provided")
    base = str(base_uri).rstrip("/")
    suffix = "/".join(str(part).strip("/") for part in parts if str(part))
    if not suffix:
        return base
    return f"{base}/{suffix}"


def _run_cmd(args: list[str], *, cwd: Path | None = None, capture_output: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=None if cwd is None else str(cwd),
        check=True,
        text=True,
        capture_output=capture_output,
        env=os.environ.copy(),
    )


def _log(message: str) -> None:
    print(message, flush=True)


def _aws_sync_exclude_args() -> list[str]:
    args: list[str] = []
    for pattern in SYNC_EXCLUDE_PATTERNS:
        args.extend(["--exclude", str(pattern)])
    return args


def _append_manifest_event(manifest_jsonl_path: Path, event: dict[str, Any]) -> None:
    manifest_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ts": _timestamp_suffix(), **event}
    with manifest_jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _load_surface_state(state_path: Path) -> dict[str, Any] | None:
    if not state_path.exists():
        return None
    return json.loads(state_path.read_text())


def _write_surface_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2))


def _update_surface_state(
    *,
    state: dict[str, Any],
    status: str,
    manifest_jsonl_path: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = dict(state)
    state["status"] = str(status)
    state["updated_at"] = _timestamp_suffix()
    if extra:
        state.update(extra)
    _write_surface_state(Path(state["state_path"]), state)
    _append_manifest_event(manifest_jsonl_path, {"relative_surface_id": state["relative_surface_id"], "status": status, **(extra or {})})
    return state


def _list_title_hunt_prefixes(source_s3_uri: str) -> list[str]:
    _require_aws_env()
    proc = _run_cmd(["aws", "s3", "ls", source_s3_uri])
    prefixes = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "PRE":
            prefix = parts[1].strip("/")
            if prefix.startswith("title_hunt_"):
                prefixes.append(prefix)
    prefixes.sort()
    return prefixes


def _sync_title_hunt_prefix(
    *,
    source_s3_uri: str,
    prefix_name: str,
    local_source_root: Path,
) -> Path:
    _require_aws_env()
    local_prefix_root = local_source_root / prefix_name
    local_prefix_root.parent.mkdir(parents=True, exist_ok=True)
    _log(f"[sync] start {prefix_name} -> {local_prefix_root}")
    _run_cmd(
        [
            "aws",
            "s3",
            "sync",
            _join_s3(source_s3_uri, prefix_name) + "/",
            str(local_prefix_root),
            "--only-show-errors",
            *_aws_sync_exclude_args(),
        ],
        capture_output=True,
    )
    _log(f"[sync] done  {prefix_name}")
    return local_prefix_root


def _discover_local_surfaces(local_source_root: Path, *, prefixes: list[str]) -> list[dict[str, str]]:
    discovered: list[dict[str, str]] = []
    for prefix_name in prefixes:
        prefix_root = local_source_root / prefix_name
        for info in list_tifxyz(prefix_root, recursive=True):
            relative_surface_id = str(info.path.relative_to(local_source_root))
            discovered.append(
                {
                    "prefix_name": prefix_name,
                    "relative_surface_id": relative_surface_id,
                    "local_path": str(info.path),
                    "uuid": str(info.uuid),
                }
            )
    discovered.sort(key=lambda item: item["relative_surface_id"])
    return discovered


def _select_dry_run_surfaces(surfaces: list[dict[str, str]], *, count: int) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    seen_prefixes: set[str] = set()
    for surface in surfaces:
        prefix = str(surface["prefix_name"])
        if prefix in seen_prefixes:
            continue
        selected.append(surface)
        seen_prefixes.add(prefix)
        if len(selected) >= int(count):
            break
    return selected


def _scale_stored_tifxyz(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    coordinate_scale_factor: float,
) -> Path:
    source = read_tifxyz(source_dir, load_mask=True, validate=True).use_stored_resolution()
    valid_mask = source._mask.copy() if source._mask is not None else np.ones_like(source._x, dtype=bool)
    scaled_x = source._x.copy()
    scaled_y = source._y.copy()
    scaled_z = source._z.copy()
    scaled_x[valid_mask] *= float(coordinate_scale_factor)
    scaled_y[valid_mask] *= float(coordinate_scale_factor)
    scaled_z[valid_mask] *= float(coordinate_scale_factor)
    scaled_bbox = None
    if source.bbox is not None:
        scaled_bbox = tuple(float(v) * float(coordinate_scale_factor) for v in source.bbox)
    scaled_surface = Tifxyz(
        _x=scaled_x,
        _y=scaled_y,
        _z=scaled_z,
        uuid=source.uuid,
        _scale=tuple(float(v) for v in source._scale),
        bbox=scaled_bbox,
        area=None if source.area is None else float(source.area) * float(coordinate_scale_factor) * float(coordinate_scale_factor),
        extra=dict(source.extra),
        _mask=valid_mask,
        path=Path(output_dir),
        interp_method=source.interp_method,
        resolution="stored",
    )
    return write_tifxyz(output_dir, scaled_surface, overwrite=True)


def _compact_extension_summary(summary: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "direction",
        "predicted_vertex_count",
        "cumulative_predicted_vertex_count",
        "final_predicted_nonseam_vertex_count",
        "final_seam_vertex_count",
        "new_band_frontier_coverage_fraction",
        "new_band_cell_coverage_fraction",
        "new_band_max_gap",
        "new_band_gap_spans",
        "first_uncovered_frontier_index",
        "iterations_completed",
        "stop_reason",
        "window_batch_size",
        "encode_decode_ms_per_fitted_window",
        "crop_cache_hits",
        "crop_cache_misses",
        "total_wall_ms",
        "fast_infer_enabled",
        "compile_infer_requested",
        "compile_infer_actual",
        "amp_dtype",
    ]
    compact = {key: summary.get(key) for key in keys}
    compact["summary_path"] = summary.get("summary_path")
    return compact


def _title_hunt_extension_preset(*, window_batch_size: int, show_progress: bool, distributed_infer: bool) -> dict[str, Any]:
    return {
        "prompt_strips": int(TITLE_HUNT_PROMPT_STRIPS),
        "predict_strips_per_iter": int(TITLE_HUNT_PREDICT_STRIPS_PER_ITER),
        "window_strip_length": int(TITLE_HUNT_WINDOW_STRIP_LENGTH),
        "window_overlap": int(TITLE_HUNT_WINDOW_OVERLAP),
        "window_batch_size": int(window_batch_size),
        "max_crop_fit_retries": 3,
        "show_progress": bool(show_progress),
        "fast_infer": True,
        "compile_infer": False,
        "amp_dtype": "bf16",
        "distributed_infer": bool(distributed_infer),
        "distributed_shard_mode": "strided",
        "distributed_gather_mode": "object",
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _run_direction_to_exhaustion(
    *,
    input_tifxyz_path: Path,
    output_dir: Path,
    direction: str,
    volume_uri: str,
    dino_backbone: str,
    autoreg_checkpoint: str,
    window_batch_size: int,
    max_extension_iters_per_call: int,
    show_progress: bool,
    max_calls: int | None = None,
    distributed_infer: bool = False,
    enforce_dry_run_quality_gate: bool = False,
) -> DirectionRunResult:
    current_input = Path(input_tifxyz_path)
    call_idx = 0
    call_summaries: list[dict[str, Any]] = []
    stage_stop_reason = "completed"
    preset = _title_hunt_extension_preset(
        window_batch_size=int(window_batch_size),
        show_progress=bool(show_progress),
        distributed_infer=bool(distributed_infer),
    )
    while True:
        call_dir = output_dir / f"{direction}_call_{call_idx:03d}"
        _log(f"[extend:{direction}] call {call_idx:03d} input={current_input}")
        summary = extend_tifxyz_mesh(
            tifxyz_path=current_input,
            volume_uri=volume_uri,
            dino_backbone=dino_backbone,
            autoreg_checkpoint=autoreg_checkpoint,
            out_dir=call_dir,
            device="cuda",
            grow_direction=direction,
            prompt_strips=int(preset["prompt_strips"]),
            predict_strips_per_iter=int(preset["predict_strips_per_iter"]),
            window_strip_length=int(preset["window_strip_length"]),
            window_overlap=int(preset["window_overlap"]),
            window_batch_size=int(preset["window_batch_size"]),
            max_extension_iters=int(max_extension_iters_per_call),
            max_crop_fit_retries=int(preset["max_crop_fit_retries"]),
            show_progress=bool(preset["show_progress"]),
            fast_infer=bool(preset["fast_infer"]),
            compile_infer=bool(preset["compile_infer"]),
            amp_dtype=str(preset["amp_dtype"]),
            distributed_infer=bool(preset["distributed_infer"]),
            distributed_shard_mode=str(preset["distributed_shard_mode"]),
            distributed_gather_mode=str(preset["distributed_gather_mode"]),
        )
        compact_summary = _compact_extension_summary(summary)
        call_summaries.append(compact_summary)
        if enforce_dry_run_quality_gate:
            _validate_dry_run_call(compact_summary, direction=direction, call_idx=call_idx)
        current_input = Path(summary["tifxyz_path"])
        if str(summary["stop_reason"]) != "max_extension_iters":
            stage_stop_reason = str(summary["stop_reason"])
            break
        call_idx += 1
        if max_calls is not None and call_idx >= int(max_calls):
            stage_stop_reason = "max_calls_reached"
            break
    stage_summary = {
        "direction": str(direction),
        "call_count": int(len(call_summaries)),
        "final_tifxyz_path": str(current_input),
        "stage_stop_reason": str(stage_stop_reason),
        "extension_preset": {key: value for key, value in preset.items() if key != "show_progress"},
        "calls": call_summaries,
        "final": call_summaries[-1],
    }
    return DirectionRunResult(final_tifxyz_path=current_input, stage_summary=stage_summary)


def _validate_dry_run_call(summary: dict[str, Any], *, direction: str, call_idx: int) -> None:
    if int(summary.get("predicted_vertex_count", 0)) <= 0:
        raise RuntimeError(f"Dry-run {direction} call {call_idx:03d} produced no added geometry")
    if float(summary.get("new_band_frontier_coverage_fraction", 0.0)) < float(TITLE_HUNT_DRY_RUN_MIN_FRONTIER_COVERAGE):
        raise RuntimeError(
            f"Dry-run {direction} call {call_idx:03d} has low frontier coverage "
            f"({summary.get('new_band_frontier_coverage_fraction')})"
        )
    if int(summary.get("new_band_max_gap", 999999)) > int(TITLE_HUNT_DRY_RUN_MAX_GAP):
        raise RuntimeError(
            f"Dry-run {direction} call {call_idx:03d} has excessive gap "
            f"({summary.get('new_band_max_gap')})"
        )


def _upload_surface_output(paths: SurfacePaths) -> None:
    _require_aws_env()
    _run_cmd(
        [
            "aws",
            "s3",
            "cp",
            "--recursive",
            str(paths.final_tifxyz_dir),
            paths.s3_final_tifxyz_uri,
            "--only-show-errors",
        ]
    )
    _run_cmd(
        [
            "aws",
            "s3",
            "cp",
            str(paths.manifest_path),
            paths.s3_manifest_uri,
            "--only-show-errors",
        ]
    )


def _new_surface_state(
    *,
    relative_surface_id: str,
    state_path: Path,
    timestamp_suffix: str,
    coordinate_scale_factor: float,
    source_local_path: str | Path,
    source_s3_uri: str,
    output_s3_uri: str,
) -> dict[str, Any]:
    return {
        "relative_surface_id": str(relative_surface_id),
        "state_path": str(state_path),
        "timestamp_suffix": str(timestamp_suffix),
        "coordinate_scale_factor": float(coordinate_scale_factor),
        "source_local_path": str(source_local_path),
        "source_s3_uri": str(source_s3_uri),
        "output_s3_uri": str(output_s3_uri),
        "status": "downloaded",
        "created_at": _timestamp_suffix(),
        "updated_at": _timestamp_suffix(),
    }


def _process_surface(
    *,
    surface: dict[str, str],
    source_s3_uri: str,
    output_s3_uri: str,
    volume_uri: str,
    dino_backbone: str,
    autoreg_checkpoint: str,
    local_output_root: Path,
    state_root: Path,
    manifest_jsonl_path: Path,
    coordinate_scale_factor: float,
    max_extension_iters_per_call: int,
    window_batch_size: int,
    show_progress: bool,
    dry_run_mode: bool,
    dry_run_max_calls_per_direction: int,
    distributed_infer: bool,
) -> dict[str, Any]:
    relative_surface_id = str(surface["relative_surface_id"])
    state_path = state_root / f"{_safe_state_name(relative_surface_id)}.json"
    state = _load_surface_state(state_path)
    if state is None:
        state = _new_surface_state(
            relative_surface_id=relative_surface_id,
            state_path=state_path,
            timestamp_suffix=_timestamp_suffix(),
            coordinate_scale_factor=coordinate_scale_factor,
            source_local_path=surface["local_path"],
            source_s3_uri=_join_s3(source_s3_uri, relative_surface_id),
            output_s3_uri=output_s3_uri,
        )
        _write_surface_state(state_path, state)
    paths = _surface_paths(
        relative_surface_id=relative_surface_id,
        timestamp_suffix=str(state["timestamp_suffix"]),
        local_output_root=local_output_root,
        state_root=state_root,
        output_s3_uri=output_s3_uri,
    )
    if str(state.get("status")) == "uploaded":
        return state

    if str(state.get("status")) == "downloaded":
        _log(f"[surface] scaling {relative_surface_id}")
        scaled_input_path = _scale_stored_tifxyz(
            surface["local_path"],
            paths.scaled_input_dir,
            coordinate_scale_factor=float(coordinate_scale_factor),
        )
        state = _update_surface_state(
            state=state,
            status="scaled",
            manifest_jsonl_path=manifest_jsonl_path,
            extra={"scaled_input_path": str(scaled_input_path)},
        )

    if str(state.get("status")) in {"scaled", "up_running"}:
        state = _update_surface_state(
            state=state,
            status="up_running",
            manifest_jsonl_path=manifest_jsonl_path,
        )
        up_input = Path(state["scaled_input_path"])
        up_output_dir = paths.output_dir / "up_stage"
        up_result = _run_direction_to_exhaustion(
            input_tifxyz_path=up_input,
            output_dir=up_output_dir,
            direction="up",
            volume_uri=volume_uri,
            dino_backbone=dino_backbone,
            autoreg_checkpoint=autoreg_checkpoint,
            window_batch_size=window_batch_size,
            max_extension_iters_per_call=max_extension_iters_per_call,
            show_progress=show_progress,
            max_calls=int(dry_run_max_calls_per_direction) if dry_run_mode else None,
            distributed_infer=bool(distributed_infer),
            enforce_dry_run_quality_gate=bool(dry_run_mode),
        )
        up_final_tifxyz = up_result.final_tifxyz_path
        up_summary = up_result.stage_summary
        _write_json(paths.summaries_dir / "summary_up.json", up_summary)
        state = _update_surface_state(
            state=state,
            status="up_done",
            manifest_jsonl_path=manifest_jsonl_path,
            extra={"up_final_tifxyz_path": str(up_final_tifxyz), "summary_up_path": str(paths.summaries_dir / "summary_up.json")},
        )

    if str(state.get("status")) in {"up_done", "down_running"}:
        state = _update_surface_state(
            state=state,
            status="down_running",
            manifest_jsonl_path=manifest_jsonl_path,
        )
        down_input = Path(state["up_final_tifxyz_path"])
        down_output_dir = paths.output_dir / "down_stage"
        down_result = _run_direction_to_exhaustion(
            input_tifxyz_path=down_input,
            output_dir=down_output_dir,
            direction="down",
            volume_uri=volume_uri,
            dino_backbone=dino_backbone,
            autoreg_checkpoint=autoreg_checkpoint,
            window_batch_size=window_batch_size,
            max_extension_iters_per_call=max_extension_iters_per_call,
            show_progress=show_progress,
            max_calls=int(dry_run_max_calls_per_direction) if dry_run_mode else None,
            distributed_infer=bool(distributed_infer),
            enforce_dry_run_quality_gate=bool(dry_run_mode),
        )
        down_final_tifxyz = down_result.final_tifxyz_path
        down_summary = down_result.stage_summary
        _write_json(paths.summaries_dir / "summary_down.json", down_summary)
        write_tifxyz(paths.final_tifxyz_dir, read_tifxyz(down_final_tifxyz, load_mask=True, validate=True).use_stored_resolution(), overwrite=True)
        state = _update_surface_state(
            state=state,
            status="down_done",
            manifest_jsonl_path=manifest_jsonl_path,
            extra={"down_final_tifxyz_path": str(paths.final_tifxyz_dir), "summary_down_path": str(paths.summaries_dir / "summary_down.json")},
        )

    if str(state.get("status")) in {"down_done", "uploaded"}:
        summary_up = json.loads(Path(state["summary_up_path"]).read_text())
        summary_down = json.loads(Path(state["summary_down_path"]).read_text())
        manifest = {
            "relative_surface_id": relative_surface_id,
            "timestamp_suffix": str(state["timestamp_suffix"]),
            "coordinate_scale_factor": float(coordinate_scale_factor),
            "source_s3_uri": str(state["source_s3_uri"]),
            "source_local_path": str(state["source_local_path"]),
            "local_output_dir": str(paths.output_dir),
            "s3_surface_prefix": paths.s3_surface_prefix,
            "final_tifxyz_dir": str(paths.final_tifxyz_dir),
            "summary_up": summary_up["final"],
            "summary_down": summary_down["final"],
            "extension_preset": summary_up.get("extension_preset") or _title_hunt_extension_preset(
                window_batch_size=int(window_batch_size),
                show_progress=bool(show_progress),
                distributed_infer=bool(distributed_infer),
            ),
            "summary_up_path": str(paths.summaries_dir / "summary_up.json"),
            "summary_down_path": str(paths.summaries_dir / "summary_down.json"),
            "final_predicted_vertex_count": int(summary_down["final"].get("predicted_vertex_count", 0)),
            "final_stop_reason_up": str(summary_up["final"].get("stop_reason")),
            "final_stop_reason_down": str(summary_down["final"].get("stop_reason")),
            "up_call_count": int(summary_up["call_count"]),
            "down_call_count": int(summary_down["call_count"]),
        }
        _write_json(paths.manifest_path, manifest)
        _upload_surface_output(paths)
        state = _update_surface_state(
            state=state,
            status="uploaded",
            manifest_jsonl_path=manifest_jsonl_path,
            extra={"manifest_path": str(paths.manifest_path), "s3_surface_prefix": paths.s3_surface_prefix},
        )

    return state


def run_batch_extend_title_hunt(
    *,
    source_s3_uri: str,
    output_s3_uri: str,
    volume_uri: str,
    dino_backbone: str,
    autoreg_checkpoint: str,
    local_source_root: Path,
    local_output_root: Path,
    state_root: Path,
    coordinate_scale_factor: float,
    max_extension_iters_per_call: int,
    window_batch_size: int,
    dry_run_surface_count: int,
    auto_continue: bool,
    show_progress: bool,
    dry_run_max_calls_per_direction: int,
    distributed_infer: bool,
) -> dict[str, Any]:
    _require_aws_env()
    local_source_root.mkdir(parents=True, exist_ok=True)
    local_output_root.mkdir(parents=True, exist_ok=True)
    state_root.mkdir(parents=True, exist_ok=True)
    manifest_jsonl_path = state_root / "manifest.jsonl"

    prefixes = _list_title_hunt_prefixes(source_s3_uri)
    dry_run_prefixes = prefixes[: int(dry_run_surface_count)]
    for prefix_name in dry_run_prefixes:
        _sync_title_hunt_prefix(source_s3_uri=source_s3_uri, prefix_name=prefix_name, local_source_root=local_source_root)
    dry_run_surfaces = _discover_local_surfaces(local_source_root, prefixes=dry_run_prefixes)
    dry_run_surfaces = _select_dry_run_surfaces(dry_run_surfaces, count=int(dry_run_surface_count))
    dry_run_ids = {item["relative_surface_id"] for item in dry_run_surfaces}
    ordered_surfaces = list(dry_run_surfaces)
    if auto_continue:
        remaining_prefixes = [prefix for prefix in prefixes if prefix not in dry_run_prefixes]
        for prefix_name in remaining_prefixes:
            _sync_title_hunt_prefix(source_s3_uri=source_s3_uri, prefix_name=prefix_name, local_source_root=local_source_root)
        surfaces = _discover_local_surfaces(local_source_root, prefixes=prefixes)
        ordered_surfaces.extend([surface for surface in surfaces if surface["relative_surface_id"] not in dry_run_ids])
    else:
        surfaces = list(dry_run_surfaces)

    processed: list[dict[str, Any]] = []
    for surface in ordered_surfaces:
        is_dry_run_surface = surface["relative_surface_id"] in dry_run_ids
        _log(f"[surface] start {surface['relative_surface_id']} dry_run={is_dry_run_surface}")
        state = _process_surface(
            surface=surface,
            source_s3_uri=source_s3_uri,
            output_s3_uri=output_s3_uri,
            volume_uri=volume_uri,
            dino_backbone=dino_backbone,
            autoreg_checkpoint=autoreg_checkpoint,
            local_output_root=local_output_root,
            state_root=state_root,
            manifest_jsonl_path=manifest_jsonl_path,
            coordinate_scale_factor=coordinate_scale_factor,
            max_extension_iters_per_call=max_extension_iters_per_call,
            window_batch_size=window_batch_size,
            show_progress=show_progress,
            dry_run_mode=is_dry_run_surface,
            dry_run_max_calls_per_direction=int(dry_run_max_calls_per_direction),
            distributed_infer=bool(distributed_infer),
        )
        processed.append(
            {
                "relative_surface_id": surface["relative_surface_id"],
                "status": state["status"],
                "timestamp_suffix": state["timestamp_suffix"],
            }
        )
        if is_dry_run_surface and state["status"] != "uploaded":
            raise RuntimeError(f"Dry-run surface failed: {surface['relative_surface_id']}")
        if (not auto_continue) and is_dry_run_surface and len([item for item in processed if item["relative_surface_id"] in dry_run_ids]) >= len(dry_run_surfaces):
            break

    return {
        "source_s3_uri": source_s3_uri,
        "output_s3_uri": output_s3_uri,
        "volume_uri": volume_uri,
        "prefix_count": len(prefixes),
        "surface_count": len(surfaces),
        "dry_run_surface_count": len(dry_run_surfaces),
        "dry_run_prefix_count": len(dry_run_prefixes),
        "dry_run_max_calls_per_direction": int(dry_run_max_calls_per_direction),
        "auto_continue": bool(auto_continue),
        "distributed_infer": bool(distributed_infer),
        "processed": processed,
        "manifest_jsonl_path": str(manifest_jsonl_path),
    }


@click.command()
@click.option("--source-s3-uri", type=str, required=True)
@click.option("--output-s3-uri", type=str, required=True)
@click.option("--volume-uri", type=str, default=DEFAULT_VOLUME_URI, show_default=True)
@click.option("--dinov2-backbone", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--autoreg-ckpt", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--local-source-root", type=click.Path(path_type=Path), default=DEFAULT_LOCAL_SOURCE_ROOT, show_default=True)
@click.option("--local-output-root", type=click.Path(path_type=Path), default=DEFAULT_LOCAL_OUTPUT_ROOT, show_default=True)
@click.option("--state-root", type=click.Path(path_type=Path), default=DEFAULT_STATE_ROOT, show_default=True)
@click.option("--coordinate-scale-factor", type=float, default=DEFAULT_COORDINATE_SCALE_FACTOR, show_default=True)
@click.option("--max-extension-iters-per-call", type=int, default=DEFAULT_MAX_EXTENSION_ITERS_PER_CALL, show_default=True)
@click.option("--window-batch-size", type=int, default=DEFAULT_WINDOW_BATCH_SIZE, show_default=True)
@click.option("--dry-run-surface-count", type=int, default=DEFAULT_DRY_RUN_SURFACE_COUNT, show_default=True)
@click.option("--dry-run-max-calls-per-direction", type=int, default=DEFAULT_DRY_RUN_MAX_CALLS_PER_DIRECTION, show_default=True)
@click.option("--auto-continue/--stop-after-dry-run", default=True, show_default=True)
@click.option("--show-progress/--no-show-progress", default=True, show_default=True)
@click.option("--distributed-infer/--no-distributed-infer", default=False, show_default=True)
def main(
    source_s3_uri: str,
    output_s3_uri: str,
    volume_uri: str,
    dinov2_backbone: Path,
    autoreg_ckpt: Path,
    local_source_root: Path,
    local_output_root: Path,
    state_root: Path,
    coordinate_scale_factor: float,
    max_extension_iters_per_call: int,
    window_batch_size: int,
    dry_run_surface_count: int,
    dry_run_max_calls_per_direction: int,
    auto_continue: bool,
    show_progress: bool,
    distributed_infer: bool,
) -> None:
    result = run_batch_extend_title_hunt(
        source_s3_uri=source_s3_uri,
        output_s3_uri=output_s3_uri,
        volume_uri=volume_uri,
        dino_backbone=str(dinov2_backbone),
        autoreg_checkpoint=str(autoreg_ckpt),
        local_source_root=local_source_root,
        local_output_root=local_output_root,
        state_root=state_root,
        coordinate_scale_factor=float(coordinate_scale_factor),
        max_extension_iters_per_call=int(max_extension_iters_per_call),
        window_batch_size=int(window_batch_size),
        dry_run_surface_count=int(dry_run_surface_count),
        dry_run_max_calls_per_direction=int(dry_run_max_calls_per_direction),
        auto_continue=bool(auto_continue),
        show_progress=bool(show_progress),
        distributed_infer=bool(distributed_infer),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
