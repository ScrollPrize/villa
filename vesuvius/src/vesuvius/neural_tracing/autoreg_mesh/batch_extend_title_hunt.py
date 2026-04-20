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

from vesuvius.neural_tracing.autoreg_mesh.extend_tifxyz import (
    ATTENTION_SCALING_MODES,
    ATTENTION_SCALING_STANDARD,
    ExtensionStageResult,
    evaluate_extension_planner,
    extend_tifxyz_mesh,
    run_extension_to_exhaustion,
)
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
TITLE_HUNT_PROMPT_STRIPS = 4
TITLE_HUNT_PREDICT_STRIPS_PER_ITER = 1
TITLE_HUNT_WINDOW_STRIP_LENGTH = 16
TITLE_HUNT_WINDOW_OVERLAP = 8
TITLE_HUNT_PREFLIGHT_MIN_FITTED_PLANS = 18
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


def _measure_vertex_spacing(z: np.ndarray, y: np.ndarray, x: np.ndarray, mask: np.ndarray) -> float:
    h, w = mask.shape
    mid_r, mid_c = h // 2, w // 2
    window = min(50, h // 4, w // 4)
    dists = []
    for r in range(max(0, mid_r - window), min(h - 1, mid_r + window)):
        for c in range(max(0, mid_c - window), min(w, mid_c + window)):
            if mask[r, c] and mask[r + 1, c]:
                d = float(np.sqrt((z[r+1,c]-z[r,c])**2 + (y[r+1,c]-y[r,c])**2 + (x[r+1,c]-x[r,c])**2))
                dists.append(d)
            if c + 1 < w and mask[r, c] and mask[r, c + 1]:
                d = float(np.sqrt((z[r,c+1]-z[r,c])**2 + (y[r,c+1]-y[r,c])**2 + (x[r,c+1]-x[r,c])**2))
                dists.append(d)
    return float(np.median(dists)) if dists else 1.0


def _resample_grid(z: np.ndarray, y: np.ndarray, x: np.ndarray, mask: np.ndarray, *, factor: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from scipy.ndimage import zoom
    factor = max(1, int(factor))
    if factor == 1:
        return z.copy(), y.copy(), x.copy(), mask.copy()
    h, w = z.shape
    new_h, new_w = h * factor, w * factor
    fill_z = np.where(mask, z, np.nan).astype(np.float64)
    fill_y = np.where(mask, y, np.nan).astype(np.float64)
    fill_x = np.where(mask, x, np.nan).astype(np.float64)
    mask_f = mask.astype(np.float64)
    zoomed_mask = zoom(mask_f, factor, order=1)
    zoomed_z = np.full_like(zoomed_mask, np.nan)
    zoomed_y = np.full_like(zoomed_mask, np.nan)
    zoomed_x = np.full_like(zoomed_mask, np.nan)
    weight = zoom(mask_f, factor, order=1)
    weight = np.clip(weight, 0, None)
    valid_weight = weight > 0.01
    for arr, fill in [(zoomed_z, fill_z), (zoomed_y, fill_y), (zoomed_x, fill_x)]:
        safe = np.where(mask, fill, 0.0)
        numer = zoom(safe * mask_f, factor, order=1)
        arr[valid_weight] = numer[valid_weight] / weight[valid_weight]
    new_mask = valid_weight & np.isfinite(zoomed_z) & np.isfinite(zoomed_y) & np.isfinite(zoomed_x)
    return (
        np.where(new_mask, zoomed_z, -1.0).astype(np.float32),
        np.where(new_mask, zoomed_y, -1.0).astype(np.float32),
        np.where(new_mask, zoomed_x, -1.0).astype(np.float32),
        new_mask,
    )


def _scale_stored_tifxyz(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    coordinate_scale_factor: float,
    target_vertex_spacing: float = 0.0,
) -> tuple[Path, int]:
    source = read_tifxyz(source_dir, load_mask=True, validate=True).use_stored_resolution()
    valid_mask = source._mask.copy() if source._mask is not None else np.ones_like(source._x, dtype=bool)
    scaled_x = source._x.copy()
    scaled_y = source._y.copy()
    scaled_z = source._z.copy()
    scaled_x[valid_mask] *= float(coordinate_scale_factor)
    scaled_y[valid_mask] *= float(coordinate_scale_factor)
    scaled_z[valid_mask] *= float(coordinate_scale_factor)

    resample_factor = 1
    if float(target_vertex_spacing) > 0:
        current_spacing = _measure_vertex_spacing(scaled_z, scaled_y, scaled_x, valid_mask)
        resample_factor = max(1, int(round(current_spacing / float(target_vertex_spacing))))
    if resample_factor > 1:
        print("[resample] vertex spacing=%.1f, target=%.1f, resampling %dx (grid %s -> %s)" % (
            current_spacing, target_vertex_spacing, resample_factor,
            valid_mask.shape,
            (valid_mask.shape[0] * resample_factor, valid_mask.shape[1] * resample_factor),
        ))
        scaled_z, scaled_y, scaled_x, valid_mask = _resample_grid(
            scaled_z, scaled_y, scaled_x, valid_mask, factor=resample_factor,
        )

    scaled_bbox = None
    if source.bbox is not None:
        scaled_bbox = tuple(float(v) * float(coordinate_scale_factor) for v in source.bbox)
    new_scale = tuple(float(v) / float(max(1, resample_factor)) for v in source._scale)
    scaled_surface = Tifxyz(
        _x=scaled_x,
        _y=scaled_y,
        _z=scaled_z,
        uuid=source.uuid,
        _scale=new_scale,
        bbox=scaled_bbox,
        area=None if source.area is None else float(source.area) * float(coordinate_scale_factor) * float(coordinate_scale_factor),
        extra=dict(source.extra),
        _mask=valid_mask,
        path=Path(output_dir),
        interp_method=source.interp_method,
        resolution="stored",
    )
    path = write_tifxyz(output_dir, scaled_surface, overwrite=True)
    return path, int(resample_factor)


def _compact_extension_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return dict(summary)


def _title_hunt_extension_preset(
    *,
    window_batch_size: int,
    show_progress: bool,
    distributed_infer: bool,
    device: str = "cuda",
    attention_scaling_mode: str = ATTENTION_SCALING_STANDARD,
) -> dict[str, Any]:
    return {
        "prompt_strips": int(TITLE_HUNT_PROMPT_STRIPS),
        "predict_strips_per_iter": int(TITLE_HUNT_PREDICT_STRIPS_PER_ITER),
        "window_strip_length": int(TITLE_HUNT_WINDOW_STRIP_LENGTH),
        "window_overlap": int(TITLE_HUNT_WINDOW_OVERLAP),
        "window_batch_size": int(window_batch_size),
        "max_crop_fit_retries": 3,
        "device": str(device),
        "show_progress": bool(show_progress),
        "fast_infer": True,
        "compile_infer": False,
        "amp_dtype": "bf16",
        "distributed_infer": bool(distributed_infer),
        "distributed_shard_mode": "strided",
        "distributed_gather_mode": "object",
        "attention_scaling_mode": str(attention_scaling_mode),
    }


def _title_hunt_preflight_candidates() -> list[dict[str, Any]]:
    return [
        {
            "name": "restored_baseline",
            "coordinate_scale_factor": 4.0,
            "target_vertex_spacing": 0.0,
            "prompt_strips": 4,
            "predict_strips_per_iter": 1,
            "window_strip_length": 16,
            "window_overlap": 8,
        },
        {
            "name": "expanded_context",
            "coordinate_scale_factor": 4.0,
            "target_vertex_spacing": 0.0,
            "prompt_strips": 8,
            "predict_strips_per_iter": 2,
            "window_strip_length": 24,
            "window_overlap": 12,
        },
        {
            "name": "restored_resampled",
            "coordinate_scale_factor": 4.0,
            "target_vertex_spacing": 20.0,
            "prompt_strips": 4,
            "predict_strips_per_iter": 1,
            "window_strip_length": 16,
            "window_overlap": 8,
        },
    ]


def _planner_candidate_sort_key(result: dict[str, Any]) -> tuple[float, float, float, int]:
    return (
        float(result.get("fitted_plans", 0)),
        float(result.get("covered_frontier", 0)),
        float(result.get("covered_frontier_fraction", 0.0)),
        -int(result.get("planning_stats", {}).get("crop_fit_failed_count", 0)),
    )


def run_title_hunt_planner_preflight(
    *,
    source_tifxyz_path: str | Path,
    output_dir: str | Path,
    grow_direction: str = "decreasing-z",
    max_crop_fit_retries: int = 3,
    min_fitted_plans: int = TITLE_HUNT_PREFLIGHT_MIN_FITTED_PLANS,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_results: list[dict[str, Any]] = []
    fixed_lattice_direction: str | None = None
    for candidate in _title_hunt_preflight_candidates():
        candidate_dir = output_dir / str(candidate["name"])
        scaled_path, resample_factor = _scale_stored_tifxyz(
            source_tifxyz_path,
            candidate_dir / "scaled_input_tifxyz",
            coordinate_scale_factor=float(candidate["coordinate_scale_factor"]),
            target_vertex_spacing=float(candidate["target_vertex_spacing"]),
        )
        scaled_prompt = int(candidate["prompt_strips"]) * int(resample_factor)
        scaled_predict = int(candidate["predict_strips_per_iter"]) * int(resample_factor)
        scaled_window = int(candidate["window_strip_length"]) * int(resample_factor)
        scaled_overlap = int(candidate["window_overlap"]) * int(resample_factor)
        requested_direction = str(grow_direction if fixed_lattice_direction is None else fixed_lattice_direction)
        evaluation = evaluate_extension_planner(
            tifxyz_path=scaled_path,
            grow_direction=requested_direction,
            prompt_strips=scaled_prompt,
            predict_strips_per_iter=scaled_predict,
            window_strip_length=scaled_window,
            window_overlap=scaled_overlap,
            max_crop_fit_retries=int(max_crop_fit_retries),
        )
        if fixed_lattice_direction is None:
            fixed_lattice_direction = str(evaluation["direction"])
        candidate_results.append(
            {
                **candidate,
                "scaled_input_path": str(scaled_path),
                "resample_factor": int(resample_factor),
                "scaled_prompt_strips": int(scaled_prompt),
                "scaled_predict_strips_per_iter": int(scaled_predict),
                "scaled_window_strip_length": int(scaled_window),
                "scaled_window_overlap": int(scaled_overlap),
                **evaluation,
            }
        )
    candidate_results.sort(key=_planner_candidate_sort_key, reverse=True)
    best = candidate_results[0]
    summary = {
        "source_tifxyz_path": str(source_tifxyz_path),
        "grow_direction": str(grow_direction),
        "resolved_lattice_direction": str(fixed_lattice_direction or grow_direction),
        "min_fitted_plans": int(min_fitted_plans),
        "best_candidate": best,
        "candidates": candidate_results,
    }
    summary_path = output_dir / "planner_preflight.json"
    _write_json(summary_path, summary)
    summary["planner_preflight_path"] = str(summary_path)
    if int(best.get("fitted_plans", 0)) < int(min_fitted_plans):
        raise RuntimeError(
            f"planner preflight failed: best candidate {best['name']} only reached "
            f"{best.get('fitted_plans', 0)} fitted plans"
        )
    return summary


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
    device: str = "cuda",
    distributed_infer: bool = False,
    attention_scaling_mode: str = ATTENTION_SCALING_STANDARD,
    enforce_dry_run_quality_gate: bool = False,
    prompt_strips_override: int | None = None,
    predict_strips_override: int | None = None,
    window_strip_length_override: int | None = None,
    window_overlap_override: int | None = None,
) -> DirectionRunResult:
    preset = _title_hunt_extension_preset(
        window_batch_size=int(window_batch_size),
        show_progress=bool(show_progress),
        distributed_infer=bool(distributed_infer),
        device=str(device),
        attention_scaling_mode=str(attention_scaling_mode),
    )
    if prompt_strips_override is not None:
        preset["prompt_strips"] = int(prompt_strips_override)
    if predict_strips_override is not None:
        preset["predict_strips_per_iter"] = int(predict_strips_override)
    if window_strip_length_override is not None:
        preset["window_strip_length"] = int(window_strip_length_override)
    if window_overlap_override is not None:
        preset["window_overlap"] = int(window_overlap_override)
    result = run_extension_to_exhaustion(
        input_tifxyz_path=input_tifxyz_path,
        output_dir=output_dir,
        grow_direction=direction,
        volume_uri=volume_uri,
        dino_backbone=dino_backbone,
        autoreg_checkpoint=autoreg_checkpoint,
        device=str(preset["device"]),
        prompt_strips=int(preset["prompt_strips"]),
        predict_strips_per_iter=int(preset["predict_strips_per_iter"]),
        window_strip_length=int(preset["window_strip_length"]),
        window_overlap=int(preset["window_overlap"]),
        window_batch_size=int(preset["window_batch_size"]),
        max_extension_iters_per_call=int(max_extension_iters_per_call),
        max_crop_fit_retries=int(preset["max_crop_fit_retries"]),
        max_calls=max_calls,
        show_progress=bool(preset["show_progress"]),
        fast_infer=bool(preset["fast_infer"]),
        compile_infer=bool(preset["compile_infer"]),
        amp_dtype=str(preset["amp_dtype"]),
        distributed_infer=bool(preset["distributed_infer"]),
        distributed_shard_mode=str(preset["distributed_shard_mode"]),
        distributed_gather_mode=str(preset["distributed_gather_mode"]),
        attention_scaling_mode=str(preset["attention_scaling_mode"]),
        call_validator=(
            (lambda summary, call_idx: _validate_dry_run_call(summary, direction=direction, call_idx=call_idx))
            if enforce_dry_run_quality_gate else None
        ),
        extend_impl=extend_tifxyz_mesh,
    )
    stage_summary = dict(result.stage_summary)
    stage_summary["extension_preset"] = {key: value for key, value in preset.items() if key != "show_progress"}
    return DirectionRunResult(final_tifxyz_path=result.final_tifxyz_path, stage_summary=stage_summary)


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


def _normalize_direction_sequence(extend_directions: list[str] | None) -> list[str]:
    values = extend_directions or ["up", "down"]
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        direction = str(value).strip()
        if not direction:
            continue
        if direction in seen:
            continue
        seen.add(direction)
        ordered.append(direction)
    if not ordered:
        raise ValueError("extend_directions must contain at least one direction")
    return ordered


def _count_valid_vertices(path: str | Path) -> int:
    surface = read_tifxyz(path, load_mask=True, validate=True).use_stored_resolution()
    mask = surface._mask
    if mask is not None:
        return int(np.asarray(mask, dtype=bool).sum())
    valid = np.isfinite(surface._x) & np.isfinite(surface._y) & np.isfinite(surface._z)
    return int(valid.sum())


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
    device: str = "cuda",
    attention_scaling_mode: str = ATTENTION_SCALING_STANDARD,
    extend_directions: list[str] | None = None,
    target_vertex_spacing: float = 0.0,
) -> dict[str, Any]:
    active_directions = _normalize_direction_sequence(extend_directions)
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

    resample_factor = 1
    if str(state.get("status")) == "downloaded":
        _log(f"[surface] scaling {relative_surface_id}")
        scaled_input_path, resample_factor = _scale_stored_tifxyz(
            surface["local_path"],
            paths.scaled_input_dir,
            coordinate_scale_factor=float(coordinate_scale_factor),
            target_vertex_spacing=float(target_vertex_spacing),
        )
        state = _update_surface_state(
            state=state,
            status="scaled",
            manifest_jsonl_path=manifest_jsonl_path,
            extra={"scaled_input_path": str(scaled_input_path), "resample_factor": int(resample_factor)},
        )
    resample_factor = int(state.get("resample_factor", 1))

    current_input = Path(state["scaled_input_path"])
    direction_summaries: dict[str, dict[str, Any]] = {}
    scaled_window_strip_length = int(TITLE_HUNT_WINDOW_STRIP_LENGTH) * resample_factor
    scaled_window_overlap = int(TITLE_HUNT_WINDOW_OVERLAP) * resample_factor
    scaled_prompt_strips = int(TITLE_HUNT_PROMPT_STRIPS) * resample_factor
    scaled_predict_strips = int(TITLE_HUNT_PREDICT_STRIPS_PER_ITER) * resample_factor
    for direction in active_directions:
        final_key = f"{direction}_final_tifxyz_path"
        summary_key = f"summary_{direction}_path"
        existing_final = state.get(final_key)
        existing_summary = state.get(summary_key)
        if existing_final and existing_summary and Path(existing_final).exists() and Path(existing_summary).exists():
            current_input = Path(existing_final)
            direction_summaries[direction] = json.loads(Path(existing_summary).read_text())
            continue
        stage_status = f"{direction}_running"
        done_status = f"{direction}_done"
        state = _update_surface_state(state=state, status=stage_status, manifest_jsonl_path=manifest_jsonl_path)
        stage_output_dir = paths.output_dir / f"{direction}_stage"
        stage_result = _run_direction_to_exhaustion(
            input_tifxyz_path=current_input,
            output_dir=stage_output_dir,
            direction=direction,
            volume_uri=volume_uri,
            dino_backbone=dino_backbone,
            autoreg_checkpoint=autoreg_checkpoint,
            window_batch_size=window_batch_size,
            max_extension_iters_per_call=max_extension_iters_per_call,
            show_progress=show_progress,
            max_calls=int(dry_run_max_calls_per_direction) if dry_run_mode else None,
            device=str(device),
            distributed_infer=bool(distributed_infer),
            attention_scaling_mode=str(attention_scaling_mode),
            enforce_dry_run_quality_gate=bool(dry_run_mode),
            prompt_strips_override=scaled_prompt_strips if resample_factor > 1 else None,
            predict_strips_override=scaled_predict_strips if resample_factor > 1 else None,
            window_strip_length_override=scaled_window_strip_length if resample_factor > 1 else None,
            window_overlap_override=scaled_window_overlap if resample_factor > 1 else None,
        )
        current_input = stage_result.final_tifxyz_path
        summary_path = paths.summaries_dir / f"summary_{direction}.json"
        _write_json(summary_path, stage_result.stage_summary)
        direction_summaries[direction] = stage_result.stage_summary
        state = _update_surface_state(
            state=state,
            status=done_status,
            manifest_jsonl_path=manifest_jsonl_path,
            extra={f"{direction}_final_tifxyz_path": str(current_input), f"summary_{direction}_path": str(summary_path)},
        )

    write_tifxyz(paths.final_tifxyz_dir, read_tifxyz(current_input, load_mask=True, validate=True).use_stored_resolution(), overwrite=True)
    original_vertex_count = _count_valid_vertices(state["scaled_input_path"])
    final_vertex_count = _count_valid_vertices(paths.final_tifxyz_dir)
    cumulative_predicted_vertex_count = max(0, int(final_vertex_count - original_vertex_count))
    last_dir = active_directions[-1]
    last_summary = direction_summaries.get(last_dir, {"final": {}})
    first_summary = direction_summaries.get(active_directions[0], {"final": {}})
    manifest = {
        "relative_surface_id": relative_surface_id,
        "timestamp_suffix": str(state["timestamp_suffix"]),
        "coordinate_scale_factor": float(coordinate_scale_factor),
        "target_vertex_spacing": float(target_vertex_spacing),
        "source_s3_uri": str(state["source_s3_uri"]),
        "source_local_path": str(state["source_local_path"]),
        "local_output_dir": str(paths.output_dir),
        "s3_surface_prefix": paths.s3_surface_prefix,
        "final_tifxyz_dir": str(paths.final_tifxyz_dir),
        "directions": list(active_directions),
        "direction_summaries": {d: s.get("final", {}) for d, s in direction_summaries.items()},
        "extension_preset": first_summary.get("extension_preset") or _title_hunt_extension_preset(
            window_batch_size=int(window_batch_size),
            show_progress=bool(show_progress),
            distributed_infer=bool(distributed_infer),
            device=str(device),
            attention_scaling_mode=str(attention_scaling_mode),
        ),
        "attention_scaling_mode": str(attention_scaling_mode),
        "resample_factor": int(resample_factor),
        "original_vertex_count": int(original_vertex_count),
        "final_vertex_count": int(final_vertex_count),
        "final_predicted_vertex_count": int(cumulative_predicted_vertex_count),
        "cumulative_predicted_vertex_count": int(cumulative_predicted_vertex_count),
        "direction_call_counts": {d: int(s.get("call_count", 0)) for d, s in direction_summaries.items()},
        "final_stage_stop_reason": str(last_summary.get("stage_stop_reason", "")),
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
    surface_filter: str | None = None,
    extend_directions: list[str] | None = None,
    show_progress: bool,
    dry_run_max_calls_per_direction: int,
    distributed_infer: bool,
    device: str = "cuda",
    attention_scaling_mode: str = ATTENTION_SCALING_STANDARD,
    target_vertex_spacing: float = 0.0,
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
    if surface_filter is not None:
        dry_run_surfaces = [s for s in dry_run_surfaces if str(surface_filter) in s["relative_surface_id"]]
    dry_run_surfaces = _select_dry_run_surfaces(dry_run_surfaces, count=int(dry_run_surface_count))
    dry_run_ids = {item["relative_surface_id"] for item in dry_run_surfaces}
    ordered_surfaces = list(dry_run_surfaces)
    if auto_continue:
        remaining_prefixes = [prefix for prefix in prefixes if prefix not in dry_run_prefixes]
        for prefix_name in remaining_prefixes:
            _sync_title_hunt_prefix(source_s3_uri=source_s3_uri, prefix_name=prefix_name, local_source_root=local_source_root)
        surfaces = _discover_local_surfaces(local_source_root, prefixes=prefixes)
        if surface_filter is not None:
            surfaces = [s for s in surfaces if str(surface_filter) in s["relative_surface_id"]]
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
            extend_directions=extend_directions or ["up", "down"],
            state_root=state_root,
            manifest_jsonl_path=manifest_jsonl_path,
            coordinate_scale_factor=coordinate_scale_factor,
            max_extension_iters_per_call=max_extension_iters_per_call,
            window_batch_size=window_batch_size,
            show_progress=show_progress,
            device=str(device),
            dry_run_mode=is_dry_run_surface,
            dry_run_max_calls_per_direction=int(dry_run_max_calls_per_direction),
            distributed_infer=bool(distributed_infer),
            attention_scaling_mode=str(attention_scaling_mode),
            target_vertex_spacing=float(target_vertex_spacing),
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
        "device": str(device),
        "distributed_infer": bool(distributed_infer),
        "attention_scaling_mode": str(attention_scaling_mode),
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
@click.option("--device", type=str, default="cuda", show_default=True)
@click.option("--show-progress/--no-show-progress", default=True, show_default=True)
@click.option("--distributed-infer/--no-distributed-infer", default=False, show_default=True)
@click.option("--attention-scaling-mode", type=click.Choice(list(ATTENTION_SCALING_MODES)), default=ATTENTION_SCALING_STANDARD, show_default=True)
@click.option("--surface-filter", type=str, default=None, help="Only process surfaces whose relative_surface_id contains this substring")
@click.option("--extend-directions", type=str, default="up,down", show_default=True, help="Comma-separated directions to extend (e.g. 'up' or 'up,down')")
@click.option("--target-vertex-spacing", type=float, default=0.0, show_default=True, help="If >0, resample the lattice so vertex spacing matches this (voxels). Set to 20 to match training data.")
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
    device: str,
    show_progress: bool,
    distributed_infer: bool,
    attention_scaling_mode: str,
    surface_filter: str | None,
    extend_directions: str,
    target_vertex_spacing: float,
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
        device=str(device),
        show_progress=bool(show_progress),
        distributed_infer=bool(distributed_infer),
        attention_scaling_mode=str(attention_scaling_mode),
        surface_filter=surface_filter,
        extend_directions=[d.strip() for d in str(extend_directions).split(",") if d.strip()],
        target_vertex_spacing=float(target_vertex_spacing),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
