"""HTTP service wrapping the 3D fit optimizer for use by VC3D.

Start with:
    python fit_service.py [--port PORT] [--data-dir PATH]

Endpoints:
    GET  /health          -> {"status": "ok"}
    GET  /status          -> current job state
    GET  /datasets        -> available .lasagna.json datasets from --data-dir
    POST /jobs            -> queue an optimization job (JSON body)
    GET  /jobs            -> list queued/running/finished jobs
    GET  /jobs/{id}       -> one job state
    GET  /jobs/{id}/results -> download one job's results
    POST /jobs/{id}/cancel -> cancel an upload/waiting/running job
    POST /jobs/reorder    -> reorder upload/waiting jobs
    POST /optimize        -> legacy queue wrapper
    POST /stop            -> request cancellation of the running job
    POST /export_vis      -> export multi-layer OBJ visualization (JSON body)
"""
from __future__ import annotations

import argparse
import atexit
import base64
import getpass
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import volume_scale


# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------

_data_dir: str | None = None  # Set via --data-dir CLI flag
_object_store_dir: Path | None = None  # Set via --object-store-dir CLI flag
_gpu_pause_enabled: bool = True  # Set via --no-gpu-pause CLI flag
_sparse_prefetch_backend: str = "tensorstore"  # Set via --sparse-prefetch-backend
_API_VERSION = "2"
_API_VERSION_HEADER = "X-Fit-Service-API-Version"
_VC3D_SOURCE_HEADER = "X-VC3D-Source"

# One debug switch for sparse coverage and coordinate sanity guards. The service
# enables it by default so optimizer jobs fail before CUDA indexing asserts.
os.environ.setdefault("LASAGNA_CHECK_SPARSE_CACHE", "1")


def _mib(n_bytes: int) -> float:
    return float(n_bytes) / (1024.0 * 1024.0)


def _mib_per_s(n_bytes: int, seconds: float) -> float:
    return _mib(n_bytes) / seconds if seconds > 0.0 else 0.0


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        root_path = Path(root)
        for name in files:
            try:
                total += (root_path / name).stat().st_size
            except OSError:
                pass
    return total


def _result_archive_child_name(child_name: str, child_count: int, output_name: str) -> str:
    requested = str(output_name or "").strip()
    if requested and int(child_count) == 1:
        return requested
    return child_name


def _normalize_single_tifxyz_output(output_dir: Path, output_name: str) -> Path | None:
    requested = str(output_name or "").strip()
    if not requested:
        return None
    if not requested.endswith(".tifxyz"):
        requested = f"{requested}.tifxyz"

    children = [p for p in output_dir.iterdir() if p.name != ".lasagna_results.tar.gz"]
    tifxyz_children = [p for p in children if p.is_dir() and p.name.endswith(".tifxyz")]
    if len(children) != 1 or len(tifxyz_children) != 1:
        return None

    child = tifxyz_children[0]
    final = output_dir / requested
    if child != final:
        if final.exists():
            raise FileExistsError(f"requested output_name already exists: {final}")
        child.rename(final)
    meta_path = final / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if isinstance(meta, dict):
            meta["uuid"] = requested
            meta["name"] = requested
            meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(
        f"[fit-service] normalized tifxyz output: output_name={requested!r} path={final}",
        flush=True,
    )
    return final


def _default_object_store_dir() -> Path:
    return Path.home() / ".cache" / "lasagna" / "fit_service" / "objects"


def _object_store_root() -> Path:
    return _object_store_dir or _default_object_store_dir()


def _hash_bytes(data: bytes) -> str:
    return "md5:" + hashlib.md5(data).hexdigest()


_SINGLE_FILE_OBJECT_TYPES = {"lasagna_model", "line", "line-map", "atlas", "atlas-pred-snap"}
_DIRECTORY_OBJECT_TYPES = {"tifxyz_segment", "atlas-base"}
_SUPPORTED_OBJECT_TYPES = _SINGLE_FILE_OBJECT_TYPES | _DIRECTORY_OBJECT_TYPES
_SUPPORTED_OBJECT_FORMATS = {
    "line": {"vc3d_fiber_json"},
    "line-map": {"vc3d_atlas_fiber_mapping_json"},
    "atlas-pred-snap": {"vc3d_atlas_pred_snap_points_json"},
    "atlas-base": {"tifxyz"},
    "atlas": {"lasagna_atlas_json"},
}


def _validate_object_ref(ref: Any) -> dict[str, str]:
    if not isinstance(ref, dict):
        raise ValueError("object ref must be an object")
    obj_type = str(ref.get("type") or "").strip()
    name = str(ref.get("name") or "").strip()
    digest = str(ref.get("hash") or "").strip().lower()
    fmt = str(ref.get("format") or "").strip()
    if obj_type not in _SUPPORTED_OBJECT_TYPES:
        raise ValueError(f"unsupported object type: {obj_type or '<missing>'}")
    if not name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("object name must be a non-empty relative path without '..'")
    if not digest.startswith("md5:") or len(digest) != 36:
        raise ValueError("object hash must be md5:<32 hex chars>")
    int(digest[4:], 16)
    if fmt:
        supported = _SUPPORTED_OBJECT_FORMATS.get(obj_type)
        if supported is not None and fmt not in supported:
            raise ValueError(f"unsupported object format for {obj_type}: {fmt}")
        return {"type": obj_type, "name": name, "hash": digest, "format": fmt}
    return {"type": obj_type, "name": name, "hash": digest}


def _object_identity(ref: dict[str, str]) -> dict[str, str]:
    return {k: ref[k] for k in ("type", "name", "hash")}


def _object_dir(ref: dict[str, str]) -> Path:
    # Object identity is type + name + hash.  The name is percent-encoded so
    # refs like "sheet.tifxyz/model.pt" do not create nested store paths.
    return _object_store_root() / ref["type"] / ref["hash"][4:] / quote(ref["name"], safe="")


def _object_metadata_path(ref: dict[str, str]) -> Path:
    return _object_dir(ref) / "object.json"


def _object_payload_path(ref: dict[str, str]) -> Path:
    base = _object_dir(ref)
    if ref["type"] == "lasagna_model":
        return base / "model.pt"
    if ref["type"] in {"line", "line-map", "atlas", "atlas-pred-snap"}:
        return base / "data.json"
    return base / "segment"


def _object_present(ref_raw: Any) -> bool:
    try:
        ref = _validate_object_ref(ref_raw)
    except Exception:
        return False
    meta_path = _object_metadata_path(ref)
    payload_path = _object_payload_path(ref)
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if {k: meta.get(k) for k in ("type", "name", "hash")} != _object_identity(ref):
        return False
    return payload_path.is_file() if ref["type"] in _SINGLE_FILE_OBJECT_TYPES else payload_path.is_dir()


def _segment_manifest_hash(files: dict[str, bytes]) -> str:
    lines: list[str] = []
    for rel in sorted(files):
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts or not rel:
            raise ValueError(f"invalid artifact file path: {rel!r}")
        lines.append(f"{rel}\t{_hash_bytes(files[rel])}\n")
    manifest = "".join(lines).encode("utf-8")
    return _hash_bytes(manifest)


def _store_uploaded_object(body: dict[str, Any]) -> dict[str, str]:
    ref = _validate_object_ref(body.get("object", body))
    tmp_parent = _object_store_root() / ".tmp"
    tmp_parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="upload_", dir=str(tmp_parent)))
    final_dir = _object_dir(ref)
    try:
        if ref["type"] in _SINGLE_FILE_OBJECT_TYPES:
            data_b64 = body.get("data")
            if not isinstance(data_b64, str):
                raise ValueError(f"{ref['type']} upload requires base64 data")
            data = base64.b64decode(data_b64)
            actual_hash = _hash_bytes(data)
            if actual_hash != ref["hash"]:
                raise ValueError(f"object hash mismatch: declared {ref['hash']} actual {actual_hash}")
            if ref["type"] in {"line", "line-map", "atlas", "atlas-pred-snap"}:
                try:
                    json.loads(data.decode("utf-8"))
                except Exception as exc:
                    raise ValueError(f"{ref['type']} upload data must be UTF-8 JSON") from exc
            (tmp_dir / _object_payload_path(ref).name).write_bytes(data)
        else:
            files_raw = body.get("files")
            if not isinstance(files_raw, dict) or not files_raw:
                raise ValueError(f"{ref['type']} upload requires non-empty files object")
            segment_dir = tmp_dir / "segment"
            segment_dir.mkdir(parents=True, exist_ok=True)
            decoded: dict[str, bytes] = {}
            for rel, data_b64 in files_raw.items():
                if not isinstance(data_b64, str):
                    raise ValueError(f"artifact file {rel!r} must be base64 text")
                decoded[str(rel)] = base64.b64decode(data_b64)
            actual_hash = _segment_manifest_hash(decoded)
            if actual_hash != ref["hash"]:
                raise ValueError(f"segment hash mismatch: declared {ref['hash']} actual {actual_hash}")
            for rel, data in decoded.items():
                dst = segment_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(data)
        (tmp_dir / "object.json").write_text(json.dumps(ref, indent=2) + "\n", encoding="utf-8")
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        if final_dir.exists():
            shutil.rmtree(final_dir)
        tmp_dir.rename(final_dir)
        return ref
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def _resolve_object_ref(ref_raw: Any) -> Path:
    ref = _validate_object_ref(ref_raw)
    if not _object_present(ref):
        raise ValueError(f"missing object: {ref['type']} {ref['name']} {ref['hash']}")
    return _object_payload_path(ref)


def _resolve_lasagna_atlas_ref(ref_raw: Any) -> dict[str, Any]:
    atlas_path = _resolve_object_ref(ref_raw)
    try:
        atlas_obj = json.loads(atlas_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"failed to load atlas JSON for object {ref_raw!r}: {exc}") from exc
    if not isinstance(atlas_obj, dict):
        raise ValueError("atlas object payload must be a JSON object")
    if atlas_obj.get("type") != "lasagna_atlas" or int(atlas_obj.get("version", 0)) != 1:
        raise ValueError("unsupported atlas payload; expected type=lasagna_atlas version=1")

    resolved = json.loads(json.dumps(atlas_obj))
    base = resolved.get("base")
    if not isinstance(base, dict) or not isinstance(base.get("ref"), dict):
        raise ValueError("atlas JSON requires base.ref")
    base["path"] = str(_resolve_object_ref(base["ref"]))

    objects = resolved.get("objects")
    if not isinstance(objects, dict):
        raise ValueError("atlas JSON requires objects")
    line_by_id: dict[str, dict[str, Any]] = {}
    line_entries = objects.get("line", [])
    if not isinstance(line_entries, list):
        raise ValueError("atlas JSON objects.line must be a list")
    for i, item in enumerate(line_entries):
        if not isinstance(item, dict):
            raise ValueError(f"atlas JSON objects.line[{i}] must be an object")
        ref = item.get("ref")
        if not isinstance(ref, dict):
            raise ValueError(f"atlas JSON objects.line[{i}] requires ref")
        item["path"] = str(_resolve_object_ref(ref))
        line_id = str(item.get("id") or ref.get("name") or "")
        if line_id:
            line_by_id[line_id] = item

    maps = resolved.get("maps", [])
    if not isinstance(maps, list):
        raise ValueError("atlas JSON maps must be a list")
    for i, item in enumerate(maps):
        if not isinstance(item, dict):
            raise ValueError(f"atlas JSON maps[{i}] must be an object")
        if str(item.get("object_type") or "line") != "line":
            raise ValueError(f"atlas JSON maps[{i}] unsupported object_type")
        map_ref = item.get("map_ref")
        if not isinstance(map_ref, dict):
            raise ValueError(f"atlas JSON maps[{i}] requires map_ref")
        item["map_path"] = str(_resolve_object_ref(map_ref))
        pred_snap_ref = item.get("pred_snap_ref")
        if isinstance(pred_snap_ref, dict):
            item["pred_snap_path"] = str(_resolve_object_ref(pred_snap_ref))
        obj_ref = item.get("object_ref")
        if isinstance(obj_ref, dict):
            item["object_path"] = str(_resolve_object_ref(obj_ref))
        else:
            object_id = str(item.get("object_id") or "")
            line_obj = line_by_id.get(object_id)
            if line_obj is not None and line_obj.get("path"):
                item["object_path"] = str(line_obj["path"])
    return resolved


def _object_refs_from_value(value: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            try:
                ref = _validate_object_ref(item)
            except Exception:
                ref = None
            if ref is not None:
                key = (ref["type"], ref["name"], ref["hash"])
                if key not in seen:
                    seen.add(key)
                    out.append(ref)
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return out


def _checkpoint_object_refs(model_path: str | None) -> dict[str, Any] | None:
    if not model_path:
        return None
    try:
        import torch
        st = torch.load(str(model_path), map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"[fit-service] WARNING: failed to inspect model object refs: {exc}", flush=True)
        return None
    if not isinstance(st, dict):
        return None
    refs = st.get("_object_refs_")
    return refs if isinstance(refs, dict) else None


def _normalize_object_refs(job_spec: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(job_spec, dict):
        return None
    return {
        "version": 1,
        "objects": _object_refs_from_value(job_spec),
        "job_spec": json.loads(json.dumps(job_spec)),
    }


def _truthy_config_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _request_volume_shape_zyx(body: dict[str, Any]) -> tuple[int, int, int] | None:
    shape = volume_scale.parse_shape_zyx(body.get("volume_shape_zyx"), name="volume_shape_zyx")
    spec = body.get("job_spec")
    if isinstance(spec, dict):
        spec_shape = volume_scale.parse_shape_zyx(
            spec.get("volume_shape_zyx"), name="job_spec.volume_shape_zyx")
        if shape is not None and spec_shape is not None and tuple(shape) != tuple(spec_shape):
            raise ValueError("volume_shape_zyx and job_spec.volume_shape_zyx must match")
        if shape is None:
            shape = spec_shape
    return shape


def _approval_inpaint_enabled(args_section: dict[str, Any]) -> bool:
    return _truthy_config_bool(args_section.get("approval-inpaint", False))


def _config_enables_pred_dt_flow_gate(cfg: dict[str, Any]) -> bool:
    stages = cfg.get("stages")
    if not isinstance(stages, list):
        return False
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        opt_cfg = stage.get("global_opt")
        if not isinstance(opt_cfg, dict):
            opt_cfg = stage.get("opt")
        if not isinstance(opt_cfg, dict):
            opt_cfg = stage
        args = opt_cfg.get("args")
        if not isinstance(args, dict):
            continue
        gate = args.get("pred_dt_flow_gate")
        if isinstance(gate, dict) and bool(gate.get("enabled", False)):
            return True
    return False


def _set_pred_dt_flow_gate_debug_out_dir(cfg: dict[str, Any], out_dir: str) -> None:
    stages = cfg.get("stages")
    if not isinstance(stages, list):
        return
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        opt_cfg = stage.get("global_opt")
        if not isinstance(opt_cfg, dict):
            opt_cfg = stage.get("opt")
        if not isinstance(opt_cfg, dict):
            opt_cfg = stage
        args = opt_cfg.get("args")
        if not isinstance(args, dict):
            continue
        gate = args.get("pred_dt_flow_gate")
        if isinstance(gate, dict) and bool(gate.get("enabled", False)):
            gate.setdefault("debug_out_dir", out_dir)


def _set_snap_surf_map_debug_obj_dir(cfg: dict[str, Any], out_dir: str) -> None:
    stages = cfg.get("stages")
    if not isinstance(stages, list):
        return
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        opt_cfg = stage.get("global_opt")
        if not isinstance(opt_cfg, dict):
            opt_cfg = stage.get("opt")
        if not isinstance(opt_cfg, dict):
            opt_cfg = stage
        args = opt_cfg.get("args")
        if not isinstance(args, dict):
            continue
        params = opt_cfg.get("params", [])
        if isinstance(params, str):
            params = [params]
        if isinstance(params, list) and any(str(p) in {"map_surf_affine", "map_surf_ms"} for p in params):
            if args.get("debug_obj_dir"):
                args["debug_obj_dir"] = out_dir
        snap_map = args.get("snap_surf_map")
        if isinstance(snap_map, dict):
            map_opt = snap_map.get("map_opt")
            if isinstance(map_opt, dict):
                map_args = map_opt.get("args")
                if not isinstance(map_args, dict):
                    map_args = {}
                    map_opt["args"] = map_args
                if map_args.get("debug_obj_dir"):
                    map_args["debug_obj_dir"] = out_dir


def _single_external_surface_path(cfg: dict[str, Any], *, consumer: str) -> str:
    external_surfaces = cfg.get("external_surfaces")
    if not isinstance(external_surfaces, list) or len(external_surfaces) != 1:
        count = len(external_surfaces) if isinstance(external_surfaces, list) else 0
        raise ValueError(f"{consumer} requires exactly one external_surfaces entry, got {count}")
    surface = external_surfaces[0]
    if not isinstance(surface, dict) or not surface.get("path"):
        raise ValueError(f"{consumer} external_surfaces[0] requires path")
    return str(surface["path"])


def _wire_external_surface_for_request(
    *,
    cfg: dict[str, Any],
    args_section: dict[str, Any],
    model_init: str,
    ext_offset_enabled: bool,
    global_map_enabled: bool = False,
) -> str | None:
    """Wire external_surfaces[0] into fit.py args for consumers."""
    approval_enabled = _approval_inpaint_enabled(args_section)
    if approval_enabled and model_init != "seed":
        raise ValueError("args.approval-inpaint is only valid with args.model-init=seed")

    tifxyz_dir: str | None = None
    if model_init == "ext":
        tifxyz_dir = _single_external_surface_path(cfg, consumer="model-init=ext")
        args_section["tifxyz-init"] = tifxyz_dir
    if approval_enabled:
        tifxyz_dir = _single_external_surface_path(cfg, consumer="approval-inpaint")
        args_section["approval-inpaint-tifxyz"] = tifxyz_dir
    if model_init == "flatten":
        _single_external_surface_path(cfg, consumer="model-init=flatten")
    if ext_offset_enabled or global_map_enabled:
        loss_names = []
        if ext_offset_enabled:
            loss_names.append("ext_offset")
        if global_map_enabled:
            loss_names.append("snap_surf_map/global_map")
        _single_external_surface_path(cfg, consumer="/".join(loss_names))

    return tifxyz_dir


def _decode_model_for_request(
    *,
    body: dict[str, Any],
    tmp_dir: str,
    model_init: str,
) -> str | None:
    """Decode model transport data only when the selected init mode consumes it."""
    import base64

    model_input = body.get("model_input")
    model_data = body.get("model_data")

    if model_init != "model":
        if model_input or model_data:
            print("[fit-service] ignoring surplus model transport data", flush=True)
        return None

    if model_data:
        model_bytes = base64.b64decode(model_data)
        decoded_model_input = str(Path(tmp_dir) / "model_input.pt")
        Path(decoded_model_input).write_bytes(model_bytes)
        print(f"[fit-service] received model data ({len(model_bytes)} bytes)", flush=True)
        return decoded_model_input

    if model_input:
        return str(model_input)

    return None


def _body_with_resolved_job_spec(body: dict[str, Any]) -> dict[str, Any]:
    """Resolve a Lasagna job spec into local runner transport fields."""
    spec = body.get("job_spec")
    if not isinstance(spec, dict):
        return body

    cfg = spec.get("config", {})
    if not isinstance(cfg, dict):
        raise ValueError("job_spec.config must be an object")
    atlas_reopt_cfg = cfg.get("atlas_reopt")
    if not isinstance(atlas_reopt_cfg, dict):
        atlas_reopt_cfg = {}
    atlas_ref_source = str(atlas_reopt_cfg.get("source") or "").strip().lower()
    prefer_request_atlas = atlas_ref_source in {"request", "selected_atlas", "current_atlas"}

    resolved = dict(body)
    resolved_cfg = dict(cfg)
    resolved_cfg.pop("atlas_reopt", None)
    external_surfaces_raw = resolved_cfg.get("external_surfaces")
    if external_surfaces_raw is not None:
        if not isinstance(external_surfaces_raw, list):
            raise ValueError("job_spec.config.external_surfaces must be a list")
        external_surfaces: list[dict[str, Any]] = []
        for i, surface_raw in enumerate(external_surfaces_raw):
            if not isinstance(surface_raw, dict):
                raise ValueError(f"job_spec.config.external_surfaces[{i}] must be an object")
            surface = dict(surface_raw)
            if all(k in surface for k in ("type", "name", "hash")):
                surface["path"] = str(_resolve_object_ref(surface))
            external_surfaces.append(surface)
        resolved_cfg["external_surfaces"] = external_surfaces

    model_ref = spec.get("model")
    model_object_refs: dict[str, Any] | None = None
    if model_ref not in (None, {}, ""):
        resolved_model_path = str(_resolve_object_ref(model_ref))
        resolved["model_input"] = resolved_model_path
        model_object_refs = _checkpoint_object_refs(resolved_model_path)

    model_job_spec = (
        model_object_refs.get("job_spec")
        if isinstance(model_object_refs, dict) and isinstance(model_object_refs.get("job_spec"), dict)
        else None
    )
    atlas_ref = spec.get("atlas")
    model_atlas_ref = model_job_spec.get("atlas") if isinstance(model_job_spec, dict) else None
    if prefer_request_atlas and atlas_ref not in (None, {}, ""):
        if model_atlas_ref not in (None, {}, "") and model_atlas_ref != atlas_ref:
            print("[fit-service] request job_spec.atlas overrides model _object_refs_ atlas", flush=True)
    elif model_atlas_ref not in (None, {}, ""):
        if atlas_ref not in (None, {}, "") and atlas_ref != model_atlas_ref:
            print("[fit-service] model _object_refs_ atlas overrides request job_spec.atlas", flush=True)
        atlas_ref = model_atlas_ref

    effective_job_spec = {
        "model": spec.get("model"),
        "atlas": atlas_ref,
        "linked_surfaces": spec.get("linked_surfaces", []),
        "config": cfg,
    }
    if "volume_shape_zyx" in spec:
        effective_job_spec["volume_shape_zyx"] = spec.get("volume_shape_zyx")
    resolved["config"] = resolved_cfg
    resolved["_job_spec_"] = effective_job_spec

    if atlas_ref not in (None, {}, ""):
        resolved_cfg["atlas"] = _resolve_lasagna_atlas_ref(atlas_ref)

    linked_refs = spec.get("linked_surfaces", [])
    if linked_refs is None:
        linked_refs = []
    if not isinstance(linked_refs, list):
        raise ValueError("job_spec.linked_surfaces must be a list")

    object_refs = _normalize_object_refs(effective_job_spec)
    if object_refs is not None:
        # Include explicit object_refs from the request as query-only refs. They
        # may contain refs inherited from an exported tifxyz meta.json that are
        # not part of the new job_spec surface area.
        request_object_refs = spec.get("object_refs")
        if isinstance(request_object_refs, dict):
            merged = list(object_refs["objects"])
            seen = {(r["type"], r["name"], r["hash"]) for r in merged}
            for ref in _object_refs_from_value(request_object_refs.get("objects", [])):
                key = (ref["type"], ref["name"], ref["hash"])
                if key not in seen:
                    seen.add(key)
                    merged.append(ref)
            object_refs["objects"] = merged
        resolved["_object_refs_"] = object_refs

    return resolved


def _config_effective_loss_enabled(cfg: dict[str, Any], term_name: str) -> bool:
    map_loss_names = {
        "map_dist",
        "map_vec_normal",
        "map_surface_normal",
        "map_smooth",
        "map_bend",
        "map_jac",
        "map_metric_smooth",
        "map_area_smooth",
        "map_dense_prior",
        "map_station_t",
    }
    map_params = {"map_surf_affine", "map_surf_ms"}
    model_params = {"mesh_ms", "amp", "bias", "cyl_params", "map_flatten_ms"}
    base_cfg = cfg.get("base")
    base_term = 0.0
    if isinstance(base_cfg, dict):
        try:
            base_term = float(base_cfg.get(term_name, 0.0))
        except (TypeError, ValueError):
            base_term = 0.0

    stages = cfg.get("stages")
    if not isinstance(stages, list):
        return False
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        opt_cfg = stage.get("global_opt")
        if not isinstance(opt_cfg, dict):
            opt_cfg = stage.get("opt")
        if not isinstance(opt_cfg, dict):
            opt_cfg = stage
        try:
            steps = int(opt_cfg.get("steps", 0))
        except (TypeError, ValueError):
            steps = 0
        if steps <= 0:
            continue

        eff = base_term
        params = _stage_params_list(opt_cfg)
        kind = "map" if (set(params) & map_params) and not (set(params) & model_params) else "model"
        w_fac = opt_cfg.get("w_fac")
        has_term_wfac = isinstance(w_fac, dict) and term_name in w_fac
        default_mul = opt_cfg.get("default_mul")
        if default_mul is not None:
            try:
                eff = base_term * float(default_mul)
            except (TypeError, ValueError):
                eff = 0.0
        if isinstance(w_fac, (int, float)):
            scalar_applies = (kind == "map" and term_name in map_loss_names) or (
                kind == "model" and term_name not in map_loss_names
            )
            if scalar_applies:
                try:
                    eff = base_term * float(w_fac)
                except (TypeError, ValueError):
                    eff = 0.0
        if has_term_wfac and w_fac.get(term_name) is not None:
            try:
                eff = base_term * float(w_fac.get(term_name))
            except (TypeError, ValueError):
                eff = 0.0
        if abs(eff) > 0.0:
            return True
    return False


def _config_effective_ext_offset_enabled(cfg: dict[str, Any]) -> bool:
    return _config_effective_loss_enabled(cfg, "ext_offset")


def _stage_params_list(opt_cfg: dict[str, Any]) -> list[str]:
    params = opt_cfg.get("params", [])
    if isinstance(params, str):
        return [params]
    if isinstance(params, list):
        return [str(p) for p in params]
    return []


def _config_global_map_enabled(cfg: dict[str, Any]) -> bool:
    args = cfg.get("args")
    if isinstance(args, dict):
        self_map_init = args.get("self-map-init", args.get("self_map_init", "off"))
        if str(self_map_init if self_map_init is not None else "off").strip().lower().replace("-", "_") != "off":
            return False
    if _config_effective_loss_enabled(cfg, "snap_surf_map"):
        return True
    stages = cfg.get("stages")
    if not isinstance(stages, list):
        return False
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        opt_cfg = stage.get("global_opt")
        if not isinstance(opt_cfg, dict):
            opt_cfg = stage.get("opt")
        if not isinstance(opt_cfg, dict):
            opt_cfg = stage
        params = set(_stage_params_list(opt_cfg))
        if params & {"map_surf_affine", "map_surf_ms"}:
            return True
        args = opt_cfg.get("args")
        if not isinstance(args, dict):
            continue
        snap_map = args.get("snap_surf_map")
        if isinstance(snap_map, dict) and snap_map.get("map_opt") is not None:
            return True
    return False


# ---------------------------------------------------------------------------
# Service announcement (file-based discovery)
# ---------------------------------------------------------------------------

_ANNOUNCE_DIR = Path.home() / ".fit_services"
_announce_file: Path | None = None


def _list_datasets() -> list[dict[str, str]]:
    """Return available .lasagna.json datasets from _data_dir."""
    if not _data_dir:
        return []
    data_path = Path(_data_dir)
    if not data_path.is_dir():
        return []
    datasets = []
    for entry in sorted(data_path.iterdir()):
        if entry.is_file() and entry.name.endswith(".lasagna.json"):
            datasets.append({"name": entry.name, "path": str(entry.resolve())})
    return datasets


def _clean_stale_announcements() -> None:
    """Remove announcement files whose PIDs are no longer alive."""
    if not _ANNOUNCE_DIR.is_dir():
        return
    for f in _ANNOUNCE_DIR.glob("*.json"):
        try:
            info = json.loads(f.read_text())
            pid = info.get("pid", -1)
            # Check if process is alive
            os.kill(pid, 0)
        except (OSError, json.JSONDecodeError, TypeError):
            try:
                f.unlink()
            except OSError:
                pass


def _write_announcement(host: str, port: int) -> None:
    """Write a service announcement file for discovery."""
    global _announce_file
    _ANNOUNCE_DIR.mkdir(parents=True, exist_ok=True)
    _clean_stale_announcements()

    pid = os.getpid()
    datasets = _list_datasets()
    info = {
        "host": host,
        "port": port,
        "pid": pid,
        "data_dir": _data_dir or "",
        "datasets": [d["name"] for d in datasets],
    }
    _announce_file = _ANNOUNCE_DIR / f"{pid}.json"
    _announce_file.write_text(json.dumps(info, indent=2))


def _remove_announcement() -> None:
    """Remove the announcement file on shutdown."""
    global _announce_file
    if _announce_file is not None:
        try:
            _announce_file.unlink(missing_ok=True)
        except OSError:
            pass
        _announce_file = None


# ---------------------------------------------------------------------------
# mDNS announcement via avahi-publish-service
# ---------------------------------------------------------------------------

_avahi_proc: subprocess.Popen | None = None


def _start_avahi_publish(port: int) -> None:
    """Publish service via avahi-publish-service (auto-unregisters on exit)."""
    global _avahi_proc

    txt_records: list[str] = []
    if _data_dir:
        txt_records.append(f"data_dir={_data_dir}")
    datasets = _list_datasets()
    if datasets:
        txt_records.append("datasets=" + ",".join(d["name"] for d in datasets))

    cmd = [
        "avahi-publish-service",
        f"Fit Optimizer (pid {os.getpid()})",
        "_fitoptimizer._tcp",
        str(port),
    ] + txt_records

    try:
        _avahi_proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[fit-service] mDNS: registered as _fitoptimizer._tcp on port {port}",
              flush=True)
    except FileNotFoundError:
        print("[fit-service] avahi-publish-service not found, skipping mDNS",
              flush=True)


def _stop_avahi_publish() -> None:
    """Stop the avahi-publish-service subprocess."""
    global _avahi_proc
    if _avahi_proc is not None:
        _avahi_proc.terminate()
        try:
            _avahi_proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            _avahi_proc.kill()
        _avahi_proc = None


# ---------------------------------------------------------------------------
# In-memory job queue
# ---------------------------------------------------------------------------

class _JobState:
    """Thread-safe mutable state for one queued optimizer job."""

    def __init__(self, *, job_id: str, sequence: int,
                 source: str, config_name: str, output_name: str = "") -> None:
        self._lock = threading.Lock()
        self.job_id = job_id
        self.sequence = sequence
        self.source = source
        self.config_name = config_name
        self.output_name = output_name
        self._state = "upload"
        self._stage = ""
        self._step = 0
        self._total_steps = 0
        self._loss = 0.0
        self._stage_progress = 0.0
        self._overall_progress = 0.0
        self._stage_name = ""
        self._error: str | None = None
        self._cancel = False
        self._output_dir: str | None = None
        self._results_tmp: str | None = None  # temp dir to clean up after download
        self._tmp_dirs: list[str] = []
        now = time.time()
        self.submitted_at = now
        self.started_at: float | None = None
        self.finished_at: float | None = None

    def snapshot(self, queue_position: int | None = None) -> dict[str, Any]:
        with self._lock:
            return {
                "job_id": self.job_id,
                "sequence": self.sequence,
                "source": self.source,
                "config_name": self.config_name,
                "output_name": self.output_name,
                "state": self._state,
                "queue_position": queue_position,
                "stage": self._stage,
                "step": self._step,
                "total_steps": self._total_steps,
                "loss": self._loss,
                "stage_progress": self._stage_progress,
                "overall_progress": self._overall_progress,
                "stage_name": self._stage_name,
                "error": self._error,
                "output_dir": self._output_dir,
                "submitted_at": self.submitted_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            }

    def set_waiting(self) -> None:
        with self._lock:
            self._state = "waiting"

    def set_running(self, stage: str, step: int, total: int, loss: float,
                    stage_progress: float = 0.0, overall_progress: float = 0.0,
                    stage_name: str = "") -> None:
        with self._lock:
            if self.started_at is None:
                self.started_at = time.time()
            self._state = "running"
            self._stage = stage
            self._step = step
            self._total_steps = total
            self._loss = loss
            self._stage_progress = stage_progress
            self._overall_progress = overall_progress
            self._stage_name = stage_name

    def set_finished(self, output_dir: str, results_tmp: str | None = None) -> None:
        with self._lock:
            self._state = "finished"
            self._output_dir = output_dir
            self._results_tmp = results_tmp
            self.finished_at = time.time()

    def set_error(self, msg: str) -> None:
        with self._lock:
            self._state = "error"
            self._error = msg
            self.finished_at = time.time()

    def set_cancelled(self, msg: str = "cancelled") -> None:
        with self._lock:
            self._state = "cancelled"
            self._error = msg
            self._cancel = True
            self.finished_at = time.time()

    def add_tmp_dir(self, path: str | None) -> None:
        if not path:
            return
        with self._lock:
            self._tmp_dirs.append(path)

    def cleanup_tmp_dirs(self, keep_results: bool = True) -> None:
        with self._lock:
            tmp_dirs = list(self._tmp_dirs)
            results_tmp = self._results_tmp
            self._tmp_dirs.clear()
        for path in tmp_dirs:
            if keep_results and results_tmp and path == results_tmp:
                continue
            shutil.rmtree(path, ignore_errors=True)

    @property
    def results_tmp(self) -> str | None:
        with self._lock:
            return self._results_tmp

    def clear_results(self) -> None:
        """Clean up results temp dir after download."""
        with self._lock:
            if self._results_tmp:
                shutil.rmtree(self._results_tmp, ignore_errors=True)
                self._results_tmp = None

    def request_cancel(self) -> None:
        with self._lock:
            self._cancel = True

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancel

    @property
    def state(self) -> str:
        with self._lock:
            return self._state


class _JobQueue:
    """FIFO scheduler with one running optimizer job."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._jobs: dict[str, _JobState] = {}
        self._bodies: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []
        self._next_sequence = 1
        self._active_job_id: str | None = None
        self._generation = 0
        self._worker = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._worker.start()

    def _bump_generation_locked(self) -> None:
        self._generation += 1

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def create_upload(self, *, source: str, config_name: str, output_name: str = "") -> _JobState:
        with self._cv:
            sequence = self._next_sequence
            self._next_sequence += 1
            job_id = uuid.uuid4().hex[:12]
            while job_id in self._jobs:
                job_id = uuid.uuid4().hex[:12]
            job = _JobState(
                job_id=job_id,
                sequence=sequence,
                source=source,
                config_name=config_name,
                output_name=output_name,
            )
            self._jobs[job_id] = job
            self._order.append(job_id)
            self._bump_generation_locked()
            self._cv.notify_all()
            return job

    def enqueue_body(self, job: _JobState, body: dict[str, Any]) -> None:
        with self._cv:
            source = str(body.get("source") or job.source or "").strip()
            config_name = str(body.get("config_name") or job.config_name or "").strip()
            output_name = str(body.get("output_name") or job.output_name or "").strip()
            if source:
                job.source = source
            if config_name:
                job.config_name = config_name
            if output_name:
                job.output_name = output_name
            self._bodies[job.job_id] = body
            job.set_waiting()
            self._bump_generation_locked()
            self._cv.notify_all()

    def _waiting_ids_locked(self) -> list[str]:
        return [
            jid for jid in self._order
            if self._jobs[jid].state in {"upload", "waiting"}
        ]

    def _reorderable_ids_locked(self) -> list[str]:
        return [
            jid for jid in self._order
            if self._jobs[jid].state == "waiting"
        ]

    def queue_position(self, job_id: str) -> int | None:
        with self._lock:
            waiting = self._waiting_ids_locked()
            if job_id in waiting:
                return waiting.index(job_id) + 1
            return None

    def snapshot(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return job.snapshot(self.queue_position(job_id))

    def job(self, job_id: str) -> _JobState | None:
        with self._lock:
            return self._jobs.get(job_id)

    def snapshots(self) -> list[dict[str, Any]]:
        with self._lock:
            waiting = self._waiting_ids_locked()
            positions = {jid: i + 1 for i, jid in enumerate(waiting)}
            return [
                self._jobs[jid].snapshot(positions.get(jid))
                for jid in self._order
            ]

    def snapshot_response(self) -> dict[str, Any]:
        with self._lock:
            return {
                "queue_generation": self._generation,
                "jobs": self.snapshots(),
            }

    def legacy_status(self) -> dict[str, Any]:
        with self._lock:
            job: _JobState | None = None
            if self._active_job_id:
                job = self._jobs.get(self._active_job_id)
            if job is None:
                for jid in self._order:
                    candidate = self._jobs[jid]
                    if candidate.state == "finished":
                        job = candidate
                if job is None and self._order:
                    job = self._jobs[self._order[0]]
            if job is None:
                return {
                    "state": "idle",
                    "queue_generation": self._generation,
                    "stage": "",
                    "step": 0,
                    "total_steps": 0,
                    "loss": 0.0,
                    "stage_progress": 0.0,
                    "overall_progress": 0.0,
                    "stage_name": "",
                    "error": None,
                    "output_dir": None,
                }
            snap = job.snapshot(self.queue_position(job.job_id))
            if snap["state"] in {"upload", "waiting"}:
                snap["state"] = "queued"
            if snap["state"] == "cancelled":
                snap["state"] = "error"
            snap["queue_generation"] = self._generation
            return snap

    def latest_finished_job(self) -> _JobState | None:
        with self._lock:
            for jid in reversed(self._order):
                job = self._jobs[jid]
                if job.state == "finished":
                    return job
            return None

    def cancel(self, job_id: str) -> tuple[bool, str]:
        with self._cv:
            job = self._jobs.get(job_id)
            if job is None:
                return False, "job not found"
            if job.state in {"finished", "error", "cancelled"}:
                return False, f"cannot cancel {job.state} job"
            if job.state in {"upload", "waiting"}:
                self._bodies.pop(job_id, None)
                job.set_cancelled()
                self._bump_generation_locked()
                self._cv.notify_all()
                return True, "cancelled"
            job.request_cancel()
            self._bump_generation_locked()
            return True, "stopping"

    def cancel_active(self) -> tuple[bool, str]:
        with self._cv:
            if self._active_job_id:
                return self.cancel(self._active_job_id)
            return False, "not running"

    def reorder(self, body: dict[str, Any]) -> tuple[bool, str]:
        with self._cv:
            movable = self._reorderable_ids_locked()
            if "order" in body:
                requested = [str(x) for x in body.get("order", [])]
                if sorted(requested) != sorted(movable):
                    return False, "order must contain exactly the waiting job ids"
                self._order = [
                    jid for jid in self._order if jid not in movable
                ] + requested
                self._bump_generation_locked()
                self._cv.notify_all()
                return True, "reordered"

            job_id = str(body.get("job_id", ""))
            before_job_id = body.get("before_job_id")
            before = str(before_job_id) if before_job_id is not None else None
            if job_id not in movable:
                return False, "job is not reorderable"
            if before is not None and before not in movable:
                return False, "before_job_id is not reorderable"
            movable.remove(job_id)
            if before is None:
                movable.append(job_id)
            else:
                movable.insert(movable.index(before), job_id)
            self._order = [
                jid for jid in self._order if self._jobs[jid].state != "waiting"
            ] + movable
            self._bump_generation_locked()
            self._cv.notify_all()
            return True, "reordered"

    def _scheduler_loop(self) -> None:
        while True:
            with self._cv:
                ready_id = None
                while ready_id is None:
                    for jid in self._order:
                        if self._jobs[jid].state == "waiting" and jid in self._bodies:
                            ready_id = jid
                            break
                    if ready_id is None:
                        self._cv.wait()
                job = self._jobs[ready_id]
                body = self._bodies.pop(ready_id)
                self._active_job_id = ready_id
                job.set_running("starting", 0, 0, 0.0)
                self._bump_generation_locked()
            try:
                _run_optimization(job, body)
            finally:
                job.cleanup_tmp_dirs(keep_results=True)
                with self._cv:
                    if self._active_job_id == ready_id:
                        self._active_job_id = None
                    self._bump_generation_locked()
                    self._cv.notify_all()


_jobs = _JobQueue()


# ---------------------------------------------------------------------------
# Optimization runner (called in background thread)
# ---------------------------------------------------------------------------

def _run_optimization(job: _JobState, body: dict[str, Any]) -> None:
    """Run fit.py then fit2tifxyz.py based on the request body.

    Supports two modes:
      - Local (internal): model_input and output_dir are local paths.
      - Remote (external): model_data contains base64-encoded model bytes,
        output goes to a temp dir, caller downloads results via GET /results.
    """
    import tempfile

    print(
        f"[fit-service] optimization worker starting: output_name={str(body.get('output_name') or '').strip()!r} "
        f"keys={sorted(body.keys())}",
        flush=True,
    )

    try:
        body = _body_with_resolved_job_spec(body)
    except Exception as exc:
        job.set_error(str(exc))
        return

    job_spec = body.get("_job_spec_")
    model_output = body.get("model_output")
    data_input = body.get("data_input")
    output_dir = body.get("output_dir")
    config = body.get("config", {})

    if not isinstance(config, dict):
        job.set_error("request config must be an object")
        return
    try:
        request_volume_shape_zyx = _request_volume_shape_zyx(body)
    except Exception as exc:
        job.set_error(str(exc))
        return
    args_section_initial = config.get("args", {})
    if not isinstance(args_section_initial, dict):
        args_section_initial = {}
    model_init_requested = str(
        args_section_initial.get("model-init", args_section_initial.get("model_init", "seed"))
    ).strip().lower()
    if model_init_requested not in {"seed", "ext", "model", "flatten", "atlas"}:
        job.set_error(
            f"invalid args.model-init '{model_init_requested}' (expected seed, ext, model, flatten, or atlas)"
        )
        return
    if not data_input and model_init_requested != "flatten":
        job.set_error("missing 'data_input'")
        return

    try:
        # Use a temp directory for all intermediate files (config json,
        # model output, etc.) so nothing leaks into the volpkg paths dir.
        tmp_dir = tempfile.mkdtemp(prefix="fit_reopt_")
        job.add_tmp_dir(tmp_dir)
        service_workdir = Path.cwd()
        print(f"[fit-service] cwd: {service_workdir}", flush=True)

        # If no output_dir, create a temp dir for results (external mode)
        results_tmp = None
        if not output_dir:
            results_tmp = tempfile.mkdtemp(prefix="fit_results_")
            job.add_tmp_dir(results_tmp)
            output_dir = results_tmp

        # model_output goes into temp dir. Flatten forces --copy-model during
        # export so the resulting tifxyz remains self-contained after cleanup.
        if not model_output:
            model_output = str(Path(tmp_dir) / "model_reopt.pt")

        # Build argv for fit.py from the config dict.
        cfg = dict(config)
        if request_volume_shape_zyx is not None:
            cfg["vc3d_volume_shape_zyx"] = [int(v) for v in request_volume_shape_zyx]
        args_section_pre = cfg.get("args", {})
        if not isinstance(args_section_pre, dict):
            args_section_pre = {}
        model_init = model_init_requested
        args_section_pre.pop("model_init", None)
        args_section_pre["model-init"] = model_init
        cfg["args"] = args_section_pre
        ext_offset_enabled = _config_effective_ext_offset_enabled(cfg)
        global_map_enabled = _config_global_map_enabled(cfg)
        if model_init == "flatten" and ext_offset_enabled:
            raise ValueError("model-init=flatten does not support ext_offset")
        model_input = _decode_model_for_request(
            body=body,
            tmp_dir=tmp_dir,
            model_init=model_init,
        )
        tifxyz_dir = _wire_external_surface_for_request(
            cfg=cfg,
            args_section=args_section_pre,
            model_init=model_init,
            ext_offset_enabled=ext_offset_enabled,
            global_map_enabled=global_map_enabled,
        )

        if model_init == "model" and not model_input:
            raise ValueError("model-init=model requires request model_data or model_input")

        args_section = dict(cfg.get("args", {}))
        if model_init != "flatten":
            args_section["input"] = str(data_input)
            args_section.setdefault("sparse-prefetch-backend", _sparse_prefetch_backend)
        if model_init == "model" and model_input:
            args_section["model-input"] = str(model_input)
        args_section["model-output"] = str(model_output)
        # Only set fit.py out-dir if explicitly requested. pred_dt_flow_gate
        # debug slices use their own debug_out_dir so enabling them does not
        # make fit.py export model_final/tifxyz into the service cwd.
        if body.get("out_dir"):
            args_section["out-dir"] = str(body["out_dir"])
        elif _config_enables_pred_dt_flow_gate(cfg):
            _set_pred_dt_flow_gate_debug_out_dir(cfg, str(service_workdir))
        if global_map_enabled:
            snap_debug_dir = Path(service_workdir) / "snap_surf_objs"
            _set_snap_surf_map_debug_obj_dir(cfg, str(snap_debug_dir))
        cfg["args"] = args_section
        cfg_path = str(Path(tmp_dir) / "fit_config.json")
        has_corr = "corr_points" in cfg
        n_corr_cols = len(cfg["corr_points"].get("collections", {})) if has_corr and isinstance(cfg.get("corr_points"), dict) else 0
        print(f"[fit-service] writing config: corr_points={has_corr} ({n_corr_cols} collections)", flush=True)
        Path(cfg_path).write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        # Monkey-patch the optimizer to report progress & check cancellation.
        import optimizer as opt_mod

        _orig_optimize = opt_mod.optimize

        def _check_cancel() -> None:
            if job.cancelled:
                raise KeyboardInterrupt("cancelled by user")

        def _patched_optimize(**kwargs: Any) -> Any:
            orig_snapshot = kwargs.get("snapshot_fn")
            orig_progress = kwargs.get("progress_fn")

            def _wrapped_snapshot(*, stage: str, step: int, loss: float, **kw: Any) -> None:
                _check_cancel()
                if orig_snapshot is not None:
                    orig_snapshot(stage=stage, step=step, loss=loss, **kw)

            def _wrapped_progress(*, step: int, total: int, loss: float, **kw: Any) -> None:
                job.set_running(
                    "optimizing", step, total, loss,
                    stage_progress=float(kw.get("stage_progress", 0.0)),
                    overall_progress=float(kw.get("overall_progress", 0.0)),
                    stage_name=str(kw.get("stage_name", "")),
                )
                _check_cancel()
                if orig_progress is not None:
                    orig_progress(step=step, total=total, loss=loss, **kw)

            kwargs["snapshot_fn"] = _wrapped_snapshot
            kwargs["progress_fn"] = _wrapped_progress
            kwargs["cancel_fn"] = _check_cancel
            return _orig_optimize(**kwargs)

        from contextlib import nullcontext
        from gpu_pause import gpu_pause_context

        opt_mod.optimize = _patched_optimize
        with (gpu_pause_context() if _gpu_pause_enabled else nullcontext()):
            try:
                import fit as fit_mod
                job.set_running("loading", 0, 0, 0.0)
                fit_mod.main([cfg_path])
                if isinstance(job_spec, dict) and Path(model_output).is_file():
                    import torch
                    st = torch.load(str(model_output), map_location="cpu", weights_only=False)
                    if isinstance(st, dict):
                        st["_job_spec_"] = job_spec
                        object_refs = body.get("_object_refs_")
                        if isinstance(object_refs, dict):
                            st["_object_refs_"] = object_refs
                        torch.save(st, str(model_output))
            finally:
                opt_mod.optimize = _orig_optimize

            if job.cancelled:
                job.set_cancelled()
                return

            save_t0 = time.perf_counter()
            job.set_running("exporting", 0, 0, 0.0)
            import fit2tifxyz
            export_argv = ["--input", str(model_output), "--output", str(output_dir)]
            if body.get("single_segment"):
                export_argv.append("--single-segment")
            if body.get("copy_model") or model_init == "flatten":
                export_argv.append("--copy-model")
            output_name = body.get("output_name")
            if output_name:
                export_argv.extend(["--output-name", str(output_name)])
            print(
                f"[fit-service] exporting tifxyz: output_name={str(output_name or '').strip()!r} "
                f"output_dir={output_dir}",
                flush=True,
            )
            voxel_size_um = config.get("voxel_size_um")
            if voxel_size_um is not None:
                export_argv.extend(["--voxel-size-um", str(float(voxel_size_um))])
            if request_volume_shape_zyx is not None:
                export_argv.extend([
                    "--target-volume-shape-zyx",
                    str(int(request_volume_shape_zyx[0])),
                    str(int(request_volume_shape_zyx[1])),
                    str(int(request_volume_shape_zyx[2])),
                ])
            _check_cancel()
            fit2tifxyz.main(export_argv, cancel_fn=_check_cancel)
            _normalize_single_tifxyz_output(Path(output_dir), str(output_name or ""))
            _check_cancel()
            save_s = time.perf_counter() - save_t0
            try:
                saved_bytes = _dir_size_bytes(Path(output_dir))
            except OSError:
                saved_bytes = 0
            print(
                f"[fit-service] results saved: {_mib(saved_bytes):.3f} MiB "
                f"in {save_s:.3f}s",
                flush=True,
            )

        # Clean up intermediate files (but keep results_tmp for download)
        shutil.rmtree(tmp_dir, ignore_errors=True)

        job.set_finished(str(output_dir), results_tmp=results_tmp)
        print(f"[fit-service] optimization finished, output: {output_dir}", flush=True)

    except KeyboardInterrupt:
        job.set_cancelled()
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[fit-service] error: {tb}", file=sys.stderr, flush=True)
        job.set_error(str(exc))


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):

    def _send_json(self, obj: Any, status: int = 200) -> None:
        if isinstance(obj, dict):
            obj.setdefault("api_version", _API_VERSION)
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header(_API_VERSION_HEADER, _API_VERSION)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _validate_api_version(self) -> bool:
        got = self.headers.get(_API_VERSION_HEADER)
        if got == _API_VERSION:
            return True
        self._send_json({
            "error": (
                f"fit-service API version mismatch: expected {_API_VERSION_HEADER}="
                f"{_API_VERSION}, got {got or '<missing>'}"
            ),
            "expected_api_version": _API_VERSION,
            "received_api_version": got,
        }, 426)
        return False

    def _read_json(self, label: str = "request") -> Any:
        length = int(self.headers.get("Content-Length", 0))
        print(f"[fit-service] {label}: reading {_mib(length):.3f} MiB", flush=True)
        read_t0 = time.perf_counter()
        raw = self.rfile.read(length)
        read_s = time.perf_counter() - read_t0
        parse_t0 = time.perf_counter()
        obj = json.loads(raw) if raw else {}
        parse_s = time.perf_counter() - parse_t0
        print(
            f"[fit-service] {label}: read {_mib(len(raw)):.3f} MiB "
            f"in {read_s:.3f}s ({_mib_per_s(len(raw), read_s):.3f} MiB/s), "
            f"json_parse={parse_s:.3f}s",
            flush=True,
        )
        return obj

    def _client_source(self) -> str:
        try:
            user = getpass.getuser()
        except Exception:
            user = "unknown"
        host = self.client_address[0] if self.client_address else ""
        try:
            host = socket.gethostbyaddr(host)[0]
        except Exception:
            pass
        return f"{user}@{host}" if host else user

    def _request_source(self) -> str:
        return str(self.headers.get(_VC3D_SOURCE_HEADER) or self._client_source()).strip()

    def _job_path_parts(self) -> list[str]:
        path = urlparse(self.path).path
        return [part for part in path.split("/") if part]

    def do_GET(self) -> None:  # noqa: N802
        if not self._validate_api_version():
            return
        parts = self._job_path_parts()
        path = "/" + "/".join(parts)

        if path == "/health":
            self._send_json({"status": "ok"})
        elif path == "/status":
            self._send_json(_jobs.legacy_status())
        elif path == "/datasets":
            self._send_json({"datasets": _list_datasets()})
        elif path == "/results":
            job = _jobs.latest_finished_job()
            if job is None:
                self._send_json({"error": "no finished results available"}, 404)
                return
            self._handle_results(job)
        elif parts == ["jobs"]:
            self._send_json(_jobs.snapshot_response())
        elif len(parts) == 2 and parts[0] == "jobs":
            snap = _jobs.snapshot(parts[1])
            if snap is None:
                self._send_json({"error": "job not found"}, 404)
            else:
                snap["queue_generation"] = _jobs.generation
                self._send_json(snap)
        elif len(parts) == 3 and parts[0] == "jobs" and parts[2] == "results":
            job = _jobs.job(parts[1])
            if job is None:
                self._send_json({"error": "job not found"}, 404)
                return
            self._handle_results(job)
        else:
            self._send_json({"error": "not found"}, 404)

    def _handle_results(self, job: _JobState) -> None:
        """Package finished optimization results as tar.gz and send them."""
        import tarfile
        import io

        snap = job.snapshot(_jobs.queue_position(job.job_id))
        if snap["state"] != "finished":
            self._send_json({"error": "no finished results available"}, 404)
            return

        output_dir = snap["output_dir"]
        if not output_dir or not Path(output_dir).is_dir():
            self._send_json({"error": "output directory not found"}, 404)
            return

        # Create tar.gz in memory.  Archive paths are relative to output_dir
        # so the tar contains e.g. "winding_combined_v004.tifxyz/meta.json".
        # Extracting in the local paths dir recreates the tifxyz subdirectory.
        pack_t0 = time.perf_counter()
        buf = io.BytesIO()
        out_path = Path(output_dir)
        children = sorted(out_path.iterdir())
        requested_output_name = str(snap.get("output_name") or "").strip()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for child in children:
                arcname = _result_archive_child_name(child.name, len(children), requested_output_name)
                tar.add(str(child), arcname=arcname)

        data = buf.getvalue()
        pack_s = time.perf_counter() - pack_t0
        self.send_response(200)
        self.send_header("Content-Type", "application/gzip")
        self.send_header(_API_VERSION_HEADER, _API_VERSION)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        send_t0 = time.perf_counter()
        self.wfile.write(data)
        self.wfile.flush()
        send_s = time.perf_counter() - send_t0

        print(
            f"[fit-service] results packed: {_mib(len(data)):.3f} MiB "
            f"in {pack_s:.3f}s",
            flush=True,
        )
        print(
            f"[fit-service] results sent: {_mib(len(data)):.3f} MiB "
            f"in {send_s:.3f}s ({_mib_per_s(len(data), send_s):.3f} MiB/s)",
            flush=True,
        )
        job.clear_results()

    def do_POST(self) -> None:  # noqa: N802
        if not self._validate_api_version():
            return
        parts = self._job_path_parts()
        path = "/" + "/".join(parts)

        if path == "/optimize":
            print("[fit-service] /optimize POST received", flush=True)
            try:
                body = self._read_json("optimize request")
            except Exception as exc:
                self._send_json({"error": f"bad json: {exc}"}, 400)
                return
            job = _jobs.create_upload(
                source=str(body.get("source") or self._client_source()),
                config_name=str(body.get("config_name") or ""),
                output_name=str(body.get("output_name") or ""),
            )
            _jobs.enqueue_body(job, body)
            print(
                f"[fit-service] /optimize accepted as queued job {job.job_id}: "
                f"output_name={job.output_name!r}",
                flush=True,
            )
            self._send_json({
                "status": "started",
                "job_id": job.job_id,
                "sequence": job.sequence,
                "source": job.source,
                "config_name": job.config_name,
                "output_name": job.output_name,
                "queue_position": _jobs.queue_position(job.job_id),
                "queue_generation": _jobs.generation,
            })

        elif path == "/objects/query":
            try:
                body = self._read_json("objects query")
                refs = body.get("objects", []) if isinstance(body, dict) else []
                if not isinstance(refs, list):
                    raise ValueError("objects must be a list")
                present = []
                missing = []
                for ref_raw in refs:
                    ref = _validate_object_ref(ref_raw)
                    (present if _object_present(ref) else missing).append(ref)
            except Exception as exc:
                self._send_json({"error": str(exc)}, 400)
                return
            self._send_json({"present": present, "missing": missing})

        elif path == "/objects":
            try:
                body = self._read_json("object upload")
                if not isinstance(body, dict):
                    raise ValueError("upload body must be an object")
                ref = _store_uploaded_object(body)
            except Exception as exc:
                self._send_json({"error": str(exc)}, 400)
                return
            self._send_json({"status": "stored", "object": ref})

        elif path == "/stop":
            ok, msg = _jobs.cancel_active()
            if ok:
                self._send_json({"status": msg})
            else:
                self._send_json({"status": "not running"})

        elif path == "/jobs":
            print("[fit-service] /jobs POST received", flush=True)
            job = _jobs.create_upload(source=self._request_source(), config_name="")
            try:
                body = self._read_json(f"job {job.job_id} request")
                if not isinstance(body, dict):
                    raise ValueError("job request must be an object")
                body = _body_with_resolved_job_spec(body)
            except Exception as exc:
                job.set_error(f"bad json: {exc}")
                self._send_json({"error": f"bad json: {exc}", "job_id": job.job_id}, 400)
                return
            _jobs.enqueue_body(job, body)
            self._send_json({
                "status": "queued",
                "job_id": job.job_id,
                "sequence": job.sequence,
                "source": job.source,
                "config_name": job.config_name,
                "output_name": job.output_name,
                "queue_position": _jobs.queue_position(job.job_id),
                "queue_generation": _jobs.generation,
            })

        elif len(parts) == 3 and parts[0] == "jobs" and parts[2] == "cancel":
            ok, msg = _jobs.cancel(parts[1])
            self._send_json(
                {"status": msg, "queue_generation": _jobs.generation}
                if ok else {"error": msg, "queue_generation": _jobs.generation},
                200 if ok else 409,
            )

        elif parts == ["jobs", "reorder"]:
            try:
                body = self._read_json("reorder request")
            except Exception as exc:
                self._send_json({"error": f"bad json: {exc}"}, 400)
                return
            ok, msg = _jobs.reorder(body if isinstance(body, dict) else {})
            response = _jobs.snapshot_response()
            response["status"] = msg
            self._send_json(response if ok else {"error": msg, "queue_generation": _jobs.generation},
                            200 if ok else 409)

        elif path == "/export_vis":
            try:
                body = self._read_json("export_vis request")
            except Exception as exc:
                self._send_json({"error": f"bad json: {exc}"}, 400)
                return
            self._handle_export_vis(body)

        else:
            self._send_json({"error": "not found"}, 404)

    def _handle_export_vis(self, body: dict[str, Any]) -> None:
        """Synchronously export multi-layer OBJ visualization.

        Returns the exported files as a tar.gz binary response.
        """
        import base64
        import io
        import shutil
        import tarfile
        import tempfile

        model_input = body.get("model_input")
        model_data = body.get("model_data")
        data_input = body.get("data_input")

        tmp_model = None
        tmp_dir = None
        try:
            tmp_dir = tempfile.mkdtemp(prefix="fit_vis_")

            if model_data:
                model_bytes = base64.b64decode(model_data)
                tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
                tmp.write(model_bytes)
                tmp.close()
                model_input = tmp.name
                tmp_model = tmp.name
            elif not model_input:
                self._send_json({"error": "missing model_input or model_data"}, 400)
                return

            # If data_input not provided, extract from checkpoint's _fit_config_
            if not data_input:
                import torch
                st = torch.load(str(model_input), map_location="cpu", weights_only=False)
                fit_cfg = st.get("_fit_config_", {}) or {}
                fit_args = fit_cfg.get("args", {}) or {}
                data_input = fit_args.get("input")
                if not data_input:
                    self._send_json({"error": "missing 'data_input' and checkpoint has no _fit_config_.args.input"}, 400)
                    return
                # Resolve relative paths against _data_dir if available
                if _data_dir and not Path(data_input).is_absolute():
                    candidate = Path(_data_dir) / data_input
                    if candidate.exists():
                        data_input = str(candidate)

            import lasagna_analyze
            from contextlib import nullcontext
            from gpu_pause import gpu_pause_context
            with (gpu_pause_context() if _gpu_pause_enabled else nullcontext()):
                lasagna_analyze.export_vis_obj(
                    model_path=str(model_input),
                    data_path=str(data_input),
                    output_dir=tmp_dir,
                    slices=body.get("slices", []),
                    channels=body.get("channels", []),
                    losses=body.get("losses", []),
                    include_mesh=bool(body.get("include_mesh", True)),
                    include_connections=bool(body.get("include_connections", True)),
                    device=body.get("device", "cuda"),
                )

            # Package as tar.gz
            buf = io.BytesIO()
            out_path = Path(tmp_dir)
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                for child in sorted(out_path.iterdir()):
                    tar.add(str(child), arcname=child.name)

            data = buf.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "application/gzip")
            self.send_header(_API_VERSION_HEADER, _API_VERSION)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

            print(f"[fit-service] export_vis done ({len(data)} bytes)", flush=True)
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[fit-service] export_vis error: {tb}", file=sys.stderr, flush=True)
            self._send_json({"error": str(exc)}, 500)
        finally:
            if tmp_model:
                try:
                    os.unlink(tmp_model)
                except OSError:
                    pass
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def log_message(self, fmt: str, *args: Any) -> None:
        msg = fmt % args
        if "/status" in msg:
            return
        print(f"[fit-service] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global _data_dir, _object_store_dir, _gpu_pause_enabled, _sparse_prefetch_backend

    p = argparse.ArgumentParser(description="Fit optimizer HTTP service for VC3D")
    p.add_argument("--port", type=int, default=9999, help="Port (default 9999)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--data-dir", default=None,
                   help="Directory containing .lasagna.json datasets")
    p.add_argument("--object-store-dir", default=None,
                   help="Directory for content-addressed VC3D artifacts")
    p.add_argument("--no-gpu-pause", action="store_true", default=False,
                   help="Disable automatic GPU pause/resume of training")
    p.add_argument("--sparse-prefetch-backend",
                   choices=("tensorstore", "python-zarr"),
                   default="tensorstore",
                   help="Sparse streaming prefetch backend for fit jobs")
    args = p.parse_args()

    if args.data_dir:
        _data_dir = str(Path(args.data_dir).resolve())
    if args.object_store_dir:
        _object_store_dir = Path(args.object_store_dir).resolve()
    if args.no_gpu_pause:
        _gpu_pause_enabled = False
    _sparse_prefetch_backend = str(args.sparse_prefetch_backend)

    datasets = _list_datasets()
    if not datasets:
        data_dir_msg = _data_dir if _data_dir else "<not set>"
        print(
            f"[fit-service] error: no .lasagna.json datasets found in --data-dir {data_dir_msg}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(2)

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    actual_port = server.server_address[1]

    # Write service announcement for discovery (file-based + mDNS)
    _write_announcement(args.host, actual_port)
    _start_avahi_publish(actual_port)
    atexit.register(_remove_announcement)
    atexit.register(_stop_avahi_publish)

    # This exact format is parsed by FitServiceManager on the C++ side
    print(f"listening on http://{args.host}:{actual_port}", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _stop_avahi_publish()
        _remove_announcement()
        server.server_close()


if __name__ == "__main__":
    main()
