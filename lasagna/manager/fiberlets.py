"""Managed whole-volume Fiberlet preprocessing jobs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import getpass
from typing import Any, Sequence

from .config import ManagerConfig, config_path
from .runs import (
    SCHEMA_VERSION, _git_revision, _process_matches, _runtime_info,
    _runtime_python, atomic_json, launch_recorded_job, read_runs,
    reserve_managed_run, resolve_run, utc_now,
)
from .tmux import Tmux


_MANAGER_OWNED_OPTIONS = {
    "--anchor-cache",
    "--normal-manifest",
    "--source-context",
}


def _validate_native_options(values: Sequence[str], *, source: str) -> None:
    for value in values:
        option = str(value).split("=", 1)[0]
        if option in _MANAGER_OWNED_OPTIONS:
            raise ValueError(
                f"{source} cannot override manager-owned Fiberlet option {option}"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _completed_dependency(
    config: ManagerConfig, selector: str, *, expected_kind: str,
) -> dict[str, Any]:
    run_dir, record = resolve_run(config, selector)
    phase = str(record.get("active_lifecycle_phase") or "inference")
    if record.get("status") != "completed" or record.get("lifecycle", {}).get(phase) != "completed":
        raise ValueError(f"input run {record.get('run_name', selector)!r} is not completed")
    if record.get("artifact_kind") != expected_kind:
        raise ValueError(
            f"input run {record.get('run_name', selector)!r} has artifact_kind "
            f"{record.get('artifact_kind')!r}, expected {expected_kind!r}"
        )
    artifacts = record.get("artifacts") if isinstance(record.get("artifacts"), dict) else {}
    bundle = run_dir / str(artifacts.get("root", "artifacts"))
    provenance_path = run_dir / str(artifacts.get("provenance", "artifacts/inference.json"))
    provenance = _load_json(provenance_path, "portable provenance")
    if provenance.get("status") != "completed" or provenance.get("artifact_kind") != expected_kind:
        raise ValueError(f"input run {record.get('run_name', selector)!r} has incomplete portable provenance")
    manifest_entries = [
        entry for entry in provenance.get("artifacts", ())
        if isinstance(entry, dict) and entry.get("kind") == "manifest"
    ]
    if len(manifest_entries) != 1 or not isinstance(manifest_entries[0].get("path"), str):
        raise ValueError(f"input run {record.get('run_name', selector)!r} must contain exactly one manifest")
    relative = Path(manifest_entries[0]["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("input manifest path is not portable")
    manifest_path = bundle / relative
    manifest = _load_json(manifest_path, "Lasagna manifest")
    channels = {
        str(channel)
        for group in manifest.get("groups", {}).values()
        if isinstance(group, dict)
        for channel in group.get("channels", ())
    }
    required = {"presence", "nx", "ny"} if expected_kind == "fiber3d-prediction" else {"nx", "ny"}
    missing = sorted(required - channels)
    if missing:
        raise ValueError(
            f"input run {record.get('run_name', selector)!r} is missing channels: {', '.join(missing)}"
        )
    crops = manifest.get("crops") or ([] if manifest.get("crop_xyzwhd") is None else [manifest["crop_xyzwhd"]])
    inference = provenance.get("inference") if isinstance(provenance.get("inference"), dict) else {}
    if crops or inference.get("crop_xyzwhd_base") is not None:
        raise ValueError("whole-volume Fiberlet preprocessing does not accept cropped prediction bundles")
    source = provenance.get("source") if isinstance(provenance.get("source"), dict) else {}
    model = provenance.get("model") if isinstance(provenance.get("model"), dict) else {}
    source_scale = provenance.get("source_scale") if isinstance(provenance.get("source_scale"), dict) else {}
    return {
        "run_dir": run_dir,
        "run_name": str(record.get("run_name") or run_dir.name),
        "run_uuid": str(record.get("run_uuid") or ""),
        "artifact_kind": expected_kind,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256(manifest_path),
        "manifest": manifest,
        "source": source,
        "model": model,
        "source_scale": source_scale,
    }


def resolve_fiberlet_inputs(
    config: ManagerConfig, fiber_selector: str, normal_selector: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fiber = _completed_dependency(config, fiber_selector, expected_kind="fiber3d-prediction")
    normal = _completed_dependency(config, normal_selector, expected_kind="lasagna")
    for field in ("sample_id", "volume_id"):
        if not fiber["source"].get(field) or fiber["source"].get(field) != normal["source"].get(field):
            raise ValueError(f"Fiber and normal inputs must identify the same source {field}")
    fiber_shape = fiber["manifest"].get("base_shape_zyx")
    normal_shape = normal["manifest"].get("base_shape_zyx")
    if not fiber_shape or list(fiber_shape) != list(normal_shape or ()):
        raise ValueError("Fiber and normal manifests must have the same base_shape_zyx")
    if float(fiber["manifest"].get("source_to_base", 1.0)) != float(normal["manifest"].get("source_to_base", 1.0)):
        raise ValueError("Fiber and normal manifests use incompatible base coordinate frames")
    return fiber, normal


def _resolve_binary(config: ManagerConfig) -> Path:
    configured = config.fiberlet_binary.strip()
    if configured:
        candidate = Path(os.path.expandvars(configured)).expanduser()
        if not candidate.is_absolute():
            candidate = config_path().parent / candidate
        candidate = candidate.resolve()
    else:
        found = shutil.which("vc_fiberlets")
        if found is None:
            raise FileNotFoundError(
                "vc_fiberlets is not on PATH; set fiberlet_binary in the las_manager config"
            )
        candidate = Path(found).resolve()
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise FileNotFoundError(f"configured Fiberlet executable is not executable: {candidate}")
    return candidate


def _dependency_identity(value: dict[str, Any]) -> dict[str, Any]:
    model = value["model"]
    return {
        "run_uuid": value["run_uuid"],
        "artifact_kind": value["artifact_kind"],
        "manifest_sha256": value["manifest_sha256"],
        "model_id": model.get("atlas_model_id"),
        "model_sha256": model.get("sha256"),
        "model_run": model.get("run"),
        "snapshot": model.get("snapshot"),
        "requested_group": value["source_scale"].get("requested_group", value["source"].get("requested_group")),
    }


def launch_fiberlet(
    config: ManagerConfig,
    fiber_selector: str,
    normal_selector: str,
    *,
    original_argv: Sequence[str],
    extra_args: Sequence[str] = (),
    tmux: Tmux | None = None,
) -> Path:
    fiber, normal = resolve_fiberlet_inputs(config, fiber_selector, normal_selector)
    _validate_native_options(config.fiberlet_params, source="fiberlet_params")
    _validate_native_options(extra_args, source="explicit arguments")
    binary = _resolve_binary(config)
    client = tmux or Tmux()
    sample_id = str(fiber["source"]["sample_id"])
    volume_id = str(fiber["source"]["volume_id"])
    run_uuid, run_name, session, run_dir = reserve_managed_run(
        config,
        name_stem=f"{sample_id}-{volume_id}-fiberlets",
        session_prefix="flt",
        tmux=client,
    )
    artifacts = run_dir / "artifacts"
    artifacts.mkdir()
    cache = run_dir / "cache"
    cache.mkdir()
    output = artifacts / "fiberlets.zarr"
    context_path = run_dir / "source_context.json"
    fiber_identity = _dependency_identity(fiber)
    normal_identity = _dependency_identity(normal)
    source_context = {
        "source_volume": {"sample_id": sample_id, "volume_id": volume_id},
        "fiber_prediction": fiber_identity,
        "normal_prediction": normal_identity,
    }
    atomic_json(context_path, source_context)
    effective_args = (
        "--threads", str(config.fiberlet_threads),
        *config.fiberlet_params,
        *extra_args,
    )
    command = [
        str(binary), "preprocess-volume", str(fiber["manifest_path"]), str(output),
        "--normal-manifest", str(normal["manifest_path"]),
        "--anchor-cache", str(cache / "fiberlets.anchors.zarr"),
        "--source-context", str(context_path),
        *effective_args,
    ]
    python = _runtime_python(config)
    source_license = fiber["source"].get("license")
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "job_kind": "fiberlet",
        "active_lifecycle_phase": "fiberlet_preprocess",
        "run_uuid": run_uuid,
        "run_name": run_name,
        "backend": "vc_fiberlets",
        "artifact_kind": "fiberlets",
        "status": "created",
        "created_at": utc_now(),
        "started_at": None,
        "ended_at": None,
        "exit_code": None,
        "pid": None,
        "process_start_time": None,
        "tmux_session": session,
        "tmux_window_id": None,
        "private": {"hostname": socket.gethostname(), "user": getpass.getuser()},
        "manager": {"version": "0.1", **_git_revision(Path(__file__).resolve().parents[2])},
        "runtime": _runtime_info(python),
        "source": {
            "sample_id": sample_id,
            "volume_id": volume_id,
            "license": source_license,
        },
        "dependencies": {
            "fiber_prediction": {**fiber_identity, "run_name": fiber["run_name"]},
            "normal_prediction": {**normal_identity, "run_name": normal["run_name"]},
        },
        "processor": {
            "path": str(binary),
            "sha256": _sha256(binary),
            "arguments": list(effective_args),
        },
        "source_context_path": "source_context.json",
        "command_path": "command.json",
        "log_path": "run.log",
        "artifacts": {
            "root": "artifacts",
            "provenance": "artifacts/fiberlets.json",
            "dataset": "artifacts/fiberlets.zarr",
            "inventory": [],
        },
        "lifecycle": {
            "fiberlet_preprocess": "created",
            "staging_upload": "not_started",
            "atlas_ingest": "not_started",
            "atlas_publication": "not_started",
        },
    }
    command_record = {
        "schema_version": 1,
        "original_argv": list(original_argv),
        "resolved_argv": command,
        "display": " ".join(shlex.quote(value) for value in command),
        "cwd": str(run_dir),
        "venv_activation": f"source {config.resolved_path('venv', required=True)}/bin/activate",
        "prefetch": None,
    }
    return launch_recorded_job(
        config,
        run_dir=run_dir,
        record=record,
        command_record=command_record,
        window_name=f"flt-{sample_id}-{run_uuid[:4]}"[:24],
        tmux=client,
    )


def finalize_fiberlet_provenance(run_dir: Path, record: dict[str, Any]) -> None:
    dataset = run_dir / str(record["artifacts"]["dataset"])
    for relative in (".zgroup", "anchors/.zarray", "prefix/.zarray", "routes/.zarray"):
        if not (dataset / relative).is_file():
            raise ValueError(f"native Fiberlet output is missing {relative}")
    attrs = _load_json(dataset / ".zattrs", "Fiberlet dataset metadata")
    if (
        attrs.get("vc_format") != "fiberlet_dataset"
        or attrs.get("format_version") != 2
        or attrs.get("dataset_kind") != "combined"
    ):
        raise ValueError("native output is not a combined Fiberlet dataset")
    if not isinstance(attrs.get("processing"), dict):
        raise ValueError("native Fiberlet output has no processing contract")
    dataset_fingerprint = str(attrs.get("dataset_fingerprint") or "")
    if len(dataset_fingerprint) != 64 or any(
        value not in "0123456789abcdef" for value in dataset_fingerprint.lower()
    ):
        raise ValueError("native Fiberlet output has an invalid dataset fingerprint")
    sources = attrs.get("sources") if isinstance(attrs.get("sources"), dict) else {}
    for name in ("fiber_prediction", "normal_prediction"):
        stored = sources.get(name) if isinstance(sources.get(name), dict) else {}
        expected = record["dependencies"][name]
        for field in ("run_uuid", "manifest_sha256"):
            if stored.get(field) != expected.get(field):
                raise ValueError(
                    f"native Fiberlet output {name}.{field} differs from the recorded input"
                )
    document = {
        "schema_version": 1,
        "artifact_kind": "fiberlets",
        "status": "completed",
        "run_uuid": record["run_uuid"],
        "generated_at": utc_now(),
        "source": record["source"],
        "dependencies": record["dependencies"],
        "processor": {
            "sha256": record["processor"]["sha256"],
            "arguments": record["processor"]["arguments"],
        },
        "fiberlet_dataset": {
            "dataset_fingerprint": attrs.get("dataset_fingerprint"),
            "algorithm_fingerprint": attrs.get("algorithm_fingerprint"),
            "format_version": attrs.get("format_version"),
        },
        "artifacts": [{
            "kind": "fiberlet-zarr",
            "path": "fiberlets.zarr",
            "metadata_sha256": _sha256(dataset / ".zattrs"),
        }],
    }
    atomic_json(run_dir / str(record["artifacts"]["provenance"]), document)


def resume_fiberlet(
    config: ManagerConfig, selector: str, *, tmux: Tmux | None = None,
) -> Path:
    run_dir, record = resolve_run(config, selector)
    if record.get("job_kind") != "fiberlet":
        raise ValueError(f"run {selector!r} is not a Fiberlet job")
    if _process_matches(record):
        raise ValueError(f"Fiberlet run {record.get('run_name')!r} is still active")
    dependencies = record.get("dependencies", {})
    fiber = dependencies.get("fiber_prediction", {})
    normal = dependencies.get("normal_prediction", {})
    resolved_fiber, resolved_normal = resolve_fiberlet_inputs(
        config, str(fiber.get("run_name") or fiber.get("run_uuid")),
        str(normal.get("run_name") or normal.get("run_uuid")),
    )
    if _dependency_identity(resolved_fiber) != {key: fiber.get(key) for key in _dependency_identity(resolved_fiber)}:
        raise ValueError("Fiber prediction dependency changed since the original run")
    if _dependency_identity(resolved_normal) != {key: normal.get(key) for key in _dependency_identity(resolved_normal)}:
        raise ValueError("normal prediction dependency changed since the original run")
    client = tmux or Tmux()
    session = str(record.get("tmux_session") or "")
    if session and client.has_session(session):
        raise ValueError(f"Fiberlet run {record.get('run_name')!r} still has a live tmux session")
    record.update(status="created", started_at=None, ended_at=None, exit_code=None, pid=None, process_start_time=None, tmux_window_id=None)
    record.setdefault("lifecycle", {})["fiberlet_preprocess"] = "created"
    command_record = _load_json(run_dir / "command.json", "recorded command")
    return launch_recorded_job(
        config,
        run_dir=run_dir,
        record=record,
        command_record=command_record,
        window_name=str(record.get("tmux_window_name") or f"flt-{record['run_uuid'][:4]}"),
        tmux=client,
    )


def fiberlet_runs(config: ManagerConfig) -> list[tuple[Path, dict[str, Any]]]:
    return [item for item in read_runs(config) if item[1].get("job_kind") == "fiberlet"]
