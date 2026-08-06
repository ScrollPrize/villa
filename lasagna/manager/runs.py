from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import uuid
from typing import Any, Sequence

from .catalog import VolumeRecord
from .config import ManagerConfig
from .prefetch import volume_cache_root
from .snapshots import SnapshotRecord
from .tmux import Tmux


SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, default=_json_default)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _slug(value: str, limit: int) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.") or "run"
    return value[:limit].rstrip("-.")


def _git_revision(root: Path) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        ).stdout)
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def _checkpoint_config(snapshot: SnapshotRecord) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("inference launch requires torch to inspect checkpoint config") from error
    payload = torch.load(snapshot.path, map_location="cpu", mmap=True, weights_only=True)
    config = payload.get("config") if isinstance(payload, dict) else None
    if not isinstance(config, dict):
        raise ValueError(
            f"snapshot {snapshot.selector!r} has no embedded config; pass --legacy-config"
        )
    return config


def _runtime_python(config: ManagerConfig) -> Path:
    venv = config.resolved_path("venv", required=True)
    assert venv is not None
    python = venv / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(f"configured venv Python does not exist: {python}")
    return python


def _runtime_info(python: Path) -> dict[str, Any]:
    script = """import json
import platform
out = {"python": platform.python_version()}
try:
    import torch
    out.update(torch=getattr(torch, "__version__", None), cuda=getattr(torch.version, "cuda", None))
except Exception as error:
    out["torch_error"] = str(error)
print(json.dumps(out))
"""
    try:
        result = subprocess.run(
            [str(python), "-c", script], check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        )
        value = json.loads(result.stdout)
        return value if isinstance(value, dict) else {"error": "invalid runtime response"}
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return {"error": str(error)}


def build_fiber_command(
    config: ManagerConfig,
    snapshot: SnapshotRecord,
    volume: VolumeRecord,
    scale: int,
    run_dir: Path,
    *,
    extra_args: Sequence[str] = (),
    provenance_context: dict[str, Any] | None = None,
    legacy_config: str | Path | None = None,
) -> tuple[list[str], Path]:
    if scale < 0:
        raise ValueError("scale must be a non-negative OME-Zarr group index")
    if legacy_config is None:
        _checkpoint_config(snapshot)
    elif not Path(legacy_config).expanduser().is_file():
        raise FileNotFoundError(f"legacy Fiber config does not exist: {legacy_config}")
    context_file = run_dir / "provenance_context.json"
    atomic_json(context_file, provenance_context or {})
    manifest = run_dir / "artifacts" / f"{run_dir.name}.lasagna.json"
    input_path = volume_cache_root(config, volume) / str(scale)
    command = [
        str(_runtime_python(config)), "-m", "vesuvius.neural_tracing.fiber_trace_3d.infer",
        *([str(Path(legacy_config).expanduser().resolve())] if legacy_config is not None else []),
        "--input", str(input_path), "--output", str(manifest),
        "--checkpoint", snapshot.path, "--provenance-context", str(context_file),
        "--no-download", *extra_args,
    ]
    return command, manifest


def build_lasagna_command(
    config: ManagerConfig,
    snapshot: SnapshotRecord,
    volume: VolumeRecord,
    scale: int,
    run_dir: Path,
    *,
    extra_args: Sequence[str] = (),
    provenance_context: dict[str, Any] | None = None,
) -> tuple[list[str], Path]:
    if scale < 0:
        raise ValueError("scale must be a non-negative OME-Zarr group index")
    context_file = run_dir / "provenance_context.json"
    atomic_json(context_file, provenance_context or {})
    manifest = run_dir / "artifacts" / f"{run_dir.name}.lasagna.json"
    input_path = volume_cache_root(config, volume) / str(scale)
    command = [
        str(_runtime_python(config)), "-m", "preprocess_cos_omezarr", "predict3d",
        "--input", str(input_path), "--output", str(manifest),
        "--unet-checkpoint", snapshot.path,
        "--provenance-context", str(context_file), "--no-download",
        *extra_args,
    ]
    return command, manifest


def launch_inference(
    config: ManagerConfig,
    snapshot: SnapshotRecord,
    volume: VolumeRecord,
    scale: int,
    *,
    original_argv: Sequence[str],
    extra_args: Sequence[str] = (),
    legacy_config: str | Path | None = None,
    tmux: Tmux | None = None,
    now: datetime | None = None,
) -> Path:
    if snapshot.backend not in {"fiber3d", "lasagna"}:
        raise ValueError(f"unsupported inference backend: {snapshot.backend}")
    if snapshot.backend == "lasagna" and legacy_config is not None:
        raise ValueError("--legacy-config applies only to Fiber checkpoints")
    output_root = config.resolved_path("output_dir", required=True)
    assert output_root is not None
    python = _runtime_python(config)
    if snapshot.backend == "lasagna":
        runtime_config = {
            "patch_shape": snapshot.patch_shape,
            "architecture": snapshot.architecture,
            "precision_policy": snapshot.precision_policy,
        }
        runtime_config_source = "checkpoint-metadata"
    elif legacy_config is None:
        runtime_config = _checkpoint_config(snapshot)
        runtime_config_source = "checkpoint"
    else:
        legacy_path = Path(legacy_config).expanduser().resolve()
        runtime_config = json.loads(legacy_path.read_text(encoding="utf-8"))
        if not isinstance(runtime_config, dict):
            raise ValueError(f"legacy Fiber config must contain a JSON object: {legacy_path}")
        runtime_config_source = "legacy-file"
    instant = now or datetime.now(timezone.utc)
    run_uuid = str(uuid.uuid4())
    readable = f"{snapshot.backend}-{volume.sample_id}-{volume.volume_id}-sd{scale}-{snapshot.run}-{snapshot.checkpoint}-{instant:%Y%m%d_%H%M%S}-{run_uuid[:8]}"
    run_name = _slug(readable, 180)
    session = "las-" + _slug(run_name, 70)
    client = tmux or Tmux()
    if client.has_session(session):
        raise ValueError(f"tmux session already exists: {session}")
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "artifacts").mkdir()
    model_context = asdict(snapshot)
    model_context.pop("path", None)
    provenance_context = {
        "run_uuid": run_uuid,
        "source": {
            "sample_id": volume.sample_id,
            "volume_id": volume.volume_id,
            "long_id": volume.long_id,
            "license": volume.license,
            "selected_origin": volume.selected_origin,
            "origins": list(volume.origins),
            "data_entry": volume.raw,
            "requested_group": int(scale),
        },
        "catalog": {
            "sha256": volume.catalog_sha256,
            "fetched_at": volume.catalog_fetched_at,
            "metadata": volume.catalog_metadata,
        },
        "model": model_context,
        "manager": {"version": "0.1"},
    }
    if snapshot.backend == "fiber3d":
        command, manifest = build_fiber_command(
            config, snapshot, volume, scale, run_dir,
            extra_args=extra_args, provenance_context=provenance_context,
            legacy_config=legacy_config,
        )
        artifact_kind = "fiber3d-prediction"
    else:
        command, manifest = build_lasagna_command(
            config, snapshot, volume, scale, run_dir,
            extra_args=extra_args, provenance_context=provenance_context,
        )
        artifact_kind = "lasagna"
    record = {
        "schema_version": SCHEMA_VERSION,
        "run_uuid": run_uuid,
        "run_name": run_name,
        "backend": snapshot.backend,
        "artifact_kind": artifact_kind,
        "status": "created",
        "created_at": utc_now(),
        "started_at": None,
        "ended_at": None,
        "exit_code": None,
        "pid": None,
        "process_start_time": None,
        "tmux_session": session,
        "private": {"hostname": socket.gethostname(), "user": getpass.getuser()},
        "manager": {"version": "0.1", **_git_revision(Path(__file__).resolve().parents[2])},
        "runtime": _runtime_info(python),
        "environment": {name: os.environ[name] for name in ("CUDA_VISIBLE_DEVICES",) if name in os.environ},
        "source": {"volume": asdict(volume), "scale": scale, "local_path": str(volume_cache_root(config, volume) / str(scale))},
        "snapshot": asdict(snapshot),
        "runtime_config": {"source": runtime_config_source, "sha256": hashlib.sha256(json.dumps(runtime_config, sort_keys=True, default=str).encode()).hexdigest()},
        "provenance_context_path": "provenance_context.json",
        "command_path": "command.json",
        "log_path": "run.log",
        "artifacts": {
            "root": "artifacts", "manifest": str(manifest.relative_to(run_dir)),
            "provenance": "artifacts/inference.json", "inventory": [],
        },
        "lifecycle": {
            "inference": "created", "staging_upload": "not_started",
            "atlas_ingest": "not_started", "atlas_publication": "not_started",
        },
    }
    command_record = {
        "schema_version": 1,
        "original_argv": list(original_argv),
        "resolved_argv": command,
        "display": " ".join(__import__("shlex").quote(value) for value in command),
        "cwd": str(run_dir),
        "venv_activation": f"source {config.resolved_path('venv', required=True)}/bin/activate",
    }
    atomic_json(run_dir / "metadata.json", record)
    atomic_json(run_dir / "command.json", command_record)
    wrapper = [str(python), "-m", "lasagna.manager.runner", str(run_dir)]
    try:
        client.create(session, "inference", wrapper)
    except Exception:
        record["status"] = "failed"
        record["lifecycle"]["inference"] = "failed"
        record["ended_at"] = utc_now()
        atomic_json(run_dir / "metadata.json", record)
        raise
    return run_dir


def read_runs(config: ManagerConfig) -> list[tuple[Path, dict[str, Any]]]:
    root = config.resolved_path("output_dir", required=True)
    assert root is not None
    if not root.is_dir():
        return []
    records = []
    for path in root.glob("*/metadata.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                records.append((path.parent, value))
        except (OSError, ValueError):
            continue
    return sorted(records, key=lambda item: (item[1].get("created_at", ""), item[0].name), reverse=True)


def _process_matches(record: dict[str, Any]) -> bool:
    pid = record.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    expected = record.get("process_start_time")
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    if expected is None:
        return True
    try:
        actual = Path(f"/proc/{pid}/stat").read_text().split()[21]
    except (OSError, IndexError):
        return True
    return actual == expected


def reconcile_runs(config: ManagerConfig, tmux: Tmux | None = None) -> list[tuple[Path, dict[str, Any]]]:
    client = tmux or Tmux()
    records = read_runs(config)
    for run_dir, record in records:
        if record.get("status") not in {"created", "running"}:
            continue
        session = str(record.get("tmux_session") or "")
        if (session and client.has_session(session)) or _process_matches(record):
            continue
        status = "interrupted" if record.get("status") == "running" else "unknown"
        record["status"] = status
        record["ended_at"] = utc_now()
        record.setdefault("lifecycle", {})["inference"] = status
        atomic_json(run_dir / "metadata.json", record)
    return read_runs(config)


def resolve_run(config: ManagerConfig, selector: str) -> tuple[Path, dict[str, Any]]:
    records = read_runs(config)
    exact = [item for item in records if selector in (item[0].name, item[1].get("run_uuid"), item[1].get("tmux_session"))]
    if len(exact) == 1:
        return exact[0]
    matches = [item for item in records if any(str(value).startswith(selector) for value in (item[0].name, item[1].get("run_uuid", ""), item[1].get("tmux_session", "")))]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"no run matches {selector!r}")
    raise ValueError(f"ambiguous run selector {selector!r}; matches: " + ", ".join(item[0].name for item in matches))
