"""Portable, backend-neutral inference provenance helpers."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_write(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def load_context(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("provenance context must contain a JSON object")
    return value


def _relative(bundle: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(bundle.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"artifact path is outside portable bundle: {path}") from error


def structural_inventory(manifest_path: str | Path) -> list[dict[str, Any]]:
    """Describe manifest and Zarr metadata without walking data chunks."""
    manifest = Path(manifest_path)
    bundle = manifest.parent
    value = json.loads(manifest.read_text(encoding="utf-8"))
    inventory: list[dict[str, Any]] = [{
        "kind": "manifest",
        "path": manifest.name,
        "sha256": sha256_file(manifest),
    }]
    for name, group in sorted(value.get("groups", {}).items()):
        level_path = bundle / str(group["zarr"])
        root = level_path.parent
        levels = []
        for candidate in sorted(root.iterdir(), key=lambda item: int(item.name) if item.name.isdigit() else 1 << 30):
            metadata_path = candidate / ".zarray"
            if not candidate.name.isdigit() or not metadata_path.is_file():
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            levels.append({
                "level": int(candidate.name),
                "shape_zyx": [int(v) for v in metadata["shape"]],
                "chunks_zyx": [int(v) for v in metadata["chunks"]],
                "dtype": str(metadata["dtype"]),
                "compressor": metadata.get("compressor"),
                "metadata_sha256": sha256_file(metadata_path),
            })
        inventory.append({
            "kind": "ome-zarr-channel",
            "name": str(name),
            "path": _relative(bundle, root),
            "channels": list(group.get("channels", [name])),
            "levels": levels,
        })
    return inventory


def validate_portable_bundle(bundle_path: str | Path) -> dict[str, Any]:
    """Validate bounded provenance references without enumerating Zarr chunks."""
    bundle = Path(bundle_path).resolve()
    value = json.loads((bundle / "inference.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("inference.json must contain a JSON object")
    if value.get("status") != "completed":
        raise ValueError(f"inference status must be 'completed', got {value.get('status')!r}")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("inference.json must contain a non-empty artifact inventory")
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
            raise ValueError("each artifact must contain a relative path")
        relative = Path(artifact["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"artifact path is not portable: {relative}")
        target = (bundle / relative).resolve()
        try:
            target.relative_to(bundle)
        except ValueError as error:
            raise ValueError(f"artifact path escapes bundle: {relative}") from error
        if not target.exists():
            raise ValueError(f"artifact does not exist: {relative}")
    return value


def base_document(*, artifact_kind: str, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    supplied = dict(context or {})
    allowed = {key: supplied[key] for key in (
        "run_uuid", "source", "catalog", "model", "manager",
    ) if key in supplied}
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": str(artifact_kind),
        "status": "running",
        "generated_at": utc_now(),
        **allowed,
    }


def finalize_document(
    document: Mapping[str, Any],
    *,
    path: str | Path,
    status: str,
    manifest_path: str | Path | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    updated = dict(document)
    updated["status"] = str(status)
    updated["updated_at"] = utc_now()
    if manifest_path is not None and Path(manifest_path).is_file():
        updated["artifacts"] = structural_inventory(manifest_path)
    if error:
        updated["error"] = str(error)
    atomic_write(path, updated)
    return updated
