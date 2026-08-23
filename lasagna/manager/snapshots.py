from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

from .config import ManagerConfig


@dataclass(frozen=True)
class SnapshotRecord:
    backend: str
    run: str
    checkpoint: str
    selector: str
    path: str
    size: int
    mtime_ns: int
    sha256: str
    step: int | None
    metric_name: str | None
    metric_value: float | None
    patch_shape: tuple[int, ...] | None
    architecture: str | None
    option_count: int | None
    precision_policy: str | None
    atlas_model_id: str | None
    model_creation_utc: str | None
    process: str | None
    task: str
    output_schema: dict[str, Any] | None
    code_revision: str | None
    model_identifier: str | None = None


def discover_snapshot_paths(roots: Iterable[Path]) -> list[tuple[str, Path]]:
    found: dict[Path, str] = {}
    for root in roots:
        root = root.expanduser().resolve()
        if not root.is_dir():
            continue
        if root.name == "snapshots":
            candidates = root.glob("*.pt")
        elif (root / "snapshots").is_dir():
            candidates = (root / "snapshots").glob("*.pt")
        else:
            candidates = root.glob("*/snapshots/*.pt")
        for path in candidates:
            resolved = path.resolve()
            found[resolved] = resolved.parent.parent.name
    return sorted(((run, path) for path, run in found.items()), key=lambda item: (item[0], item[1].name, str(item[1])))


def _cache_path(config: ManagerConfig) -> Path:
    root = config.resolved_path("cache_dir", required=True)
    assert root is not None
    return root / "snapshots" / "index.json"


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value.get("entries", {}) if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atlas_checkpoint_identities(config: ManagerConfig) -> dict[str, dict[str, str | None]]:
    """Index registered Atlas models by their exact checkpoint digest."""
    atlas_dir = config.resolved_path("atlas_dir")
    if atlas_dir is None:
        return {}
    models_dir = atlas_dir / "data" / "models"
    if not models_dir.is_dir():
        return {}
    identities: dict[str, dict[str, str | None]] = {}
    for path in sorted(models_dir.rglob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(value, dict):
            continue
        properties = _mapping(value.get("properties"))
        digest = str(properties.get("snapshot_sha256") or "").lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            continue
        model_id = str(value.get("id") or "")
        if not model_id:
            raise ValueError(f"Atlas model with checkpoint SHA-256 has no id: {path}")
        creation = _mapping(value.get("creation"))
        identifier = properties.get("model_identifier")
        if identifier is not None and (not isinstance(identifier, str) or not identifier.strip()):
            raise ValueError(f"Atlas model has invalid model_identifier: {path}")
        identity = {
            "atlas_model_id": model_id,
            "model_creation_utc": str(creation.get("date")) if creation.get("date") else None,
            "model_identifier": identifier.strip() if isinstance(identifier, str) else None,
        }
        previous = identities.get(digest)
        if previous is not None and previous != identity:
            raise ValueError(
                f"multiple Atlas models claim checkpoint SHA-256 {digest}: "
                f"{previous['atlas_model_id']!r} and {model_id!r}"
            )
        identities[digest] = identity
    return identities


def _with_atlas_identity(
    record: SnapshotRecord,
    identities: Mapping[str, Mapping[str, str | None]],
) -> SnapshotRecord:
    identity = identities.get(record.sha256.lower())
    if identity is None:
        return record
    atlas_model_id = identity.get("atlas_model_id")
    if record.atlas_model_id and record.atlas_model_id[:14] != str(atlas_model_id)[:14]:
        raise ValueError(
            f"checkpoint {record.path} embeds Atlas model {record.atlas_model_id!r}, "
            f"but its SHA-256 is registered as {atlas_model_id!r}"
        )
    model_identifier = identity.get("model_identifier")
    if record.model_identifier and record.model_identifier != model_identifier:
        raise ValueError(
            f"checkpoint {record.path} embeds model identifier {record.model_identifier!r}, "
            f"but Atlas registers {model_identifier!r}"
        )
    return replace(
        record,
        atlas_model_id=str(atlas_model_id),
        model_creation_utc=record.model_creation_utc or identity.get("model_creation_utc"),
        model_identifier=record.model_identifier or model_identifier,
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first(config: Mapping[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        value: Any = config
        for key in path:
            value = _mapping(value).get(key)
        if value is not None:
            return value
    return None


def _checkpoint_backend(top: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    """Classify checkpoints without relying on their filename."""
    if isinstance(config.get("model_3d"), Mapping):
        return "fiber3d"
    if any(key in top for key in ("norm_type", "upsample_mode", "output_sigmoid", "model_patch_size")):
        return "lasagna"
    # Current Fiber training always embeds a config. Bare state-dict wrappers
    # are the legacy Lasagna format consumed by train_unet_3d.build_model.
    if "state_dict" in top and not config:
        return "lasagna"
    return "fiber3d"


def _lasagna_sibling_config(path: Path) -> Mapping[str, Any]:
    sibling = path.parent / "config.json"
    try:
        value = json.loads(sibling.read_text(encoding="utf-8"))
        return value if isinstance(value, Mapping) else {}
    except (OSError, ValueError):
        return {}


def _infer_lasagna_patch(top: Mapping[str, Any], path: Path) -> tuple[int, ...] | None:
    patch = top.get("patch_size") or top.get("model_patch_size")
    if patch is None:
        patch = _lasagna_sibling_config(path).get("patch_size")
    if patch is None:
        try:
            from train_unet_3d import infer_model_patch_size
        except ImportError:
            from lasagna.train_unet_3d import infer_model_patch_size
        patch = infer_model_patch_size(dict(_mapping(top.get("state_dict") or top)))
    if isinstance(patch, (list, tuple)):
        return tuple(int(value) for value in patch)
    return (int(patch),) * 3 if patch is not None else None


def _extract(path: Path, run: str, stat: os.stat_result) -> SnapshotRecord:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("snapshot listing requires torch in the configured environment") from error
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    top = _mapping(payload)
    config = _mapping(top.get("config"))
    backend = _checkpoint_backend(top, config)
    model = _mapping(config.get("model_3d") or config.get("model"))
    training = _mapping(config.get("training"))
    atlas = _mapping(config.get("atlas"))
    patch = _first(config, ("patch_shape_zyx",), ("patch_size",), ("model_3d", "patch_shape_zyx"), ("model", "patch_size"))
    patch_shape = tuple(int(v) for v in patch) if isinstance(patch, (list, tuple)) else None
    if backend == "lasagna":
        patch_shape = _infer_lasagna_patch(top, path)
    option_count = model.get("direction_branch_count")
    output_schema = {
        "output_channels": model.get("output_channels"),
        "direction_branch_count": option_count,
        "conditioned_decoder_enabled": model.get("conditioned_decoder_enabled"),
    }
    output_schema = {key: value for key, value in output_schema.items() if value is not None} or None
    architecture = model.get("architecture") or model.get("name")
    if backend == "lasagna":
        architecture = top.get("architecture") or "lasagna_3d"
    if architecture is None and model:
        architecture = "fiber_trace_3d"
    precision = training.get("mixed_precision") or config.get("mixed_precision") or top.get("precision")
    metric = top.get("metric", top.get("val_loss"))
    metric_name = top.get("metric_name")
    if metric_name is None and top.get("val_loss") is not None:
        metric_name = "validation/loss"
    atlas_model_id = atlas.get("model_id") or config.get("atlas_model_id") or top.get("atlas_model_id")
    model_identifier = (
        atlas.get("model_identifier")
        or config.get("model_identifier")
        or top.get("model_identifier")
    )
    record = SnapshotRecord(
        backend=backend,
        run=run,
        checkpoint=path.name,
        selector=f"{backend}/{run}/{path.name}",
        path=str(path),
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        sha256=_sha256(path),
        step=int(top["step"]) if top.get("step") is not None else None,
        metric_name=str(metric_name) if metric_name is not None else None,
        metric_value=float(metric) if metric is not None else None,
        patch_shape=patch_shape,
        architecture=str(architecture) if architecture is not None else None,
        option_count=int(option_count) if option_count is not None else None,
        precision_policy=str(precision) if precision is not None else None,
        atlas_model_id=str(atlas_model_id) if atlas_model_id else None,
        model_creation_utc=str(atlas.get("creation_utc") or config.get("model_creation_utc")) if (atlas.get("creation_utc") or config.get("model_creation_utc")) else None,
        process=str(atlas.get("process") or ("fiber_trace_3d.train" if backend == "fiber3d" else "train_unet_3d")),
        # Atlas stores both the original geometry model and Fiber's compatible
        # Lasagna-volume output under the existing Lasagna model task.
        task="lasagna",
        output_schema=output_schema,
        code_revision=str(top.get("code_revision") or config.get("code_revision")) if (top.get("code_revision") or config.get("code_revision")) else None,
        model_identifier=str(model_identifier) if model_identifier else None,
    )
    return record


def index_snapshots(
    config: ManagerConfig,
    *,
    cached_only: bool = False,
    write_cache: bool = True,
) -> list[SnapshotRecord]:
    cache_path = _cache_path(config)
    cache = _load_cache(cache_path)
    atlas_identities = _atlas_checkpoint_identities(config)
    updated: dict[str, dict[str, Any]] = {}
    records: list[SnapshotRecord] = []
    for run, path in discover_snapshot_paths(config.resolved_snapshot_dirs()):
        stat = path.stat()
        key = str(path)
        cached = cache.get(key)
        if cached and cached.get("size") == stat.st_size and cached.get("mtime_ns") == stat.st_mtime_ns:
            record = SnapshotRecord(**cached)
        elif cached_only:
            continue
        else:
            record = _extract(path, run, stat)
        # Cache only checkpoint-owned metadata. Atlas registration is projected
        # on every scan so registry edits take effect without touching weights.
        updated[key] = asdict(record)
        records.append(_with_atlas_identity(record, atlas_identities))
    if not cached_only and write_cache:
        _atomic_json(cache_path, {"schema_version": 1, "entries": updated})
    return sorted(records, key=lambda record: (record.run, record.checkpoint, record.path))


def completion_snapshot_candidates(
    config: ManagerConfig, *, backend: str | None = None,
) -> list[tuple[str, str]]:
    """Discover every checkpoint cheaply, enriching candidates from the index."""
    cache = _load_cache(_cache_path(config))
    atlas_identities = _atlas_checkpoint_identities(config)
    candidates: dict[str, str] = {}
    for run, path in discover_snapshot_paths(config.resolved_snapshot_dirs()):
        try:
            stat = path.stat()
        except OSError:
            continue
        cached = cache.get(str(path))
        if (
            cached
            and cached.get("size") == stat.st_size
            and cached.get("mtime_ns") == stat.st_mtime_ns
        ):
            try:
                record = SnapshotRecord(**cached)
            except TypeError:
                pass
            else:
                record = _with_atlas_identity(record, atlas_identities)
                if backend is not None and record.backend != backend:
                    continue
                candidates[record.selector] = (
                    f"step {record.step}" if record.step is not None else "snapshot"
                )
                continue
        # Backend classification requires loading the checkpoint. The
        # backend-free alias is accepted by resolve_snapshot and keeps shell
        # completion fast even for newly copied multi-gigabyte checkpoints.
        candidates[f"{run}/{path.name}"] = "unindexed snapshot"
    return sorted(candidates.items())


def resolve_snapshot(records: list[SnapshotRecord], selector: str) -> SnapshotRecord:
    aliases: dict[str, list[SnapshotRecord]] = {}
    for record in records:
        for alias in (record.selector, f"{record.run}/{record.checkpoint}", record.checkpoint):
            aliases.setdefault(alias, []).append(record)
    exact = aliases.get(selector, [])
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ValueError(_ambiguous(selector, exact))
    matches = {record.selector: record for alias, values in aliases.items() if alias.startswith(selector) for record in values}
    if len(matches) == 1:
        return next(iter(matches.values()))
    if not matches:
        raise ValueError(f"no snapshot matches {selector!r}")
    raise ValueError(_ambiguous(selector, matches.values()))


def _ambiguous(selector: str, records: Iterable[SnapshotRecord]) -> str:
    return f"ambiguous snapshot selector {selector!r}; matches: " + ", ".join(sorted({r.selector for r in records}))
