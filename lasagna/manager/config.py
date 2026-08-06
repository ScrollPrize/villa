from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import os
from pathlib import Path
import tempfile
import tomllib
from typing import Any


DEFAULT_CATALOG_URL = (
    "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.json"
)
DEFAULT_OPEN_DATA_BUCKET = "s3://vesuvius-challenge-open-data"
DEFAULT_INFERENCE_PARAMS = (
    "--tile-size", "512", "--border", "32", "--overlap", "96",
    "--devices", "all",
)


@dataclass(frozen=True)
class ManagerConfig:
    catalog_url: str = DEFAULT_CATALOG_URL
    open_data_bucket: str = DEFAULT_OPEN_DATA_BUCKET
    snapshot_dirs: tuple[str, ...] = ()
    cache_dir: str = ""
    output_dir: str = ""
    venv: str = ""
    atlas_dir: str = ""
    upload_staging_s3: str = ""
    catalog_max_age_seconds: int = 3600
    params: tuple[str, ...] = DEFAULT_INFERENCE_PARAMS

    def resolved_path(self, name: str, *, required: bool = False) -> Path | None:
        if name not in {"cache_dir", "output_dir", "venv", "atlas_dir"}:
            raise KeyError(name)
        value = str(getattr(self, name)).strip()
        if not value:
            if required:
                raise ValueError(f"config value {name!r} is required for this command")
            return None
        return _resolve_path(value, config_path().parent)

    def resolved_snapshot_dirs(self) -> tuple[Path, ...]:
        return tuple(_resolve_path(value, config_path().parent) for value in self.snapshot_dirs)


def config_path() -> Path:
    override = os.environ.get("LAS_MANAGER_CONFIG")
    if override:
        return Path(os.path.expandvars(override)).expanduser().resolve()
    root = Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser()
    return (root / "las_manager" / "config.toml").resolve()


def _resolve_path(value: str, base: Path) -> Path:
    expanded = Path(os.path.expandvars(value)).expanduser()
    if not expanded.is_absolute():
        expanded = base / expanded
    return expanded.resolve()


def _validate(raw: dict[str, Any]) -> ManagerConfig:
    known = {field.name for field in fields(ManagerConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"unknown config key(s): {', '.join(unknown)}")
    for array_name in ("snapshot_dirs", "params"):
        if array_name not in raw:
            continue
        values = raw[array_name]
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise ValueError(f"{array_name} must be an array of strings")
        raw[array_name] = tuple(values)
    for name in ("catalog_url", "open_data_bucket", "cache_dir", "output_dir", "venv", "atlas_dir", "upload_staging_s3"):
        if name in raw and not isinstance(raw[name], str):
            raise ValueError(f"{name} must be a string")
    if "catalog_max_age_seconds" in raw:
        age = raw["catalog_max_age_seconds"]
        if not isinstance(age, int) or isinstance(age, bool) or age < 0:
            raise ValueError("catalog_max_age_seconds must be a non-negative integer")
    return ManagerConfig(**raw)


def load_config(path: Path | None = None) -> ManagerConfig:
    path = path or config_path()
    if not path.is_file():
        raise FileNotFoundError(f"las_manager config does not exist: {path}; run 'las_manager config init'")
    with path.open("rb") as handle:
        return _validate(tomllib.load(handle))


def render_config(config: ManagerConfig = ManagerConfig()) -> str:
    values = asdict(config)
    snapshots = ", ".join(_toml_string(v) for v in values.pop("snapshot_dirs"))
    params = ", ".join(_toml_string(v) for v in values.pop("params"))
    lines = [
        "# las_manager global configuration",
        "# Paths may contain ~ or environment variables; relative paths use this file's directory.",
        f"catalog_url = {_toml_string(values['catalog_url'])}",
        f"open_data_bucket = {_toml_string(values['open_data_bucket'])}",
        f"snapshot_dirs = [{snapshots}]",
        f"cache_dir = {_toml_string(values['cache_dir'])}",
        f"output_dir = {_toml_string(values['output_dir'])}",
        f"venv = {_toml_string(values['venv'])}",
        f"atlas_dir = {_toml_string(values['atlas_dir'])}",
        f"upload_staging_s3 = {_toml_string(values['upload_staging_s3'])}",
        f"catalog_max_age_seconds = {values['catalog_max_age_seconds']}",
        "# Default backend arguments; arguments after `inference run ... --` override these.",
        f"params = [{params}]",
        "",
    ]
    return "\n".join(lines)


def _toml_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def atomic_write_text(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def initialize_config(*, path: Path | None = None, force: bool = False) -> Path:
    path = path or config_path()
    if path.exists() and not force:
        raise FileExistsError(f"config already exists: {path}; pass --force to replace it")
    atomic_write_text(path, render_config())
    return path


def display_config(config: ManagerConfig) -> dict[str, Any]:
    result = asdict(config)
    result["config_path"] = str(config_path())
    result["snapshot_dirs"] = [str(path) for path in config.resolved_snapshot_dirs()]
    for name in ("cache_dir", "output_dir", "venv", "atlas_dir"):
        value = config.resolved_path(name)
        result[name] = str(value) if value is not None else ""
    return result
