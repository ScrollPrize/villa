"""Session-facing configuration and dataset resolution for interactive Spiral fits.

This module intentionally has no Torch, Zarr, or VC imports.  VC3D can therefore
resolve and validate a dataset before importing the comparatively expensive fitting
stack in the service worker.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import glob
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping
import zipfile

from config import Config


# Version 16 introduces the versioned scroll specification
# (spiral-scroll.json): outward sense is a scroll property and no longer part
# of the load request, and z_begin/z_end are Config keys stored in
# checkpoint configurations.
API_VERSION = 16


def run_mutable_config(config: Mapping[str, Any]) -> dict[str, Any]:
    fields = Config.catalog()["schema"]["fields"]
    return {key: value for key, value in config.items()
            if fields[key]["runtime_impact"] in {"run_boundary", "shell_reload"}}


class PclRole(str, Enum):
    ABSOLUTE = "absolute"
    PATCH_OVERLAP = "patch_overlap"
    RELATIVE = "relative"
    SAME_WINDING = "same_winding"
    DRAWN_CONTROL_POINTS = "drawn_control_points"


@dataclass(frozen=True)
class PclInputSpec:
    path: str
    # None marks a legacy role-less input: fit_spiral then infers
    # winding_is_absolute from the file's basename, as the historical CLI did.
    role: PclRole | None
    required: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PclInputSpec":
        return cls(
            path=_normalise_path(value.get("path")),
            role=PclRole(value["role"]),
            required=bool(value.get("required", False)),
        )


@dataclass(frozen=True)
class SpiralInputPaths:
    dataset_root: str = ""
    umbilicus: str = ""
    pcls: tuple[PclInputSpec, ...] = ()
    fibers: str = ""
    tracks_dbm: str = ""
    verified_patches: str = ""
    unverified_patches: str = ""
    outer_shell: str = ""
    normal_x: str = ""
    normal_y: str = ""
    gradient_magnitude: str = ""
    surf_sdt: str = ""
    scroll_zarr: str = ""
    checkpoint: str = ""
    output_directory: str = ""
    cache_directory: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SpiralInputPaths":
        names = {item.name for item in cls.__dataclass_fields__.values()}
        kwargs = {
            name: _normalise_path(value.get(name))
            for name in names
            if name != "pcls"
        }
        kwargs["pcls"] = tuple(
            item if isinstance(item, PclInputSpec) else PclInputSpec.from_mapping(item)
            for item in value.get("pcls", ())
        )
        return cls(**kwargs)

    def manifest(self) -> dict[str, Any]:
        result = asdict(self)
        result["pcls"] = [
            {"path": item.path,
             "role": item.role.value if item.role is not None else None,
             "required": item.required}
            for item in self.pcls
        ]
        return result


@dataclass(frozen=True)
class SpiralRunConfig:
    z_begin: int
    z_end: int
    scroll_name: str = "scroll"
    voxel_size_um: float = 9.6
    lasagna_group: str = "4"
    lasagna_scale: int = 4
    storage_backend: str = "sparse_cuda"
    legacy_checkpoint_step: int = 0
    run_tag: str = ""
    render_volume_scale: int = 16
    config: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SpiralRunConfig":
        return cls(
            z_begin=int(value.get("z_begin", 0)),
            z_end=int(value.get("z_end", 0)),
            scroll_name=str(value.get("scroll_name", "scroll")),
            voxel_size_um=float(value.get("voxel_size_um", 9.6)),
            lasagna_group=str(value.get("lasagna_group", "4")),
            lasagna_scale=int(value.get("lasagna_scale", 4)),
            storage_backend=str(value.get("storage_backend", "sparse_cuda")).lower(),
            legacy_checkpoint_step=int(value.get("legacy_checkpoint_step", 0)),
            run_tag=str(value.get("run_tag", "")),
            render_volume_scale=int(value.get("render_volume_scale", 16)),
            config=dict(value.get("config", {})),
        )

    def manifest(self) -> dict[str, Any]:
        result = asdict(self)
        result["config"] = dict(self.config)
        return result


@dataclass(frozen=True)
class SpiralPreviewConfig:
    first_winding: int = 10
    variant: str = "raw"

    def manifest(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SpiralDatasetResolution:
    root: str
    resolved: dict[str, str] = field(default_factory=dict)
    pcl_inputs: list[dict[str, Any]] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)
    missing_optional: list[str] = field(default_factory=list)
    ambiguities: dict[str, list[str]] = field(default_factory=dict)
    detected_checkpoints: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Parsed spiral-scroll.json manifest (see ScrollSpec.manifest()); None when
    # the specification is missing or invalid, which is a missing_required
    # condition ("scroll_spec").
    scroll_spec: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return not self.missing_required and not self.ambiguities

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["ok"] = self.ok
        return result


# ---------------------------------------------------------------------------
# Versioned scroll specification (spiral-scroll.json)
# ---------------------------------------------------------------------------

SCROLL_SPEC_FILENAME = "spiral-scroll.json"
SCROLL_SPEC_SCHEMA_VERSION = 1

# Input keys whose paths may depart from the directory conventions. Values in
# the spec file are resolved relative to the dataset root; conventional paths
# need no entry at all.
SCROLL_SPEC_PATH_OVERRIDE_KEYS = (
    "umbilicus", "fibers", "verified_patches", "unverified_patches",
    "outer_shell", "tracks_dbm", "normal_x", "normal_y",
    "gradient_magnitude", "surf_sdt",
)

_SCROLL_SPEC_TOP_LEVEL_KEYS = (
    "schema_version", "name", "voxel_size_um", "spiral_outward_sense",
    "umbilicus", "normal_zarr_group", "surf_sdt_zarr_group", "lasagna_scale",
    "paths",
)

# Conventional dataset layout for the headless CLI, mirroring the historical
# fit_spiral module-global defaults. resolve_dataset_root() shares the same
# relative paths for the entries it discovers.
_CONVENTIONAL_INPUT_RELATIVES = {
    "umbilicus": "umbilicus.json",
    "fibers": "fibers",
    "verified_patches": "verified_patches",
    "outer_shell": "outer_shell",
    "tracks_dbm": "tracks/2um_ds2_ps256_surf_v2.dbm",
    "normal_x": "lasagna_inputs/las_008_nx.ome.zarr",
    "normal_y": "lasagna_inputs/las_008_ny.ome.zarr",
    "gradient_magnitude": "lasagna_inputs/las_008_grad_mag.ome.zarr",
    "surf_sdt": "lasagna_inputs/las_008_surf_sdt.ome.zarr",
}

_CONVENTIONAL_PCL_INPUTS = (
    ("abs_winding.json", PclRole.ABSOLUTE),
    ("patch-overlap-pcls.json", PclRole.PATCH_OVERLAP),
    ("relative_windings.json", PclRole.RELATIVE),
    ("same_windings.json", PclRole.SAME_WINDING),
    ("drawn_control_points.json", PclRole.DRAWN_CONTROL_POINTS),
)


class ScrollSpecError(ValueError):
    """A missing, malformed, or out-of-contract spiral-scroll.json."""


@dataclass(frozen=True)
class ScrollSpec:
    """Physical/dataset facts of one scroll, parsed from spiral-scroll.json.

    Torch-free and frozen: safe to resolve in the VC3D-facing service process
    and to pickle into GPU worker processes. Deployment and presentation
    values (output/cache roots, run tags, storage backend, render scale) are
    deliberately not part of the scroll file.
    """

    name: str
    voxel_size_um: float
    spiral_outward_sense: str
    umbilicus_coordinate_scale: float = 1.0
    normal_zarr_group: str = "4"
    surf_sdt_zarr_group: str = "1"
    lasagna_scale: int = 4
    # Allow-listed absolute-path overrides, (key, resolved path) pairs.
    path_overrides: tuple[tuple[str, str], ...] = ()

    def path_override(self, key: str) -> str:
        return dict(self.path_overrides).get(key, "")

    def manifest(self) -> dict[str, Any]:
        result = asdict(self)
        result["path_overrides"] = dict(self.path_overrides)
        return result


def parse_scroll_spec(document: Any, dataset_root: str | os.PathLike[str],
                      *, source: str = SCROLL_SPEC_FILENAME) -> ScrollSpec:
    """Validate a spiral-scroll.json document strictly and freeze it.

    Unknown keys are errors (named), schema_version is required, and path
    overrides are resolved relative to the dataset root.
    """
    if not isinstance(document, Mapping):
        raise ScrollSpecError(f"{source}: the scroll specification must be a JSON object")
    unknown = sorted(set(document) - set(_SCROLL_SPEC_TOP_LEVEL_KEYS))
    if unknown:
        raise ScrollSpecError(f"{source}: unknown keys: {unknown}")
    if "schema_version" not in document:
        raise ScrollSpecError(f"{source}: schema_version is required")
    if document["schema_version"] != SCROLL_SPEC_SCHEMA_VERSION:
        raise ScrollSpecError(
            f"{source}: unsupported schema_version {document['schema_version']!r} "
            f"(this build supports {SCROLL_SPEC_SCHEMA_VERSION})")
    missing = sorted(
        key for key in ("name", "voxel_size_um", "spiral_outward_sense")
        if key not in document)
    if missing:
        raise ScrollSpecError(f"{source}: missing required keys: {missing}")

    name = str(document["name"]).strip()
    if not name:
        raise ScrollSpecError(f"{source}: name must be a non-empty string")
    try:
        voxel_size_um = float(document["voxel_size_um"])
    except (TypeError, ValueError):
        raise ScrollSpecError(f"{source}: voxel_size_um must be a number") from None
    if not voxel_size_um > 0:
        raise ScrollSpecError(f"{source}: voxel_size_um must be positive")
    sense = str(document["spiral_outward_sense"]).upper()
    if sense not in ("CW", "ACW"):
        raise ScrollSpecError(f"{source}: spiral_outward_sense must be CW or ACW")

    umbilicus = document.get("umbilicus", {})
    if not isinstance(umbilicus, Mapping):
        raise ScrollSpecError(f"{source}: umbilicus must be an object")
    unknown = sorted(set(umbilicus) - {"coordinate_scale"})
    if unknown:
        raise ScrollSpecError(f"{source}: unknown umbilicus keys: {unknown}")
    try:
        coordinate_scale = float(umbilicus.get("coordinate_scale", 1.0))
    except (TypeError, ValueError):
        raise ScrollSpecError(f"{source}: umbilicus coordinate_scale must be a number") from None

    lasagna_scale = document.get("lasagna_scale", 4)
    if type(lasagna_scale) is not int or lasagna_scale <= 0:
        raise ScrollSpecError(f"{source}: lasagna_scale must be a positive integer")

    paths = document.get("paths", {})
    if not isinstance(paths, Mapping):
        raise ScrollSpecError(f"{source}: paths must be an object")
    unknown = sorted(set(paths) - set(SCROLL_SPEC_PATH_OVERRIDE_KEYS))
    if unknown:
        raise ScrollSpecError(
            f"{source}: unknown path override keys: {unknown} "
            f"(allowed: {sorted(SCROLL_SPEC_PATH_OVERRIDE_KEYS)})")
    root = Path(_normalise_path(dataset_root))
    overrides = []
    for key in sorted(paths):
        value = paths[key]
        if not isinstance(value, str) or not value.strip():
            raise ScrollSpecError(f"{source}: path override {key!r} must be a non-empty string")
        overrides.append((key, _normalise_path(value, base=root)))

    return ScrollSpec(
        name=name,
        voxel_size_um=voxel_size_um,
        spiral_outward_sense=sense,
        umbilicus_coordinate_scale=coordinate_scale,
        normal_zarr_group=str(document.get("normal_zarr_group", "4")),
        surf_sdt_zarr_group=str(document.get("surf_sdt_zarr_group", "1")),
        lasagna_scale=lasagna_scale,
        path_overrides=tuple(overrides),
    )


def load_scroll_spec(dataset_root: str | os.PathLike[str],
                     spec_path: str | os.PathLike[str] | None = None) -> ScrollSpec:
    """Load the scroll specification for a dataset.

    Discovers the single conventional file <dataset_root>/spiral-scroll.json
    unless an explicit spec_path is given. A missing or invalid file raises
    ScrollSpecError with instructions.
    """
    root = Path(_normalise_path(dataset_root)) if str(dataset_root or "").strip() else None
    if spec_path is not None:
        path = Path(_normalise_path(spec_path))
    elif root is not None:
        path = root / SCROLL_SPEC_FILENAME
    else:
        raise ScrollSpecError(
            "No dataset root given: cannot discover the scroll specification "
            f"({SCROLL_SPEC_FILENAME})")
    if not path.is_file():
        raise ScrollSpecError(
            f"No scroll specification found at {path}. Create {SCROLL_SPEC_FILENAME} "
            "in the dataset root with schema_version, name, voxel_size_um, and "
            "spiral_outward_sense (plus any non-conventional path overrides).")
    try:
        with path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, ValueError) as exc:
        raise ScrollSpecError(f"{path}: invalid scroll specification: {exc}") from exc
    return parse_scroll_spec(document, root if root is not None else path.parent,
                             source=str(path))


def conventional_input_paths(
        dataset_root: str | os.PathLike[str], spec: ScrollSpec, *,
        output_directory: str = "", cache_directory: str = "",
        checkpoint: str = "") -> SpiralInputPaths:
    """Resolve the conventional dataset layout (plus spec overrides) for the
    headless CLI, mirroring the historical fit_spiral module-global defaults.

    Unlike resolve_dataset_root() this performs no existence probing: the CLI
    fails on the specific missing input during loading, exactly as the module
    globals did. The dataset root is kept verbatim (no symlink resolution) so
    conventional paths read exactly as the caller spelled the root; explicit
    spec overrides are already normalised against the root at parse time.
    """
    root = str(dataset_root)

    def resolve(key):
        override = spec.path_override(key)
        if override:
            return override
        relative = _CONVENTIONAL_INPUT_RELATIVES.get(key)
        return f"{root}/{relative}" if relative else ""

    pcls = tuple(
        PclInputSpec(path=f"{root}/{relative}", role=role)
        for relative, role in _CONVENTIONAL_PCL_INPUTS
    )
    return SpiralInputPaths(
        dataset_root=str(root),
        umbilicus=resolve("umbilicus"),
        pcls=pcls,
        fibers=resolve("fibers"),
        tracks_dbm=resolve("tracks_dbm"),
        verified_patches=resolve("verified_patches"),
        unverified_patches=spec.path_override("unverified_patches"),
        outer_shell=resolve("outer_shell"),
        normal_x=resolve("normal_x"),
        normal_y=resolve("normal_y"),
        gradient_magnitude=resolve("gradient_magnitude"),
        surf_sdt=resolve("surf_sdt"),
        checkpoint=_normalise_path(checkpoint) if checkpoint else "",
        output_directory=_normalise_path(output_directory) if output_directory else "",
        cache_directory=_normalise_path(cache_directory) if cache_directory else "",
    )


_CONVENTIONAL_ENTRIES: tuple[tuple[str, str, str, bool], ...] = (
    ("umbilicus", "umbilicus.json", "file", True),
    ("fibers", "fibers", "directory", False),
    ("verified_patches", "verified_patches", "directory", True),
    ("outer_shell", "outer_shell", "directory", False),
    ("normal_x", "lasagna_inputs/las_008_nx.ome.zarr", "directory", False),
    ("normal_y", "lasagna_inputs/las_008_ny.ome.zarr", "directory", False),
    ("gradient_magnitude", "lasagna_inputs/las_008_grad_mag.ome.zarr", "directory", False),
    ("surf_sdt", "lasagna_inputs/las_008_surf_sdt.ome.zarr", "directory", False),
)

_PCL_ENTRIES: tuple[tuple[PclRole, str, bool], ...] = (
    (PclRole.ABSOLUTE, "abs_winding.json", False),
    (PclRole.RELATIVE, "relative_windings.json", False),
    (PclRole.SAME_WINDING, "same_windings.json", False),
    (PclRole.DRAWN_CONTROL_POINTS, "drawn_control_points.json", False),
)


def _normalise_path(value: Any, base: Path | None = None) -> str:
    if value is None or str(value).strip() == "":
        return ""
    path = Path(os.path.expandvars(os.path.expanduser(str(value).strip())))
    if base is not None and not path.is_absolute():
        path = base / path
    # strict=False is important for proposed output/cache paths.
    return str(path.resolve(strict=False))


def _has_dbm_backing(path: Path) -> bool:
    if path.is_file():
        return True
    suffixes = (".db", ".dat", ".dir", ".bak", ".pag")
    return any(Path(str(path) + suffix).is_file() for suffix in suffixes)


def resolve_logical_dbm(path: str | Path) -> str:
    """Return the DBM logical base while accepting implementation suffix files."""
    candidate = Path(path)
    text = str(candidate)
    for suffix in (".db", ".dat", ".dir", ".bak", ".pag"):
        if text.endswith(".dbm" + suffix):
            candidate = Path(text[: -len(suffix)])
            break
    return _normalise_path(candidate) if _has_dbm_backing(candidate) else ""


def validate_checkpoint_container(path: str | Path) -> None:
    """Require a complete modern torch.save archive before GPU teardown."""
    checkpoint = Path(path)
    with checkpoint.open("rb") as stream:
        signature = stream.read(4)
    if not signature.startswith(b"PK"):
        raise ValueError("Legacy pickle checkpoints are not supported; resave as a modern torch.save archive")
    if not zipfile.is_zipfile(checkpoint):
        raise ValueError("checkpoint is an incomplete or corrupt PyTorch ZIP archive")


def _dbm_candidates(root: Path) -> list[str]:
    logical: set[str] = set()
    tracks = root / "tracks"
    if not tracks.is_dir():
        return []
    for entry in sorted(tracks.iterdir(), key=lambda item: item.name):
        text = str(entry)
        if ".dbm" not in entry.name:
            continue
        base = text[: text.index(".dbm") + len(".dbm")]
        if _has_dbm_backing(Path(base)):
            logical.add(_normalise_path(base))
    return sorted(logical)


def resolve_dataset_root(
    root_value: str | os.PathLike[str],
    *,
    session_name: str = "",
) -> SpiralDatasetResolution:
    root = Path(_normalise_path(root_value))
    result = SpiralDatasetResolution(root=str(root))
    if not root.is_dir():
        result.missing_required.append("dataset_root")
        result.warnings.append(f"Dataset root is not a readable directory: {root}")
        return result

    # The scroll specification is the dataset's one required source of
    # physical facts; a dataset without it does not resolve.
    try:
        spec = load_scroll_spec(root)
    except ScrollSpecError as exc:
        spec = None
        result.missing_required.append("scroll_spec")
        result.warnings.append(str(exc))
    else:
        result.scroll_spec = spec.manifest()

    for key, relative, kind, required in _CONVENTIONAL_ENTRIES:
        override = spec.path_override(key) if spec is not None else ""
        candidate = Path(override) if override else root / relative
        found = candidate.is_file() if kind == "file" else candidate.is_dir()
        if found and os.access(candidate, os.R_OK):
            result.resolved[key] = _normalise_path(candidate)
        elif required:
            result.missing_required.append(key)
        else:
            result.missing_optional.append(key)

    for role, relative, required in _PCL_ENTRIES:
        candidate = root / relative
        if candidate.is_file() and os.access(candidate, os.R_OK):
            result.pcl_inputs.append({
                "path": _normalise_path(candidate),
                "role": role.value,
                "required": required,
            })
        elif required:
            result.missing_required.append(f"pcl:{role.value}")
        else:
            result.missing_optional.append(f"pcl:{role.value}")

    tracks_override = spec.path_override("tracks_dbm") if spec is not None else ""
    preferred = Path(tracks_override) if tracks_override \
        else root / "tracks" / "2um_ds2_ps256_surf_v2.dbm"
    preferred_logical = resolve_logical_dbm(preferred)
    if preferred_logical:
        result.resolved["tracks_dbm"] = preferred_logical
    else:
        candidates = _dbm_candidates(root)
        if len(candidates) == 1:
            result.resolved["tracks_dbm"] = candidates[0]
        elif len(candidates) > 1:
            result.ambiguities["tracks_dbm"] = candidates
        else:
            result.missing_optional.append("tracks_dbm")

    output_directory = root / "spiral_output"
    if session_name:
        output_directory /= session_name
    result.resolved["output_directory"] = _normalise_path(output_directory)
    local_cache = root / ".spiral-cache"
    parent_writable = os.access(root, os.W_OK)
    if local_cache.is_dir() or parent_writable:
        result.resolved["cache_directory"] = _normalise_path(local_cache)
    else:
        fallback = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "vc3d" / "spiral"
        result.resolved["cache_directory"] = _normalise_path(fallback)
        result.warnings.append("Dataset root is not writable; using the user Spiral cache")

    checkpoints = sorted(
        _normalise_path(path)
        for path in root.glob("*.ckpt")
        if path.is_file()
    )
    result.detected_checkpoints = checkpoints
    return result


def _expand_pcl(spec: PclInputSpec) -> list[str]:
    if not spec.path:
        return []
    if glob.has_magic(spec.path):
        return sorted(_normalise_path(path) for path in glob.glob(spec.path))
    return [spec.path]


def _validate_json_file(path: Path, label: str, errors: list[dict[str, str]]) -> None:
    try:
        with path.open("r", encoding="utf-8") as stream:
            json.load(stream)
    except Exception as exc:
        errors.append({"field": label, "message": f"Invalid JSON: {exc}"})


def validate_session_request(
    paths: SpiralInputPaths,
    run: SpiralRunConfig,
) -> list[dict[str, str]]:
    """Perform cheap, aggregate validation before any GPU allocation."""
    errors: list[dict[str, str]] = []

    def require_file(value: str, field_name: str, *, json_file: bool = False) -> None:
        path = Path(value) if value else None
        if path is None or not path.is_file():
            errors.append({"field": field_name, "message": "Required readable file is missing"})
            return
        if not os.access(path, os.R_OK):
            errors.append({"field": field_name, "message": "File is not readable"})
        elif json_file:
            _validate_json_file(path, field_name, errors)

    def optional_dir(value: str, field_name: str, required: bool = False) -> None:
        if not value and not required:
            return
        path = Path(value) if value else None
        if path is None or not path.is_dir():
            errors.append({"field": field_name, "message": "Required directory is missing" if required else "Path is not a directory"})
        elif not os.access(path, os.R_OK):
            errors.append({"field": field_name, "message": "Directory is not readable"})

    require_file(paths.umbilicus, "umbilicus", json_file=True)
    disable_patches = bool(run.config.get("input_disable_patches", False))
    optional_dir(paths.verified_patches, "verified_patches",
                 required=not disable_patches)
    if not disable_patches:
        optional_dir(paths.unverified_patches, "unverified_patches")
    optional_dir(paths.fibers, "fibers")

    shell_enabled = (
        float(run.config.get("loss_weight_shell_outer", 1.0)) > 0
        or float(run.config.get("loss_weight_shell_patch_radius", 0)) > 0
    )
    optional_dir(paths.outer_shell, "outer_shell", required=shell_enabled)

    if paths.tracks_dbm and not resolve_logical_dbm(paths.tracks_dbm):
        errors.append({"field": "tracks_dbm", "message": "DBM logical base or backing file was not found"})

    for index, spec in enumerate(paths.pcls):
        expanded = _expand_pcl(spec)
        if spec.required and not expanded:
            errors.append({"field": f"pcls[{index}]", "message": "Required PCL pattern matched no files"})
        for expanded_path in expanded:
            path = Path(expanded_path)
            if not path.is_file():
                errors.append({"field": f"pcls[{index}]", "message": f"PCL file does not exist: {path}"})
            else:
                _validate_json_file(path, f"pcls[{index}]", errors)

    # The dense-spacing mode is checked before any asset-path requirements
    # so an invalid mode errors as itself, not as a missing-file error.
    spacing_mode = str(run.config.get("dense_spacing_mode", "phase"))
    if spacing_mode not in ("phase", "grad_mag"):
        errors.append({"field": "dense_spacing_mode",
                       "message": "Must be phase or grad_mag"})
        spacing_mode = None

    use_normals = float(
        run.config.get("loss_weight_dense_normals", 100.0)) > 0
    spacing_enabled = float(run.config.get("loss_weight_dense_spacing", 12.0)) > 0
    use_phase = spacing_mode == "phase"
    use_grad_mag = spacing_mode == "grad_mag" and spacing_enabled
    # The phase bundle requires its core inputs (SDT for phase, count, and
    # attachment; both normal channels for band incidence handling) even when
    # individual sub-weights are zero, so run-mutable weights can be raised
    # at run boundaries. grad_mag never requires the SDT; normals are needed
    # only for the independent dense-normal loss.
    for value, label, required in (
        (paths.normal_x, "normal_x", use_normals or use_phase),
        (paths.normal_y, "normal_y", use_normals or use_phase),
        (paths.gradient_magnitude, "gradient_magnitude", use_grad_mag),
        (paths.surf_sdt, "surf_sdt", use_phase),
    ):
        optional_dir(value, label, required=required)

    if run.z_begin >= run.z_end:
        errors.append({"field": "z_range", "message": "z_begin must be less than z_end"})
    if run.lasagna_scale <= 0:
        errors.append({"field": "lasagna_scale", "message": "Must be positive"})
    if run.storage_backend != "sparse_cuda":
        errors.append({
            "field": "storage_backend",
            "message": "Only sparse_cuda is supported",
        })

    if not paths.output_directory:
        errors.append({"field": "output_directory", "message": "Output directory is required"})
    else:
        output = Path(paths.output_directory)
        probe_parent = output
        while not probe_parent.exists() and probe_parent != probe_parent.parent:
            probe_parent = probe_parent.parent
        if output.exists() and not output.is_dir():
            errors.append({"field": "output_directory", "message": "Output path is not a directory"})
        elif not probe_parent.is_dir() or not os.access(probe_parent, os.W_OK):
            errors.append({"field": "output_directory", "message": "Output directory is not writable"})

    if (use_normals or use_phase or use_grad_mag) and not paths.cache_directory:
        errors.append({"field": "cache_directory", "message": "Cache directory is required for Lasagna inputs"})

    if paths.checkpoint and not Path(paths.checkpoint).is_file():
        errors.append({"field": "checkpoint", "message": "Checkpoint file does not exist"})
    elif paths.checkpoint:
        try:
            validate_checkpoint_container(paths.checkpoint)
        except (OSError, ValueError) as exc:
            errors.append({"field": "checkpoint", "message": str(exc)})
    return errors


def parse_session_request(value: Mapping[str, Any]) -> tuple[SpiralInputPaths, SpiralRunConfig, SpiralPreviewConfig]:
    paths = SpiralInputPaths.from_mapping(value.get("paths", {}))
    run = SpiralRunConfig.from_mapping(value.get("run", {}))
    preview_map = value.get("preview", {})
    preview = SpiralPreviewConfig(
        first_winding=int(preview_map.get("first_winding", 10)),
        variant=str(preview_map.get("variant", "raw")),
    )
    return paths, run, preview
