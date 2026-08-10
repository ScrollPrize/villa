"""Session-facing configuration and dataset resolution for interactive Spiral fits.

This module intentionally has no Torch, Zarr, or VC imports.  VC3D can therefore
resolve and validate a dataset before importing the comparatively expensive fitting
stack in the service worker.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import datetime
from enum import Enum
import glob
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
import zipfile

from config import Config


# Version 17 binds dataset/output/cache to the service connection: --dataset
# and --output are required startup arguments, dataset resolution describes
# inputs only (the service injects the startup-resolved output/cache into the
# /dataset advertisement), and the /dataset/resolve browse endpoint is
# removed — resolution happens once at startup.
# Version 18 adds the structured /events stream and stops synthesizing
# eta_seconds inside the /session/status progress snapshot: clients derive
# the ETA from the raw step/total/elapsed fields.
# Version 19 replaces the Ready/Paused session states with a single Idle
# state, reported as both the state and the phase: a session that has never
# run and a session paused after N iterations are the same lifecycle state,
# and the difference is not one a user acts on. Clients that want to report
# how much work has happened read current_iteration.
# Version 20 adds POST /session/load-checkpoint: a strict, two-phase,
# all-rank load of a checkpoint into the resident model, valid in Idle. The
# service refuses (409) any checkpoint that is not an exact match for the
# live model domain and structure instead of rebuilding the model behind the
# client's back; a domain change stays the job of a new fit. An accepted load
# restores the checkpoint's completed_iterations and publishes it.
# Version 21 makes previews and pause-time saving explicit. POST
# /session/export-preview exports and publishes one preview generation on
# request; resuming from a checkpoint and pausing after a run no longer
# export one by themselves, so inspecting a checkpoint costs a load rather
# than a load plus a preview. The run request carries autosave_on_pause
# (default true), which decides whether that run's pause writes the durable
# autosave.
# Version 22 makes the session eager and always loaded. The service creates
# its runtime asynchronously at startup and reports Loading (then Idle, or
# Error with the cause) without any client request; there is no "Empty"
# state and DELETE /session is gone. POST /session/load is replaced by
# POST /session/rebuild, the only verb that may replace the model domain or
# structural configuration: it tears the resident session down and builds a
# fresh one (Idle|Error -> Loading), and with {"defaults": true} it rebuilds
# from launch defaults, ignoring any autosave. A startup autosave is chosen
# from explicit metadata sidecars (session namespace, dataset identity,
# completed iterations), never from filename ordering.
# Version 22 also drops the status fields nothing consumed:
# service_generation (the constant 1; process identity is /health's
# process_id), command_generation (replay is keyed by operation and command
# ID), session_replacement_in_progress and replacement_old_session_released.
# The counters that remain are session_generation (session identity),
# session_revision (configuration/input revision) and generation (status
# ordering).
API_VERSION = 22


class SessionState(str, Enum):
    """Lifecycle state of one resident fit session.

    The wire form is the member name; the member is a ``str`` so a status
    snapshot serializes and compares exactly like the string it replaces.

    Ready and Paused are one state: an idle session that has never stepped
    and an idle session paused after N iterations differ only in
    ``completed_iterations``. Operation phase and progress are reported
    separately (``phase``/``progress``) and are not part of this enum.

    ``Empty`` is deliberately absent: it describes a service with no
    session at all, not a state a session can be in.
    """

    Loading = "Loading"
    Idle = "Idle"
    Running = "Running"
    Saving = "Saving"
    ExportingPreview = "ExportingPreview"
    Error = "Error"
    Closing = "Closing"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


# States in which a session is executing something on the fitter thread and
# cannot accept a new run, a save, or a replacement.
SESSION_BUSY_STATES = frozenset({
    SessionState.Loading, SessionState.Running, SessionState.Saving,
    SessionState.ExportingPreview,
})


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


# ---------------------------------------------------------------------------
# Declarative fit-input catalog
# ---------------------------------------------------------------------------
#
# One description of every fit input, consumed by request validation
# (validate_session_request), dataset resolution (resolve_dataset_root /
# conventional_input_paths), run planning (spiral_service.plan_run), and the
# resident fitter's impact analysis (FitContext.apply_config). Each entry
# records the input's path kind, its enabling/required predicates over the
# run configuration, what must be rebuilt when the path changes, and whether
# a change breaks checkpoint compatibility.

# A path change whose input has no specific rebuild scope invalidates every
# prepared-input family: the session must be reloaded with fresh host inputs.
FULL_REBUILD_DEPENDENCIES = (
    "dense_stores", "patch_pcl", "tracks", "shell", "preview_output")


def _always(config: Mapping[str, Any]) -> bool:
    return True


def _never(config: Mapping[str, Any]) -> bool:
    return False


def _patches_enabled(config: Mapping[str, Any]) -> bool:
    return not bool(config.get("input_disable_patches", False))


def _shell_losses_enabled(config: Mapping[str, Any]) -> bool:
    return (
        float(config.get("loss_weight_shell_outer", 1.0)) > 0
        or float(config.get("loss_weight_shell_patch_radius", 0)) > 0
    )


def _dense_spacing_mode(config: Mapping[str, Any]) -> str | None:
    # An invalid mode is reported as its own validation error; the
    # mode-derived asset predicates then all read as disabled so the invalid
    # mode never masquerades as missing-file errors.
    mode = str(config.get("dense_spacing_mode", "phase"))
    return mode if mode in ("phase", "grad_mag") else None


def _phase_bundle_enabled(config: Mapping[str, Any]) -> bool:
    return _dense_spacing_mode(config) == "phase"


def _normals_required(config: Mapping[str, Any]) -> bool:
    # The phase bundle requires both normal channels (band incidence
    # handling) even when individual sub-weights are zero, so run-mutable
    # weights can be raised at run boundaries.
    return (float(config.get("loss_weight_dense_normals", 100.0)) > 0
            or _phase_bundle_enabled(config))


def _grad_mag_required(config: Mapping[str, Any]) -> bool:
    return (_dense_spacing_mode(config) == "grad_mag"
            and float(config.get("loss_weight_dense_spacing", 12.0)) > 0)


@dataclass(frozen=True)
class FitInputSpec:
    """Declarative description of one fit input."""

    # SpiralInputPaths field name (and manifest/path-change key).
    key: str
    # "file" | "directory" | "zarr-group" | "dbm" | "pcl-set".
    kind: str
    # File contents must parse as JSON.
    json_content: bool = False
    # Conventional dataset-relative location ("" = none: the input is only
    # reached through an explicit path or a scroll-spec override).
    conventional_relative: str = ""
    # A missing conventional path is missing_required at dataset resolution.
    resolve_required: bool = False
    # enabled(config): the input participates at all; inactive inputs are
    # not validated even when a path is set.
    enabled: Callable[[Mapping[str, Any]], bool] = _always
    # required(config): the input must exist for this configuration.
    required: Callable[[Mapping[str, Any]], bool] = _never
    # What a path change invalidates on a resident session:
    # "prepared_input_rebuild" reloads host inputs (session reload);
    # "shell_reload" rebuilds device shell state at a run boundary.
    runtime_impact: str = "prepared_input_rebuild"
    dependencies: tuple[str, ...] = FULL_REBUILD_DEPENDENCIES
    # Whether changing the input breaks checkpoint compatibility. No fit
    # input does today: the checkpoint domain is set by "new_fit" config
    # keys (z-range, model shape, optimizer seed), never by an input path.
    checkpoint_domain: bool = False


# Entries are ordered as validate_session_request reports them.
FIT_INPUT_CATALOG: tuple[FitInputSpec, ...] = (
    FitInputSpec("umbilicus", "file", json_content=True,
                 conventional_relative="umbilicus.json",
                 resolve_required=True, required=_always),
    FitInputSpec("verified_patches", "directory",
                 conventional_relative="verified_patches",
                 resolve_required=True, required=_patches_enabled),
    FitInputSpec("unverified_patches", "directory",
                 enabled=_patches_enabled),
    FitInputSpec("fibers", "directory", conventional_relative="fibers"),
    FitInputSpec("outer_shell", "directory",
                 conventional_relative="outer_shell",
                 required=_shell_losses_enabled,
                 runtime_impact="shell_reload", dependencies=("shell",)),
    FitInputSpec("tracks_dbm", "dbm",
                 conventional_relative="tracks/2um_ds2_ps256_surf_v2.dbm"),
    FitInputSpec("pcls", "pcl-set", json_content=True),
    FitInputSpec("normal_x", "zarr-group",
                 conventional_relative="lasagna_inputs/las_008_nx.ome.zarr",
                 required=_normals_required),
    FitInputSpec("normal_y", "zarr-group",
                 conventional_relative="lasagna_inputs/las_008_ny.ome.zarr",
                 required=_normals_required),
    FitInputSpec("gradient_magnitude", "zarr-group",
                 conventional_relative="lasagna_inputs/las_008_grad_mag.ome.zarr",
                 required=_grad_mag_required),
    FitInputSpec("surf_sdt", "zarr-group",
                 conventional_relative="lasagna_inputs/las_008_surf_sdt.ome.zarr",
                 required=_phase_bundle_enabled),
)

_FIT_INPUTS_BY_KEY = {spec.key: spec for spec in FIT_INPUT_CATALOG}


def fit_input(key: str) -> FitInputSpec | None:
    return _FIT_INPUTS_BY_KEY.get(key)


def input_change_impact(key: str) -> tuple[str, list[str]]:
    """Rebuild scope of changing one input path on a resident session.

    Unknown keys get the conservative full-rebuild scope, exactly as run
    planning has always treated paths it had no specific knowledge of.
    """
    spec = _FIT_INPUTS_BY_KEY.get(key)
    if spec is None:
        return "prepared_input_rebuild", list(FULL_REBUILD_DEPENDENCIES)
    return spec.runtime_impact, list(spec.dependencies)


def input_path_schema() -> dict[str, dict[str, Any]]:
    """Path entries advertised in the configuration catalog schema.

    Only inputs a resident session can take live (without a session reload)
    are advertised; everything else uses the implicit
    prepared-input-rebuild default.
    """
    return {
        spec.key: {"runtime_impact": spec.runtime_impact,
                   "dependencies": list(spec.dependencies)}
        for spec in FIT_INPUT_CATALOG
        if spec.runtime_impact != "prepared_input_rebuild"
    }


# PCL point-collection inputs: (role, conventional filename, discovered).
# "discovered" = resolve_dataset_root probes for the file; patch-overlap
# collections are conventional for the headless CLI but are not advertised
# by dataset resolution (matching the historical maps).
PCL_ROLE_CONVENTIONS: tuple[tuple[PclRole, str, bool], ...] = (
    (PclRole.ABSOLUTE, "abs_winding.json", True),
    (PclRole.PATCH_OVERLAP, "patch-overlap-pcls.json", False),
    (PclRole.RELATIVE, "relative_windings.json", True),
    (PclRole.SAME_WINDING, "same_windings.json", True),
    (PclRole.DRAWN_CONTROL_POINTS, "drawn_control_points.json", True),
)


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
# need no entry at all. PCL collections are per-role request entries, not
# single overridable paths.
SCROLL_SPEC_PATH_OVERRIDE_KEYS = tuple(
    spec.key for spec in FIT_INPUT_CATALOG if spec.kind != "pcl-set")

_SCROLL_SPEC_TOP_LEVEL_KEYS = (
    "schema_version", "name", "voxel_size_um", "spiral_outward_sense",
    "umbilicus", "normal_zarr_group", "surf_sdt_zarr_group", "lasagna_scale",
    "paths",
)

# Conventional dataset layout for the headless CLI, mirroring the historical
# fit_spiral module-global defaults. resolve_dataset_root() shares the same
# relative paths for the entries it discovers.
_CONVENTIONAL_INPUT_RELATIVES = {
    spec.key: spec.conventional_relative
    for spec in FIT_INPUT_CATALOG
    if spec.conventional_relative and spec.kind != "pcl-set"
}

_CONVENTIONAL_PCL_INPUTS = tuple(
    (filename, role) for role, filename, _ in PCL_ROLE_CONVENTIONS)


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


# ---------------------------------------------------------------------------
# Autosave metadata and startup selection
# ---------------------------------------------------------------------------
#
# A pause writes ``checkpoint_autosave.ckpt`` into that run's output directory
# and, beside it, a metadata sidecar naming what the file *is*. An
# always-loaded service picks its startup autosave from those sidecars alone:
# a bare ``.ckpt`` with no metadata is never selected, and the winner is the
# one with the most completed iterations, never the last filename in sort
# order.

AUTOSAVE_CHECKPOINT_NAME = "checkpoint_autosave.ckpt"
AUTOSAVE_METADATA_NAME = "checkpoint_autosave.json"
AUTOSAVE_METADATA_SCHEMA = "spiral-autosave/1"
_AUTOSAVE_DIGEST_CHUNK = 1 << 20


class AutosaveError(RuntimeError):
    """The autosave selected for startup cannot be used, and why."""


@dataclass(frozen=True)
class AutosaveCandidate:
    """One autosave that named itself through a metadata sidecar."""

    checkpoint: str
    metadata_path: str
    session_namespace: str
    dataset_root: str
    completed_iterations: int
    created: str
    sha256: str
    size: int

    def manifest(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AutosaveSelection:
    """The startup autosave decision, and every candidate that lost."""

    selected: AutosaveCandidate | None = None
    #: ``(metadata or checkpoint path, reason)`` for each rejected candidate.
    rejected: tuple[tuple[str, str], ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "selected": self.selected.manifest() if self.selected else None,
            "rejected": [{"path": path, "reason": reason}
                         for path, reason in self.rejected],
        }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(_AUTOSAVE_DIGEST_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def write_autosave_metadata(
    checkpoint_path: str | Path,
    *,
    session_namespace: str | Path,
    dataset_root: str | Path,
    completed_iterations: int,
) -> str:
    """Record what an autosave is, next to it, atomically.

    ``session_namespace`` is the service output root the autosave belongs to
    (``<output>/<session-name>`` for a named service); an autosave written
    under a different namespace is never a startup candidate for this one.
    """
    checkpoint = Path(checkpoint_path)
    metadata_path = checkpoint.with_name(AUTOSAVE_METADATA_NAME)
    document = {
        "schema": AUTOSAVE_METADATA_SCHEMA,
        "session_namespace": _normalise_path(session_namespace),
        "dataset_root": _normalise_path(dataset_root),
        "checkpoint": checkpoint.name,
        "completed_iterations": int(completed_iterations),
        "created": datetime.datetime.now(datetime.timezone.utc)
                   .replace(microsecond=0).isoformat(),
        "size": checkpoint.stat().st_size,
        "sha256": file_sha256(checkpoint),
        "api_version": API_VERSION,
    }
    temporary = metadata_path.with_name(f".{metadata_path.name}.incoming")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, metadata_path)
    return str(metadata_path)


def read_autosave_metadata(path: str | Path) -> AutosaveCandidate:
    """Parse one sidecar strictly; a malformed sidecar is not a candidate."""
    metadata_path = Path(path)
    with metadata_path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise ValueError("autosave metadata is not a JSON object")
    schema = str(document.get("schema") or "")
    if schema != AUTOSAVE_METADATA_SCHEMA:
        raise ValueError(
            f"unsupported autosave metadata schema {schema!r} "
            f"(expected {AUTOSAVE_METADATA_SCHEMA!r})")
    name = str(document.get("checkpoint") or "")
    if not name or "/" in name or name in (".", ".."):
        raise ValueError("autosave metadata does not name a sibling checkpoint")
    for key in ("session_namespace", "dataset_root", "sha256"):
        if not str(document.get(key) or "").strip():
            raise ValueError(f"autosave metadata is missing {key}")
    return AutosaveCandidate(
        checkpoint=str(metadata_path.parent / name),
        metadata_path=str(metadata_path),
        session_namespace=_normalise_path(document["session_namespace"]),
        dataset_root=_normalise_path(document["dataset_root"]),
        completed_iterations=int(document.get("completed_iterations", 0)),
        created=str(document.get("created") or ""),
        sha256=str(document["sha256"]),
        size=int(document.get("size", -1)),
    )


def validate_autosave(candidate: AutosaveCandidate) -> None:
    """Check container and checkpoint identity; raise ``AutosaveError``."""
    checkpoint = Path(candidate.checkpoint)
    if not checkpoint.is_file():
        raise AutosaveError(
            f"the autosave named by {candidate.metadata_path} is missing "
            f"({candidate.checkpoint})")
    size = checkpoint.stat().st_size
    if candidate.size >= 0 and size != candidate.size:
        raise AutosaveError(
            f"{candidate.checkpoint} is {size} bytes; its metadata records "
            f"{candidate.size}")
    try:
        validate_checkpoint_container(checkpoint)
    except (OSError, ValueError) as exc:
        raise AutosaveError(f"{candidate.checkpoint}: {exc}") from exc
    digest = file_sha256(checkpoint)
    if digest != candidate.sha256:
        raise AutosaveError(
            f"{candidate.checkpoint} has digest {digest}; its metadata "
            f"records {candidate.sha256}")


def discover_autosave_metadata(output_root: str | Path) -> list[Path]:
    """Every autosave sidecar under a service output root, run dirs included."""
    root = Path(output_root)
    if not root.is_dir():
        return []
    found = {root / AUTOSAVE_METADATA_NAME}
    found.update(root.glob(f"*/{AUTOSAVE_METADATA_NAME}"))
    return sorted(path for path in found if path.is_file())


def select_startup_autosave(
    output_root: str | Path,
    *,
    session_namespace: str | Path,
    dataset_root: str | Path,
) -> AutosaveSelection:
    """Choose the autosave an always-loaded service should resume from.

    Selection is by metadata: the sidecar must parse, must carry this
    service's session namespace, and must have been written against this
    dataset root. Among the survivors the one with the most completed
    iterations wins (ties broken by the recorded creation time), so the
    result never depends on filename ordering. Container and identity
    validation happens after selection, in ``validate_autosave``: a selected
    autosave that fails it is an error, not a reason to silently resume from
    an older one.
    """
    namespace = _normalise_path(session_namespace)
    dataset = _normalise_path(dataset_root)
    candidates: list[AutosaveCandidate] = []
    rejected: list[tuple[str, str]] = []
    for metadata_path in discover_autosave_metadata(output_root):
        try:
            candidate = read_autosave_metadata(metadata_path)
        except (OSError, ValueError, TypeError) as exc:
            rejected.append((str(metadata_path), f"unreadable metadata: {exc}"))
            continue
        if candidate.session_namespace != namespace:
            rejected.append((
                str(metadata_path),
                f"belongs to session namespace {candidate.session_namespace}"))
            continue
        if candidate.dataset_root != dataset:
            rejected.append((
                str(metadata_path),
                f"was written against dataset root {candidate.dataset_root}"))
            continue
        candidates.append(candidate)
    if not candidates:
        return AutosaveSelection(rejected=tuple(rejected))
    selected = max(candidates,
                   key=lambda item: (item.completed_iterations, item.created))
    rejected.extend(
        (candidate.metadata_path,
         f"superseded by {selected.checkpoint} at "
         f"{selected.completed_iterations} iterations")
        for candidate in candidates if candidate is not selected)
    return AutosaveSelection(selected=selected, rejected=tuple(rejected))


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


def default_user_cache_dir() -> str:
    """The one documented user cache location, outside any dataset.

    ``$XDG_CACHE_HOME/vc3d/spiral`` (``~/.cache/vc3d/spiral`` by default).
    Cache entries are content-addressed, so one shared directory serves every
    dataset; ``--cache`` overrides it per service or CLI run.
    """
    base = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
    return _normalise_path(base / "vc3d" / "spiral")


def resolve_dataset_root(
    root_value: str | os.PathLike[str],
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

    for input_spec in FIT_INPUT_CATALOG:
        # DBM inputs need backing-file probing (below) and PCL collections
        # are per-role files; neither is a plain path probe.
        if input_spec.kind in ("dbm", "pcl-set") or not input_spec.conventional_relative:
            continue
        override = spec.path_override(input_spec.key) if spec is not None else ""
        candidate = Path(override) if override \
            else root / input_spec.conventional_relative
        found = candidate.is_file() if input_spec.kind == "file" \
            else candidate.is_dir()
        if found and os.access(candidate, os.R_OK):
            result.resolved[input_spec.key] = _normalise_path(candidate)
        elif input_spec.resolve_required:
            result.missing_required.append(input_spec.key)
        else:
            result.missing_optional.append(input_spec.key)

    for role, relative, discovered in PCL_ROLE_CONVENTIONS:
        if not discovered:
            continue
        candidate = root / relative
        if candidate.is_file() and os.access(candidate, os.R_OK):
            result.pcl_inputs.append({
                "path": _normalise_path(candidate),
                "role": role.value,
                "required": False,
            })
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

    # Dataset resolution describes inputs only. The service binds the
    # startup-resolved --output/--cache into the advertised resolution
    # (spiral_service.bind_service_paths); nothing generated lives under
    # the dataset root.
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

    def check_catalog_input(spec: FitInputSpec) -> None:
        if not spec.enabled(run.config):
            return
        if spec.kind == "file":
            require_file(getattr(paths, spec.key), spec.key,
                         json_file=spec.json_content)
        elif spec.kind in ("directory", "zarr-group"):
            optional_dir(getattr(paths, spec.key), spec.key,
                         required=spec.required(run.config))
        elif spec.kind == "dbm":
            value = getattr(paths, spec.key)
            if value and not resolve_logical_dbm(value):
                errors.append({"field": spec.key, "message": "DBM logical base or backing file was not found"})
        elif spec.kind == "pcl-set":
            for index, pcl in enumerate(paths.pcls):
                expanded = _expand_pcl(pcl)
                if pcl.required and not expanded:
                    errors.append({"field": f"pcls[{index}]", "message": "Required PCL pattern matched no files"})
                for expanded_path in expanded:
                    path = Path(expanded_path)
                    if not path.is_file():
                        errors.append({"field": f"pcls[{index}]", "message": f"PCL file does not exist: {path}"})
                    else:
                        _validate_json_file(path, f"pcls[{index}]", errors)

    # The Lasagna store requirements (see the catalog's predicates: the
    # phase bundle needs SDT and both normal channels even at zero
    # sub-weights; grad_mag never needs the SDT) are checked after the
    # dense-spacing mode below, so an invalid mode errors as itself rather
    # than as missing-file errors.
    for spec in FIT_INPUT_CATALOG:
        if spec.kind != "zarr-group":
            check_catalog_input(spec)

    spacing_mode = str(run.config.get("dense_spacing_mode", "phase"))
    if spacing_mode not in ("phase", "grad_mag"):
        errors.append({"field": "dense_spacing_mode",
                       "message": "Must be phase or grad_mag"})

    for spec in FIT_INPUT_CATALOG:
        if spec.kind == "zarr-group":
            check_catalog_input(spec)

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

    lasagna_inputs_enabled = any(
        spec.required(run.config) for spec in FIT_INPUT_CATALOG
        if spec.kind == "zarr-group")
    if lasagna_inputs_enabled and not paths.cache_directory:
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
