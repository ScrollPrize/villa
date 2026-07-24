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



# Version 6 exposes the session's effective Python/checkpoint Advanced-config
# baseline independently from client overrides.
API_VERSION = 6


# Counts which describe how many training objects/points are sampled per
# optimizer step. The service exposes the post-scaling values actually used by
# the resident fitter, and Run-scoped edits set those active values directly.
RUN_MUTABLE_SAMPLING_KEYS = frozenset({
    "num_patches_per_step",
    "num_patches_per_step_for_dt",
    "num_points_per_patch",
    "unverified_num_patches_per_step",
    "unverified_num_patches_per_step_for_dt",
    "unverified_num_points_per_patch",
    "rel_winding_num_pcls",
    "rel_winding_num_patch_pairs_per_pcl",
    "abs_winding_num_pcls",
    "abs_winding_num_points_per_pcl",
    "unattached_pcl_num_per_step",
    "unattached_pcl_num_points_per_step",
    "track_num_per_step",
    "track_num_points_per_step",
    "dense_normals_num_points",
    "dense_spacing_num_pairs",
    "dense_spacing_density_extra_pairs",
    "dense_attachment_num_points",
    "min_spacing_independent_samples",
    "regularisation_num_points",
    "shell_num_samples",
})


RUN_MUTABLE_BOOLEAN_KEYS = frozenset({
    "save_png_visualizations",
})


RUN_MUTABLE_TRACK_POLICY_KEYS = frozenset({
    "track_length_bin_weights",
    "max_track_crossing_per_step",
    "track_min_sample_spacing",
    "track_max_sample_spacing",
})


def is_run_mutable_config_key(key: str) -> bool:
    """Return whether an advanced setting may change at a Run boundary."""
    return (
        key in RUN_MUTABLE_SAMPLING_KEYS
        or key in RUN_MUTABLE_BOOLEAN_KEYS
        or key in RUN_MUTABLE_TRACK_POLICY_KEYS
        or (key.startswith("loss_weight_") and key != "loss_weight_anchor")
        or key.startswith("loss_start_")
    )


def run_mutable_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Select the Run-scoped editor fields from a complete fitter config."""
    return {
        key: value for key, value in config.items()
        if is_run_mutable_config_key(key)
    }


def apply_optional_input_selection(config: dict[str, Any]) -> dict[str, Any]:
    """Force losses and sample counts off for session-disabled inputs."""
    def zero(*keys: str) -> None:
        for key in keys:
            config[key] = 0

    if not bool(config.get("use_verified_patches", True)):
        zero("loss_weight_patch_radius", "loss_weight_patch_dt",
             "loss_weight_umbilicus", "loss_weight_shell_patch_radius",
             "num_patches_per_step", "num_patches_per_step_for_dt",
             "num_points_per_patch")
    if not bool(config.get("use_unverified_patches", True)):
        zero("loss_weight_unverified_patch_radius",
             "loss_weight_unverified_patch_dt",
             "unverified_num_patches_per_step",
             "unverified_num_patches_per_step_for_dt",
             "unverified_num_points_per_patch")
    normals = bool(config.get("use_normals", True))
    sdt = bool(config.get("use_surf_sdt", True))
    if not normals:
        zero("loss_weight_dense_normals", "dense_normals_num_points")
    if not bool(config.get("use_tracks", True)):
        zero("loss_weight_track_radius", "loss_weight_track_dt",
             "track_num_per_step", "track_num_points_per_step")
    if not bool(config.get("use_fibers", True)):
        zero("loss_weight_unattached_pcl_radius", "loss_weight_unattached_pcl_dt",
             "unattached_pcl_num_per_step", "unattached_pcl_num_points_per_step")

    spacing_mode = str(config.get("dense_spacing_mode", "phase"))
    if not sdt or not normals:
        zero("loss_weight_dense_spacing_count",
             "loss_weight_dense_spacing_density",
             "loss_weight_dense_attachment", "dense_spacing_count_extra_pairs",
             "dense_spacing_density_extra_pairs", "dense_attachment_num_points")
        if spacing_mode == "phase":
            zero("loss_weight_dense_spacing", "dense_spacing_num_pairs")
    if (not bool(config.get("use_gradient_magnitude", True))
            and spacing_mode == "grad_mag"):
        zero("loss_weight_dense_spacing", "dense_spacing_num_pairs")
    return config


class PclRole(str, Enum):
    ABSOLUTE = "absolute"
    PATCH_OVERLAP = "patch_overlap"
    RELATIVE = "relative"
    SAME_WINDING = "same_winding"
    DRAWN_CONTROL_POINTS = "drawn_control_points"


@dataclass(frozen=True)
class PclInputSpec:
    path: str
    role: PclRole
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
            {"path": item.path, "role": item.role.value, "required": item.required}
            for item in self.pcls
        ]
        return result


@dataclass(frozen=True)
class SpiralRunConfig:
    z_begin: int
    z_end: int
    scroll_name: str = "scroll"
    outward_sense: str = "CW"
    voxel_size_um: float = 9.6
    lasagna_group: str = "4"
    lasagna_scale: int = 4
    storage_backend: str = "auto"
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
            outward_sense=str(value.get("outward_sense", "CW")).upper(),
            voxel_size_um=float(value.get("voxel_size_um", 9.6)),
            lasagna_group=str(value.get("lasagna_group", "4")),
            lasagna_scale=int(value.get("lasagna_scale", 4)),
            storage_backend=str(value.get("storage_backend", "auto")).lower(),
            legacy_checkpoint_step=int(value.get("legacy_checkpoint_step", 0)),
            run_tag=str(value.get("run_tag", "")),
            render_volume_scale=int(value.get("render_volume_scale", 16)),
            config=dict(value.get("config", {})),
        )


@dataclass(frozen=True)
class SpiralPreviewConfig:
    first_winding: int = 10
    variant: str = "raw"


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

    @property
    def ok(self) -> bool:
        return not self.missing_required and not self.ambiguities

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["ok"] = self.ok
        return result


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


def resolve_dataset_root(root_value: str | os.PathLike[str]) -> SpiralDatasetResolution:
    root = Path(_normalise_path(root_value))
    result = SpiralDatasetResolution(root=str(root))
    if not root.is_dir():
        result.missing_required.append("dataset_root")
        result.warnings.append(f"Dataset root is not a readable directory: {root}")
        return result

    for key, relative, kind, required in _CONVENTIONAL_ENTRIES:
        candidate = root / relative
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

    preferred = root / "tracks" / "2um_ds2_ps256_surf_v2.dbm"
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

    result.resolved["output_directory"] = _normalise_path(root / "spiral_output")
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
    disable_patches = bool(run.config.get("disable_patches", False))
    use_verified = bool(run.config.get("use_verified_patches", True)) and not disable_patches
    use_unverified = bool(run.config.get("use_unverified_patches", True)) and not disable_patches
    optional_dir(paths.verified_patches, "verified_patches", required=use_verified)
    if use_unverified:
        optional_dir(paths.unverified_patches, "unverified_patches")
    if bool(run.config.get("use_fibers", True)):
        optional_dir(paths.fibers, "fibers")

    shell_enabled = (
        float(run.config.get("loss_weight_shell_outer", 1.0)) > 0
        or float(run.config.get("loss_weight_shell_patch_radius", 0)) > 0
    )
    optional_dir(paths.outer_shell, "outer_shell", required=shell_enabled)

    if (bool(run.config.get("use_tracks", True)) and paths.tracks_dbm
            and not resolve_logical_dbm(paths.tracks_dbm)):
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

    normals_selected = bool(run.config.get("use_normals", True))
    sdt_selected = bool(run.config.get("use_surf_sdt", True))
    grad_mag_selected = bool(run.config.get("use_gradient_magnitude", True))
    use_normals = (normals_selected
                   and float(run.config.get("loss_weight_dense_normals", 100.0)) > 0)
    spacing_enabled = float(run.config.get("loss_weight_dense_spacing", 12.0)) > 0
    use_phase = spacing_mode == "phase" and normals_selected and sdt_selected
    use_grad_mag = (
        spacing_mode == "grad_mag" and spacing_enabled and grad_mag_selected)
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
    if run.outward_sense not in {"CW", "ACW"}:
        errors.append({"field": "outward_sense", "message": "Must be CW or ACW"})
    if run.lasagna_scale <= 0:
        errors.append({"field": "lasagna_scale", "message": "Must be positive"})
    if run.storage_backend not in {"auto", "mmap", "dense_cuda"}:
        errors.append({"field": "storage_backend", "message": "Must be auto, mmap, or dense_cuda"})

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


class SpiralFitSession:
    """Interface implemented by the resident fitter owned by the service worker."""

    completed_iterations: int

    def step(self, count: int, stop_event: Any, progress_callback: Any) -> Mapping[str, Any]:
        raise NotImplementedError

    def save_checkpoint(self, path: str) -> str:
        raise NotImplementedError

    def export_preview(self, generation_dir: str) -> Mapping[str, Any]:
        raise NotImplementedError
