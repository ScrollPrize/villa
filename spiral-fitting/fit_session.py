"""Session-facing configuration and dataset resolution for interactive Spiral fits.

This module intentionally has no Torch, Zarr, or VC imports.  VC3D can therefore
resolve and validate a dataset before importing the comparatively expensive fitting
stack in the service worker.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import datetime
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
import zipfile

from config import Config
from surface_orientation import spiral_outward_sense_for


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
# Version 23 removes the verbs no client called — GET /logs (every line it
# carried is already a log-kind /events record), POST /session/export-full (a
# 501 stub) and DELETE /session/inputs/<id> (abandoned uploads expire on
# their own) — and POST /service/restart, whose only remaining job was
# recovering a wedged process that an operator restarts directly.
# POST /session/save-checkpoint now takes {"name"} instead of {"path"}: a
# checkpoint was only ever allowed under the session output directory, so
# the service resolves it and the client stops guessing host paths.
# POST /session/export-preview accepts and returns instead of blocking for
# the whole export; status carries preview_exporting while it runs.
# dataset_owned is gone from /health and /session/status: --dataset is
# required, so it was always true.
# Version 25 makes the startup-resolved input manifest strictly read-only.
# Version 26 removes the stateful run-plan handshake. POST /session/run carries
# the complete request and rejects configuration that requires a rebuild;
# run-boundary configuration is applied directly.
# Version 27 makes spiral-scroll.json the only source of the facts it carries.
# The run block no longer carries scroll_name, voxel_size_um, lasagna_group or
# lasagna_scale, and a request that names any of them is rejected. Each
# restated something the dataset root already specifies and silently won over
# it, so a client could mislabel outputs, scale a fit against a resolution the
# dataset contradicts, or read the Lasagna stores at the wrong zarr level.
# Version 28 gives a rebuild two stages and folds the checkpoint rebuild into
# the load. POST /session/rebuild reports the "stage" it took ("model" keeps
# the loaded inputs and the brick pools; "all" is the whole build), and
# /configuration advertises schema.model_stage_keys so a client can say in
# advance which it would get. POST /session/load-checkpoint takes
# "allow_rebuild", and its 409 carries the preflight's "reasons" with either
# the "stage" a rebuild would need or "refused" when none would help.
# Version 29 adds the compact winding-inference input and dense-spacing mode.
# Version 30 makes session creation explicit.  A service starts in
# Uninitialized and does not import the fitting runtime, initialize CUDA, read
# fit inputs, or select an autosave until POST /session/initialize.
# Version 31 is this branch's combined live-input/editor contract. It separates
# client-local drafts from submitted inputs, adds CAS revisions for fibers and
# same-winding collections, and publishes the same-winding artifact metadata.
# Version 32 publishes a relative-winding point-cloud artifact next to the
# same-winding one and opens the collection-mutation protocol to the
# ``relative`` role, so both editable PCL roles share one client workflow.
# API 33 separates immutable transfers from revisioned input batches and
# requires a dataset editing owner for service mutations.
# Version 34 adds service-owned background preview schedules, source-iteration
# preview provenance, and active/pending publication state. Preview capture is
# an all-rank fit boundary, while flattening and artifact publication continue
# independently of the client connection and the next Run.
# Version 35 gives every Run an independent DT-loss schedule. The required
# dt_loss_schedule object controls how much of the final Run window is DT
# eligible and is transient rather than part of advanced configuration or a
# checkpoint.
# Version 36 reads patch, PCL and fiber inputs from the object store bound at
# launch (--object-store) and freezes them for a resident session. The
# editing workspace is replaced by a session lease (POST
# /session/lease/claim and /release, token header X-Spiral-Session-Token;
# status reports lease_id). The input-editing routes answer 409 with code
# inputs_frozen, naming the full-input rebuild, and uploads accept only
# checkpoints. /dataset no longer resolves patch, fiber or PCL files
# (pcl_inputs is gone); it advertises the store binding (object_store: uuid,
# path, author), the launch configuration's selection from it (inputs) and
# the service's capabilities. Status carries input_digest and input_summary
# of the frozen inputs, input_manifest lists only the inputs read by path,
# and the full object-store manifest is the input_manifest_artifact; the
# PCL display artifacts (same_winding_artifact, relative_winding_artifact)
# are gone. Previews, autosave metadata and the preview index record the
# input digest, and a published preview is re-shown only for the same inputs.
# POST /session/rebuild takes "inputs": "reload" to reread the store under an
# unchanged configuration and reports which it did ("inputs": "keep" for a
# model-only rebuild). A checkpoint refused for its inputs says so in the 409's
# "inputs" (legacy, problems, the reload rebuild and
# object_store_allow_changed_inputs).
# 37: hub positions. /dataset's object_store carries "hub" (address,
# identity, position) for a service bound to a hub replica, and
# capabilities.input_positions. Initialize and full-input rebuilds link and
# pin a hub position first, including at least the request's
# "inputs_position"; the resident fit is unchanged when that fails
# ("input_preparation_error"). Status carries "input_position" (hub,
# sequence, requested, head, attempts, source) and "input_preparation";
# lifecycle responses carry "build_id". A checkpoint resumes at the position
# it recorded. object_store_input_position is service-owned.
# 38: input updates. POST /session/update-inputs {"inputs_position",
# "expected_input_generation"} links and pins a hub position and replaces
# the resident fit's inputs with it at an iteration boundary, keeping the
# model, optimiser, scheduler and iteration (capabilities.input_updates;
# inputs_frozen is false for a hub-bound service). Status carries
# "input_update" while one runs and "input_generation"; a failed update's
# "input_preparation_error" says its "phase" and whether the fit needs
# recovery ("recovery_required").
# 39: positions without a hub. A service bound to a plain store (no
# --object-store-hub) records positions of the store itself, shared in place
# with the client's processes on its machine: /dataset's object_store carries
# "positions" (identity, position, position_error), the capabilities report
# input_positions and input_updates, and initialize, full-input rebuilds and
# update-inputs link and pin positions of the store as for a hub. Positions'
# "hub" names their owner (a hub, or the store).
# 40: the client's project. Every session request (initialize, rebuild,
# including a defaults rebuild) carries "project": {"name", "volumes": {store
# volume: {voxel_size_um, base_shape_zyx, z_direction_is_top_to_bottom,
# left_handed_coordinates, umbilicus: {uuid, revision, points_zyx}}}}, each
# fact only where the project knows it. The build fits the volume its inputs
# name with that volume's umbilicus and facts; spiral-scroll.json becomes
# optional, may state only what the project does not know, and must not
# contradict it. /dataset no longer resolves an umbilicus; session_request
# gains "project" (name, volume, facts, umbilicus identity) and "scroll".
API_VERSION = 40


class SessionState(str, Enum):
    """Lifecycle state reported by the session service.

    The wire form is the member name; the member is a ``str`` so a status
    snapshot serializes and compares exactly like the string it replaces.

    Ready and Paused are one state: an idle session that has never stepped
    and an idle session paused after N iterations differ only in
    ``completed_iterations``. Operation phase and progress are reported
    separately (``phase``/``progress``) and are not part of this enum.

    ``Uninitialized`` belongs to the service rather than a resident runtime:
    it is the intentional state before the user asks to create the first fit.
    """

    Uninitialized = "Uninitialized"
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
            if key in fields
            and fields[key]["runtime_impact"] == "run_boundary"}


_INPUT_TOGGLE_KEYS = {
    "verified_patches": "input_use_verified_patches",
    "tracks_dbm": "input_use_tracks",
    "fibers": "input_use_fibers",
    "fiber_directions": "input_use_fiber_directions",
    "normals": "input_use_normals",
    "gradient_magnitude": "input_use_gradient_magnitude",
    "winding_inference": "input_use_winding_inference",
    "outer_shell": "input_use_outer_shell",
}

def input_source_enabled(config: Mapping[str, Any], source: str) -> bool:
    """Whether a rebuild may include one optional supervision source."""
    enabled = bool(config.get(_INPUT_TOGGLE_KEYS[source], True))
    if source == "verified_patches":
        enabled = enabled and not bool(config.get("input_disable_patches", False))
    return enabled


# ---------------------------------------------------------------------------
# Declarative fit-input catalog
# ---------------------------------------------------------------------------
#
# One description of every fit input read from the dataset by path, consumed
# by request validation (validate_session_request), dataset resolution
# (resolve_dataset_root / conventional_input_paths) and run admission
# (spiral_service.ServiceState.run). Patches, point collections and fibers
# are not here: they come from the object store (object_store_inputs). Each entry records the input's path kind
# and its enabling/required predicates over the run configuration. Fit-input
# paths are static for the lifetime of a resident session, so an entry says
# nothing about rebuild scope: changing any of them rebuilds the session.


def _always(config: Mapping[str, Any]) -> bool:
    return True


def _never(config: Mapping[str, Any]) -> bool:
    return False


def _tracks_enabled(config: Mapping[str, Any]) -> bool:
    return input_source_enabled(config, "tracks_dbm")


def _shell_losses_enabled(config: Mapping[str, Any]) -> bool:
    return (input_source_enabled(config, "outer_shell")
            and float(config.get("loss_weight_shell_outer", 1.0)) > 0)


def _dense_spacing_mode(config: Mapping[str, Any]) -> str | None:
    # An invalid mode is reported as its own validation error; the
    # mode-derived asset predicates then all read as disabled so the invalid
    # mode never masquerades as missing-file errors.
    # Partial requests must select the same inputs as the fitter, which fills
    # omitted fields from Config before constructing its context.
    mode = str(config.get("dense_spacing_mode", Config().dense_spacing_mode))
    return mode if mode in ("grad_mag", "winding_model") else None


def _normals_required(config: Mapping[str, Any]) -> bool:
    return (input_source_enabled(config, "normals")
            and float(config.get("loss_weight_dense_normals", 100.0)) > 0)


def _fiber_directions_enabled(config: Mapping[str, Any]) -> bool:
    return (input_source_enabled(config, "fiber_directions")
            and float(config.get("loss_weight_fiber_directions", 0.0)) > 0)


def _grad_mag_required(config: Mapping[str, Any]) -> bool:
    return (input_source_enabled(config, "gradient_magnitude")
            and _dense_spacing_mode(config) == "grad_mag"
            and float(config.get("loss_weight_dense_spacing", 12.0)) > 0)


def _winding_model_enabled(config: Mapping[str, Any]) -> bool:
    return (
        _dense_spacing_mode(config) == "winding_model"
        and input_source_enabled(config, "winding_inference")
        and input_source_enabled(config, "outer_shell")
    )


def _outer_shell_required(config: Mapping[str, Any]) -> bool:
    return _shell_losses_enabled(config) or _winding_model_enabled(config)


def winding_inference_enabled(config: Mapping[str, Any]) -> bool:
    return _winding_model_enabled(config)


def shell_losses_enabled(config: Mapping[str, Any]) -> bool:
    return _shell_losses_enabled(config)


@dataclass(frozen=True)
class FitInputSpec:
    """Declarative description of one fit input."""

    # SpiralInputPaths field name (and manifest/path-change key).
    key: str
    # "file" | "directory" | "zarr-group" | "dbm".
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
    # Whether changing the input breaks checkpoint compatibility. No fit
    # input does today: the checkpoint domain is set by "new_fit" config
    # keys (z-range, model shape, optimizer seed), never by an input path.
    checkpoint_domain: bool = False


# Entries are ordered as validate_session_request reports them.
FIT_INPUT_CATALOG: tuple[FitInputSpec, ...] = (
    FitInputSpec("umbilicus", "file", json_content=True,
                 conventional_relative="umbilicus.json",
                 resolve_required=True, required=_always),
    FitInputSpec("fiber_directions", "file",
                 conventional_relative="fiber_directions.npz",
                 enabled=_fiber_directions_enabled,
                 required=_fiber_directions_enabled),
    FitInputSpec("outer_shell", "directory",
                 conventional_relative="outer_shell",
                 enabled=lambda config: input_source_enabled(config, "outer_shell"),
                 required=_outer_shell_required),
    FitInputSpec("tracks_dbm", "dbm",
                 conventional_relative="tracks/2um_ds2_ps256_surf_v2.dbm",
                 enabled=_tracks_enabled),
    FitInputSpec("normal_x", "zarr-group",
                 conventional_relative="lasagna_inputs/las_008_nx.ome.zarr",
                 enabled=lambda config: input_source_enabled(config, "normals"),
                 required=_normals_required),
    FitInputSpec("normal_y", "zarr-group",
                 conventional_relative="lasagna_inputs/las_008_ny.ome.zarr",
                 enabled=lambda config: input_source_enabled(config, "normals"),
                 required=_normals_required),
    FitInputSpec("gradient_magnitude", "zarr-group",
                 conventional_relative="lasagna_inputs/las_008_grad_mag.ome.zarr",
                 enabled=lambda config: input_source_enabled(
                     config, "gradient_magnitude"),
                 required=_grad_mag_required),
    FitInputSpec("winding_inference", "directory",
                 conventional_relative="winding_inference",
                 enabled=_winding_model_enabled,
                 required=_winding_model_enabled),
)

_FIT_INPUTS_BY_KEY = {spec.key: spec for spec in FIT_INPUT_CATALOG}


def fit_input(key: str) -> FitInputSpec | None:
    return _FIT_INPUTS_BY_KEY.get(key)


@dataclass(frozen=True)
class SpiralInputPaths:
    """The fit inputs read by path; the object store supplies the rest."""

    dataset_root: str = ""
    umbilicus: str = ""
    fiber_directions: str = ""
    tracks_dbm: str = ""
    outer_shell: str = ""
    normal_x: str = ""
    normal_y: str = ""
    gradient_magnitude: str = ""
    winding_inference: str = ""
    scroll_zarr: str = ""
    checkpoint: str = ""
    output_directory: str = ""
    cache_directory: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SpiralInputPaths":
        # Unknown keys (an older manifest's patch, fiber and PCL paths) are
        # not inputs any more.
        return cls(**{name: _normalise_path(value.get(name))
                      for name in cls.__dataclass_fields__})

    def manifest(self) -> dict[str, Any]:
        return asdict(self)


def lasagna_normal_inputs(normal_x, normal_y, group, scale):
    """A fit's lasagna normals as object_store_inputs.fiber_normal_source
    takes them, or None when it loads none: the normal stores, their group
    and its downsampling relative to the fit volume."""
    if not (normal_x and normal_y):
        return None
    return {'normal_x': normal_x, 'normal_y': normal_y,
            'group': group, 'scale': scale}


# Run-block keys the scroll specification owns, mapped to the ScrollSpec field
# each one used to shadow. A request that carries one is rejected rather than
# ignored: silently dropping it would let a client believe it had renamed a
# scroll, changed its resolution, or moved the Lasagna stores.
SCROLL_SPEC_OWNED_RUN_KEYS = {
    "scroll_name": "name",
    "voxel_size_um": "voxel_size_um",
    "lasagna_group": "normal_zarr_group",
    "lasagna_scale": "lasagna_scale",
}


@dataclass(frozen=True)
class SpiralRunConfig:
    """Deployment and presentation settings of one fit run.

    Nothing spiral-scroll.json specifies is here — not the scroll's physical
    facts (name, voxel size, outward sense) and not the dataset's Lasagna store
    layout. That file is the single source for all of it (see ``ScrollSpec``).
    """

    z_begin: int
    z_end: int
    storage_backend: str = "sparse_cuda"
    run_tag: str = ""
    render_volume_scale: int = 16
    config: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SpiralRunConfig":
        return cls(
            z_begin=int(value.get("z_begin", 0)),
            z_end=int(value.get("z_end", 0)),
            storage_backend=str(value.get("storage_backend", "sparse_cuda")).lower(),
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
    missing_required: list[str] = field(default_factory=list)
    missing_optional: list[str] = field(default_factory=list)
    ambiguities: dict[str, list[str]] = field(default_factory=dict)
    detected_checkpoints: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # The fields spiral-scroll.json states (load_dataset_scroll_fields; path
    # overrides as a mapping), {} without one; None when it is invalid, which
    # is a missing_required condition ("scroll_spec"). The session's scroll
    # completes them with the project's facts (SessionScroll).
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
SCROLL_SPEC_PATH_OVERRIDE_KEYS = tuple(spec.key for spec in FIT_INPUT_CATALOG)
# Overrides of inputs that now come from the object store. A specification
# written for the file-input service may still name them; they are ignored.
RETIRED_PATH_OVERRIDE_KEYS = ("verified_patches", "fibers")

# Conventional dataset layout for the headless CLI, mirroring the historical
# fit_spiral module-global defaults. resolve_dataset_root() shares the same
# relative paths for the entries it discovers.
_CONVENTIONAL_INPUT_RELATIVES = {
    spec.key: spec.conventional_relative
    for spec in FIT_INPUT_CATALOG
    if spec.conventional_relative
}


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
    # The following are in the open-data catalog metadata; together they
    # fix spiral_outward_sense (surface_orientation.spiral_outward_sense_for);
    # the z direction also decides whether exported grids are flipped so row 0 is
    # the scroll top. None when the spec leaves them out.
    z_direction_is_top_to_bottom: bool | None = None
    left_handed_coordinates: bool | None = None
    # Shape of the volume coordinate domain used by Spiral surfaces. This is
    # independent of physical voxel size and may be absent in legacy specs.
    base_shape_zyx: tuple[int, int, int] | None = None
    umbilicus_coordinate_scale: float = 1.0
    normal_zarr_group: str = "4"
    lasagna_scale: int = 4
    # The scroll's winding count, when someone has estimated it. A default for
    # the fit, not a fact the scroll owns: see scroll_spec_config_defaults.
    num_windings: int | None = None
    # Allow-listed absolute-path overrides, (key, resolved path) pairs.
    path_overrides: tuple[tuple[str, str], ...] = ()

    def path_override(self, key: str) -> str:
        return dict(self.path_overrides).get(key, "")

    def manifest(self) -> dict[str, Any]:
        result = asdict(self)
        result["path_overrides"] = dict(self.path_overrides)
        return result


def _checked_voxel_size(value: Any, source: str) -> float:
    try:
        voxel_size_um = float(value)
    except (TypeError, ValueError):
        raise ScrollSpecError(f"{source}: voxel_size_um must be a number") from None
    if not voxel_size_um > 0:
        raise ScrollSpecError(f"{source}: voxel_size_um must be positive")
    return voxel_size_um


def _checked_base_shape(value: Any, source: str) -> tuple[int, int, int]:
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(type(item) is not int or item <= 0 for item in value)):
        raise ScrollSpecError(
            f"{source}: base_shape_zyx must be a ZYX list of three positive integers")
    return tuple(value)


def _checked_flag(value: Any, key: str, source: str) -> bool:
    if type(value) is not bool:
        raise ScrollSpecError(f"{source}: {key} must be true or false")
    return value


def _scroll_spec_fields(document: Any, dataset_root: str | os.PathLike[str],
                        source: str) -> dict[str, Any]:
    """The ScrollSpec fields a spiral-scroll.json states, validated but none
    required (``_scroll_spec_from_fields`` requires them): what a project's
    facts complete in a service session (``SessionScroll``).

    Unknown top-level keys are ignored for forward compatibility,
    schema_version is required, and path overrides are resolved relative to
    the dataset root.
    """
    if not isinstance(document, Mapping):
        raise ScrollSpecError(f"{source}: the scroll specification must be a JSON object")
    if "schema_version" not in document:
        raise ScrollSpecError(f"{source}: schema_version is required")
    if document["schema_version"] != SCROLL_SPEC_SCHEMA_VERSION:
        raise ScrollSpecError(
            f"{source}: unsupported schema_version {document['schema_version']!r} "
            f"(this build supports {SCROLL_SPEC_SCHEMA_VERSION})")
    fields: dict[str, Any] = {}
    for key in ("z_direction_is_top_to_bottom", "left_handed_coordinates"):
        if document.get(key) is not None:
            fields[key] = _checked_flag(document[key], key, source)
    if "name" in document:
        fields["name"] = str(document["name"]).strip()
        if not fields["name"]:
            raise ScrollSpecError(f"{source}: name must be a non-empty string")
    if "voxel_size_um" in document:
        fields["voxel_size_um"] = _checked_voxel_size(document["voxel_size_um"], source)
    if document.get("base_shape_zyx") is not None:
        fields["base_shape_zyx"] = _checked_base_shape(document["base_shape_zyx"], source)
    if "spiral_outward_sense" in document:
        fields["spiral_outward_sense"] = str(document["spiral_outward_sense"]).upper()

    umbilicus = document.get("umbilicus", {})
    if not isinstance(umbilicus, Mapping):
        raise ScrollSpecError(f"{source}: umbilicus must be an object")
    unknown = sorted(set(umbilicus) - {"coordinate_scale"})
    if unknown:
        raise ScrollSpecError(f"{source}: unknown umbilicus keys: {unknown}")
    if "coordinate_scale" in umbilicus:
        try:
            fields["umbilicus_coordinate_scale"] = float(umbilicus["coordinate_scale"])
        except (TypeError, ValueError):
            raise ScrollSpecError(f"{source}: umbilicus coordinate_scale must be a number") from None

    if "normal_zarr_group" in document:
        fields["normal_zarr_group"] = str(document["normal_zarr_group"])
    if "lasagna_scale" in document:
        lasagna_scale = document["lasagna_scale"]
        if type(lasagna_scale) is not int or lasagna_scale <= 0:
            raise ScrollSpecError(f"{source}: lasagna_scale must be a positive integer")
        fields["lasagna_scale"] = lasagna_scale

    num_windings = document.get("num_windings")
    # Same floor as shell_outer_winding_idx, which this value becomes.
    if num_windings is not None:
        if type(num_windings) is not int or num_windings < 2:
            raise ScrollSpecError(f"{source}: num_windings must be an integer >= 2")
        fields["num_windings"] = num_windings

    paths = document.get("paths", {})
    if not isinstance(paths, Mapping):
        raise ScrollSpecError(f"{source}: paths must be an object")
    paths = {key: value for key, value in paths.items()
             if key not in RETIRED_PATH_OVERRIDE_KEYS}
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
    if overrides:
        fields["path_overrides"] = tuple(overrides)
    return fields


def _scroll_spec_from_fields(fields: Mapping[str, Any], source: str) -> ScrollSpec:
    """Require the physical facts in ``fields`` and freeze them: the outward
    sense is the catalog properties' when both are known (an explicit one must
    agree), else it must be given."""
    catalog = {key: fields.get(key)
               for key in ("z_direction_is_top_to_bottom", "left_handed_coordinates")}
    catalog_sense = (
        spiral_outward_sense_for(**catalog)
        if None not in catalog.values() else None)
    required = ["name", "voxel_size_um"]
    if catalog_sense is None:
        required.append("spiral_outward_sense")
    missing = sorted(key for key in required if key not in fields)
    if missing:
        raise ScrollSpecError(f"{source}: missing required keys: {missing}")
    sense = fields.get("spiral_outward_sense", catalog_sense)
    if sense not in ("CW", "ACW"):
        raise ScrollSpecError(f"{source}: spiral_outward_sense must be CW or ACW")
    if catalog_sense is not None and sense != catalog_sense:
        raise ScrollSpecError(
            f"{source}: spiral_outward_sense {sense} contradicts the catalog "
            f"properties (z_direction_is_top_to_bottom="
            f"{str(catalog['z_direction_is_top_to_bottom']).lower()}, "
            f"left_handed_coordinates="
            f"{str(catalog['left_handed_coordinates']).lower()} give "
            f"{catalog_sense})")
    return ScrollSpec(
        name=fields["name"],
        voxel_size_um=fields["voxel_size_um"],
        spiral_outward_sense=sense,
        z_direction_is_top_to_bottom=catalog["z_direction_is_top_to_bottom"],
        left_handed_coordinates=catalog["left_handed_coordinates"],
        base_shape_zyx=fields.get("base_shape_zyx"),
        umbilicus_coordinate_scale=fields.get("umbilicus_coordinate_scale", 1.0),
        normal_zarr_group=fields.get("normal_zarr_group", "4"),
        lasagna_scale=fields.get("lasagna_scale", 4),
        num_windings=fields.get("num_windings"),
        path_overrides=fields.get("path_overrides", ()),
    )


def parse_scroll_spec(document: Any, dataset_root: str | os.PathLike[str],
                      *, source: str = SCROLL_SPEC_FILENAME) -> ScrollSpec:
    """Validate known spiral-scroll.json fields and freeze them; the file
    must state every physical fact (``_scroll_spec_from_fields``)."""
    return _scroll_spec_from_fields(
        _scroll_spec_fields(document, dataset_root, source), source)


def _read_scroll_spec_document(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError) as exc:
        raise ScrollSpecError(f"{path}: invalid scroll specification: {exc}") from exc


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
            "either the volume's catalog properties z_direction_is_top_to_bottom "
            "and left_handed_coordinates or spiral_outward_sense (plus any "
            "non-conventional path overrides).")
    return parse_scroll_spec(_read_scroll_spec_document(path),
                             root if root is not None else path.parent,
                             source=str(path))


def load_dataset_scroll_fields(dataset_root: str | os.PathLike[str]) -> tuple[dict[str, Any], str]:
    """A service dataset's spiral-scroll.json fields and the file, or ``({},
    "")`` when it has none: in a service session the project supplies the
    physical facts it can (``SessionScroll``), so the file is optional and may
    state only the rest. An invalid file raises ScrollSpecError."""
    root = Path(_normalise_path(dataset_root))
    path = root / SCROLL_SPEC_FILENAME
    if not path.is_file():
        return {}, ""
    return _scroll_spec_fields(_read_scroll_spec_document(path), root, str(path)), str(path)


# ---------------------------------------------------------------------------
# The VC3D project a service session fits (the request's "project" block)
# ---------------------------------------------------------------------------

# Physical scroll facts a project volume may state, by ScrollSpec field. VC3D
# sends each only when it knows it exactly (a catalog volume's orientation and
# full-resolution shape, say).
PROJECT_VOLUME_FACTS = ("voxel_size_um", "base_shape_zyx",
                        "z_direction_is_top_to_bottom", "left_handed_coordinates")
# Fit inputs a service session reads from the project instead of the dataset.
PROJECT_INPUT_KEYS = ("umbilicus",)


@dataclass(frozen=True)
class ProjectUmbilicus:
    """A volume's umbilicus in the project's umbilicus store: its object, the
    revision VC3D read and its points in the volume's own voxels (ZYX)."""

    uuid: str
    revision: str
    points_zyx: tuple[tuple[float, float, float], ...]

    def manifest(self) -> dict[str, Any]:
        return {"uuid": self.uuid, "revision": self.revision,
                "points": len(self.points_zyx)}

    def to_mapping(self) -> dict[str, Any]:
        return {"uuid": self.uuid, "revision": self.revision,
                "points_zyx": [list(point) for point in self.points_zyx]}


@dataclass(frozen=True)
class ProjectVolume:
    """What the project knows of one volume; None where it does not know it
    exactly."""

    voxel_size_um: float | None = None
    base_shape_zyx: tuple[int, int, int] | None = None
    z_direction_is_top_to_bottom: bool | None = None
    left_handed_coordinates: bool | None = None
    umbilicus: ProjectUmbilicus | None = None

    def manifest(self) -> dict[str, Any]:
        result = {key: getattr(self, key) for key in PROJECT_VOLUME_FACTS
                  if getattr(self, key) is not None}
        if self.umbilicus is not None:
            result["umbilicus"] = self.umbilicus.manifest()
        return result

    def to_mapping(self) -> dict[str, Any]:
        result = {key: getattr(self, key) for key in PROJECT_VOLUME_FACTS
                  if getattr(self, key) is not None}
        if self.umbilicus is not None:
            result["umbilicus"] = self.umbilicus.to_mapping()
        return result


@dataclass(frozen=True)
class SpiralProject:
    """The VC3D project a service session fits. Its file is VC3D's and need
    not be readable by the service, so VC3D sends what the fit reads of it:
    the project's name and, per volume (keyed by the store volume its
    annotations are kept in, vc::project::AnnotationVolume), the facts it knows
    and its umbilicus."""

    name: str
    volumes: Mapping[str, ProjectVolume]

    @classmethod
    def from_mapping(cls, value: Any) -> "SpiralProject":
        if not isinstance(value, Mapping):
            raise ScrollSpecError("project must be an object")
        name = str(value.get("name") or "").strip()
        if not name:
            raise ScrollSpecError("project.name must be a non-empty string")
        entries = value.get("volumes", {})
        if not isinstance(entries, Mapping):
            raise ScrollSpecError("project.volumes must be an object keyed by volume")
        volumes = {}
        for uuid, entry in entries.items():
            source = f"project volume {uuid}"
            if not isinstance(entry, Mapping):
                raise ScrollSpecError(f"{source} must be an object")
            facts = {}
            if entry.get("voxel_size_um") is not None:
                facts["voxel_size_um"] = _checked_voxel_size(entry["voxel_size_um"], source)
            if entry.get("base_shape_zyx") is not None:
                facts["base_shape_zyx"] = _checked_base_shape(entry["base_shape_zyx"], source)
            for key in ("z_direction_is_top_to_bottom", "left_handed_coordinates"):
                if entry.get(key) is not None:
                    facts[key] = _checked_flag(entry[key], key, source)
            umbilicus = entry.get("umbilicus")
            if umbilicus is not None:
                facts["umbilicus"] = _checked_umbilicus(umbilicus, source)
            volumes[str(uuid)] = ProjectVolume(**facts)
        return cls(name=name, volumes=volumes)

    def to_mapping(self) -> dict[str, Any]:
        """The request block ``from_mapping`` reads."""
        return {"name": self.name,
                "volumes": {uuid: volume.to_mapping()
                            for uuid, volume in sorted(self.volumes.items())}}

    def record(self, volume_uuid: str) -> dict[str, Any]:
        """What a session fitting ``volume_uuid`` records of the project."""
        return {"name": self.name, "volume": volume_uuid,
                **self.volumes[volume_uuid].manifest()}


def _checked_umbilicus(value: Any, source: str) -> ProjectUmbilicus:
    if not isinstance(value, Mapping):
        raise ScrollSpecError(f"{source}: umbilicus must be an object")
    uuid = str(value.get("uuid") or "").strip()
    revision = str(value.get("revision") or "").strip()
    if not uuid or not revision:
        raise ScrollSpecError(f"{source}: umbilicus needs its uuid and revision")
    points = value.get("points_zyx")
    try:
        points = tuple(tuple(float(coordinate) for coordinate in point) for point in points)
    except (TypeError, ValueError):
        points = ()
    # The fit interpolates z -> yx between control points: two at least.
    if len(points) < 2 or any(len(point) != 3 or not all(map(math.isfinite, point))
                              for point in points):
        raise ScrollSpecError(
            f"{source}: umbilicus points_zyx must be two or more finite ZYX points")
    return ProjectUmbilicus(uuid=uuid, revision=revision, points_zyx=points)


def _same_fact(stated: Any, known: Any) -> bool:
    if isinstance(known, float):
        return math.isclose(float(stated), known, rel_tol=1e-6)
    return stated == known


@dataclass(frozen=True)
class SessionScroll:
    """A service session's scroll before its fit volume is known: the
    project's facts and the dataset's optional spiral-scroll.json fields.
    ``resolve`` completes it once the inputs name the fit volume."""

    project: SpiralProject
    spec_fields: Mapping[str, Any] = field(default_factory=dict)
    spec_source: str = ""

    # The packed Lasagna inputs remain the dataset's, so their layout is the
    # dataset's specification's (or the default).
    @property
    def normal_zarr_group(self) -> str:
        return str(self.spec_fields.get("normal_zarr_group", "4"))

    @property
    def lasagna_scale(self) -> int:
        return int(self.spec_fields.get("lasagna_scale", 4))

    def volume(self, volume_uuid: str) -> ProjectVolume:
        try:
            return self.project.volumes[volume_uuid]
        except KeyError:
            raise ScrollSpecError(
                f"the fit volume {volume_uuid} is not a volume of the project "
                f"{self.project.name!r}: attach it to the project") from None

    def resolve(self, volume_uuid: str) -> ScrollSpec:
        """The scroll of a fit of ``volume_uuid``: the project volume's facts,
        the specification's where the project does not know them (a
        specification stating one differently is refused, not overruled),
        and the project's name. The umbilicus is the project's, in the fit
        volume's own voxels."""
        volume = self.volume(volume_uuid)
        fields = dict(self.spec_fields)
        for key in PROJECT_VOLUME_FACTS:
            known = getattr(volume, key)
            if known is None:
                continue
            stated = fields.get(key)
            if stated is not None and not _same_fact(stated, known):
                raise ScrollSpecError(
                    f"{self.spec_source} states {key} {stated!r}, but the project's "
                    f"volume {volume_uuid} has {known!r}; remove it from "
                    f"{SCROLL_SPEC_FILENAME}")
            fields[key] = known
        fields["name"] = self.project.name
        fields["umbilicus_coordinate_scale"] = 1.0
        source = f"project {self.project.name!r} (volume {volume_uuid})"
        if self.spec_source:
            source += f" with {self.spec_source}"
        return _scroll_spec_from_fields(fields, source)

    def umbilicus(self, volume_uuid: str) -> ProjectUmbilicus:
        umbilicus = self.volume(volume_uuid).umbilicus
        if umbilicus is None:
            raise ScrollSpecError(
                f"the project {self.project.name!r} has no umbilicus for the fit "
                f"volume {volume_uuid}: import one (File > Import Umbilicus)")
        return umbilicus


def write_project_umbilicus(directory: str | os.PathLike[str], volume_uuid: str,
                            umbilicus: ProjectUmbilicus) -> str:
    """Write ``umbilicus`` under ``directory`` as the control-point JSON the
    fit reads (umbilicus.json_umbilicus_z_to_yx), in the fit volume's voxels;
    returns its path. Named by its revision: a session reads the revision it
    was built with."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"umbilicus-{umbilicus.revision}.json"
    document = {
        "control_points": [{"z": z, "y": y, "x": x} for z, y, x in umbilicus.points_zyx],
        "source": {"volume_uuid": volume_uuid, "object_uuid": umbilicus.uuid,
                   "revision": umbilicus.revision},
    }
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(document), encoding="utf-8")
    temporary.replace(path)
    return str(path)


def scroll_spec_config_defaults(spec: ScrollSpec | Mapping[str, Any] | None,
                                base_config: Mapping[str, Any]) -> dict[str, Any]:
    """Fit-config defaults a scroll specification supplies.

    The generic default for the outer winding and the exporters' physical
    winding estimate is Scroll 1's count (130), which on a scroll with fewer
    windings exports windings that lie outside the papyrus. An optional
    num_windings in spiral-scroll.json replaces that default for this scroll.
    The gap-lattice capacity is raised only when the count would not fit.

    Callers apply the result between the Python defaults and any explicit
    configuration, so an explicit setting still wins. Accepts a ScrollSpec or
    its manifest() dict; returns {} when the scroll gives no count.
    """
    if spec is None:
        return {}
    num_windings = (spec.get("num_windings") if isinstance(spec, Mapping)
                    else spec.num_windings)
    if num_windings is None:
        return {}
    defaults = {
        "shell_outer_winding_idx": num_windings,
        "model_gap_expander_num_windings": num_windings,
    }
    capacity = base_config.get("model_gap_expander_capacity_windings")
    if capacity is not None and capacity < num_windings + 3:
        defaults["model_gap_expander_capacity_windings"] = num_windings + 3
    return defaults

def conventional_input_paths(
        dataset_root: str | os.PathLike[str], spec: ScrollSpec, *,
        output_directory: str = "", cache_directory: str = "",
        checkpoint: str = "") -> SpiralInputPaths:
    """Resolve the conventional dataset layout (plus spec overrides) for the
    headless CLI, mirroring the historical fit_spiral module-global defaults.

    Patches, fibers and point collections are not paths: the CLI reads them
    from an object store (config object_store_path), never from the dataset.

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

    return SpiralInputPaths(
        dataset_root=str(root),
        umbilicus=resolve("umbilicus"),
        fiber_directions=resolve("fiber_directions"),
        tracks_dbm=resolve("tracks_dbm"),
        outer_shell=resolve("outer_shell"),
        normal_x=resolve("normal_x"),
        normal_y=resolve("normal_y"),
        gradient_magnitude=resolve("gradient_magnitude"),
        winding_inference=resolve("winding_inference"),
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
# Autosave metadata and selection
# ---------------------------------------------------------------------------
#
# A pause writes ``checkpoint_autosave.ckpt`` into that run's output directory
# and, beside it, a metadata sidecar naming what the file *is*. An
# selector can identify it from those sidecars alone: a bare ``.ckpt`` with no
# metadata is never selected, and the winner is the one with the most
# completed iterations, never the last filename in sort order. The interactive
# service advertises autosaves as checkpoints but does not select one itself.

AUTOSAVE_CHECKPOINT_NAME = "checkpoint_autosave.ckpt"
# Both the headless fitter and the interactive runtime refresh their
# resumable checkpoint at this cadence, in addition to any final write.
AUTOSAVE_INTERVAL_ITERATIONS = 1000
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
    model_state_sha256: str | None = None,
    input_digest: str | None = None,
    input_position: Mapping[str, Any] | None = None,
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
    if model_state_sha256:
        # The surface identity the checkpoint payload carries, readable
        # without loading the archive.
        document["model_state_sha256"] = str(model_state_sha256)
    if input_digest:
        # The object-store inputs it was fitted to (the checkpoint's
        # input_manifest.object_store_digest), likewise.
        document["input_digest"] = str(input_digest)
    if input_position:
        # The hub position they were read at ({hub, sequence}).
        document["input_position"] = dict(input_position)
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
    """Choose the newest matching autosave from validated metadata.

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

    # A service session's project supplies the physical facts it knows
    # (SessionScroll), so the scroll specification is optional and may state
    # only the rest; an invalid one does not resolve.
    overrides = {}
    try:
        fields, _ = load_dataset_scroll_fields(root)
    except ScrollSpecError as exc:
        result.missing_required.append("scroll_spec")
        result.warnings.append(str(exc))
    else:
        overrides = dict(fields.get("path_overrides", ()))
        result.scroll_spec = {**fields, "path_overrides": overrides}

    for input_spec in FIT_INPUT_CATALOG:
        # DBM inputs need backing-file probing (below); the project supplies
        # PROJECT_INPUT_KEYS.
        if (input_spec.kind == "dbm" or not input_spec.conventional_relative
                or input_spec.key in PROJECT_INPUT_KEYS):
            continue
        override = overrides.get(input_spec.key, "")
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

    tracks_override = overrides.get("tracks_dbm", "")
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


def _validate_json_file(path: Path, label: str, errors: list[dict[str, str]]) -> None:
    try:
        with path.open("r", encoding="utf-8") as stream:
            json.load(stream)
    except Exception as exc:
        errors.append({"field": label, "message": f"Invalid JSON: {exc}"})


def validate_session_request(
    paths: SpiralInputPaths,
    run: SpiralRunConfig,
    *,
    supplied: Iterable[str] = (),
) -> list[dict[str, str]]:
    """Perform cheap, aggregate validation before any GPU allocation.

    ``supplied`` names inputs another source provides once the fit volume is
    known (a service session's PROJECT_INPUT_KEYS); they are not checked here.
    """
    supplied = frozenset(supplied)
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
        if not spec.enabled(run.config) or spec.key in supplied:
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

    # The Lasagna store requirements (see the catalog's predicates) are
    # checked after the dense-spacing mode below, so an invalid mode errors
    # as itself rather than as missing-file errors.
    for spec in FIT_INPUT_CATALOG:
        if spec.kind != "zarr-group":
            check_catalog_input(spec)

    if _dense_spacing_mode(run.config) is None:
        errors.append({"field": "dense_spacing_mode",
                       "message": "Must be grad_mag or winding_model"})

    for spec in FIT_INPUT_CATALOG:
        if spec.kind == "zarr-group":
            check_catalog_input(spec)

    if _winding_model_enabled(run.config) and paths.winding_inference:
        manifest = Path(paths.winding_inference) / "manifest.json"
        if not manifest.is_file():
            errors.append({"field": "winding_inference",
                           "message": "manifest.json is missing"})
        else:
            try:
                metadata = json.loads(manifest.read_text())
                if metadata.get("artifact_type") != "winding_inference_crossings":
                    raise ValueError("unexpected artifact_type")
                if int(metadata.get("format_version", -1)) != 1:
                    raise ValueError("unsupported format_version")
            except Exception as exc:
                errors.append({"field": "winding_inference",
                               "message": f"Invalid inference manifest: {exc}"})

    if run.z_begin >= run.z_end:
        errors.append({"field": "z_range", "message": "z_begin must be less than z_end"})
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
