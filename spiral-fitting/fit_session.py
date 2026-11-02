"""Session-facing configuration and input resolution for Spiral fits.

This module intentionally has no Torch, Zarr, or VC imports, so the service
and the command-line fit can resolve and validate their inputs before
importing the comparatively expensive fitting stack. Both resolve into one
input contract (SpiralInputPaths and a ScrollSpec, checked by
validate_fit_inputs): the command line from an explicit input manifest
(load_input_manifest), a service from the client's project (SessionScroll)
and its own host bindings (spiral_service.ServiceBindings).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
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
from lasagna_source import (LASAGNA_CHANNELS, LasagnaSource, LasagnaSourceError,
                            check_identity, fiber_directions_path,
                            is_remote_location, lasagna_scale_for,
                            load_lasagna_source, read_lasagna_manifest,
                            url_identity, write_lasagna_source)
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
# 41: the project's Lasagna dataset and the outward sense. The project block
# carries "lasagna" ({location, group, normal_x, normal_y, gradient_magnitude}
# stores, or {location, error}) and each volume its "lasagna_scale"; the
# session's Lasagna stores, group and scale are the project's, and /dataset no
# longer resolves them. The fit setting spiral_outward_sense gives the sense
# where the fit volume's orientation does not (and must agree where it does).
# A service no longer reads spiral-scroll.json's scroll facts or Lasagna
# layout (they are the command-line fit's); its path overrides and
# num_windings still apply.
# 42: the outer shell is the object store's surface tagged spiral-outer-shell,
# read with the selection (never a patch). /dataset no longer resolves an
# outer_shell path; the input manifest carries "outer_shell" (its provenance,
# or null) and input_summary its UUID.
# 43: Lasagna datasets by identity. The project block's "lasagna" carries a
# local dataset's "identity" (its provenance run_uuid, else sha256-<the
# manifest's SHA-256>); a remote one's is its manifest URL's, which the
# service computes (lasagna_source.url_identity). The fit reads the resident
# pools of that identity from the service's cache (<cache>/lasagna/<identity>/)
# and never builds them, refusing a build whose packs are missing. paths
# carries one "lasagna" (the dataset's source document) instead of normal_x,
# normal_y and gradient_magnitude. It also carries "fiber_inference"
# ({location, identity} of the selected fiber-inference dataset, identified
# the same way, or {location, error}) and each
# volume its "fiber_scale" (base voxels of that dataset per volume voxel); a
# fit with the fiber-direction loss reads the samples extracted for that
# identity and scale from the cache (<cache>/fiber-directions/), not the
# dataset's fiber_directions.npz.
# 44: tracks come from the object store. paths no longer carries "tracks_dbm"
# (neither /dataset nor a load request names a track database); the fit reads
# the selected tracks (object_store_track_uuids/_tags, input_use_tracks) and
# their crossing links, which its link refresh makes. The input manifest lists
# tracks and track_crossing relationships like any other input.
# 45: explicit inputs. A service has no dataset: --dataset, spiral-scroll.json
# and conventional-directory discovery are gone. GET /dataset is replaced by
# GET /bindings, which advertises the service's host bindings ("bindings":
# output_directory, cache_directory and winding_inference {bound, path,
# available, error}), capabilities.winding_inference, the session checkpoints,
# the object store and its selection. There are no dataset-root checkpoints
# (detected_checkpoints), and /health carries no dataset_root. Winding
# inference is read from the host path bound at launch (--winding-inference),
# never one a client names; a fit needing it without a usable binding fails
# preparation. A request's paths carry only "checkpoint"; a session's paths
# record the fit volume ("volume_uuid") and the inputs' "provenance" instead
# of dataset_root. The configuration catalog takes no num_windings from a
# host scroll specification.
API_VERSION = 45


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
    "tracks": "input_use_tracks",
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
# One description of every fit input read by path, consumed by input
# validation (validate_fit_inputs, for the command line and a service alike)
# and run admission (spiral_service.ServiceState.run). Patches, point
# collections, fibers, tracks and the outer shell are not here: they come
# from the object store (object_store_inputs). Each entry records the input's path kind
# and its enabling/required predicates over the run configuration. Fit-input
# paths are static for the lifetime of a resident session, so an entry says
# nothing about rebuild scope: changing any of them rebuilds the session.


def _always(config: Mapping[str, Any]) -> bool:
    return True


def _never(config: Mapping[str, Any]) -> bool:
    return False


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


def fiber_directions_required(config: Mapping[str, Any]) -> bool:
    return (input_source_enabled(config, "fiber_directions")
            and float(config.get("loss_weight_fiber_directions", 0.0)) > 0)


def _grad_mag_required(config: Mapping[str, Any]) -> bool:
    return (input_source_enabled(config, "gradient_magnitude")
            and _dense_spacing_mode(config) == "grad_mag"
            and float(config.get("loss_weight_dense_spacing", 12.0)) > 0)


def lasagna_enabled(config: Mapping[str, Any]) -> bool:
    return (input_source_enabled(config, "normals")
            or input_source_enabled(config, "gradient_magnitude"))


def _lasagna_required(config: Mapping[str, Any]) -> bool:
    return _normals_required(config) or _grad_mag_required(config)


def lasagna_packs_required(config: Mapping[str, Any]) -> list[str]:
    """The Lasagna packs (lasagna_source.PACK_INPUTS) a fit of ``config``
    loads."""
    return [kind for kind, required in (("normals", _normals_required(config)),
                                        ("grad_mag", _grad_mag_required(config)))
            if required]


def _winding_model_enabled(config: Mapping[str, Any]) -> bool:
    return (
        _dense_spacing_mode(config) == "winding_model"
        and input_source_enabled(config, "winding_inference")
        and input_source_enabled(config, "outer_shell")
    )


def outer_shell_required(config: Mapping[str, Any]) -> bool:
    """Whether the fit needs the outer shell (the object store's surface
    tagged spiral-outer-shell): shell losses or winding-model supervision."""
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
    # "file" | "directory" | "lasagna".
    kind: str
    # File contents must parse as JSON.
    json_content: bool = False
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
    FitInputSpec("umbilicus", "file", json_content=True, required=_always),
    FitInputSpec("fiber_directions", "file",
                 enabled=fiber_directions_required,
                 required=fiber_directions_required),
    # The Lasagna dataset (lasagna_source): a *.lasagna.json manifest, or the
    # source document write_lasagna_source wrote for it.
    FitInputSpec("lasagna", "lasagna",
                 enabled=lasagna_enabled, required=_lasagna_required),
    FitInputSpec("winding_inference", "directory",
                 enabled=_winding_model_enabled,
                 required=_winding_model_enabled),
)

_FIT_INPUTS_BY_KEY = {spec.key: spec for spec in FIT_INPUT_CATALOG}


def fit_input(key: str) -> FitInputSpec | None:
    return _FIT_INPUTS_BY_KEY.get(key)


@dataclass(frozen=True)
class SpiralInputPaths:
    """The resolved inputs a fit reads by path; the object store supplies the
    rest. The command line (load_input_manifest) and a service (its project
    and host bindings) resolve into this one structure, which session and
    checkpoint metadata record.

    ``volume_uuid`` is the reference volume the inputs are in: the fit volume
    the selected store objects must name (check_fit_volume). ``provenance``
    says where the inputs came from (the manifest and its digest, or the
    project and the service's bindings), for the record only."""

    volume_uuid: str = ""
    umbilicus: str = ""
    fiber_directions: str = ""
    lasagna: str = ""
    winding_inference: str = ""
    scroll_zarr: str = ""
    checkpoint: str = ""
    output_directory: str = ""
    cache_directory: str = ""
    provenance: Mapping[str, Any] = field(default_factory=dict)

    # Fields that are not host paths.
    _RECORD_FIELDS = ("volume_uuid", "provenance")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SpiralInputPaths":
        # Unknown keys (an older manifest's dataset root, or its patch, fiber
        # and PCL paths) are not inputs any more. URLs remain URLs.
        fields = {name: (str(value[name]).strip()
                         if is_remote_location(value.get(name) or "")
                         else _normalise_path(value.get(name)))
                  for name in cls.__dataclass_fields__
                  if name not in cls._RECORD_FIELDS}
        return cls(volume_uuid=str(value.get("volume_uuid") or ""),
                   provenance=dict(value.get("provenance") or {}), **fields)

    def manifest(self) -> dict[str, Any]:
        result = asdict(self)
        result["provenance"] = json.loads(json.dumps(self.provenance))
        return result


def lasagna_normal_inputs(source: LasagnaSource | None, config: Mapping[str, Any], scale,
                          cache_directory: str | os.PathLike[str] | None):
    """A fit's lasagna normals as object_store_inputs.fiber_normal_source
    takes them, or None when it has none (no dataset, input_use_normals off,
    or no normals pack of the dataset in the cache): the pack and its
    array's downsampling relative to the fit volume."""
    if source is None or not cache_directory or not input_source_enabled(config, "normals"):
        return None
    pack = source.pack_directory(cache_directory, "normals")
    if not os.path.isfile(os.path.join(pack, "meta.json")):
        return None
    return {'pool': pack, 'scale': scale}


# Run-block keys the scroll (a service session's project) owns, mapped to the
# ScrollSpec field each one used to shadow. A request that carries one is
# rejected rather than ignored: silently dropping it would let a client
# believe it had renamed a scroll, changed its resolution, or moved the
# Lasagna stores.
SCROLL_SPEC_OWNED_RUN_KEYS = {
    "scroll_name": "name",
    "voxel_size_um": "voxel_size_um",
    "lasagna_group": "lasagna.group",
    "lasagna_scale": "lasagna_scale",
}


@dataclass(frozen=True)
class SpiralRunConfig:
    """Deployment and presentation settings of one fit run.

    Nothing the scroll specifies is here — not its physical facts (name,
    voxel size, outward sense) and not the Lasagna dataset's layout. The
    resolved inputs are the single source for all of it (see ``ScrollSpec``).
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


# ---------------------------------------------------------------------------
# Scroll facts and the command-line input manifest
# ---------------------------------------------------------------------------

# The command-line fit's explicit input manifest (load_input_manifest).
INPUT_MANIFEST_SCHEMA_VERSION = 1


class ScrollSpecError(ValueError):
    """Missing, malformed or contradictory scroll facts or fit inputs."""


@dataclass(frozen=True)
class ScrollSpec:
    """Physical facts of one scroll: the input manifest's (command line) or
    the project volume's (service).

    Torch-free and frozen: safe to resolve in the VC3D-facing service process
    and to pickle into GPU worker processes. Deployment and presentation
    values (output/cache roots, run tags, storage backend, render scale) are
    deliberately not part of it.
    """

    name: str
    voxel_size_um: float
    spiral_outward_sense: str
    # The following are in the open-data catalog metadata; together they
    # fix spiral_outward_sense (surface_orientation.spiral_outward_sense_for);
    # the z direction also decides whether exported grids are flipped so row 0 is
    # the scroll top. None when the inputs leave them out.
    z_direction_is_top_to_bottom: bool | None = None
    left_handed_coordinates: bool | None = None
    # Shape of the volume coordinate domain used by Spiral surfaces. This is
    # independent of physical voxel size and may be unknown.
    base_shape_zyx: tuple[int, int, int] | None = None
    umbilicus_coordinate_scale: float = 1.0
    # The Lasagna arrays' downsampling relative to the fit volume; the arrays
    # themselves are the Lasagna dataset's (lasagna_source). None when the
    # inputs do not state it: resolve_lasagna_scale derives it from the
    # dataset's frame where it can, and requires it where it cannot.
    lasagna_scale: int | None = None
    # The scroll's winding count, when someone has estimated it. A default for
    # the fit, not a fact the scroll owns: see scroll_spec_config_defaults.
    num_windings: int | None = None

    def manifest(self) -> dict[str, Any]:
        return asdict(self)


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


# The keys of the input manifest's scroll block, as ScrollSpec fields.
SCROLL_KEYS = ("name", "voxel_size_um", "base_shape_zyx", "spiral_outward_sense",
               "z_direction_is_top_to_bottom", "left_handed_coordinates",
               "lasagna_scale", "num_windings")


def _scroll_fields(document: Any, source: str) -> dict[str, Any]:
    """The ScrollSpec fields a scroll block states, validated but none
    required (``_scroll_spec_from_fields`` requires them)."""
    if not isinstance(document, Mapping):
        raise ScrollSpecError(f"{source} must be an object")
    unknown = sorted(set(document) - set(SCROLL_KEYS))
    if unknown:
        raise ScrollSpecError(
            f"{source}: unknown keys {unknown} (allowed: {sorted(SCROLL_KEYS)})")
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
    return fields


def _scroll_spec_from_fields(fields: Mapping[str, Any], source: str, *,
                             outward_sense: str | None = None) -> ScrollSpec:
    """Require the physical facts in ``fields`` and freeze them: the outward
    sense is the catalog properties' when both are known (a stated one, or the
    configured ``outward_sense`` fit setting, must agree), else the stated or
    configured one (which must agree with each other)."""
    catalog = {key: fields.get(key)
               for key in ("z_direction_is_top_to_bottom", "left_handed_coordinates")}
    catalog_sense = (
        spiral_outward_sense_for(**catalog)
        if None not in catalog.values() else None)
    required = ["name", "voxel_size_um"]
    if catalog_sense is None and outward_sense is None:
        required.append("spiral_outward_sense")
    missing = sorted(key for key in required if key not in fields)
    if missing:
        raise ScrollSpecError(
            f"{source}: missing required keys: {missing}"
            + (" (or set the fit setting spiral_outward_sense)"
               if "spiral_outward_sense" in missing else ""))
    stated = fields.get("spiral_outward_sense")
    if stated is not None and outward_sense is not None and stated != outward_sense:
        raise ScrollSpecError(
            f"{source}: spiral_outward_sense {stated} contradicts the fit setting "
            f"spiral_outward_sense {outward_sense}")
    sense = stated or outward_sense or catalog_sense
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
        lasagna_scale=fields.get("lasagna_scale"),
        num_windings=fields.get("num_windings"),
    )


def parse_scroll(document: Any, *, source: str = "scroll",
                 outward_sense: str | None = None) -> ScrollSpec:
    """Validate an input manifest's scroll block and freeze it; it must state
    every physical fact except an outward sense the fit setting
    ``outward_sense`` gives (``_scroll_spec_from_fields``)."""
    return _scroll_spec_from_fields(_scroll_fields(document, source), source,
                                    outward_sense=outward_sense)


def resolve_lasagna_scale(spec: ScrollSpec, source: LasagnaSource) -> ScrollSpec:
    """``spec`` with the downsampling of the Lasagna dataset ``source``
    relative to the fit volume: derived from the volume's ``base_shape_zyx``
    and the dataset's frame where both are known (lasagna_scale_for), a
    stated ``lasagna_scale`` then only a cross-check that must agree; else
    the stated one, which is then required. The derivation takes the two
    grids to cover one aligned extent."""
    derived, how = lasagna_scale_for(source, spec.base_shape_zyx)
    if derived is None:
        if spec.lasagna_scale is None:
            raise ScrollSpecError(
                f"cannot derive the Lasagna arrays' downsampling relative to the fit "
                f"volume: {how}; state scroll.lasagna_scale in the input manifest, "
                "or the volume's scroll.base_shape_zyx there and the dataset's in "
                "its Lasagna manifest")
        return spec
    if spec.lasagna_scale is not None and spec.lasagna_scale != derived:
        raise ScrollSpecError(
            f"lasagna_scale {spec.lasagna_scale} contradicts the downsampling of "
            f"{source.location} relative to the fit volume, {derived} ({how})")
    return replace(spec, lasagna_scale=derived)


# The input manifest's top-level keys. Geometry (patches, point collections,
# fibers, tracks and the outer shell) is the object store's alone: it has no
# entry here.
INPUT_MANIFEST_KEYS = ("schema_version", "volume_uuid", "scroll", "umbilicus",
                       "lasagna", "fiber_directions", "winding_inference",
                       "scroll_zarr")
# Keys a manifest may carry by mistake, with what to do instead: a former
# spiral-scroll.json's, and geometry the object store holds.
_MISPLACED_MANIFEST_KEYS = {
    **{key: "it belongs in the scroll block" for key in SCROLL_KEYS},
    "umbilicus_coordinate_scale": "state it as umbilicus.coordinate_scale",
    "paths": ("a spiral-scroll.json's path overrides are retired: name each "
              "input at the top level, relative to the manifest"),
    "normal_zarr_group": "the Lasagna manifest (lasagna.manifest) names the arrays",
    **{key: "geometry is read from --object-store only"
       for key in ("verified_patches", "patches", "fibers", "pcls",
                   "point_collections", "outer_shell", "tracks", "tracks_dbm")},
}


def _manifest_path(value: Any, base: Path, key: str, source: str, *,
                   remote: bool = False) -> str:
    """One input-manifest location: a filesystem path, resolved against the
    manifest's directory when relative, or (where ``remote``) a URL, kept
    as it is."""
    if not isinstance(value, str) or not value.strip():
        raise ScrollSpecError(f"{source}: {key} must be a non-empty string")
    if is_remote_location(value):
        if not remote:
            raise ScrollSpecError(
                f"{source}: {key} must be a local path, not the URL {value.strip()}")
        return value.strip()
    return _normalise_path(value, base=base)


def _manifest_block(document: Mapping[str, Any], key: str, allowed: tuple[str, ...],
                    source: str) -> Mapping[str, Any] | None:
    block = document.get(key)
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise ScrollSpecError(f"{source}: {key} must be an object")
    unknown = sorted(set(block) - set(allowed))
    if unknown:
        raise ScrollSpecError(
            f"{source}: unknown {key} keys {unknown} (allowed: {sorted(allowed)})")
    return block


def load_input_manifest(path: str | os.PathLike[str], *,
                        cache_directory: str | os.PathLike[str],
                        outward_sense: str | None = None,
                        ) -> tuple[ScrollSpec, SpiralInputPaths]:
    """Resolve a command-line fit's explicit input manifest into its scroll
    and the inputs it reads by path (``(ScrollSpec, SpiralInputPaths)``).

    Every input is named; nothing is discovered. Relative paths resolve
    against the manifest's directory, so a fit does not depend on its working
    directory. ``volume_uuid``, ``scroll`` and ``umbilicus`` are required;
    the supervision inputs are optional here and required by
    validate_fit_inputs when the configuration's losses read them. The scroll
    must decide its outward sense, unless the fit setting ``outward_sense``
    does (and then agree with it).

    ``lasagna.manifest`` names the local Lasagna manifest. With
    ``lasagna.original_url`` it is a local copy of the remote dataset there:
    the fit reads it as that dataset (its identity, so its packs are the ones
    a service fitting the URL shares), from a source document written under
    ``cache_directory``. Nothing is downloaded.
    """
    manifest_path = Path(_normalise_path(path))
    source = str(manifest_path)
    try:
        data = manifest_path.read_bytes()
        document = json.loads(data)
    except (OSError, ValueError) as exc:
        raise ScrollSpecError(f"{source}: cannot read the input manifest: {exc}") from exc
    if not isinstance(document, Mapping):
        raise ScrollSpecError(f"{source}: the input manifest must be a JSON object")
    unknown = sorted(set(document) - set(INPUT_MANIFEST_KEYS))
    if unknown:
        raise ScrollSpecError(
            f"{source}: unknown input-manifest keys: "
            + "; ".join(f"{key} ({_MISPLACED_MANIFEST_KEYS[key]})"
                        if key in _MISPLACED_MANIFEST_KEYS else key
                        for key in unknown)
            + f" (allowed: {sorted(INPUT_MANIFEST_KEYS)})")
    if document.get("schema_version") != INPUT_MANIFEST_SCHEMA_VERSION:
        raise ScrollSpecError(
            f"{source}: schema_version must be {INPUT_MANIFEST_SCHEMA_VERSION} "
            f"(got {document.get('schema_version')!r})")
    volume_uuid = document.get("volume_uuid")
    if not isinstance(volume_uuid, str) or not volume_uuid.strip():
        raise ScrollSpecError(
            f"{source}: volume_uuid must name the reference volume the inputs "
            "are in (the volume the selected store objects declare)")
    if "scroll" not in document:
        raise ScrollSpecError(f"{source}: the scroll block is required")
    fields = _scroll_fields(document["scroll"], f"{source}: scroll")
    base = manifest_path.parent

    umbilicus = _manifest_block(document, "umbilicus", ("path", "coordinate_scale"),
                                source)
    if umbilicus is None:
        raise ScrollSpecError(f"{source}: umbilicus (its path) is required")
    umbilicus_path = _manifest_path(umbilicus.get("path"), base, "umbilicus.path", source)
    if "coordinate_scale" in umbilicus:
        scale = umbilicus["coordinate_scale"]
        if (isinstance(scale, bool) or not isinstance(scale, (int, float))
                or not (math.isfinite(scale) and scale > 0)):
            raise ScrollSpecError(
                f"{source}: umbilicus.coordinate_scale must be a positive number")
        fields["umbilicus_coordinate_scale"] = float(scale)
    scroll = _scroll_spec_from_fields(fields, f"{source}: scroll",
                                      outward_sense=outward_sense)

    provenance = {"input_manifest": source,
                  "sha256": hashlib.sha256(data).hexdigest()}
    lasagna = ""
    block = _manifest_block(document, "lasagna", ("manifest", "original_url"), source)
    if block is not None:
        lasagna = _manifest_path(block.get("manifest"), base, "lasagna.manifest", source)
        url = block.get("original_url")
        if url is not None:
            if not isinstance(url, str) or not is_remote_location(url):
                raise ScrollSpecError(
                    f"{source}: lasagna.original_url must be the URL of the remote "
                    "dataset the local manifest is a copy of")
            provenance["lasagna_original_url"] = url.strip()
            # A missing copy is reported, when the fit needs it, by
            # validate_fit_inputs.
            if os.path.isfile(lasagna):
                try:
                    copy = read_lasagna_manifest(lasagna, url=url.strip())
                except LasagnaSourceError as exc:
                    raise ScrollSpecError(f"{source}: lasagna: {exc}") from None
                lasagna = write_lasagna_source(
                    copy.dataset_directory(cache_directory), copy)
    optional = {key: (_manifest_path(document[key], base, key, source,
                                     remote=key == "scroll_zarr")
                      if document.get(key) is not None else "")
                for key in ("fiber_directions", "winding_inference", "scroll_zarr")}
    return scroll, SpiralInputPaths(volume_uuid=volume_uuid.strip(),
                                    umbilicus=umbilicus_path, lasagna=lasagna,
                                    provenance=provenance, **optional)


def check_fit_volume(paths: SpiralInputPaths, volume_uuid: str | None) -> None:
    """Refuse inputs resolved in another reference volume than the fit volume
    ``volume_uuid`` the selected store objects name."""
    if paths.volume_uuid and volume_uuid and paths.volume_uuid != volume_uuid:
        raise ScrollSpecError(
            f"the inputs are in the volume {paths.volume_uuid}, but the selected "
            f"store objects are fitted in {volume_uuid}: name that volume's inputs, "
            "or select objects of this one")


# ---------------------------------------------------------------------------
# The VC3D project a service session fits (the request's "project" block)
# ---------------------------------------------------------------------------

# Physical scroll facts a project volume may state, by ScrollSpec field. VC3D
# sends each only when it knows it exactly (a catalog volume's orientation and
# full-resolution shape, say).
PROJECT_VOLUME_FACTS = ("voxel_size_um", "base_shape_zyx",
                        "z_direction_is_top_to_bottom", "left_handed_coordinates")
# The project's Lasagna dataset, by fit input.
PROJECT_LASAGNA_INPUTS = ("lasagna",)
# Fit inputs a service session reads from the project: the Lasagna dataset, and once the build knows the fit volume its umbilicus
# and the fiber directions extracted at its scale (VOLUME_INPUT_KEYS).
VOLUME_INPUT_KEYS = ("umbilicus", "fiber_directions")
PROJECT_INPUT_KEYS = (*PROJECT_LASAGNA_INPUTS, *VOLUME_INPUT_KEYS)


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
class ProjectLasagna:
    """The project's selected Lasagna dataset as the fit reads it: its
    identity (lasagna_source), its manifest's location, the OME-Zarr stores
    of its nx, ny and grad_mag channels and the array (pyramid group) of each
    that the dataset declares; ``error`` says why the fit cannot read it (a
    multichannel dataset, say). The fit reads the identity's packs; the
    stores are where VC3D found the dataset."""

    location: str = ""
    identity: str = ""
    stores: Mapping[str, str] = field(default_factory=dict)  # LASAGNA_CHANNELS
    group: str = ""
    error: str = ""

    def manifest(self) -> dict[str, Any]:
        result = {"location": self.location}
        if self.error:
            result["error"] = self.error
        else:
            result.update(identity=self.identity, group=self.group,
                          **dict(sorted(self.stores.items())))
        return result

    to_mapping = manifest

    def source(self) -> LasagnaSource:
        return LasagnaSource(identity=self.identity, location=self.location,
                             group=self.group, stores=dict(self.stores))


@dataclass(frozen=True)
class ProjectFiberInference:
    """The project's selected fiber-inference dataset: its identity
    (lasagna_source), keying the fiber-direction samples extracted from it;
    ``error`` says why VC3D could not read it."""

    location: str = ""
    identity: str = ""
    error: str = ""

    def manifest(self) -> dict[str, Any]:
        result = {"location": self.location}
        if self.error:
            result["error"] = self.error
        else:
            result["identity"] = self.identity
        return result

    to_mapping = manifest


@dataclass(frozen=True)
class ProjectVolume:
    """What the project knows of one volume; None where it does not know it
    exactly. ``lasagna_scale`` is the project's Lasagna arrays' downsampling
    relative to this volume's voxels; ``fiber_scale`` the base voxels of the
    project's fiber-inference dataset per voxel of this volume."""

    voxel_size_um: float | None = None
    base_shape_zyx: tuple[int, int, int] | None = None
    z_direction_is_top_to_bottom: bool | None = None
    left_handed_coordinates: bool | None = None
    lasagna_scale: int | None = None
    fiber_scale: float | None = None
    umbilicus: ProjectUmbilicus | None = None

    def _facts(self) -> dict[str, Any]:
        return {key: getattr(self, key)
                for key in (*PROJECT_VOLUME_FACTS, "lasagna_scale", "fiber_scale")
                if getattr(self, key) is not None}

    def manifest(self) -> dict[str, Any]:
        result = self._facts()
        if self.umbilicus is not None:
            result["umbilicus"] = self.umbilicus.manifest()
        return result

    def to_mapping(self) -> dict[str, Any]:
        result = self._facts()
        if self.umbilicus is not None:
            result["umbilicus"] = self.umbilicus.to_mapping()
        return result


@dataclass(frozen=True)
class SpiralProject:
    """The VC3D project a service session fits. Its file is VC3D's and need
    not be readable by the service, so VC3D sends what the fit reads of it:
    the project's name, its selected Lasagna dataset and, per volume (keyed by
    the store volume its annotations are kept in,
    vc::project::AnnotationVolume), the facts it knows and its umbilicus."""

    name: str
    volumes: Mapping[str, ProjectVolume]
    lasagna: ProjectLasagna | None = None
    fiber_inference: ProjectFiberInference | None = None

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
            if entry.get("lasagna_scale") is not None:
                scale = entry["lasagna_scale"]
                if type(scale) is not int or scale <= 0:
                    raise ScrollSpecError(f"{source}: lasagna_scale must be a positive integer")
                facts["lasagna_scale"] = scale
            if entry.get("fiber_scale") is not None:
                scale = entry["fiber_scale"]
                if (isinstance(scale, bool) or not isinstance(scale, (int, float))
                        or not (math.isfinite(scale) and scale > 0)):
                    raise ScrollSpecError(f"{source}: fiber_scale must be a positive number")
                facts["fiber_scale"] = float(scale)
            umbilicus = entry.get("umbilicus")
            if umbilicus is not None:
                facts["umbilicus"] = _checked_umbilicus(umbilicus, source)
            volumes[str(uuid)] = ProjectVolume(**facts)
        return cls(name=name, volumes=volumes,
                   lasagna=_checked_lasagna(value.get("lasagna")),
                   fiber_inference=_checked_fiber_inference(value.get("fiber_inference")))

    def to_mapping(self) -> dict[str, Any]:
        """The request block ``from_mapping`` reads."""
        result = {"name": self.name,
                  "volumes": {uuid: volume.to_mapping()
                              for uuid, volume in sorted(self.volumes.items())}}
        if self.lasagna is not None:
            result["lasagna"] = self.lasagna.to_mapping()
        if self.fiber_inference is not None:
            result["fiber_inference"] = self.fiber_inference.to_mapping()
        return result

    def record(self, volume_uuid: str) -> dict[str, Any]:
        """What a session fitting ``volume_uuid`` records of the project."""
        result = {"name": self.name, "volume": volume_uuid,
                  **self.volumes[volume_uuid].manifest()}
        if self.lasagna is not None:
            result["lasagna"] = self.lasagna.manifest()
        if self.fiber_inference is not None:
            result["fiber_inference"] = self.fiber_inference.manifest()
        return result

    def lasagna_error(self) -> str:
        """Why the fit cannot read Lasagna stores from this project."""
        if self.lasagna is None:
            return (f"the project {self.name!r} selects no Lasagna dataset: attach "
                    "one and select it")
        return self.lasagna.error


def _project_dataset_identity(value: Mapping[str, Any], source: str) -> str:
    """A project dataset's identity: a remote one's is its URL's
    (lasagna_source.url_identity), which VC3D does not send; a local one's
    the identity VC3D sends."""
    location = str(value.get("location") or "")
    try:
        if is_remote_location(location):
            if value.get("identity") not in (None, url_identity(location)):
                raise LasagnaSourceError(
                    f"the remote dataset {location} is identified by its URL, not "
                    f"{value.get('identity')!r}")
            return url_identity(location)
        return check_identity(value.get("identity"))
    except LasagnaSourceError as exc:
        raise ScrollSpecError(f"{source}: {exc}") from None


def _checked_lasagna(value: Any) -> ProjectLasagna | None:
    if value is None:
        return None
    source = "project lasagna"
    if not isinstance(value, Mapping):
        raise ScrollSpecError(f"{source} must be an object")
    if value.get("error"):
        return ProjectLasagna(location=str(value.get("location") or ""),
                              error=str(value["error"]))
    stores = {key: str(value[key]) for key in LASAGNA_CHANNELS if value.get(key)}
    group = str(value.get("group") or "")
    if not stores or not group:
        raise ScrollSpecError(f"{source} needs its stores and their array group")
    identity = _project_dataset_identity(value, source)
    return ProjectLasagna(location=str(value.get("location") or ""), identity=identity,
                          stores=stores, group=group)


def _checked_fiber_inference(value: Any) -> ProjectFiberInference | None:
    if value is None:
        return None
    source = "project fiber_inference"
    if not isinstance(value, Mapping):
        raise ScrollSpecError(f"{source} must be an object")
    location = str(value.get("location") or "")
    if value.get("error"):
        return ProjectFiberInference(location=location, error=str(value["error"]))
    return ProjectFiberInference(location=location,
                                 identity=_project_dataset_identity(value, source))


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


@dataclass(frozen=True)
class SessionScroll:
    """A service session's scroll before its fit volume is known: the
    project's. ``resolve`` completes it once the inputs name the fit volume.
    The service adapter of the shared input contract, as load_input_manifest
    is the command line's."""

    project: SpiralProject

    def volume(self, volume_uuid: str) -> ProjectVolume:
        try:
            return self.project.volumes[volume_uuid]
        except KeyError:
            raise ScrollSpecError(
                f"the fit volume {volume_uuid} is not a volume of the project "
                f"{self.project.name!r}: attach it to the project") from None

    def resolve(self, volume_uuid: str, outward_sense: str | None = None,
                *, lasagna: bool = False) -> ScrollSpec:
        """The scroll of a fit of ``volume_uuid``: the project volume's facts
        and the project's name, with the configured ``outward_sense`` (the fit
        setting spiral_outward_sense) where the volume's orientation does not
        decide it. With ``lasagna`` (the fit reads the Lasagna dataset) the
        volume gives the dataset's scale. The umbilicus is the project's, in
        the fit volume's own voxels."""
        volume = self.volume(volume_uuid)
        fields = {key: getattr(volume, key) for key in PROJECT_VOLUME_FACTS
                  if getattr(volume, key) is not None}
        fields["name"] = self.project.name
        fields["umbilicus_coordinate_scale"] = 1.0
        if lasagna:
            if error := self.project.lasagna_error():
                raise ScrollSpecError(error)
            if volume.lasagna_scale is None:
                raise ScrollSpecError(
                    f"the project's Lasagna dataset {self.project.lasagna.location} "
                    f"has no scale relative to the fit volume {volume_uuid} (neither "
                    "it nor the volume declares its full-resolution shape)")
            fields["lasagna_scale"] = volume.lasagna_scale
        return _scroll_spec_from_fields(
            fields, f"project {self.project.name!r} (volume {volume_uuid})",
            outward_sense=outward_sense)

    def fiber_directions(self, volume_uuid: str,
                         cache_directory: str | os.PathLike[str]) -> str:
        """The fiber-direction samples a fit of ``volume_uuid`` reads: those
        extracted from the project's fiber-inference dataset at the volume's
        scale, in the cache (fiber_direction_samples.py --cache)."""
        path = self.fiber_directions_cache_path(volume_uuid, cache_directory)
        if not os.path.isfile(path):
            dataset = self.project.fiber_inference
            scale = self.volume(volume_uuid).fiber_scale
            raise ScrollSpecError(
                f"no fiber directions of {dataset.location} (dataset {dataset.identity}) "
                f"at scale {scale:g} in the cache: extract them with "
                f"fiber_direction_samples.py {dataset.location} --cache <cache> "
                f"--output-scale {scale:g} (or --adopt an existing artifact) and "
                f"copy them to {path}")
        return path

    def fiber_directions_cache_path(self, volume_uuid: str,
                                    cache_directory: str | os.PathLike[str]) -> str:
        """Where the cache keeps the fiber-direction samples of a fit of
        ``volume_uuid``, present or not."""
        dataset = self.project.fiber_inference
        if dataset is None:
            raise ScrollSpecError(
                f"the project {self.project.name!r} selects no fiber-inference "
                "dataset: attach one and select it")
        if dataset.error:
            raise ScrollSpecError(dataset.error)
        scale = self.volume(volume_uuid).fiber_scale
        if scale is None:
            raise ScrollSpecError(
                f"the fiber-inference dataset {dataset.location} has no scale relative "
                f"to the fit volume {volume_uuid} (neither it nor the volume declares "
                "its full-resolution shape)")
        return fiber_directions_path(cache_directory, dataset.identity, scale)

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
    """Fit-config defaults the command-line input manifest's scroll supplies.

    The generic default for the outer winding and the exporters' physical
    winding estimate is Scroll 1's count (130), which on a scroll with fewer
    windings exports windings that lie outside the papyrus. An optional
    scroll.num_windings in the input manifest replaces that default for this
    scroll. A service takes winding counts from its fit configuration alone.
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

def _normalise_path(value: Any, base: Path | None = None) -> str:
    if value is None or str(value).strip() == "":
        return ""
    path = Path(os.path.expandvars(os.path.expanduser(str(value).strip())))
    if base is not None and not path.is_absolute():
        path = base / path
    # strict=False is important for proposed output/cache paths.
    return str(path.resolve(strict=False))


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
    # The fit volume it was fitted in ('' when the sidecar does not say).
    volume_uuid: str
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
    volume_uuid: str | None,
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
        "volume_uuid": str(volume_uuid or ""),
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
    for key in ("session_namespace", "sha256"):
        if not str(document.get(key) or "").strip():
            raise ValueError(f"autosave metadata is missing {key}")
    return AutosaveCandidate(
        checkpoint=str(metadata_path.parent / name),
        metadata_path=str(metadata_path),
        session_namespace=_normalise_path(document["session_namespace"]),
        volume_uuid=str(document.get("volume_uuid") or ""),
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
    volume_uuid: str | None = None,
) -> AutosaveSelection:
    """Choose the newest matching autosave from validated metadata.

    Selection is by metadata: the sidecar must parse, must carry this
    service's session namespace and, given ``volume_uuid``, must not have
    been fitted in another volume. Among the survivors the one with the most completed
    iterations wins (ties broken by the recorded creation time), so the
    result never depends on filename ordering. Container and identity
    validation happens after selection, in ``validate_autosave``: a selected
    autosave that fails it is an error, not a reason to silently resume from
    an older one.
    """
    namespace = _normalise_path(session_namespace)
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
        if volume_uuid and candidate.volume_uuid and candidate.volume_uuid != volume_uuid:
            rejected.append((
                str(metadata_path),
                f"was fitted in the volume {candidate.volume_uuid}"))
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


def default_user_cache_dir() -> str:
    """The one documented user cache location.

    ``$XDG_CACHE_HOME/vc3d/spiral`` (``~/.cache/vc3d/spiral`` by default).
    Cache entries are content-addressed, so one shared directory serves every
    scroll; ``--cache`` overrides it per service or CLI run.
    """
    base = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
    return _normalise_path(base / "vc3d" / "spiral")


def _validate_json_file(path: Path, label: str, errors: list[dict[str, str]]) -> None:
    try:
        with path.open("r", encoding="utf-8") as stream:
            json.load(stream)
    except Exception as exc:
        errors.append({"field": label, "message": f"Invalid JSON: {exc}"})


def validate_fit_inputs(
    paths: SpiralInputPaths,
    config: Mapping[str, Any],
    *,
    supplied: Iterable[str] = (),
    build_lasagna_packs: bool = True,
    missing: Mapping[str, str] | None = None,
) -> list[dict[str, str]]:
    """Check the inputs a fit of ``config`` reads by path, before any GPU
    allocation: the shared validation of the command line's and a service's
    resolved inputs.

    ``supplied`` names inputs another source provides once the fit volume is
    known (a service session's VOLUME_INPUT_KEYS); they are not checked here.
    ``missing`` maps an input to what its adapter says when one the
    configuration needs is not named at all. Without ``build_lasagna_packs``
    (a service) the Lasagna packs the fit loads must already be in the cache
    directory.
    """
    supplied = frozenset(supplied)
    missing = dict(missing or {})
    errors: list[dict[str, str]] = []

    def unnamed(field_name: str, message: str) -> None:
        hint = missing.get(field_name)
        errors.append({"field": field_name,
                       "message": f"{message}: {hint}" if hint else message})

    def require_file(value: str, field_name: str, *, json_file: bool = False) -> None:
        if not value:
            unnamed(field_name, "Required file is not named")
            return
        path = Path(value)
        if not path.is_file():
            errors.append({"field": field_name,
                           "message": f"Required readable file is missing: {path}"})
            return
        if not os.access(path, os.R_OK):
            errors.append({"field": field_name, "message": f"File is not readable: {path}"})
        elif json_file:
            _validate_json_file(path, field_name, errors)

    def optional_dir(value: str, field_name: str, required: bool = False) -> None:
        if not value:
            if required:
                unnamed(field_name, "Required directory is not named")
            return
        path = Path(value)
        if not path.is_dir():
            errors.append({"field": field_name, "message":
                           f"{'Required directory is missing' if required else 'Path is not a directory'}: {path}"})
        elif not os.access(path, os.R_OK):
            errors.append({"field": field_name, "message": f"Directory is not readable: {path}"})

    def check_lasagna(value: str, field_name: str, required: bool) -> None:
        if not value:
            if required:
                unnamed(field_name, "No Lasagna dataset, and the configuration's losses read one")
            return
        try:
            source = load_lasagna_source(value)
            packs = lasagna_packs_required(config)
            for kind in packs:
                source.arrays(kind)
        except LasagnaSourceError as exc:
            errors.append({"field": field_name, "message": str(exc)})
            return
        if build_lasagna_packs or not paths.cache_directory:
            return
        for kind in packs:
            pack = source.pack_directory(paths.cache_directory, kind)
            if not Path(pack, "meta.json").is_file():
                errors.append({"field": field_name, "message":
                               f"The {kind} pack of Lasagna dataset {source.identity} "
                               f"({source.location}) is not in the cache: build it with "
                               f"pack_resident_pools.py and copy it to {pack}"})

    def check_catalog_input(spec: FitInputSpec) -> None:
        if not spec.enabled(config) or spec.key in supplied:
            return
        if spec.kind == "file":
            require_file(getattr(paths, spec.key), spec.key,
                         json_file=spec.json_content)
        elif spec.kind == "directory":
            optional_dir(getattr(paths, spec.key), spec.key,
                         required=spec.required(config))
        elif spec.kind == "lasagna":
            check_lasagna(getattr(paths, spec.key), spec.key,
                          required=spec.required(config))

    # The Lasagna store requirements (see the catalog's predicates) are
    # checked after the dense-spacing mode below, so an invalid mode errors
    # as itself rather than as missing-file errors.
    for spec in FIT_INPUT_CATALOG:
        if spec.kind != "lasagna":
            check_catalog_input(spec)

    if _dense_spacing_mode(config) is None:
        errors.append({"field": "dense_spacing_mode",
                       "message": "Must be grad_mag or winding_model"})

    for spec in FIT_INPUT_CATALOG:
        if spec.kind == "lasagna":
            check_catalog_input(spec)

    if (_winding_model_enabled(config) and paths.winding_inference
            and Path(paths.winding_inference).is_dir()):
        error = winding_inference_error(paths.winding_inference)
        if error:
            errors.append({"field": "winding_inference", "message": error})

    if _lasagna_required(config) and not paths.cache_directory:
        errors.append({"field": "cache_directory", "message": "Cache directory is required for Lasagna inputs"})
    return errors


def winding_inference_error(path: str | os.PathLike[str]) -> str:
    """Why the winding-inference artifact at ``path`` cannot be read, or ''."""
    manifest = Path(path) / "manifest.json"
    if not Path(path).is_dir():
        return f"Required directory is missing: {path}"
    if not manifest.is_file():
        return f"manifest.json is missing in {path}"
    try:
        metadata = json.loads(manifest.read_text())
        if metadata.get("artifact_type") != "winding_inference_crossings":
            raise ValueError("unexpected artifact_type")
        if int(metadata.get("format_version", -1)) != 1:
            raise ValueError("unsupported format_version")
    except Exception as exc:
        return f"Invalid inference manifest {manifest}: {exc}"
    return ""


def validate_session_request(
    paths: SpiralInputPaths,
    run: SpiralRunConfig,
    *,
    supplied: Iterable[str] = (),
    build_lasagna_packs: bool = True,
    missing: Mapping[str, str] | None = None,
) -> list[dict[str, str]]:
    """Perform cheap, aggregate validation of a service session request
    before any GPU allocation: its inputs (validate_fit_inputs) and its run
    block, output directory and checkpoint."""
    errors = validate_fit_inputs(paths, run.config, supplied=supplied,
                                 build_lasagna_packs=build_lasagna_packs,
                                 missing=missing)

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
