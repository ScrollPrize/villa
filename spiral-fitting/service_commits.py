"""Commit a published Spiral preview generation to the object store.

A published generation (an HTTP preview artifact, lasagna_publish) is
disposable. Committing one makes it one store surface: its flattened
geometry, validity and winding-ID map, bound to the fit volume, with
provenance back to the inputs it was fitted to. The surface's channels and
meta replace the artifact's manifest, ``meta.json`` and winding map, so they
agree by construction.

A commit is written once and never updated. Committing the same model state,
inputs and flatten profile again finds the existing surface
(``find_committed_fit``) instead of creating another. It holds no run
difference and no loss maps: those diagnose a live fit, not a result.

The surface is tagged ``spiral-fit`` (object_store_inputs.SPIRAL_FIT_TAG),
which keeps it out of every fit's input selection.

On a hosted binding the service's replica may hold these commits unpushed,
and nothing else besides detector links: a commit is a new object with a
fresh UUID, so it can never conflict on the hub. It stays pending until a
push succeeds, and every later push (the next commit, link refresh or
startup) carries it.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from lasagna_publish import _surface_xyz, _validate_tifxyz_output_step
from object_store_inputs import SPIRAL_FIT_TAG

COMMIT_TOOL = "spiral-fitting"
# The published manifest schema a commit reads (lasagna_publish.publish).
PUBLISHED_SCHEMA_VERSION = 3
WINDING_CHANNEL = "winding_id"
NO_WINDING = -1


class CommitRefused(ValueError):
    """This generation cannot be committed, and why."""


@dataclasses.dataclass(frozen=True)
class CommitKey:
    """What makes two commits the same surface."""

    model_state_sha256: str
    input_digest: str
    flatten_profile: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class PublishedGeneration:
    """One published generation, read whole from its artifact directory."""

    key: CommitKey
    xyz: np.ndarray  # (rows, columns, 3) float32 xyz, -1 at invalid vertices
    valid: np.ndarray  # (rows, columns) bool, by the flattened and the store's rule
    winding_ids: np.ndarray  # (rows, columns) int32, NO_WINDING outside
    scale: tuple[float, float]
    winding_bounds: list
    input_position: Mapping[str, Any] | None
    source_fit_iteration: int | None
    base_shape_zyx: tuple[int, int, int] | None
    # Vertices valid in the flattened surface that the store cannot hold
    # (store_valid), and so are invalid in the commit.
    unrepresentable: int = 0


@dataclasses.dataclass(frozen=True)
class CommitContext:
    """What a commit records besides the generation's own files: the session's.

    ``config`` is the fit configuration; with ``input_position`` it
    reproduces the selection (object_store_inputs.request_from_config).
    ``sources`` are the objects the fit is anchored to, never the whole
    selection, since meta is loaded with every catalog listing.
    ``base_shape_zyx`` is the fit volume's full-resolution shape, when the
    project knows it.
    """

    volume_uuid: str
    name: str
    author: str
    config: Mapping[str, Any]
    input_digest: str
    sources: tuple[str, ...] = ()
    checkpoint: str | None = None
    base_shape_zyx: tuple[int, int, int] | None = None
    umbilicus: Mapping[str, Any] | None = None
    api_version: int | None = None


@dataclasses.dataclass(frozen=True)
class CommitResult:
    surface_uuid: str
    geometry_revision: str
    reused: bool
    # Whether the surface is on the hub; None for a store without one.
    pushed: bool | None = None
    push_error: str | None = None


def store_valid(xyz):
    """The store's vertex rule (validSurfacePoint): every coordinate finite
    and not -1, and z positive. It invalidates everything the flattened
    rule (lasagna_publish._surface_xyz) does, so the commit needs no
    validity channel of its own; it can only invalidate more."""
    return (np.isfinite(xyz).all(axis=-1) & (xyz != -1.0).all(axis=-1)
            & (xyz[..., 2] > 0))


def _read_winding_ids(path, shape):
    """The published float32 winding map as int32 (lasagna_publish writes
    float32 because OpenCV/Qt cannot portably decode signed-int TIFFs)."""
    with Image.open(path) as image:
        values = np.asarray(image, dtype=np.float32)
    if values.shape != tuple(shape):
        raise CommitRefused(
            f"the winding map is {values.shape[1]}x{values.shape[0]}, not the "
            f"surface's {shape[1]}x{shape[0]}")
    if not np.all(np.isfinite(values)) or not np.array_equal(values, np.round(values)):
        raise CommitRefused("the winding map holds values that are not integers")
    return values.astype(np.int32)


def _shape(value, name):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise CommitRefused(f"{name} is not a ZYX shape: {value!r}")
    return tuple(int(v) for v in value)


def read_published_generation(root, manifest_name="manifest.json"):
    """The generation published under ``root``, checked to be in the fit frame.

    Refuses a generation published before commits existed (no flatten
    profile), one fitted without object-store inputs, and one whose Lasagna
    export did not keep the fit frame (``export_coordinate_scale`` other
    than exactly 1.0, or absent).
    """
    root = Path(root)
    manifest = json.loads((root / manifest_name).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != PUBLISHED_SCHEMA_VERSION:
        raise CommitRefused(
            f"the generation's manifest has schema {manifest.get('schema_version')!r}, "
            f"not {PUBLISHED_SCHEMA_VERSION}")
    profile = manifest.get("flatten_profile")
    if not isinstance(profile, Mapping):
        raise CommitRefused(
            "the generation was published before commits recorded the flatten "
            "profile; publish a new preview and commit that")
    model_state = str(manifest.get("model_state_sha256") or "")
    input_digest = str(manifest.get("input_digest") or "")
    if not model_state:
        raise CommitRefused("the generation records no model state")
    if not input_digest:
        raise CommitRefused("the generation was not fitted to object-store inputs")
    surface_dir = root / str(manifest.get("surface_id") or "")
    if not manifest.get("surface_id") or not surface_dir.is_dir():
        raise CommitRefused("the generation has no flattened surface")
    meta = json.loads((surface_dir / "meta.json").read_text(encoding="utf-8"))
    export_scale = meta.get("export_coordinate_scale")
    # Exactly 1.0: the flattened surface is then in the fit frame, voxels of
    # the fit volume at full resolution. A missing value is refused too, so
    # a new path to a non-unit factor cannot pass unnoticed.
    if type(export_scale) not in (int, float) or float(export_scale) != 1.0:
        raise CommitRefused(
            f"the flattened surface records export_coordinate_scale {export_scale!r}, "
            "not 1.0: it is not in the fit frame")
    step = float(profile.get("output_step_vx") or 0.0)
    scale = _validate_tifxyz_output_step(meta, step)
    base_shape = _shape(manifest.get("base_shape_zyx"), "the generation's base_shape_zyx")
    if _shape(meta.get("base_shape_zyx"), "the surface's base_shape_zyx") != base_shape:
        raise CommitRefused(
            "the flattened surface and the generation's manifest disagree on base_shape_zyx")
    xyz, flattened_valid = _surface_xyz(surface_dir)
    valid = flattened_valid & store_valid(xyz)
    xyz[~valid] = -1.0
    winding_name = str(manifest.get("winding_id_map") or "")
    if not winding_name:
        raise CommitRefused("the generation has no winding map")
    winding_ids = _read_winding_ids(root / winding_name, valid.shape)
    winding_ids[~valid] = NO_WINDING
    iteration = manifest.get("source_fit_iteration")
    return PublishedGeneration(
        key=CommitKey(model_state, input_digest, dict(profile)),
        xyz=xyz, valid=valid, winding_ids=winding_ids,
        scale=(float(scale[0]), float(scale[1])),
        winding_bounds=list(manifest.get("winding_bounds") or ()),
        input_position=manifest.get("input_position"),
        source_fit_iteration=None if iteration is None else int(iteration),
        base_shape_zyx=base_shape,
        unrepresentable=int(np.count_nonzero(flattened_valid & ~valid)))


def find_committed_fit(store, key):
    """The active committed fit surface of ``key`` in ``store``, or None.

    A linear scan of the tag: commits are user-driven and few. Revisit if
    ``spiral-fit`` grows to thousands."""
    for record in store.objects_by_tag(SPIRAL_FIT_TAG):
        meta = record.get("meta") or {}
        if (record.get("kind") == "surface"
                and meta.get("model_state_sha256") == key.model_state_sha256
                and meta.get("input_digest") == key.input_digest
                and meta.get("flatten_profile") == dict(key.flatten_profile)):
            return record
    return None


def commit_meta(generation, context):
    """The committed surface's meta, beyond what the store records itself
    (coordinates and provenance)."""
    meta = {
        "winding_bounds": generation.winding_bounds,
        "component_winding_ids": [item["winding"] for item in generation.winding_bounds],
        "flatten_profile": dict(generation.key.flatten_profile),
        "model_state_sha256": generation.key.model_state_sha256,
        "source_fit_iteration": generation.source_fit_iteration,
        "input_digest": generation.key.input_digest,
        "input_position": generation.input_position,
        "spiral_api_version": context.api_version,
    }
    if context.umbilicus is not None:
        # The project's umbilicus lives in its own store, so it is named here
        # rather than among the provenance sources.
        meta["umbilicus"] = dict(context.umbilicus)
    return meta


def check_commit(generation, context):
    """Refuse a generation the session cannot vouch for."""
    if generation.key.input_digest != context.input_digest:
        raise CommitRefused(
            "the generation was fitted to inputs the session has since replaced")
    if (context.base_shape_zyx is not None
            and generation.base_shape_zyx != tuple(context.base_shape_zyx)):
        raise CommitRefused(
            f"the generation's base_shape_zyx {generation.base_shape_zyx} is not the "
            f"fit volume's shape {tuple(context.base_shape_zyx)}")


def write_commit(store_path, generation, context):
    """Create the committed surface in the store at ``store_path``; return
    its UUID. ``create_surface`` opens and closes its own handle, so it never
    keeps a handle a later hub sync needs idle."""
    from vc_object_store.surfaces import create_surface
    parameters = {"config": dict(context.config)}
    if context.checkpoint:
        parameters["checkpoint"] = context.checkpoint
    return create_surface(
        store_path, generation.xyz, name=context.name, scale=generation.scale,
        volume=context.volume_uuid, voxel_scale=1.0, tool=COMMIT_TOOL,
        parameters=parameters, sources=context.sources,
        meta=commit_meta(generation, context),
        channels={WINDING_CHANNEL: generation.winding_ids.astype("<i4")},
        tags=(SPIRAL_FIT_TAG,), author=context.author)


def commit_in_place(store_path, generation, context):
    """Commit to a store shared in place (no hub). The caller serializes
    store writes, so a concurrent request for the same key finds this one."""
    from vc_object_store import ObjectStore
    check_commit(generation, context)
    with ObjectStore.open(str(store_path), author=context.author) as store:
        existing = find_committed_fit(store, generation.key)
    if existing is not None:
        surface_uuid, reused = existing["uuid"], True
    else:
        surface_uuid, reused = write_commit(store_path, generation, context), False
    with ObjectStore.open(str(store_path), read_only=True) as store:
        revision = store.current_geometry(surface_uuid)["uuid"]
    return CommitResult(surface_uuid, revision, reused)


def commit_to_replica(store_path, generation, context):
    """Commit to the service's hub replica, then publish with a plain push.

    The caller serializes store writes (no link refresh runs meanwhile).
    Unpublished links (a refresh whose revert failed) are reverted first, so
    the push carries only commits; a revert that raises fails the commit
    before anything is written. A pull then makes other services' commits
    visible to the reuse lookup; an unreachable hub still commits locally.
    The push is a plain one, not tied to a refresh's head: a new object
    cannot conflict and needs no lease. When it fails the commit stays
    pending (``pushed`` False) and a later push carries it.
    """
    from vc_object_store import ObjectStore
    check_commit(generation, context)
    with ObjectStore.open(str(store_path), author=context.author) as store:
        store.hub_revert()
        try:
            store.hub_sync(pull=True, push=False)
        except Exception:  # noqa: BLE001 - the push below reports the hub
            pass
        existing = find_committed_fit(store, generation.key)
    if existing is not None:
        surface_uuid, reused = existing["uuid"], True
    else:
        surface_uuid, reused = write_commit(store_path, generation, context), False
    with ObjectStore.open(str(store_path), author=context.author) as store:
        revision = store.current_geometry(surface_uuid)["uuid"]
        if surface_uuid not in store.hub_status()["pending"]:
            return CommitResult(surface_uuid, revision, reused, pushed=True)
        error = push_pending(store)
        pushed = surface_uuid not in store.hub_status()["pending"]
    return CommitResult(surface_uuid, revision, reused, pushed=pushed,
                        push_error=None if pushed else (error or "the push left it pending"))


def push_pending(store):
    """Publish a replica's pending records with a plain push; return why it
    failed, or None. ``store`` must be an idle handle."""
    try:
        store.hub_sync(pull=True, push=True)
    except Exception as exc:  # noqa: BLE001 - reported; the records stay pending
        return f"{type(exc).__name__}: {exc}"
    return None


def pending_records(store):
    """A replica's pending records as ``(commits, others)`` UUID lists:
    committed fits (surfaces tagged spiral-fit) and everything else."""
    commits, others = [], []
    for uuid in store.hub_status()["pending"]:
        record = store.object(uuid)
        is_commit = (record is not None and record.get("kind") == "surface"
                     and SPIRAL_FIT_TAG in (record.get("tags") or ()))
        (commits if is_commit else others).append(uuid)
    return commits, others
