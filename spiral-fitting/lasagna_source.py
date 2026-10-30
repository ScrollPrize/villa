"""The Lasagna dataset a fit reads, and where its resident pools live.

A fit reads Lasagna normals and gradient magnitude from resident-pool packs
(pack_resident_pools.py), never from the source arrays. Packs are derived
data, kept in the fit's cache directory and keyed by the dataset's identity,
so the same dataset reached by different paths (a local copy, a remote
manifest, another host's mount) shares one pack, and a host needs only the
packs to fit:

    <cache>/lasagna/<identity>/normals_g<group>/    nx and ny, one pool
    <cache>/lasagna/<identity>/grad_mag_g<group>/   grad_mag

A dataset's identity is the ``run_uuid`` its provenance record (the
manifest's ``"provenance"`` file, e.g. inference.json) states for that
manifest, else ``sha256-`` and the SHA-256 of the manifest file's bytes. VC3D
computes the same from the same bytes (SpiralProjectInputs).

A fit names its dataset with a ``*.lasagna.json`` manifest (the command-line
fit) or with a source document a service writes from the project VC3D sends
(``write_lasagna_source``); ``load_lasagna_source`` reads either.

Fiber-direction samples (fiber_direction_samples.py) extracted from a
fiber-inference dataset are kept the same way, by that dataset's identity and
the fit-volume scale their positions are in:

    <cache>/fiber-directions/<identity>/scale<base voxels per fit voxel>.npz
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

# The fit's Lasagna inputs by the manifest channel each reads.
LASAGNA_CHANNELS = {"normal_x": "nx", "normal_y": "ny", "gradient_magnitude": "grad_mag"}
# Pack kinds and the inputs each packs, in channel order.
PACK_INPUTS = {"normals": ("normal_x", "normal_y"), "grad_mag": ("gradient_magnitude",)}

SOURCE_FORMAT = "spiral-lasagna-source"
SOURCE_VERSION = 1
MANIFEST_SUFFIX = ".lasagna.json"

# An identity names a cache directory: a run UUID or sha256-<hex>.
_IDENTITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class LasagnaSourceError(ValueError):
    """A Lasagna manifest or source document the fit cannot read."""


@dataclass(frozen=True)
class LasagnaSource:
    """One Lasagna dataset as a fit reads it: its identity, its manifest's
    location, the array name its channels share and, by input
    (LASAGNA_CHANNELS), the OME-Zarr store holding that channel; a store is
    ``<store>/<group>``. The stores are where the dataset was found and need
    not be readable here: only building a pack reads them."""

    identity: str
    location: str
    group: str
    stores: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        check_identity(self.identity)
        if not self.group or "/" in self.group:
            raise LasagnaSourceError(f"{self.location}: invalid array name {self.group!r}")

    def manifest(self) -> dict[str, Any]:
        return {"identity": self.identity, "location": self.location,
                "group": self.group, "stores": dict(sorted(self.stores.items()))}

    def pack_directory(self, cache_directory: str | os.PathLike[str], kind: str) -> str:
        """The resident pool of pack ``kind`` (PACK_INPUTS) of this dataset."""
        if kind not in PACK_INPUTS:
            raise ValueError(f"unknown Lasagna pack {kind!r}")
        return os.path.join(str(cache_directory), "lasagna", self.identity,
                            f"{kind}_g{self.group}")

    def arrays(self, kind: str) -> list[str]:
        """The source arrays pack ``kind`` holds, in channel order."""
        missing = [key for key in PACK_INPUTS[kind] if not self.stores.get(key)]
        if missing:
            raise LasagnaSourceError(
                f"{self.location} has no "
                + ", ".join(LASAGNA_CHANNELS[key] for key in missing) + " channel")
        return [os.path.join(self.stores[key], self.group) for key in PACK_INPUTS[kind]]


def check_identity(identity: str) -> str:
    if not isinstance(identity, str) or not _IDENTITY.fullmatch(identity):
        raise LasagnaSourceError(f"invalid Lasagna dataset identity {identity!r}")
    return identity


def manifest_identity(manifest_bytes: bytes, manifest_name: str,
                      provenance: Mapping[str, Any] | None) -> str:
    """The identity of a manifest: its provenance's ``run_uuid`` when the
    provenance names this manifest (or names none), else the SHA-256 of its
    bytes."""
    if isinstance(provenance, Mapping):
        run_uuid = provenance.get("run_uuid")
        named = provenance.get("manifest")
        if (isinstance(run_uuid, str) and _IDENTITY.fullmatch(run_uuid)
                and named in (None, manifest_name)):
            return run_uuid
    return "sha256-" + hashlib.sha256(manifest_bytes).hexdigest()


def read_lasagna_manifest(path: str | os.PathLike[str]) -> LasagnaSource:
    """A local ``*.lasagna.json`` manifest (version 1 or 2) as a
    LasagnaSource: each channel's store and the array name they share, which
    must be one single-channel array per channel at one scaledown."""
    path = Path(path)
    try:
        data = path.read_bytes()
        document = json.loads(data)
    except (OSError, ValueError) as exc:
        raise LasagnaSourceError(f"{path}: cannot read the Lasagna manifest: {exc}") from exc
    if not isinstance(document, Mapping) or not isinstance(document.get("groups"), Mapping):
        raise LasagnaSourceError(f"{path}: not a Lasagna manifest (no groups)")
    base = path.parent

    # A provenance record the manifest names but that cannot be read is an
    # error, not a fallback: the identity must not depend on what was copied.
    provenance = None
    if isinstance(document.get("provenance"), str) and document["provenance"].strip():
        record = base / document["provenance"]
        try:
            provenance = json.loads(record.read_text())
        except (OSError, ValueError) as exc:
            raise LasagnaSourceError(
                f"{path}: cannot read its provenance record {record}: {exc}") from exc

    stores, group, scaledown = {}, "", None
    for key, channel in LASAGNA_CHANNELS.items():
        entries = [(name, entry) for name, entry in document["groups"].items()
                   if isinstance(entry, Mapping)
                   and channel in (entry.get("channels") or [name])]
        if not entries:
            continue
        name, entry = entries[0]
        channels = entry.get("channels") or [name]
        if len(channels) != 1:
            raise LasagnaSourceError(
                f"{path}: Spiral reads single-channel Lasagna arrays; channel "
                f"{channel} shares group {name}")
        array = Path(str(entry.get("zarr") or ""))
        if not array.name or array.is_absolute():
            raise LasagnaSourceError(f"{path}: group {name} has no relative zarr array")
        if (group and array.name != group) or (
                scaledown is not None and entry.get("scaledown") != scaledown):
            raise LasagnaSourceError(
                f"{path}: Spiral reads the Lasagna channels at one array; this "
                "manifest stores them at different ones")
        stores[key] = os.path.normpath(str(base / array.parent))
        group, scaledown = array.name, entry.get("scaledown")
    if not stores:
        raise LasagnaSourceError(f"{path} has no nx, ny or grad_mag channel")
    return LasagnaSource(identity=manifest_identity(data, path.name, provenance),
                         location=str(path), group=group, stores=stores)


def write_lasagna_source(directory: str | os.PathLike[str], source: LasagnaSource) -> str:
    """Write ``source`` under ``directory`` as the document
    ``load_lasagna_source`` reads; returns its path. Named by its identity."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"lasagna-{source.identity}.json"
    document = {"format": SOURCE_FORMAT, "version": SOURCE_VERSION, **source.manifest()}
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    os.replace(temporary, path)
    return str(path)


def load_lasagna_source(path: str | os.PathLike[str]) -> LasagnaSource:
    """The dataset ``path`` names: a source document write_lasagna_source
    wrote, or a Lasagna manifest."""
    try:
        document = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise LasagnaSourceError(f"{path}: cannot read the Lasagna dataset: {exc}") from exc
    if not (isinstance(document, Mapping) and document.get("format") == SOURCE_FORMAT):
        return read_lasagna_manifest(path)
    if document.get("version") != SOURCE_VERSION:
        raise LasagnaSourceError(f"{path}: unsupported version {document.get('version')!r}")
    stores = document.get("stores") or {}
    if not isinstance(stores, Mapping) or set(stores) - set(LASAGNA_CHANNELS):
        raise LasagnaSourceError(f"{path}: stores must be keyed by {sorted(LASAGNA_CHANNELS)}")
    return LasagnaSource(identity=str(document.get("identity") or ""),
                         location=str(document.get("location") or ""),
                         group=str(document.get("group") or ""),
                         stores={key: str(value) for key, value in stores.items() if value})


def fiber_directions_path(cache_directory: str | os.PathLike[str], identity: str,
                          output_scale: float) -> str:
    """The fiber-direction samples of the fiber-inference dataset
    ``identity`` in a fit volume of ``output_scale`` base voxels per voxel."""
    if not output_scale > 0:
        raise LasagnaSourceError(f"invalid fiber-direction scale {output_scale!r}")
    return os.path.join(str(cache_directory), "fiber-directions", check_identity(identity),
                        f"scale{float(output_scale):g}.npz")


def find_dataset_manifest(directory: str | os.PathLike[str]) -> str:
    """The one ``*.lasagna.json`` in ``directory``, or '' when there is none
    or more than one."""
    try:
        found = sorted(entry for entry in os.listdir(directory)
                       if entry.endswith(MANIFEST_SUFFIX))
    except OSError:
        return ""
    return os.path.join(str(directory), found[0]) if len(found) == 1 else ""
