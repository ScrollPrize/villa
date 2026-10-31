"""The Lasagna dataset a fit reads, and where its resident pools live.

A fit reads Lasagna normals and gradient magnitude from resident-pool packs
(pack_resident_pools.py), never from the source arrays. Packs are derived
data, kept in the fit's cache directory and keyed by the dataset's identity,
so the same dataset reached by different paths (a local copy, a remote
manifest, another host's mount) shares one pack, and a host needs only the
packs to fit:

    <cache>/lasagna/<identity>/normals_g<group>/    nx and ny, one pool
    <cache>/lasagna/<identity>/grad_mag_g<group>/   grad_mag

A remote dataset (an ``s3://``, ``http(s)://`` or ``sftp://`` manifest; the
data at a URL is taken never to change) is identified by its manifest's URL:
``url-`` and a digest of the URL (``url_identity``), computed here only; VC3D
sends the URL. A local copy of one is packed under that URL's identity by
naming the URL (``read_lasagna_manifest(..., url=)``, pack_resident_pools.py
``--as``, fit_spiral.py ``--lasagna-url``). Any other local dataset's identity
is the ``run_uuid`` its provenance record (the manifest's ``"provenance"``
file, e.g. inference.json) states for that manifest, else ``sha256-`` and the
SHA-256 of the manifest file's bytes. VC3D computes the same from the same
bytes (SpiralProjectInputs).

A fit names its dataset with a ``*.lasagna.json`` manifest (the command-line
fit) or with a source document a service writes from the project VC3D sends
(``write_lasagna_source``); ``load_lasagna_source`` reads either.

A manifest that states its ``base_shape_zyx`` gives the dataset's frame: the
arrays' downsampling relative to that base grid (``source_to_base`` times
2^scaledown). With the fit volume's own shape that fixes the arrays'
downsampling relative to the fit volume (``lasagna_scale_for``), as VC3D's
SpiralProjectInputs computes it; it assumes the two grids cover one aligned
extent, which shapes alone cannot show.

Fiber-direction samples (fiber_direction_samples.py) extracted from a
fiber-inference dataset are kept the same way, by that dataset's identity and
the fit-volume scale their positions are in:

    <cache>/fiber-directions/<identity>/scale<base voxels per fit voxel>.npz
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

# The fit's Lasagna inputs by the manifest channel each reads.
LASAGNA_CHANNELS = {"normal_x": "nx", "normal_y": "ny", "gradient_magnitude": "grad_mag"}
# Pack kinds and the inputs each packs, in channel order.
PACK_INPUTS = {"normals": ("normal_x", "normal_y"), "grad_mag": ("gradient_magnitude",)}

SOURCE_FORMAT = "spiral-lasagna-source"
SOURCE_VERSION = 1
MANIFEST_SUFFIX = ".lasagna.json"

# An identity names a cache directory: a run UUID, sha256-<hex> or url-<hex>.
_IDENTITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class LasagnaSourceError(ValueError):
    """A Lasagna manifest or source document the fit cannot read."""


@dataclass(frozen=True)
class LasagnaSource:
    """One Lasagna dataset as a fit reads it: its identity, its manifest's
    location, the array name its channels share and, by input
    (LASAGNA_CHANNELS), the OME-Zarr store holding that channel; a store is
    ``<store>/<group>``. The stores are where the dataset was found and need
    not be readable here: only building a pack reads them.
    ``base_shape_zyx`` and ``base_voxels_per_array_voxel`` are the dataset's
    frame, None where its manifest does not state it."""

    identity: str
    location: str
    group: str
    stores: Mapping[str, str] = field(default_factory=dict)
    base_shape_zyx: tuple[int, int, int] | None = None
    base_voxels_per_array_voxel: float | None = None

    def __post_init__(self):
        check_identity(self.identity)
        if not self.group or "/" in self.group:
            raise LasagnaSourceError(f"{self.location}: invalid array name {self.group!r}")

    def manifest(self) -> dict[str, Any]:
        document = {"identity": self.identity, "location": self.location,
                    "group": self.group, "stores": dict(sorted(self.stores.items()))}
        if self.base_shape_zyx is not None:
            document["base_shape_zyx"] = list(self.base_shape_zyx)
        if self.base_voxels_per_array_voxel is not None:
            document["base_voxels_per_array_voxel"] = self.base_voxels_per_array_voxel
        return document

    def dataset_directory(self, cache_directory: str | os.PathLike[str]) -> str:
        """Where a cache keeps this dataset's packs (and, written by the
        command-line packer and fit, its source document)."""
        return os.path.join(str(cache_directory), "lasagna", self.identity)

    def pack_directory(self, cache_directory: str | os.PathLike[str], kind: str) -> str:
        """The resident pool of pack ``kind`` (PACK_INPUTS) of this dataset."""
        if kind not in PACK_INPUTS:
            raise ValueError(f"unknown Lasagna pack {kind!r}")
        return os.path.join(self.dataset_directory(cache_directory), f"{kind}_g{self.group}")

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


def is_remote_location(location: str) -> bool:
    """Whether ``location`` is a URL (vc::project::isLocationRemote)."""
    scheme = urlsplit(str(location).strip()).scheme.lower()
    return scheme in ("s3", "http", "https", "sftp") or scheme.startswith("s3+")


def dataset_url(location: str) -> str:
    """``location``, a remote dataset's manifest URL, in the one spelling its
    identity is taken from: scheme and host lower-cased, no fragment, no
    trailing slash."""
    if not is_remote_location(location):
        raise LasagnaSourceError(f"{location!r} is not a remote dataset URL")
    parts = urlsplit(str(location).strip())
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(),
                       parts.path.rstrip("/"), parts.query, ""))


def url_identity(location: str) -> str:
    """The identity of the remote dataset whose manifest is at ``location``."""
    return "url-" + hashlib.sha256(dataset_url(location).encode()).hexdigest()[:32]


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


def read_lasagna_manifest(path: str | os.PathLike[str],
                          url: str | None = None) -> LasagnaSource:
    """A local ``*.lasagna.json`` manifest (version 1 or 2) as a
    LasagnaSource: each channel's store and the array name they share, which
    must be one single-channel array per channel at one scaledown. With
    ``url``, the manifest is a local copy of the remote dataset there, and has
    that dataset's identity and location (its stores stay the local ones)."""
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
    if url is None and isinstance(document.get("provenance"), str) and document["provenance"].strip():
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

    # The frame, read as vc::lasagna's manifest reader reads it.
    source_to_base = document.get("source_to_base", 1.0)
    if (type(source_to_base) not in (int, float) or not math.isfinite(source_to_base)
            or source_to_base <= 0):
        raise LasagnaSourceError(f"{path}: source_to_base must be a positive number")
    scaledown = 0 if scaledown is None else scaledown
    if type(scaledown) is not int or not 0 <= scaledown <= 30:
        raise LasagnaSourceError(f"{path}: invalid scaledown {scaledown!r}")
    base_shape = document.get("base_shape_zyx")
    if base_shape is not None:
        base_shape = _checked_shape(base_shape, f"{path}: base_shape_zyx")
    frame = {"base_shape_zyx": base_shape,
             "base_voxels_per_array_voxel": float(source_to_base) * 2 ** scaledown}

    if url is not None:
        return LasagnaSource(identity=url_identity(url), location=dataset_url(url),
                             group=group, stores=stores, **frame)
    return LasagnaSource(identity=manifest_identity(data, path.name, provenance),
                         location=str(path), group=group, stores=stores, **frame)


def _checked_shape(value: Any, what: str) -> tuple[int, int, int]:
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(type(extent) is not int or extent <= 0 for extent in value)):
        raise LasagnaSourceError(f"{what} must be a ZYX list of three positive integers")
    return tuple(value)


def _same_level(fine: int, coarse: int, factor: int) -> bool:
    """Whether ``coarse`` is ``fine`` downsampled ``factor`` times: either
    ceil or floor halving; at one level, the voxel-count and inclusive
    maximum-coordinate conventions (one apart)."""
    if factor == 1:
        return abs(fine - coarse) <= 1
    return coarse in (-(-fine // factor), max(1, fine // factor))


def dyadic_coordinate_scale(from_shape_zyx, to_shape_zyx, maximum_level: int = 30) -> float:
    """``to_shape_zyx`` voxels per ``from_shape_zyx`` voxel: the one power of
    two (within 2^±``maximum_level``) at which the shapes are levels of one
    grid (vc::lasagna::dyadicCoordinateScaleBetweenShapes). ValueError when
    no single level matches."""
    if not 0 <= maximum_level <= 30:
        raise ValueError("maximum dyadic coordinate level must be between 0 and 30")
    if any(int(extent) <= 0 for extent in (*from_shape_zyx, *to_shape_zyx)):
        raise ValueError("coordinate-domain shapes must be positive")
    matches = [
        level for level in range(-maximum_level, maximum_level + 1)
        if all(_same_level(*((to, frm) if level >= 0 else (frm, to)), 1 << abs(level))
               for frm, to in zip(from_shape_zyx, to_shape_zyx))]
    if len(matches) != 1:
        raise ValueError(
            f"shapes {list(from_shape_zyx)} and {list(to_shape_zyx)} do not identify "
            "exactly one power-of-two scale")
    return math.ldexp(1.0, matches[0])


def array_shape_matches(volume_shape_zyx, array_shape_zyx, downsampling: int) -> bool:
    """Whether arrays of ``array_shape_zyx`` are ``volume_shape_zyx``
    downsampled ``downsampling`` times, by the rule of
    ``dyadic_coordinate_scale``."""
    return all(_same_level(int(fine), int(coarse), int(downsampling))
               for fine, coarse in zip(volume_shape_zyx, array_shape_zyx))


def lasagna_scale_for(source: LasagnaSource, volume_shape_zyx) -> tuple[int | None, str]:
    """The downsampling of ``source``'s arrays relative to a fit volume of
    shape ``volume_shape_zyx`` (its full-resolution shape), as VC3D's
    SpiralProjectInputs derives it, and how it was found; ``(None, why)``
    when it cannot be derived. Valid only where the volume and the dataset's
    base grid cover the same aligned extent."""
    if volume_shape_zyx is None:
        return None, "the fit volume's shape (base_shape_zyx) is not known"
    if source.base_shape_zyx is None or source.base_voxels_per_array_voxel is None:
        return None, f"{source.location} does not state its base_shape_zyx"
    try:
        base_per_volume_voxel = dyadic_coordinate_scale(volume_shape_zyx, source.base_shape_zyx)
    except ValueError as exc:
        return None, (f"the fit volume's shape {list(volume_shape_zyx)} and the "
                      f"dataset's base shape {list(source.base_shape_zyx)} are not "
                      f"levels of one grid ({exc})")
    scale = source.base_voxels_per_array_voxel / base_per_volume_voxel
    how = (f"{source.base_voxels_per_array_voxel:g} base voxels per array voxel, "
           f"{base_per_volume_voxel:g} per fit-volume voxel")
    if scale < 1 or scale != round(scale):
        return None, f"{how} is not a whole downsampling ({scale:g})"
    return int(scale), how


def write_lasagna_source(directory: str | os.PathLike[str], source: LasagnaSource) -> str:
    """Write ``source`` under ``directory`` as the document
    ``load_lasagna_source`` reads; returns its path. Named by its identity."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"lasagna-{source.identity}.json"
    document = {"format": SOURCE_FORMAT, "version": SOURCE_VERSION, **source.manifest()}
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
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
    base_shape = document.get("base_shape_zyx")
    per_array_voxel = document.get("base_voxels_per_array_voxel")
    if per_array_voxel is not None and (
            type(per_array_voxel) not in (int, float) or not per_array_voxel > 0):
        raise LasagnaSourceError(f"{path}: invalid base_voxels_per_array_voxel")
    return LasagnaSource(identity=str(document.get("identity") or ""),
                         location=str(document.get("location") or ""),
                         group=str(document.get("group") or ""),
                         stores={key: str(value) for key, value in stores.items() if value},
                         base_shape_zyx=(None if base_shape is None else
                                         _checked_shape(base_shape, f"{path}: base_shape_zyx")),
                         base_voxels_per_array_voxel=(
                             None if per_array_voxel is None else float(per_array_voxel)))


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
