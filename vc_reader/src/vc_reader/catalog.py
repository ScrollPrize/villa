"""Parsed, queryable view of the Vesuvius Challenge open-data ``metadata.json``.

Layout of the file (see metadata.schema.json next to it on S3)::

    samples[<sample id>] = {sample, scans{}, volumes{}, segments{}}
    models[<model id>]

Conventions relied on here, all from the schema:

* Points are ``[x, y, z]`` in level-0 voxels of the volume they refer to;
  array shapes are ``[z, y, x]``.
* OME-Zarr level N is a 2^N downsample: voxel size ``pixel_size_um * 2^N``.
* ``sample.properties.volume_transforms`` holds the transitive closure of
  registered transforms between a sample's volumes (level-0 to level-0).
* Cross-references may name volumes that are not in the published file.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from . import _fetch
from .affine import Affine
from .artifacts import Artifact, ArtifactType, parse_artifacts
from .errors import AmbiguousMatchError, NotFoundError
from .names import COMMUNITY_ALIASES, SampleNameIndex

#: Relative tolerance when matching a requested resolution (µm) to a voxel size.
#: Wide enough for rounded figures in names (2.399 vs 2.4), narrower than the
#: gap between distinct pyramid levels.
DEFAULT_RESOLUTION_REL_TOL = 0.005

#: OME-Zarr volumes are published with pyramid levels 0..5.
PYRAMID_LEVELS = 6


def _num(x: Any) -> float | None:
    return None if x is None else float(x)


def _resolution_matches(value: float | None, wanted: float, rel_tol: float) -> bool:
    return value is not None and math.isclose(value, wanted, rel_tol=rel_tol)


def _scan_name(md: Mapping[str, Any]) -> str | None:
    """The facility's name for a scan, from ``creation.metadata``.

    ESRF records the acquisition's ``scanRadix`` (``2.4um_PHerc-Paris4_B_HA``),
    DLS a ``scan_name`` (``PHercParis4 54keV stitched_part_2, …``).
    """
    acq = ((md.get("metadata") or {}).get("tomo") or {}).get("acquisition") or {}
    return acq.get("scanRadix") or md.get("scan_name") or None


# --------------------------------------------------------------------------
# Small value types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VolumeCoverage:
    """How much of a segment lies inside a volume (``segment.properties.volume_coverage``)."""

    volume_id: str
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    overlap_ratio: float
    computed_at: str | None = None

    @classmethod
    def from_json(cls, volume_id: str, d: Mapping[str, Any]) -> VolumeCoverage:
        lo, hi = d["bbox_transformed"]
        return cls(volume_id, tuple(lo), tuple(hi), float(d["overlap_ratio"]), d.get("computed_at"))  # type: ignore[arg-type]


@dataclass(frozen=True)
class License:
    name: str
    url: str


# --------------------------------------------------------------------------
# Entities
# --------------------------------------------------------------------------


class Model:
    """An ML model referenced by predictions (``parameters.model_id``)."""

    def __init__(self, d: Mapping[str, Any]):
        self.raw = d
        self.id: str = d["id"]
        self.long_id: str | None = d.get("long_id")
        self.name: str | None = d.get("suffix") or None
        self.properties: Mapping[str, Any] = d.get("properties") or {}
        self.task: str | None = self.properties.get("task")
        self.architecture: str | None = self.properties.get("architecture")

    def __repr__(self) -> str:
        return f"Model({self.id!r}, name={self.name!r}, task={self.task!r})"


class Scan:
    """An X-ray CT acquisition. Raw scan data is not published."""

    def __init__(self, sample: Sample, d: Mapping[str, Any]):
        self.sample = sample
        self.raw = d
        self.id: str = d["id"]
        self.long_id: str | None = d.get("long_id")
        self.creation: Mapping[str, Any] = d.get("creation") or {}
        p = d.get("properties") or {}
        self.pixel_size_um = _num(p.get("pixel_size_um"))
        self.energy_kev = _num(p.get("energy_keV"))
        self.detector_distance_mm = _num(p.get("detector_distance_mm"))
        self.name: str | None = _scan_name(self.creation.get("metadata") or {})

    @property
    def date(self) -> str | None:
        return self.creation.get("date")

    @property
    def volumes(self) -> list[Volume]:
        return [v for v in self.sample.volume_list() if v.scan_id == self.id]

    def __repr__(self) -> str:
        return f"Scan({self.long_id or self.id!r})"


class Volume:
    """A 3D reconstruction of a scan, published as an OME-Zarr pyramid."""

    def __init__(self, sample: Sample, d: Mapping[str, Any]):
        self.sample = sample
        self.raw = d
        self.id: str = d["id"]
        self.long_id: str | None = d.get("long_id")
        self.suffix: str | None = d.get("suffix")
        self.scan_id: str | None = d.get("scan_id")
        self.creation: Mapping[str, Any] = d.get("creation") or {}
        p = self.properties = d.get("properties") or {}
        self.pixel_size_um = _num(p.get("pixel_size_um"))
        self.energy_kev = _num(p.get("energy_keV"))
        self.detector_distance_mm = _num(p.get("detector_distance_mm"))
        self.dtype: str | None = p.get("data_format")
        shape = p.get("shape")
        self.shape_zyx: tuple[int, int, int] | None = tuple(shape) if shape else None  # type: ignore[assignment]
        lic = p.get("license")
        self.license = License(lic["name"], lic["url"]) if lic else None
        sup = p.get("superseded_by")
        self.superseded_by_id: str | None = sup["volume_id"] if sup else None
        self.render_level: int = int(p.get("render_level") or 0)
        self.render_segments: bool | None = p.get("render_segments")
        self.left_handed_coordinates: bool | None = p.get("left_handed_coordinates")
        self.z_direction_is_top_to_bottom: bool | None = p.get("z_direction_is_top_to_bottom")
        self.text_side: str | None = p.get("text_side")
        self.artifacts = parse_artifacts(d.get("data"))

    def __repr__(self) -> str:
        return f"Volume({self.long_id or self.id!r})"

    # -- relations ------------------------------------------------------------

    @property
    def scan(self) -> Scan | None:
        return self.sample.scans.get(self.scan_id) if self.scan_id else None

    @property
    def superseded_by(self) -> Volume | None:
        return self.sample.volumes.get(self.superseded_by_id) if self.superseded_by_id else None

    @property
    def is_superseded(self) -> bool:
        return self.superseded_by_id is not None

    @property
    def is_secondary(self) -> bool:
        """``render_segments: false``: segments are no longer rendered on this volume.

        The pipeline sets it on experimental, ROI or redundant exports.
        """
        return self.render_segments is False

    def transform_to(self, other: Volume | str) -> Affine | None:
        """Level-0 ``[x, y, z]`` of this volume -> level-0 of ``other``; None if unregistered."""
        return self.sample.transform(self.id, _volume_id(other))

    # -- artifacts --------------------------------------------------------------

    def artifacts_of(self, *types: str) -> list[Artifact]:
        return [a for a in self.artifacts if a.type in types]

    @property
    def zarr(self) -> Artifact | None:
        """The ``ome-zarr`` artifact holding the volume itself."""
        found = self.artifacts_of(ArtifactType.OME_ZARR)
        return found[0] if found else None

    def url(self, scheme: str | None = "https") -> str | None:
        """URL of the OME-Zarr root (with trailing slash)."""
        return self.zarr.url(scheme) if self.zarr else None

    def surface_predictions(self) -> list[Artifact]:
        return self.artifacts_of(ArtifactType.SURFACE_PREDICTION_ZARR)

    def normal_grids(self) -> list[Artifact]:
        return self.artifacts_of(ArtifactType.NORMAL_GRIDS)

    def lasagna(self) -> list[Artifact]:
        return self.artifacts_of(ArtifactType.LASAGNA)

    def ink_3d_predictions(self) -> list[Artifact]:
        return self.artifacts_of(ArtifactType.INK_DETECTION_3D_ZARR)

    def umbilicus(self) -> list[Artifact]:
        return self.artifacts_of(ArtifactType.UMBILICUS)

    def fiberlets(self) -> list[Artifact]:
        return self.artifacts_of(ArtifactType.FIBERLETS)

    def derived_artifacts(self) -> list[Artifact]:
        """Volume-level artifacts other than the volume's own OME-Zarr."""
        return [a for a in self.artifacts if a.type != ArtifactType.OME_ZARR]

    # -- pyramid levels -------------------------------------------------------

    @property
    def levels(self) -> tuple[VolumeLevel, ...]:
        return tuple(VolumeLevel(self, n) for n in range(PYRAMID_LEVELS))

    def level(
        self,
        level: int | None = None,
        *,
        um: float | None = None,
        rel_tol: float = DEFAULT_RESOLUTION_REL_TOL,
    ) -> VolumeLevel:
        """One pyramid level, by index or by voxel size in µm (``um=9.6``)."""
        if (level is None) == (um is None):
            raise TypeError("pass exactly one of level or um")
        if level is not None:
            if not 0 <= level < PYRAMID_LEVELS:
                raise NotFoundError(f"{self}: level {level} outside 0..{PYRAMID_LEVELS - 1}")
            return VolumeLevel(self, level)
        if self.pixel_size_um is None:
            raise NotFoundError(f"{self}: pixel size unknown, cannot select a level by µm")
        for lvl in self.levels:
            if _resolution_matches(lvl.voxel_size_um, um, rel_tol):  # type: ignore[arg-type]
                return lvl
        available = ", ".join(f"L{lvl.level}={lvl.voxel_size_um:g}um" for lvl in self.levels)
        raise NotFoundError(f"{self}: no level at {um:g} um (available: {available})")

    # -- segments and labels --------------------------------------------------

    def segments(self, *, min_coverage: float | None = None) -> list[SegmentOnVolume]:
        """Segments with a canonical mesh (``tifxyz-transformed``) on this volume.

        With ``min_coverage``, keep only segments whose recorded overlap ratio
        with this volume is at least that value.
        """
        out = []
        for seg in self.sample.segment_list():
            if seg.mesh_on(self.id) is None:
                continue
            view = SegmentOnVolume(seg, self)
            if min_coverage is not None and (view.coverage is None or view.coverage.overlap_ratio < min_coverage):
                continue
            out.append(view)
        return out

    def ink_labels(
        self,
        *,
        mappable: bool = True,
        label_set: str | None = None,
        latest_only: bool = True,
        dedupe: bool = True,
        min_coverage: float = 0.0,
    ) -> list[InkLabelMatch]:
        """Ink labels that are on, or can be mapped onto, this volume.

        * ``direct``: the labels' ``target_volume`` is this volume.
        * ``transform``: the labels are on another volume of the sample that has
          a registered transform to this one, and the segment's recorded overlap
          with this volume is above ``min_coverage`` (or not recorded). Mapping
          means: label pixel -> 3D point via the labels' own mesh
          (``InkLabelSet.mesh``) -> ``InkLabelMatch.transform`` into this volume.

        With ``dedupe`` (default), each (segment, label_set) appears once,
        preferring a direct match, then labels on a volume with the same voxel
        frame (superseded pair), then natively annotated labels, then the lowest
        target volume id.
        """
        out: list[InkLabelMatch] = []
        for seg in self.sample.segment_list():
            coverage = seg.coverage.get(self.id)
            matches: list[InkLabelMatch] = []
            for labels in seg.ink_labels(label_set=label_set, latest_only=latest_only):
                if labels.target_volume_id == self.id:
                    matches.append(InkLabelMatch(labels, self, "direct", Affine.identity(), coverage))
                    continue
                if not mappable:
                    continue
                if coverage is not None and coverage.overlap_ratio <= min_coverage:
                    continue
                tf = self.sample.transform(labels.target_volume_id, self.id)
                if tf is not None:
                    matches.append(InkLabelMatch(labels, self, "transform", tf, coverage))
            if dedupe:
                best: dict[str | None, InkLabelMatch] = {}
                for m in sorted(matches, key=lambda m: m._preference):
                    best.setdefault(m.labels.label_set, m)
                matches = list(best.values())
            out.extend(matches)
        return out


class VolumeLevel:
    """One pyramid level of a volume, i.e. a working coordinate scale."""

    def __init__(self, volume: Volume, level: int):
        self.volume = volume
        self.level = level

    def __repr__(self) -> str:
        um = f"{self.voxel_size_um:g}um" if self.voxel_size_um is not None else "?um"
        return f"VolumeLevel({self.volume.long_id or self.volume.id!r}, level={self.level}, {um})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, VolumeLevel) and (self.volume is other.volume and self.level == other.level)

    def __hash__(self) -> int:
        return hash((self.volume.id, self.level))

    @property
    def scale_factor(self) -> int:
        """Level-0 voxels per voxel of this level along each axis."""
        return 2**self.level

    @property
    def voxel_size_um(self) -> float | None:
        px = self.volume.pixel_size_um
        return None if px is None else px * self.scale_factor

    @property
    def expected_shape_zyx(self) -> tuple[int, int, int] | None:
        """``ceil(level-0 shape / 2^level)`` from the recorded level-0 shape.

        The zarr array metadata is authoritative; this is what to expect.
        """
        s = self.volume.shape_zyx
        f = self.scale_factor
        return None if s is None else tuple(-(-n // f) for n in s)  # type: ignore[return-value]

    def url(self, scheme: str | None = "https") -> str | None:
        """URL of this level's array (OME-Zarr dataset path ``"<level>"``)."""
        root = self.volume.url(scheme)
        return None if root is None else root.rstrip("/") + f"/{self.level}/"

    def open(self, scheme: str = "https", **kwargs: Any):
        """A :class:`~vc_reader.reader.ZarrReader` for this level's array.

        Keyword arguments go to :meth:`ZarrReader.open` (``cache_dir``,
        ``memory_cache``, ``threads``, ...). Requires the ``reader`` extra.
        """
        from .reader import ZarrReader

        url = self.url(scheme)
        if url is None:
            raise NotFoundError(f"{self.volume} has no OME-Zarr location")
        return ZarrReader.open(url, **kwargs)

    @property
    def to_level0(self) -> Affine:
        """Voxel ``[x, y, z]`` at this level -> level-0 voxel of the same volume."""
        return Affine.scale(self.scale_factor)

    @property
    def from_level0(self) -> Affine:
        """Level-0 voxel ``[x, y, z]`` -> voxel at this level."""
        return Affine.scale(1.0 / self.scale_factor)

    def transform_to(self, other: VolumeLevel | Volume) -> Affine | None:
        """Voxel ``[x, y, z]`` at this level -> voxel of ``other`` (a level, or level 0 of a volume)."""
        dst = other if isinstance(other, VolumeLevel) else VolumeLevel(other, 0)
        between = self.volume.transform_to(dst.volume)
        if between is None:
            return None
        return dst.from_level0 @ between @ self.to_level0

    # -- paired data ------------------------------------------------------------

    def predictions(self) -> list[PairedArtifact]:
        """Volume-level derived artifacts, with the sub-level aligned to this scale where one exists."""
        return [PairedArtifact.for_level(a, self) for a in self.volume.derived_artifacts()]

    def segments(self, *, min_coverage: float | None = None) -> list[SegmentOnVolume]:
        return self.volume.segments(min_coverage=min_coverage)

    def ink_labels(self, **kwargs: Any) -> list[InkLabelMatch]:
        """Same as :meth:`Volume.ink_labels`; use :attr:`from_level0` on mapped coordinates."""
        return self.volume.ink_labels(**kwargs)

    def paired(self, **ink_label_kwargs: Any) -> PairedData:
        """Everything published for this volume, viewed at this scale."""
        return PairedData(
            level=self,
            predictions=self.predictions(),
            segments=self.segments(),
            ink_labels=self.ink_labels(**ink_label_kwargs),
        )


@dataclass(frozen=True)
class PairedArtifact:
    """A volume-level artifact seen from one pyramid level of its volume.

    ``sublevel`` is the level *inside the artifact's own pyramid* whose grid
    coincides with the requested level, or None when the artifact is not a
    pyramid on the volume grid (lasagna bundles, normal grids, umbilicus) or
    is coarser than the requested level.
    """

    artifact: Artifact
    level: VolumeLevel
    sublevel: int | None

    @classmethod
    def for_level(cls, artifact: Artifact, level: VolumeLevel) -> PairedArtifact:
        g = artifact.grid_level
        sub = level.level - g if g is not None and level.level >= g else None
        return cls(artifact, level, sub)

    @property
    def aligned(self) -> bool:
        return self.sublevel is not None

    def open(self, scheme: str = "https", **kwargs: Any):
        """A :class:`~vc_reader.reader.ZarrReader` for the aligned sub-level (see :meth:`VolumeLevel.open`)."""
        from .reader import ZarrReader

        url = self.url(scheme)
        if url is None:
            raise NotFoundError(f"{self.artifact} has no array aligned with {self.level}")
        return ZarrReader.open(url, **kwargs)

    def url(self, scheme: str | None = "https") -> str | None:
        """URL of the aligned sub-level array, or None if not aligned."""
        root = self.artifact.url(scheme)
        if root is None or self.sublevel is None:
            return None
        return root.rstrip("/") + f"/{self.sublevel}/"


@dataclass(frozen=True)
class PairedData:
    level: VolumeLevel
    predictions: list[PairedArtifact]
    segments: list[SegmentOnVolume]
    ink_labels: list[InkLabelMatch]


class Segment:
    """A surface traced in ``original_volume_id``, plus everything derived from it."""

    def __init__(self, sample: Sample, d: Mapping[str, Any]):
        self.sample = sample
        self.raw = d
        self.id: str = d["id"]
        self.long_id: str | None = d.get("long_id")
        self.suffix: str | None = d.get("suffix") or None
        self.original_volume_id: str = d["original_volume_id"]
        self.creation: Mapping[str, Any] = d.get("creation") or {}
        p = self.properties = d.get("properties") or {}
        self.width: int | None = p.get("width")
        self.height: int | None = p.get("height")
        self.original_volume_downscale: int = int(p.get("original_volume_downscale") or 1)
        self.coverage: dict[str, VolumeCoverage] = {
            vid: VolumeCoverage.from_json(vid, c) for vid, c in (p.get("volume_coverage") or {}).items()
        }
        self.artifacts = parse_artifacts(d.get("data"))

    def __repr__(self) -> str:
        return f"Segment({self.long_id or self.id!r})"

    @property
    def original_volume(self) -> Volume | None:
        return self.sample.volumes.get(self.original_volume_id)

    def artifacts_of(self, *types: str, target_volume: Volume | str | None = None) -> list[Artifact]:
        tv = _volume_id(target_volume) if target_volume is not None else None
        return [a for a in self.artifacts if a.type in types and (tv is None or a.target_volume_id == tv)]

    @property
    def target_volume_ids(self) -> list[str]:
        """Volumes this segment has any per-volume artifact for (meshes, renders, labels)."""
        return sorted({a.target_volume_id for a in self.artifacts if a.target_volume_id})

    def mesh_on(self, volume: Volume | str) -> Artifact | None:
        """The canonical mesh (``tifxyz-transformed``) in level-0 voxels of ``volume``."""
        found = self.artifacts_of(ArtifactType.TIFXYZ_TRANSFORMED, target_volume=volume)
        return found[0] if found else None

    def on(self, volume: Volume | str) -> SegmentOnVolume:
        vol = volume if isinstance(volume, Volume) else self.sample.volume(volume)
        return SegmentOnVolume(self, vol)

    def ink_labels(self, *, label_set: str | None = None, latest_only: bool = True) -> list[InkLabelSet]:
        sets = [InkLabelSet(self, a) for a in self.artifacts_of(ArtifactType.INK_LABELS)]
        if label_set is not None:
            sets = [s for s in sets if s.label_set == label_set]
        if latest_only:
            superseded = {(s.target_volume_id, s.supersedes) for s in sets if s.supersedes}
            sets = [s for s in sets if (s.target_volume_id, s.label_set) not in superseded]
        return sorted(sets, key=lambda s: (s.target_volume_id or "", s.label_set or ""))


class SegmentOnVolume:
    """A segment's artifacts for one target volume (mesh, renders, predictions, labels)."""

    def __init__(self, segment: Segment, volume: Volume):
        self.segment = segment
        self.volume = volume

    def __repr__(self) -> str:
        return f"SegmentOnVolume({self.segment.long_id or self.segment.id!r}, on={self.volume.id!r})"

    def _of(self, *types: str) -> list[Artifact]:
        return self.segment.artifacts_of(*types, target_volume=self.volume)

    @property
    def coverage(self) -> VolumeCoverage | None:
        return self.segment.coverage.get(self.volume.id)

    @property
    def mesh(self) -> Artifact | None:
        return self.segment.mesh_on(self.volume)

    @property
    def layers(self) -> list[Artifact]:
        """Surface volumes rendered along the mesh (``layers-zarr``, legacy ``layers-tif``)."""
        return self._of(ArtifactType.LAYERS_ZARR, ArtifactType.LAYERS_TIF)

    @property
    def ink_detections(self) -> list[Artifact]:
        return self._of(ArtifactType.INK_DETECTION)

    @property
    def ink_3d_projections(self) -> list[Artifact]:
        return self._of(ArtifactType.INK_3D_PREDICTION)

    @property
    def alpha_renders(self) -> list[Artifact]:
        return self._of(ArtifactType.ALPHA_RENDER)

    @property
    def previews(self) -> list[Artifact]:
        return self._of(ArtifactType.ALPHA_RENDER_DOWNSAMPLED, ArtifactType.INK_DETECTION_DOWNSAMPLED)

    @property
    def ink_labels(self) -> list[InkLabelSet]:
        return [s for s in self.segment.ink_labels() if s.target_volume_id == self.volume.id]

    @property
    def artifacts(self) -> list[Artifact]:
        """Every artifact of the segment whose ``target_volume`` is this volume."""
        return [a for a in self.segment.artifacts if a.target_volume_id == self.volume.id]

    def render_pixel_size_um(self, artifact: Artifact) -> float | None:
        """Effective pixel size of a render artifact: ``pixel_size_um * 2^render_level``."""
        px = self.volume.pixel_size_um
        return None if px is None else px * 2**artifact.render_level


class InkLabelSet:
    """One ``ink-labels`` artifact: annotations on the render of a segment on ``target_volume``.

    The directory holds ``inklabels.zarr``, ``supervision.zarr`` and optionally
    ``validation.zarr`` (rank-2 ``[y, x]`` on the render pixel grid).
    """

    def __init__(self, segment: Segment, artifact: Artifact):
        self.segment = segment
        self.artifact = artifact
        p = artifact.properties
        self.target_volume_id: str = artifact.parameters["target_volume"]
        self.label_set: str | None = artifact.label_set
        self.registration: str | None = p.get("registration")
        self.annotated_on: str | None = p.get("annotated_on")
        self.mapping_coverage: float | None = _num(p.get("mapping_coverage"))
        self.supersedes: str | None = p.get("supersedes")
        self.arrays: tuple[str, ...] = tuple(p.get("arrays") or ())
        self.labelled_px: int | None = p.get("labelled_px")
        self.supervised_px: int | None = p.get("supervised_px")

    def __repr__(self) -> str:
        return (
            f"InkLabelSet(segment={self.segment.id!r}, target_volume={self.target_volume_id!r}, "
            f"label_set={self.label_set!r}, registration={self.registration!r})"
        )

    @property
    def target_volume(self) -> Volume | None:
        return self.segment.sample.volumes.get(self.target_volume_id)

    @property
    def mesh(self) -> Artifact | None:
        """Mesh whose render grid the labels are on (lifts label pixels to 3D)."""
        return self.segment.mesh_on(self.target_volume_id)

    @property
    def render(self) -> Artifact | None:
        """The ``layers-zarr`` render the labels are registered to."""
        found = self.segment.artifacts_of(ArtifactType.LAYERS_ZARR, target_volume=self.target_volume_id)
        return found[0] if found else None

    def url(self, scheme: str | None = "https") -> str | None:
        return self.artifact.url(scheme)

    def array_url(self, name: str = "inklabels", scheme: str | None = "https") -> str | None:
        """URL of one of the label arrays: ``inklabels``, ``supervision`` or ``validation``."""
        root = self.url(scheme)
        return None if root is None else root.rstrip("/") + f"/{name}.zarr/"


@dataclass(frozen=True)
class InkLabelMatch:
    """Ink labels relative to a requested volume.

    ``transform`` maps level-0 voxels of the labels' target volume to level-0
    voxels of ``volume`` (identity for ``relation == "direct"``). ``coverage`` is
    the segment's recorded coverage of ``volume``, if any.
    """

    labels: InkLabelSet
    volume: Volume
    relation: Literal["direct", "transform"]
    transform: Affine
    coverage: VolumeCoverage | None

    @property
    def segment(self) -> Segment:
        return self.labels.segment

    @property
    def mesh_on_volume(self) -> Artifact | None:
        """The segment's canonical mesh on the requested volume, if one was published."""
        return self.segment.mesh_on(self.volume)

    @property
    def same_frame(self) -> bool:
        """True when the labels' volume shares this volume's voxel frame (direct, or superseded pair)."""
        return self.relation == "direct" or self.volume.sample.same_frame(self.labels.target_volume_id, self.volume)

    @property
    def _preference(self) -> tuple[Any, ...]:
        return (
            self.relation != "direct",
            not self.same_frame,
            self.labels.registration != "native",
            self.labels.target_volume_id,
        )


class Sample:
    """A scroll or fragment, with its scans, volumes and segments."""

    def __init__(self, catalog: Catalog, tree: Mapping[str, Any]):
        self.catalog = catalog
        self.raw = tree
        s = tree["sample"]
        self.id: str = s["id"]
        self.description: str | None = s.get("description")
        p = self.properties = s.get("properties") or {}
        self.type: str | None = p.get("type")
        self.display_name: str | None = p.get("description")
        self.legacy_data_url: str | None = p.get("legacy_data_url")
        self.aliases: tuple[str, ...] = tuple(a for a, sid in COMMUNITY_ALIASES.items() if sid == self.id)
        self.artifacts = parse_artifacts(s.get("data"))
        self.scans: dict[str, Scan] = {k: Scan(self, v) for k, v in (tree.get("scans") or {}).items()}
        self.volumes: dict[str, Volume] = {k: Volume(self, v) for k, v in (tree.get("volumes") or {}).items()}
        self.segments: dict[str, Segment] = {k: Segment(self, v) for k, v in (tree.get("segments") or {}).items()}
        self._transforms: dict[tuple[str, str], Affine] = {}
        for group in p.get("volume_transforms") or ():
            src = group["from_volume_id"]
            for t in group.get("transforms") or ():
                self._transforms[(src, t["to_volume_id"])] = Affine.from_rows(t["matrix"], t.get("derivation_path"))

    def __repr__(self) -> str:
        return f"Sample({self.id!r}, volumes={len(self.volumes)}, segments={len(self.segments)})"

    # Entity maps keep file order, which the schema says is not significant.
    # Lists are sorted by id so query results are deterministic.
    def volume_list(self) -> list[Volume]:
        return [self.volumes[k] for k in sorted(self.volumes)]

    def segment_list(self) -> list[Segment]:
        return [self.segments[k] for k in sorted(self.segments)]

    @property
    def photo(self) -> Artifact | None:
        found = [a for a in self.artifacts if a.type == ArtifactType.PHOTO]
        return found[0] if found else None

    # -- volumes -----------------------------------------------------------------

    def find_volumes(
        self,
        query: str | None = None,
        *,
        resolution: float | None = None,
        energy_kev: float | None = None,
        scan: Scan | str | None = None,
        include_superseded: bool = False,
        rel_tol: float = DEFAULT_RESOLUTION_REL_TOL,
    ) -> list[Volume]:
        """Volumes matching every given criterion.

        ``query`` is a volume id or long id. ``resolution`` is the level-0 voxel
        size in µm: volumes whose pixel size equals it are returned if there are
        any, otherwise all within relative tolerance ``rel_tol`` (so ``2.399``
        finds a ``2.4`` volume, but does not also return ``2.403`` volumes when a
        ``2.399`` one exists). ``energy_kev`` matches to within 0.5 keV.
        ``scan`` is a scan, scan id, long id or name. Superseded volumes are
        skipped unless named by ``query`` or ``include_superseded``. Results are
        ordered by distance to ``resolution``, then id.
        """
        vols = self.volume_list()
        if query is not None:
            vols = [v for v in vols if query in (v.id, v.long_id, (v.long_id or "").removesuffix(".zarr"))]
        elif not include_superseded:
            vols = [v for v in vols if not v.is_superseded]
        if scan is not None:
            sid = self._scan_id(scan)
            vols = [v for v in vols if v.scan_id == sid]
        if resolution is not None:
            vols = [v for v in vols if _resolution_matches(v.pixel_size_um, resolution, rel_tol)]
            exact = [v for v in vols if v.pixel_size_um == resolution]
            vols = exact or sorted(vols, key=lambda v: abs(v.pixel_size_um - resolution))  # type: ignore[operator]
        if energy_kev is not None:
            vols = [v for v in vols if v.energy_kev is not None and math.isclose(v.energy_kev, energy_kev, abs_tol=0.5)]
        return vols

    def volume(
        self,
        query: str | None = None,
        *,
        resolution: float | None = None,
        energy_kev: float | None = None,
        scan: Scan | str | None = None,
        include_superseded: bool = False,
        rel_tol: float = DEFAULT_RESOLUTION_REL_TOL,
    ) -> Volume:
        """Exactly one volume; see :meth:`find_volumes`. Raises if none or several match.

        A scan can have several exports at one resolution. When some of a
        scan's matching exports are secondary (:attr:`Volume.is_secondary`) and
        some are not, the secondary ones are dropped. Volumes of different scans
        are never preferred over each other: they usually cover different
        regions (scan parts, ROIs).
        """
        if query is None and resolution is None and energy_kev is None and scan is None:
            raise TypeError("pass a volume id/long id, resolution=, energy_kev= or scan=")
        vols = self.find_volumes(
            query,
            resolution=resolution,
            energy_kev=energy_kev,
            scan=scan,
            include_superseded=include_superseded,
            rel_tol=rel_tol,
        )
        criteria = ", ".join(
            f"{k}={v!r}"
            for k, v in (
                ("query", query),
                ("resolution", resolution),
                ("energy_kev", energy_kev),
                ("scan", _scan_label(scan)),
            )
            if v is not None
        )
        if not vols:
            listing = "\n".join("  " + _describe_volume(v) for v in self.volume_list())
            raise NotFoundError(f"no volume of {self.id} matches {criteria}. Volumes:\n{listing}")
        main_scans = {v.scan_id for v in vols if not v.is_secondary}
        vols = [v for v in vols if not v.is_secondary or v.scan_id not in main_scans]
        if len(vols) > 1:
            narrow = [
                name
                for name, values in (("energy_kev=", {v.energy_kev for v in vols}), ("scan=", {v.scan_id for v in vols}))
                if len(values) > 1
            ]
            listing = "\n".join("  " + _describe_volume(v) for v in vols)
            raise AmbiguousMatchError(
                f"{len(vols)} volumes of {self.id} match {criteria}; "
                f"narrow it with {' or '.join([*narrow, 'a volume id'])}:\n{listing}",
                vols,
            )
        return vols[0]

    def _scan_id(self, scan: Scan | str) -> str:
        if isinstance(scan, Scan):
            return scan.id
        hits = [s.id for s in self.scans.values() if scan in (s.long_id, s.name)]
        return hits[0] if hits else scan

    def transform(self, from_volume: Volume | str, to_volume: Volume | str) -> Affine | None:
        """Level-0 ``[x, y, z]`` of ``from_volume`` -> level-0 of ``to_volume``.

        Uses ``volume_transforms`` (inverse and transitive closure of the
        registrations). A volume and its superseding volume share an identical
        voxel frame, so the identity is returned for that pair if no matrix is
        listed. None when the volumes are not registered to each other.
        """
        a, b = _volume_id(from_volume), _volume_id(to_volume)
        if a == b:
            return Affine.identity()
        tf = self._transforms.get((a, b))
        if tf is not None:
            return tf
        return Affine.identity() if self.same_frame(a, b) else None

    def same_frame(self, a: Volume | str, b: Volume | str) -> bool:
        """True if ``a`` and ``b`` are the same volume or one supersedes the other."""
        a, b = _volume_id(a), _volume_id(b)
        va, vb = self.volumes.get(a), self.volumes.get(b)
        return a == b or bool(va and va.superseded_by_id == b) or bool(vb and vb.superseded_by_id == a)

    # -- segments and labels ---------------------------------------------------

    def segment(self, query: str) -> Segment:
        """A segment by id, long id or suffix (tool name/uuid)."""
        hits = [s for s in self.segment_list() if query in (s.id, s.long_id, s.suffix)]
        if not hits:
            raise NotFoundError(f"no segment of {self.id} matches {query!r}")
        if len(hits) > 1:
            raise AmbiguousMatchError(f"{len(hits)} segments of {self.id} match {query!r}", hits)
        return hits[0]

    def ink_labels(
        self,
        *,
        target_volume: Volume | str | None = None,
        label_set: str | None = None,
        latest_only: bool = True,
    ) -> list[InkLabelSet]:
        """Every ink-label set of the sample, as published (no mapping).

        Use :meth:`Volume.ink_labels` to also find labels that can be mapped
        onto a particular volume.
        """
        tv = _volume_id(target_volume) if target_volume is not None else None
        return [
            s
            for seg in self.segment_list()
            for s in seg.ink_labels(label_set=label_set, latest_only=latest_only)
            if tv is None or s.target_volume_id == tv
        ]


class Catalog:
    """The whole metadata file: samples and models."""

    def __init__(self, data: Mapping[str, Any], source: str | None = None):
        self.raw = data
        self.source = source
        self.schema_url: str | None = data.get("$schema")
        self.models: dict[str, Model] = {k: Model(v) for k, v in (data.get("models") or {}).items()}
        self.samples: dict[str, Sample] = {k: Sample(self, v) for k, v in (data.get("samples") or {}).items()}
        self._names = SampleNameIndex()
        for s in self.samples.values():
            self._names.add(s.id, (s.display_name, *s.aliases))

    def __repr__(self) -> str:
        return f"Catalog(samples={len(self.samples)}, models={len(self.models)}, source={self.source!r})"

    # -- construction ------------------------------------------------------------

    @classmethod
    def load(
        cls,
        source: str | os.PathLike[str] = _fetch.DEFAULT_METADATA_URL,
        *,
        cache_dir: str | os.PathLike[str] | None = None,
        use_cache: bool = True,
        max_age: float = 3600.0,
        refresh: bool = False,
        timeout: float = 60.0,
    ) -> Catalog:
        """Load from an http(s) URL (cached on disk) or a local file (plain or gzipped JSON)."""
        src = os.fspath(source)
        if src.startswith(("http://", "https://")):
            data = _fetch.fetch_metadata(
                src, cache_dir=cache_dir, use_cache=use_cache, max_age=max_age, refresh=refresh, timeout=timeout
            )
        else:
            data = _fetch.read_metadata_file(src)
        return cls(data, source=src)

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> Catalog:
        return cls(_fetch.read_metadata_file(path), source=os.fspath(path))

    # -- lookups -----------------------------------------------------------------

    def sample(self, query: str) -> Sample:
        """A sample by id or common name: ``"PHercParis4"``, ``"pherc paris 4"``, ``"Scroll 1"``, ``"PHerc 332"``."""
        ids = self._names.resolve(query)
        if not ids:
            hint = self._names.suggest(query)
            raise NotFoundError(
                f"no sample matches {query!r}" + (f"; did you mean {', '.join(hint)}?" if hint else "")
            )
        if len(ids) > 1:
            raise AmbiguousMatchError(f"{query!r} matches several samples: {sorted(ids)}", sorted(ids))
        return self.samples[ids.pop()]

    def find_samples(self, query: str | None = None, *, type: str | None = None) -> list[Sample]:
        """Samples whose names contain ``query`` (normalised), optionally of a ``type`` (scroll/fragment)."""
        ids = self._names.search(query) if query else set(self.samples)
        return [self.samples[i] for i in sorted(ids) if type is None or self.samples[i].type == type]

    def model(self, model_id: str) -> Model:
        try:
            return self.models[model_id]
        except KeyError:
            raise NotFoundError(f"no model {model_id!r}") from None

    def volume(self, volume_id: str) -> Volume:
        """A volume by id or long id, across all samples."""
        hits = [v for s in self.samples.values() for v in s.find_volumes(volume_id)]
        if not hits:
            raise NotFoundError(f"no volume {volume_id!r}")
        if len(hits) > 1:
            raise AmbiguousMatchError(f"{len(hits)} volumes match {volume_id!r}", hits)
        return hits[0]


def _volume_id(v: Volume | str) -> str:
    return v.id if isinstance(v, Volume) else v


def _scan_label(scan: Scan | str | None) -> str | None:
    return scan.id if isinstance(scan, Scan) else scan


def _describe_volume(v: Volume) -> str:
    scan = v.scan
    bits = [v.long_id or v.id, f"scan={v.scan_id}" if v.scan_id else "", f"({scan.name})" if scan and scan.name else ""]
    if v.shape_zyx:
        bits.append("shape=" + "x".join(map(str, v.shape_zyx)))
    bits.append(f"segments={len(v.segments())}")
    if v.is_superseded:
        bits.append(f"superseded by {v.superseded_by_id}")
    if v.is_secondary:
        bits.append("secondary (render_segments=false)")
    return "  ".join(b for b in bits if b)


__all__: Sequence[str] = [
    "Catalog",
    "Sample",
    "Scan",
    "Volume",
    "VolumeLevel",
    "Segment",
    "SegmentOnVolume",
    "InkLabelSet",
    "InkLabelMatch",
    "PairedArtifact",
    "PairedData",
    "Model",
    "VolumeCoverage",
    "License",
]
