"""Catalog metadata records and the scroll-centered browsing entry point.

Records retain source relationships; contextual views share these records.
Parsing and registrations are shared with the native C++ catalog.
"""
from __future__ import annotations
import os
import json
from . import _catalog
from dataclasses import dataclass
from typing import Any, Mapping
from . import _fetch
from .affine import Affine
from .artifacts import Artifact, parse_artifacts
from .names import COMMUNITY_ALIASES, SampleNameIndex
DEFAULT_RESOLUTION_REL_TOL = 0.005
PYRAMID_LEVELS = 6

def _num(x: Any) -> float | None:
    return None if x is None else float(x)

def _scan_name(md: Mapping[str, Any]) -> str | None:
    """The facility's name for a scan, from ``creation.metadata``.

    ESRF records the acquisition's ``scanRadix`` (``2.4um_PHerc-Paris4_B_HA``),
    DLS a ``scan_name`` (``PHercParis4 54keV stitched_part_2, …``).
    """
    acq = ((md.get("metadata") or {}).get("tomo") or {}).get("acquisition") or {}
    return acq.get("scanRadix") or md.get("scan_name") or None

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
        return cls(volume_id, tuple(lo), tuple(hi), float(d["overlap_ratio"]), d.get("computed_at"))

@dataclass(frozen=True)
class License:
    name: str
    url: str

class Model:
    """An ML model referenced by predictions (``parameters.model_id``)."""

    def __init__(self, d: Mapping[str, Any]):
        self.raw = d
        self.id: str = d['id']
        self.long_id: str | None = d.get('long_id')
        self.name: str | None = d.get('suffix') or None
        self.properties: Mapping[str, Any] = d.get('properties') or {}
        self.task: str | None = self.properties.get('task')
        self.architecture: str | None = self.properties.get('architecture')

    def __repr__(self) -> str:
        return f'Model({self.id!r}, name={self.name!r}, task={self.task!r})'

class Scan:

    def __init__(self, sample: _SampleRecord, native):
        d = native.raw
        self.sample = sample
        self.raw = d
        self.id: str = native.id
        self.long_id: str | None = d.get('long_id')
        self.creation: Mapping[str, Any] = d.get('creation') or {}
        p = d.get('properties') or {}
        self.pixel_size_um = native.pixel_size_um
        self.energy_kev = _num(p.get('energy_keV'))
        self.detector_distance_mm = _num(p.get('detector_distance_mm'))
        self.name: str | None = _scan_name(self.creation.get('metadata') or {})

    @property
    def date(self) -> str | None:
        return self.creation.get('date')

    def __repr__(self) -> str:
        return f'Scan({self.long_id or self.id!r})'

class _VolumeRecord:

    def __init__(self, sample: _SampleRecord, native):
        d = native.raw
        self.sample = sample
        self.raw = d
        self.id: str = native.id
        self.long_id: str | None = d.get('long_id')
        self.suffix: str | None = d.get('suffix')
        self.scan_id: str | None = native.scan_id or None
        self.creation: Mapping[str, Any] = d.get('creation') or {}
        p = self.properties = d.get('properties') or {}
        self.pixel_size_um = native.pixel_size_um
        self.energy_kev = native.energy_kev
        self.detector_distance_mm = native.detector_distance_mm
        self.dtype: str | None = native.dtype or None
        shape = native.shape_zyx
        self.shape_zyx: tuple[int, int, int] | None = tuple(shape) if shape else None
        lic = p.get('license')
        self.license = License(lic['name'], lic['url']) if lic else None
        sup = p.get('superseded_by')
        self.superseded_by_id: str | None = sup['volume_id'] if sup else None
        self.render_level: int = int(p.get('render_level') or 0)
        self.render_segments: bool | None = p.get('render_segments')
        self.left_handed_coordinates: bool | None = p.get('left_handed_coordinates')
        self.z_direction_is_top_to_bottom: bool | None = p.get('z_direction_is_top_to_bottom')
        self.text_side: str | None = p.get('text_side')
        self.artifacts = tuple(Artifact.from_native(a) for a in native.artifacts)

    def __repr__(self) -> str:
        return f'_VolumeRecord({self.long_id or self.id!r})'

    @property
    def scan(self) -> Scan | None:
        return self.sample.scans.get(self.scan_id) if self.scan_id else None

    @property
    def is_superseded(self) -> bool:
        return self.superseded_by_id is not None

class _SegmentRecord:

    def __init__(self, sample: _SampleRecord, native):
        d = native.raw
        self.sample = sample
        self.raw = d
        self.id: str = native.id
        self.long_id: str | None = d.get('long_id')
        self.suffix: str | None = d.get('suffix') or None
        self.original_volume_id: str = native.original_volume_id
        self.creation: Mapping[str, Any] = d.get('creation') or {}
        p = self.properties = d.get('properties') or {}
        self.width: int | None = native.width
        self.height: int | None = native.height
        self.original_volume_downscale: int = int(p.get('original_volume_downscale') or 1)
        self.coverage: dict[str, VolumeCoverage] = {vid: VolumeCoverage.from_json(vid, c) for vid, c in (p.get('volume_coverage') or {}).items()}
        self.artifacts = tuple(Artifact.from_native(a) for a in native.artifacts)

    def __repr__(self) -> str:
        return f'_SegmentRecord({self.long_id or self.id!r})'

    def artifacts_of(self, *types: str, target_volume: _VolumeRecord | str | None=None) -> list[Artifact]:
        tv = _volume_id(target_volume) if target_volume is not None else None
        return [a for a in self.artifacts if a.type in types and (tv is None or a.target_volume_id == tv)]

class _SampleRecord:

    def __init__(self, catalog: Catalog, native):
        tree = native.raw
        self.catalog = catalog
        self.raw = tree
        s = tree.get('sample') or tree
        self.id: str = native.id
        self.description: str | None = s.get('description')
        p = self.properties = native.properties
        self.type: str | None = native.type or None
        self.display_name: str | None = p.get('description')
        self.legacy_data_url: str | None = p.get('legacy_data_url')
        self.aliases: tuple[str, ...] = tuple((a for a, sid in COMMUNITY_ALIASES.items() if sid == self.id))
        self.artifacts = tuple(Artifact.from_native(a) for a in native.artifacts)
        self.scans: dict[str, Scan] = {v.id: Scan(self, v) for v in native.scans}
        self.volumes: dict[str, _VolumeRecord] = {v.id: _VolumeRecord(self, v) for v in native.volumes}
        self.segments: dict[str, _SegmentRecord] = {v.id: _SegmentRecord(self, v) for v in native.segments}

    def __repr__(self) -> str:
        return f'_SampleRecord({self.id!r}, volumes={len(self.volumes)}, segments={len(self.segments)})'

    def volume_list(self) -> list[_VolumeRecord]:
        return [self.volumes[k] for k in sorted(self.volumes)]

    def segment_list(self) -> list[_SegmentRecord]:
        return [self.segments[k] for k in sorted(self.segments)]

    def transform(self, from_volume: _VolumeRecord | str, to_volume: _VolumeRecord | str) -> Affine | None:
        """Level-0 ``[x, y, z]`` of ``from_volume`` -> level-0 of ``to_volume``.

        Uses ``volume_transforms`` (inverse and transitive closure of the
        registrations). A volume and its superseding volume share an identical
        voxel frame, so the identity is returned for that pair if no matrix is
        listed. None when the volumes are not registered to each other.
        """
        result = self.catalog._native.transform(self.id, _volume_id(from_volume), _volume_id(to_volume))
        return None if result is None else Affine.from_rows(result['matrix'], result['derivation_path'])

    def same_frame(self, a: _VolumeRecord | str, b: _VolumeRecord | str) -> bool:
        """True if ``a`` and ``b`` are the same volume or one supersedes the other."""
        a, b = (_volume_id(a), _volume_id(b))
        va, vb = (self.volumes.get(a), self.volumes.get(b))
        return a == b or bool(va and va.superseded_by_id == b) or bool(vb and vb.superseded_by_id == a)

class Catalog:

    def __init__(self, data: Mapping[str, Any], source: str | None=None):
        self._native = data if isinstance(data, _catalog.Catalog) else _catalog.parse(json.dumps(data))
        data = self._native.raw
        if 'metadata' in data:
            data = data['metadata']
        self.raw = data
        self.source = source
        self.schema_url: str | None = data.get('$schema')
        self.models: dict[str, Model] = {v.id: Model(dict(v.raw, id=v.id)) for v in self._native.models}
        self._samples: dict[str, _SampleRecord] = {v.id: _SampleRecord(self, v) for v in self._native.samples}
        self._names = SampleNameIndex()
        for s in self._samples.values():
            self._names.add(s.id, (s.display_name, *s.aliases))

    def __repr__(self) -> str:
        return f'Catalog(samples={len(self._samples)}, models={len(self.models)}, source={self.source!r})'

    @classmethod
    def load(cls, source: str | os.PathLike[str]=_fetch.DEFAULT_METADATA_URL, *, cache_dir: str | os.PathLike[str] | None=None, use_cache: bool=True, max_age: float=3600.0, refresh: bool=False, timeout: float=60.0) -> Catalog:
        """Load from an http(s) URL (cached on disk) or a local file (plain or gzipped JSON)."""
        src = os.fspath(source)
        import hashlib
        cache = ''
        if use_cache:
            cache = os.fspath((_fetch.default_cache_dir() if cache_dir is None else __import__('pathlib').Path(cache_dir)) /
                              ('metadata-' + hashlib.sha256(src.encode()).hexdigest()[:16] + '.json'))
        return cls(_catalog.load(src, cache, refresh, max_age, timeout), source=src)

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> Catalog:
        return cls(_catalog.load(os.fspath(path)), source=os.fspath(path))

    def __reduce__(self):
        return (Catalog, (self.raw, self.source))

    @property
    def _index(self):
        if not hasattr(self, '_relationship_index'):
            from .discovery import RelationshipIndex
            self._relationship_index = RelationshipIndex(self)
        return self._relationship_index

    @property
    def samples(self):
        from .browsing import Collection, ScrollSource
        return Collection((ScrollSource._view(self._index, s.id) for s in self._samples.values()))

    @property
    def scrolls(self):
        return self.samples.filter(type='scroll')

def _volume_id(v: _VolumeRecord | str) -> str:
    return v.id if isinstance(v, _VolumeRecord) else v

