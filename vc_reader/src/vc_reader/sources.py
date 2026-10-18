"""Immutable, stdlib-only descriptions of storage, values, and registration.

Constructing or deserializing these objects never opens storage. Unknown fields
remain None; ``unresolved_fields`` is a declaration diagnostic, not an I/O probe.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Iterator
from dataclasses import dataclass, field, fields, replace
from typing import Any

from .affine import Affine
from .catalog import VolumeCoverage


@dataclass(frozen=True)
class FrozenMapping(Mapping):
    """Small, pickleable immutable JSON mapping used for provenance."""

    items_tuple: tuple = ()

    def __getitem__(self, key):
        for k, value in self.items_tuple:
            if k == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (k for k, _ in self.items_tuple)

    def __len__(self):
        return len(self.items_tuple)


def _freeze(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite declaration")
        return value
    if isinstance(value, _Value):
        return value
    if isinstance(value, Affine):
        matrix = _freeze(value.matrix)
        if len(matrix) != 3 or any(len(row) != 4 for row in matrix):
            raise ValueError("affine must have shape 3x4")
        return Affine(matrix, value.derivation_path)
    if isinstance(value, VolumeCoverage):
        return VolumeCoverage(**{f.name: _freeze(getattr(value, f.name)) for f in fields(value)})
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise TypeError("metadata keys must be strings")
        return FrozenMapping(tuple((k, _freeze(v)) for k, v in sorted(value.items())))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(v) for v in value)
    raise TypeError(f"unsupported declaration value: {type(value).__name__}")


class _Value:
    def __post_init__(self):
        for f in fields(self):
            object.__setattr__(self, f.name, _freeze(getattr(self, f.name)))
        self._validate()

    def _validate(self):
        pass


def _name(value, field_name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a nonempty string")


def _index(value, field_name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer")


def _axes(axes):
    if axes is not None and not isinstance(axes, tuple):
        raise TypeError("axes must be a sequence of axis names")
    if axes is not None and (not axes or len(set(axes)) != len(axes)):
        raise ValueError("axes must be nonempty and unique")
    for axis in axes or ():
        _name(axis, "axis")


def _vector(vector, name):
    if not isinstance(vector, tuple) or len(vector) != 3 or any(type(v) not in (int, float) for v in vector):
        raise ValueError(f"{name} must contain three finite numbers")


def _affine(value):
    if value is not None:
        if not isinstance(value, Affine):
            raise TypeError("transform must be an Affine")
        value.inverse()  # Reject singular declarations, without NumPy.


def _uv(value):
    if value is not None:
        if len(value) != 2 or any(len(row) != 3 for row in value):
            raise ValueError("pixel-to-UV transform must have shape 2x3")
        if value[0][0] * value[1][1] == value[0][1] * value[1][0]:
            raise ValueError("pixel-to-UV transform is singular")


@dataclass(frozen=True)
class Frame(_Value):
    sample_id: str
    volume_id: str
    level: int = 0

    def _validate(self):
        _name(self.sample_id, "sample_id")
        _name(self.volume_id, "volume_id")
        _index(self.level, "level")


@dataclass(frozen=True)
class LevelRequest(_Value):
    """Requested reference resolution; a negative sublevel means unavailable."""

    reference: Frame
    sublevel: int | None = None

    def _validate(self):
        if not isinstance(self.reference, Frame):
            raise TypeError("reference must be a Frame")
        if self.sublevel is not None and type(self.sublevel) is not int:
            raise ValueError("sublevel must be an integer or None")


@dataclass(frozen=True)
class Selection(_Value):
    array_path: str | None = None
    level: int | None = None
    series: str | int | None = None
    channels: tuple[str | int, ...] | None = None
    time: str | int | None = None
    corresponding: LevelRequest | None = None

    def _validate(self):
        if sum(v is not None for v in (self.array_path, self.level, self.corresponding)) > 1:
            raise ValueError("select only one of array_path, level, or corresponding")
        if self.array_path is not None:
            _name(self.array_path, "array_path")
        if self.level is not None:
            _index(self.level, "level")
        if self.corresponding is not None and not isinstance(self.corresponding, LevelRequest):
            raise TypeError("corresponding must be a LevelRequest")
        if self.channels is not None and not isinstance(self.channels, tuple):
            raise TypeError("channels must be a sequence of indices or names")
        for name, values in (("series", (self.series,)), ("time", (self.time,)),
                             ("channels", self.channels or ())):
            for value in values:
                if value is None:
                    continue
                if isinstance(value, str):
                    _name(value, name)
                else:
                    _index(value, name)
        if self.channels is not None and (not self.channels or len(set(self.channels)) != len(self.channels)):
            raise ValueError("channels must be nonempty and unique")


@dataclass(frozen=True)
class Storage(_Value):
    location: str | None = None
    format: str | None = None
    locations: tuple[str, ...] = ()
    alternatives: tuple[str, ...] = ()
    selection: Selection = field(default_factory=Selection)
    pyramid: bool = False
    grid_level: int | None = None

    def _validate(self):
        if not isinstance(self.locations, tuple) or not isinstance(self.alternatives, tuple):
            raise TypeError("locations and alternatives must be sequences of locations")
        if self.location is not None and self.locations:
            raise ValueError("use location or ordered stack locations, not both")
        for value in (*self.locations, *self.alternatives):
            _name(value, "location")
        if self.location is not None:
            _name(self.location, "location")
        if self.format is not None:
            _name(self.format, "format")
        if self.locations and self.format != "stack":
            raise ValueError("ordered locations require format='stack'")
        if self.format == "stack" and not self.locations:
            raise ValueError("a stack needs explicit ordered locations")
        if not isinstance(self.selection, Selection):
            raise TypeError("selection must be a Selection")
        if type(self.pyramid) is not bool:
            raise TypeError("pyramid must be boolean")
        if self.grid_level is not None:
            _index(self.grid_level, "grid_level")


@dataclass(frozen=True)
class Plane(_Value):
    """XY pixel centers and one-voxel normal footprint in volume level-0 XYZ."""

    origin: tuple[float, float, float]
    basis_x: tuple[float, float, float]
    basis_y: tuple[float, float, float]
    thickness: tuple[float, float, float]

    def _validate(self):
        for name in ("origin", "basis_x", "basis_y", "thickness"):
            _vector(getattr(self, name), name)
        x, y, n = self.basis_x, self.basis_y, self.thickness
        cross = (x[1]*y[2]-x[2]*y[1], x[2]*y[0]-x[0]*y[2], x[0]*y[1]-x[1]*y[0])
        if not any(cross) or not any(n):
            raise ValueError("plane needs independent pixel bases and nonzero thickness")
        # Geometric validation allows ordinary rounding of an explicitly computed
        # normal, not a change to sample support or interpolation tolerances.
        for basis in (x, y):
            dot = sum(a*b for a, b in zip(basis, n))
            scale = math.sqrt(sum(a*a for a in basis) * sum(a*a for a in n))
            if abs(dot) > 1e-12 * scale:
                raise ValueError("plane thickness must be perpendicular to its pixel bases")


@dataclass(frozen=True)
class VolumeRegistration(_Value):
    sample_id: str
    volume_id: str
    affine: Affine | None = None
    plane: Plane | None = None
    mode: str = "array"
    storage_frame: str | None = None
    declared_grid: Selection | None = None
    origin: str = "explicit"
    storage_units: str | None = None
    volume_spacing_um: tuple[float, float, float] | None = None

    def _validate(self):
        _name(self.sample_id, "sample_id")
        _name(self.volume_id, "volume_id")
        _affine(self.affine)
        if self.mode not in ("array", "storage"):
            raise ValueError("registration mode must be 'array' or 'storage'")
        if self.mode == "storage":
            _name(self.storage_frame, "storage_frame")
        elif self.storage_frame is not None:
            raise ValueError("storage_frame requires mode='storage'")
        if self.plane is not None:
            if not isinstance(self.plane, Plane) or self.affine is not None or self.mode != "array":
                raise ValueError("plane placement requires a Plane and no affine/storage mode")
        if self.declared_grid is not None and not isinstance(self.declared_grid, Selection):
            raise TypeError("declared_grid must be a Selection")
        if self.storage_units is not None:
            _name(self.storage_units, "storage_units")
            if self.mode != "storage":
                raise ValueError("storage_units requires storage placement")
        if self.volume_spacing_um is not None:
            _vector(self.volume_spacing_um, "volume_spacing_um")
            if any(v <= 0 for v in self.volume_spacing_um):
                raise ValueError("volume spacing must be positive")
        _name(self.origin, "registration origin")


@dataclass(frozen=True)
class ValueFrame(_Value):
    kind: str = "volume"
    sample_id: str | None = None
    volume_id: str | None = None
    affine: Affine | None = None
    level: int | None = None
    name: str | None = None
    units: str | None = None

    def _validate(self):
        if self.kind == "volume":
            _name(self.sample_id, "sample_id")
            _name(self.volume_id, "volume_id")
            if self.affine is not None and self.level is not None:
                raise ValueError("value frame selects an affine or a level")
            _affine(self.affine)
            if self.level is not None:
                _index(self.level, "level")
        elif self.kind == "external":
            _name(self.name, "external frame name")
            _name(self.units, "external frame units")
            if any(v is not None for v in (self.sample_id, self.volume_id, self.affine, self.level)):
                raise ValueError("external frame cannot declare a volume transform")
        else:
            raise ValueError("value frame kind must be 'volume' or 'external'")


@dataclass(frozen=True)
class Encoding(_Value):
    kind: str
    pairs: tuple
    dtype: str
    fallback: int | None = None

    def _validate(self):
        if self.kind not in ("integer", "color"):
            raise ValueError("encoding kind must be 'integer' or 'color'")
        if not isinstance(self.pairs, tuple):
            raise TypeError("encoding pairs must be a sequence")
        dtypes = {f'{prefix}{bits}': (-(2**(bits-1)), 2**(bits-1)-1) if prefix == 'int'
                  else (0, 2**bits-1) for prefix in ('int', 'uint') for bits in (8, 16, 32, 64)}
        if self.dtype not in dtypes:
            raise ValueError("encoding needs an integer output dtype")
        keys = set()
        color_size = None
        values = [] if self.fallback is None else [self.fallback]
        for pair in self.pairs:
            if len(pair) != 2:
                raise ValueError("encoding pairs contain a key and output value")
            key, value = pair
            if self.kind == "integer":
                if type(key) is not int:
                    raise ValueError("integer lookup keys must be integers")
            else:
                if not isinstance(key, tuple) or not key or any(type(v) is not int or v < 0 for v in key):
                    raise ValueError("color keys must be tuples of nonnegative integers")
                color_size = color_size or len(key)
                if len(key) != color_size:
                    raise ValueError("color keys must have equal component counts")
            if key in keys:
                raise ValueError("duplicate encoding key")
            keys.add(key)
            values.append(value)
        lo, hi = dtypes[self.dtype]
        if any(type(v) is not int or not lo <= v <= hi for v in values):
            raise ValueError("encoding output is outside its integer dtype")


@dataclass(frozen=True)
class Surface(_Value):
    id: str
    geometry: Storage
    sample_id: str
    volume_id: str
    axes: tuple[str, ...] | None = None
    component_order: tuple[str, ...] | None = None
    xyz_to_volume: Affine | None = None
    pixel_to_uv: tuple | None = None
    spacing_um: tuple[float, float, float] | None = None
    mask: Source | None = None
    normals: Source | None = None

    def _validate(self):
        for name in ("id", "sample_id", "volume_id"):
            _name(getattr(self, name), name)
        if not isinstance(self.geometry, Storage):
            raise TypeError("geometry must be Storage")
        _axes(self.axes)
        _components(self.component_order)
        _affine(self.xyz_to_volume)
        _uv(self.pixel_to_uv)
        if self.spacing_um is not None:
            _vector(self.spacing_um, "spacing_um")
            if any(v <= 0 for v in self.spacing_um):
                raise ValueError("physical spacing must be positive")
        for source in (self.mask, self.normals):
            if source is not None and not isinstance(source, Source):
                raise TypeError("surface mask and normals must be Sources")


@dataclass(frozen=True)
class SurfaceRegistration(_Value):
    surface: Surface
    pixel_to_uv: tuple | None = None
    origin: str = "explicit"
    declared_grid: Selection | None = None

    def _validate(self):
        if not isinstance(self.surface, Surface):
            raise TypeError("surface must be a Surface")
        _uv(self.pixel_to_uv)
        _name(self.origin, "registration origin")
        if self.declared_grid is not None and not isinstance(self.declared_grid, Selection):
            raise TypeError("declared_grid must be a Selection")

    @property
    def sample_id(self):
        return self.surface.sample_id

    @property
    def volume_id(self):
        return self.surface.volume_id


@dataclass(frozen=True)
class Companion(_Value):
    name: str
    source: Source
    same_grid: bool = False

    def _validate(self):
        _name(self.name, "companion name")
        if self.name in ("coverage", "conflict"):
            raise ValueError(f"reserved companion name: {self.name}")
        if not isinstance(self.source, Source) or self.source.role not in ("label", "scalar"):
            raise ValueError("companion must be a label or scalar Source")
        if type(self.same_grid) is not bool:
            raise TypeError("same_grid must be boolean")
        if self.same_grid and self.source.registration is not None:
            raise ValueError("same_grid companion cannot also declare registration")
        if not self.same_grid and self.source.registration is None:
            raise ValueError("companion needs registration or explicit same_grid=True")


@dataclass(frozen=True)
class QueryContext(_Value):
    reference: Frame
    relation: str
    transform: Affine
    coverage: VolumeCoverage | None = None

    def _validate(self):
        if not isinstance(self.reference, Frame):
            raise TypeError("reference must be a Frame")
        if self.relation not in ("direct", "transform"):
            raise ValueError("unknown query relation")
        if not isinstance(self.transform, Affine):
            raise TypeError("query transform must be an Affine")
        if self.coverage is not None and not isinstance(self.coverage, VolumeCoverage):
            raise TypeError("query coverage must be VolumeCoverage")
        _affine(self.transform)


def _components(value):
    if value is not None and (not isinstance(value, tuple) or len(value) != 3 or set(value) != {"x", "y", "z"}):
        raise ValueError("component_order must be a permutation of x, y, z")


@dataclass(frozen=True)
class Source(_Value):
    id: str
    storage: Storage = field(default_factory=Storage)
    role: str | None = None
    axes: tuple[str, ...] | None = None
    registration: VolumeRegistration | SurfaceRegistration | None = None
    component_order: tuple[str, ...] | None = None
    value_frame: ValueFrame | None = None
    normalize: bool | None = None
    encoding: Encoding | None = None
    companions: tuple[Companion, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=FrozenMapping)
    artifact_id: str | None = None
    artifact_kind: str | None = None
    sample_id: str | None = None
    owner_id: str | None = None
    model_id: str | None = None
    label_set: str | None = None
    supersedes: tuple[str, ...] = ()
    coverage: tuple[VolumeCoverage, ...] = ()
    derived: bool = False
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    fill_value: int | float | None = None
    query: QueryContext | None = None

    def _validate(self):
        _name(self.id, "source ID")
        if not self.id.startswith(("catalog:", "user:")):
            object.__setattr__(self, "id", "user:" + self.id)
        if not isinstance(self.storage, Storage):
            raise TypeError(f"{self.id}: storage must be Storage")
        if self.role not in (None, "image", "label", "scalar", "vector", "normal", "coordinate"):
            raise ValueError(f"{self.id}: unknown role {self.role!r}")
        _axes(self.axes)
        _components(self.component_order)
        for value, cls, name in ((self.registration, (VolumeRegistration, SurfaceRegistration), "registration"),
                                 (self.value_frame, ValueFrame, "value_frame"),
                                 (self.encoding, Encoding, "encoding"), (self.query, QueryContext, "query")):
            if value is not None and not isinstance(value, cls):
                raise TypeError(f"{self.id}: invalid {name}")
        reg = self.registration
        if reg is not None and reg.declared_grid is None and (
            isinstance(reg, SurfaceRegistration) or reg.mode == "array"
        ):
            object.__setattr__(self, "registration", replace(reg, declared_grid=self.storage.selection))
        if self.normalize is not None and type(self.normalize) is not bool:
            raise TypeError(f"{self.id}: normalize must be boolean")
        if type(self.derived) is not bool:
            raise TypeError(f"{self.id}: derived must be boolean")
        if self.shape is not None:
            if not isinstance(self.shape, tuple) or not self.shape:
                raise ValueError(f"{self.id}: shape must be a nonempty sequence")
            for size in self.shape:
                _index(size, "shape")
            if self.axes is not None and len(self.shape) != len(self.axes):
                raise ValueError(f"{self.id}: shape and axes have different ranks")
        if not isinstance(self.provenance, Mapping):
            raise TypeError(f"{self.id}: provenance must be a mapping")
        for name in ("companions", "coverage", "supersedes"):
            if not isinstance(getattr(self, name), tuple):
                raise TypeError(f"{self.id}: {name} must be a sequence")
        for c in self.companions:
            if not isinstance(c, Companion):
                raise TypeError(f"{self.id}: companions must be Companion objects")
        if len({c.name for c in self.companions}) != len(self.companions):
            raise ValueError(f"{self.id}: duplicate companion name")
        for c in self.coverage:
            if not isinstance(c, VolumeCoverage):
                raise TypeError(f"{self.id}: coverage must contain VolumeCoverage objects")
        for identifier in self.supersedes:
            _name(identifier, "superseded source ID")

    @property
    def unresolved_fields(self) -> tuple[str, ...]:
        missing = []
        if self.storage.location is None and not self.storage.locations:
            missing.append("storage.location")
        for name, value in (("storage.format", self.storage.format), ("role", self.role), ("axes", self.axes)):
            if value is None:
                missing.append(name)
        reg = self.registration
        if reg is None:
            missing.append("registration")
        elif isinstance(reg, VolumeRegistration):
            if reg.affine is None and reg.plane is None:
                missing.append("registration.placement")
        else:
            for name, value in (("pixel_to_uv", reg.pixel_to_uv), ("surface.xyz_to_volume", reg.surface.xyz_to_volume),
                                ("surface.pixel_to_uv", reg.surface.pixel_to_uv),
                                ("surface.geometry.location", reg.surface.geometry.location),
                                ("surface.geometry.format", reg.surface.geometry.format),
                                ("surface.axes", reg.surface.axes),
                                ("surface.component_order", reg.surface.component_order)):
                if value is None:
                    missing.append("registration." + name)
        if self.storage.pyramid and self.storage.selection == Selection():
            missing.append("storage.selection")
        if self.storage.selection.corresponding is not None:
            missing.append("storage.selection.corresponding")
        if self.role in ("vector", "normal", "coordinate"):
            if self.component_order is None:
                missing.append("component_order")
            if self.value_frame is None:
                missing.append("value_frame")
            elif self.value_frame.kind == "volume" and self.value_frame.affine is None and self.value_frame.level is None:
                missing.append("value_frame.placement")
        if self.role == "vector" and self.normalize is None:
            missing.append("normalize")
        for companion in self.companions:
            missing.extend(f"companions.{companion.name}.{name}" for name in companion.source.unresolved_fields
                           if not (companion.same_grid and name == "registration"))
        return tuple(missing)

    def with_selection(self, **kwargs) -> Source:
        """Replace storage selection; the declaration's original grid stays fixed."""
        return replace(self, storage=replace(self.storage, selection=Selection(**kwargs)))

    def open(self, **reader_options):
        """Open a source for spatial crops; optional readers load only here."""
        from .reader.source_reader import SourceReader

        return SourceReader(self, **reader_options)

    def to_dict(self) -> dict:
        return {"schema_version": 1, "source": _encode(self)}

    @classmethod
    def from_dict(cls, value: Mapping) -> Source:
        if set(value) != {"schema_version", "source"} or type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("unsupported source schema version or envelope")
        source = _decode(value["source"])
        if not isinstance(source, cls):
            raise ValueError("serialized root must be a Source")
        return source


@dataclass
class SampleResult:
    """One unbatched result; arrays and runtime validation arrive with sampling.

    Dense data are CZYX; valid/mask arrays have one channel. Every data key has
    a valid entry and a masks dictionary (possibly empty). Metadata is optional
    and should be omitted or converted before default DataLoader collation.
    """

    data: dict[str, Any]
    valid: dict[str, Any]
    masks: dict[str, dict[str, Any]]
    metadata: dict | None = None


_TYPES = {cls.__name__: cls for cls in (Frame, LevelRequest, Selection, Storage, Plane,
          VolumeRegistration, ValueFrame, Encoding, Surface, SurfaceRegistration,
          Companion, QueryContext, Source, Affine, VolumeCoverage)}


def _encode(value):
    if isinstance(value, (_Value, Affine, VolumeCoverage)):
        return {"type": type(value).__name__, "fields": {f.name: _encode(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, Mapping):
        return {"type": "mapping", "items": [[k, _encode(v)] for k, v in value.items()]}
    if isinstance(value, tuple):
        return [_encode(v) for v in value]
    return value


def _decode(value):
    if isinstance(value, list):
        return tuple(_decode(v) for v in value)
    if isinstance(value, Mapping):
        tag = value.get("type")
        if tag == "mapping" and set(value) == {"type", "items"}:
            pairs = value["items"]
            if any(not isinstance(p, (list, tuple)) or len(p) != 2 or not isinstance(p[0], str) for p in pairs):
                raise ValueError("invalid mapping entries")
            if len({k for k, _ in pairs}) != len(pairs):
                raise ValueError("duplicate mapping key")
            return _freeze({k: _decode(v) for k, v in pairs})
        if tag not in _TYPES or set(value) != {"type", "fields"}:
            raise ValueError(f"unknown descriptor tag: {tag!r}")
        return _freeze(_TYPES[tag](**{k: _decode(v) for k, v in value["fields"].items()}))
    return _freeze(value)
