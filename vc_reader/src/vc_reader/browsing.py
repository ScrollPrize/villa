"""Scroll-centered views over shared metadata records and storage descriptors."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import copy
import math
import operator

from .affine import Affine
from .artifacts import ArtifactType
from .catalog import Catalog, DEFAULT_RESOLUTION_REL_TOL, PYRAMID_LEVELS
from .errors import AmbiguousMatchError, NotFoundError
from .sources import Source, SurfaceRegistration, VolumeRegistration


def _id(value):
    return value.id if hasattr(value, 'id') else value


def _surface_volumes(images):
    return Collection((a for a in images._all
                       if a.artifact_kind in (ArtifactType.LAYERS_ZARR, ArtifactType.LAYERS_TIF)),
                      include_superseded=images.include_superseded, ordered=True)


class Collection:
    """Deterministic views with explicit filtering and unambiguous selection.

    Iteration hides only explicitly superseded entities. ID lookup also finds
    historical entities; filter(include_superseded=True) includes them in queries.
    """
    def __init__(self, values=(), *, include_superseded=False, ordered=False):
        values = tuple(values)
        self._all = values if ordered else tuple(sorted(values, key=lambda v: (v.id, getattr(v, 'sample_id', '') or '')))
        self.include_superseded = include_superseded

    def __iter__(self):
        return (v for v in self._all if self.include_superseded or not getattr(v, 'is_superseded', False))

    def __len__(self):
        return sum(1 for _ in self)

    def __bool__(self):
        return next(iter(self), None) is not None

    def __getitem__(self, identifier):
        matches = [v for v in self._all if v.id == identifier]
        return self._one(matches, f'ID {identifier!r}')

    def _one(self, matches, description):
        if len(matches) == 1:
            return matches[0]
        candidates = [v.id for v in (matches or self._all)]
        message = f'{description}: {len(matches)} matches; candidates: {candidates}'
        if not matches:
            raise NotFoundError(message)
        raise AmbiguousMatchError(message, matches)

    def filter(self, *, include_superseded=None, **criteria):
        if include_superseded is not None and type(include_superseded) is not bool:
            raise TypeError('include_superseded must be boolean')
        def matches(value):
            return all(_id(getattr(value, key)) == _id(wanted) for key, wanted in criteria.items())
        return Collection((v for v in self._all if matches(v)), ordered=True,
                          include_superseded=self.include_superseded if include_superseded is None else include_superseded)

    def get(self, **criteria):
        """Return the item matching all criteria, requiring a unique match.

        Raise NotFoundError for no match or AmbiguousMatchError for multiple
        matches. With no criteria, the collection must contain exactly one item.
        """
        return self._one(list(self.filter(**criteria)), repr(criteria))

    def ink(self, **criteria):
        """Return the matching ink label, raising for missing or ambiguous matches.

        Use ``filter(artifact_kind='ink-labels')`` to browse multiple labels.
        """
        return self.filter(artifact_kind='ink-labels').get(**criteria)

    def zarr(self, **criteria):
        """Return the matching Zarr asset, raising for missing or ambiguous matches.

        Use ``filter(format='zarr')`` to browse multiple assets.
        """
        return self.filter(format='zarr').get(**criteria)

    @property
    def volumes(self):
        # Aggregation retains each owning scroll even when volume IDs coincide.
        return Collection((v for s in self for v in s.volumes._all), include_superseded=self.include_superseded)

    def at_resolution(self, um, *, rel_tol=DEFAULT_RESOLUTION_REL_TOL):
        if not math.isfinite(um) or um <= 0 or not math.isfinite(rel_tol) or rel_tol < 0:
            raise ValueError('resolution must be positive and tolerance nonnegative, both finite')
        matches = []
        for volume in self:
            if not isinstance(volume, Volume):
                raise TypeError('at_resolution requires a volume collection')
            if volume.pixel_size_um is None:
                continue
            for asset in volume.images:
                count = PYRAMID_LEVELS if asset.source.storage.pyramid else 1
                for level in range(count):
                    estimate = volume.pixel_size_um * 2**level
                    if math.isclose(estimate, um, rel_tol=rel_tol):
                        matches.append(ResolutionCandidate(volume, asset.source, level, estimate))
        return Collection(sorted(matches, key=lambda m: (abs(m.estimated_resolution_um-um), m.scroll.id, m.volume.id, m.level, m.source.id)), ordered=True)

    def __repr__(self):
        return f'Collection({len(self)} entities)'


class _Context:
    def _assets(self, kind=None):
        if isinstance(self, ScrollSource):
            identifiers = self._index.by_sample.get(self.id, ()) if self._volume is None else self._index.by_volume.get((self.id, self._volume), ())
        elif isinstance(self, Volume):
            identifiers = self._index.by_volume.get((self.sample_id, self.id), ())
        else:
            identifiers = self._index.by_segment.get((self.sample_id, self.id), ())
            if self._volume is not None:
                identifiers = tuple(i for i in identifiers if self._index.volume_ids[i] == self._volume)
        return Collection(Asset(self._index, i) for i in identifiers if kind is None or self._index.kinds[i] == kind)

    @property
    def artifacts(self):
        return self._assets()

    @property
    def labels(self):
        return self._assets('label')

    @property
    def images(self):
        return self._assets('image')

    @property
    def surface_volumes(self):
        """Rendered layer stacks associated with this sample, volume or segment.

        Depth follows sampled surface layers, rather than the CT voxel Z axis.
        """
        return _surface_volumes(self._assets('image'))

    @property
    def representations(self):
        return self._assets('representation')

    @property
    def geometry(self):
        return self._assets('geometry')


class ScrollSource(_Context):
    """A sample with optional direct-volume context; navigation requests no arrays."""
    def __init__(self, scroll, volume=None, *, catalog=None):
        catalog = Catalog.load() if catalog is None else catalog
        ids = catalog._names.resolve(scroll)
        if scroll in catalog._samples:
            ids = {scroll}
        if not ids:
            raise NotFoundError(f'no sample {scroll!r}')
        if len(ids) != 1:
            raise AmbiguousMatchError(f'ambiguous sample {scroll!r}: {sorted(ids)}', sorted(ids))
        self._index, self.id, self._volume = catalog._index, next(iter(ids)), None
        if volume is not None:
            self._volume = self.volumes[_id(volume)].id

    @classmethod
    def _view(cls, index, identifier, volume=None):
        result = object.__new__(cls)
        result._index, result.id, result._volume = index, identifier, volume
        return result

    @property
    def catalog(self):
        return self._index.catalog

    @property
    def record(self):
        return self.catalog._samples[self.id]

    def __getattr__(self, name):
        if name not in ('raw', 'properties', 'description', 'type', 'display_name', 'aliases', 'scans'):
            raise AttributeError(name)
        return getattr(self.record, name)

    @property
    def volume(self):
        return None if self._volume is None else self.volumes[self._volume]

    @property
    def volumes(self):
        identifiers = set(self.record.volumes)
        identifiers.update(e['volume'] for e in self._index.overlays if e['sample'] == self.id and e['kind'] == 'volume')
        return Collection(Volume(self._index, self.id, v) for v in identifiers)

    @property
    def segments(self):
        identifiers = set(self.record.segments)
        identifiers.update(e['segment'] for e in self._index.overlays if e['sample'] == self.id and e['segment'])
        if self._volume is not None:
            identifiers &= {self._index.segment_ids[i] for i in self._index.by_volume.get((self.id, self._volume), ())}
        return Collection(Segment(self._index, self.id, s, self._volume) for s in identifiers)

    def on_volume(self, identifier):
        identifier = self.volumes[_id(identifier)].id
        return self._view(self._index, self.id, identifier)

    def transform(self, source, destination):
        return self.record.transform(_id(source), _id(destination))

    def with_source(self, source, *, kind, volume=None, segment=None, image=None, label=None,
                    same_grid=False, pixel_affine=None):
        """Attach a descriptor to a new view without mutating published metadata.

        kind is volume/image/label/representation/geometry/artifact. Pairing
        requires image= (for labels) or label= (for images) and exactly one of
        same_grid=True or a 2x3 label-XY-to-image-XY pixel_affine.
        """
        from .discovery import RelationshipIndex
        if not isinstance(source, Source) or not source.id.startswith('user:'):
            raise ValueError('custom sources require the user: namespace')
        if kind not in ('volume', 'image', 'label', 'representation', 'geometry', 'artifact'):
            raise ValueError(f'unknown asset kind {kind!r}')
        if type(same_grid) is not bool:
            raise TypeError('same_grid must be boolean')
        if source.sample_id not in (None, self.id):
            raise ValueError('source sample differs from owning scroll')
        volume, segment = _id(volume), _id(segment)
        if kind == 'volume':
            volume = volume or source.id
            if volume in self.record.volumes or not volume.startswith('user:'):
                raise ValueError('new volume IDs must use the user: namespace')
        else:
            volume = volume or self._volume
            if volume is not None:
                self.volumes[volume]
        if segment is not None and segment not in self.record.segments and not segment.startswith('user:'):
            raise ValueError('new segment IDs must use the user: namespace')
        if source.role is None and kind in ('volume', 'image', 'label'):
            source = replace(source, role='label' if kind == 'label' else 'image')
        reg = source.registration
        if reg is not None and (reg.sample_id != self.id or (volume is not None and reg.volume_id != volume)):
            raise ValueError('source registration contradicts owning sample/volume')
        volume = volume or (reg.volume_id if reg is not None else None)
        if kind == 'volume' and reg is None:
            source = replace(source, registration=VolumeRegistration(self.id, volume, affine=Affine.identity()))
        source = replace(source, sample_id=self.id, owner_id=segment or volume or self.id)
        pairs = []
        if image is not None or label is not None:
            if (image is not None and (kind != 'label' or label is not None)) or (label is not None and kind != 'image'):
                raise ValueError('pair a label with image=, or an image with label=')
            if same_grid == (pixel_affine is not None):
                raise ValueError('declare exactly one of same_grid=True or pixel_affine')
            matrix = ((1., 0., 0.), (0., 1., 0.)) if same_grid else pixel_affine
            from .sources import _uv
            matrix = tuple(tuple(row) for row in matrix)
            _uv(matrix)
            Affine.from_rows(((matrix[0][0], matrix[0][1], 0, matrix[0][2]),
                              (matrix[1][0], matrix[1][1], 0, matrix[1][2]), (0, 0, 1, 0))).inverse()
            pairs.append(dict(label=source.id if kind == 'label' else _id(label),
                              image=_id(image) if kind == 'label' else source.id, affine=matrix, same_grid=same_grid))
        elif same_grid or pixel_affine is not None:
            raise ValueError('pixel mapping requires an explicit paired asset')
        entry = dict(source=source.to_dict(), kind=kind, sample=self.id, volume=volume, segment=segment, pairs=pairs)
        index = RelationshipIndex(self.catalog, (*self._index.overlays, entry))
        return self._view(index, self.id, self._volume)

    def to_dict(self):
        return dict(schema_version=1, catalog=copy.deepcopy(self.catalog.raw), overlays=copy.deepcopy(list(self._index.overlays)), scroll=self.id, volume=self._volume)

    @classmethod
    def from_dict(cls, value):
        from .discovery import RelationshipIndex
        if set(value) != {'schema_version', 'catalog', 'overlays', 'scroll', 'volume'} or type(value.get('schema_version')) is not int or value['schema_version'] != 1:
            raise ValueError('unsupported scroll view schema version')
        catalog = Catalog(value['catalog'])
        index = RelationshipIndex(catalog, value['overlays'])
        if value['scroll'] not in catalog._samples:
            raise ValueError('overlay view names an unknown sample')
        view = cls._view(index, value['scroll'], value['volume'])
        if value['volume'] is not None:
            view.volumes[value['volume']]
        return view

    def __repr__(self):
        return f'ScrollSource({self.id!r}, volume={self._volume!r})'


class Volume(_Context):
    def __init__(self, index, sample, identifier):
        self._index, self.sample_id, self.id = index, sample, identifier

    @property
    def scroll(self):
        return ScrollSource._view(self._index, self.sample_id, self.id)

    @property
    def record(self):
        return self._index.catalog._samples[self.sample_id].volumes.get(self.id)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        record = self.record
        if record is None and name in ('pixel_size_um', 'shape_zyx', 'is_superseded', 'long_id', 'energy_kev', 'scan_id'):
            if name == 'pixel_size_um':
                sources = list(self.images)
                reg = sources[0].source.registration if len(sources) == 1 else None
                spacing = reg.volume_spacing_um if isinstance(reg, VolumeRegistration) else None
                return spacing[0] if spacing is not None and len(set(spacing)) == 1 else None
            return False if name == 'is_superseded' else None
        return getattr(record, name)

    @property
    def images(self):
        return self._assets('volume')

    @property
    def source(self):
        return self.images.get().source

    @property
    def segments(self):
        return self.scroll.segments

    def level(self, level):
        return VolumeLevel(self, level)

    def open(self, *, level=0, **options):
        return self.level(level).open(**options)

    def open_aligned(self, targets=None, *, level=0, backend='cpu', **options):
        from .sampling import open_sample
        return open_sample(self.level(level), {k: v.source if isinstance(v, Asset) else v for k, v in (targets or {}).items()},
                           backend=backend, **options)

    def transform_to(self, other):
        return self.scroll.transform(self.id, _id(other))

    def __repr__(self):
        return f'Volume({self.sample_id!r}, {self.id!r})'


class Segment(_Context):
    def __init__(self, index, sample, identifier, volume=None):
        self._index, self.sample_id, self.id, self._volume = index, sample, identifier, volume

    @property
    def record(self):
        return self._index.catalog._samples[self.sample_id].segments.get(self.id)

    @property
    def scroll(self):
        return ScrollSource._view(self._index, self.sample_id, self._volume)

    @property
    def volumes(self):
        ids = {self._index.volume_ids[i] for i in self._index.by_segment.get((self.sample_id, self.id), ())}
        return Collection(Volume(self._index, self.sample_id, v) for v in ids if v is not None)

    def on_volume(self, volume):
        volume = self.volumes[_id(volume)].id
        return Segment(self._index, self.sample_id, self.id, volume)

    def __repr__(self):
        return f'Segment({self.id!r}, volume={self._volume!r})'


class Asset:
    def __init__(self, index, identifier, *, source=None, parent=None):
        self._index, self.id, self._source, self._parent = index, identifier, source, parent

    @property
    def source(self):
        return self._source if self._source is not None else self._index.sources[self.id]

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(self.source, name)

    @property
    def kind(self):
        return self._index.kinds.get(self.id, 'mask')

    @property
    def format(self):
        return self.source.storage.format

    @property
    def is_superseded(self):
        return self.id in self._index.hidden

    @property
    def volume(self):
        parent = self._parent or self.id
        volume = self._index.volume_ids[parent]
        return None if volume is None else Volume(self._index, self._index.owners[parent], volume)

    @property
    def segment(self):
        parent = self._parent or self.id
        segment = self._index.segment_ids[parent]
        return None if segment is None else Segment(self._index, self._index.owners[parent], segment, self._index.volume_ids[parent])

    @property
    def scroll(self):
        return ScrollSource._view(self._index, self._index.owners[self._parent or self.id])

    @property
    def images(self):
        return Collection(Asset(self._index, p.image_id) for p in self._index.pairs.get(self.id, ()))

    @property
    def surface_volumes(self):
        """Surface layer stacks paired with this label; volume is the source CT."""
        return _surface_volumes(self.images)

    @property
    def labels(self):
        return Collection(Asset(self._index, p.label_id) for p in self._index.reverse_pairs.get(self.id, ()))

    @property
    def masks(self):
        return Collection(Asset(self._index, c.source.id, source=c.source, parent=self.id) for c in self.source.companions)

    @property
    def depth_semantics(self):
        return 'sampled_layers' if self.kind == 'image' else None

    @property
    def mappable_volumes(self):
        """Volume affines reachable from this asset's source volume.

        Surface labels retain their source volume even when pixel-to-UV fields
        are unresolved. Inspect required_operations before opening a destination.
        This property reads metadata only.
        """
        source_volume = self.volume
        if source_volume is None or self.source.registration is None:
            return Collection()
        result = []
        for volume in self.scroll.volumes._all:
            transform = self.scroll.transform(source_volume.id, volume.id)
            if transform is None:
                continue
            coverage = next((c for c in self.source.coverage if c.volume_id == volume.id), None)
            operations = ('surface_projection',) if isinstance(self.source.registration, SurfaceRegistration) else ()
            if volume.id != source_volume.id:
                operations += ('volume_transform',)
            result.append(VolumeMapping(volume, transform, coverage, operations))
        return Collection(result)

    def summary(self):
        """JSON-ready source ownership, volume mappings and unresolved fields.

        Mapping availability and pixel/geometry declarations are independent.
        Coverage describes published segment bounds, not a crop guarantee.
        No storage headers or image arrays are requested.
        """
        source_volume, segment = self.volume, self.segment
        return dict(id=self.id, kind=self.kind, sample=self.scroll.id,
                    segment=None if segment is None else segment.id,
                    source_volume=None if source_volume is None else source_volume.id,
                    unresolved_fields=list(self.source.unresolved_fields),
                    mappable_volumes=[dict(volume=m.id, required_operations=list(m.required_operations),
                        affine=m.transform.matrix, derivation_path=m.transform.derivation_path,
                        coverage=None if m.coverage is None else asdict(m.coverage))
                        for m in self.mappable_volumes])

    def open(self, *, level=None, **options):
        source = self.source if level is None else _select_level(self.source, level)
        return source.open(**options)

    def open_paired(self, *, image=None, level=0, layers=slice(None), backend='cpu', **options):
        from .paired import PairedReader
        if self.kind != 'label':
            raise TypeError('open_paired requires a label')
        if image is None:
            image = self.images.get()
        if not isinstance(image, Asset) or self._index.sources.get(image.id) != image.source:
            raise ValueError('image must belong to the same scroll view')
        matches = [p for p in self._index.pairs.get(self.id, ()) if p.image_id == image.id]
        if len(matches) != 1:
            raise ValueError('label/image relationship is absent or ambiguous')
        return PairedReader(self.source, image.source, matches[0], level=level, layers=layers, backend=backend, **options)

    def __repr__(self):
        source_volume = self.volume
        details = f', source_volume={source_volume.id!r}' if source_volume is not None else ''
        if self.kind == 'label':
            details += f', mappable_volumes={[m.id for m in self.mappable_volumes]!r}'
        return f'Asset({self.id!r}, kind={self.kind!r}{details})'


@dataclass(frozen=True)
class VolumeMapping:
    volume: Volume
    transform: Affine
    coverage: object
    required_operations: tuple

    @property
    def id(self):
        return self.volume.id

    @property
    def is_superseded(self):
        return self.volume.is_superseded


def _select_level(source, level):
    level = operator.index(level)
    if level < 0:
        raise ValueError('level must be nonnegative')
    if source.storage.pyramid:
        selection = replace(source.storage.selection, level=level, array_path=None, corresponding=None)
        return replace(source, storage=replace(source.storage, selection=selection))
    if level:
        raise ValueError('a single-resolution array has only level 0')
    return source


@dataclass(frozen=True)
class VolumeLevel:
    volume: Volume
    level: int

    def __post_init__(self):
        if operator.index(self.level) < 0:
            raise ValueError('level must be nonnegative')

    @property
    def source(self):
        return _select_level(self.volume.source, self.level)

    @property
    def voxel_size_um(self):
        return None if self.volume.pixel_size_um is None else self.volume.pixel_size_um * 2**self.level

    def open(self, **options):
        return self.source.open(**options)


@dataclass(frozen=True)
class ResolutionCandidate:
    volume: Volume
    source: Source
    level: int
    estimated_resolution_um: float
    verified: bool = False
    selection: object = None
    spacing_um: tuple | None = None
    shape: tuple | None = None

    @property
    def id(self):
        return f'{self.volume.sample_id}/{self.volume.id}/{self.level}/{self.source.id}'

    @property
    def scroll(self):
        return self.volume.scroll

    def _verify(self, reader):
        resolved = reader.resolved
        matrix = resolved.storage_from_array
        if matrix is not None and resolved.storage_units == 'micrometer':
            spacing = tuple(math.sqrt(sum(matrix.linear[k][i]**2 for k in range(3))) for i in (2, 1, 0))
        else:
            reg = resolved.registration
            if not isinstance(reg, VolumeRegistration) or reg.affine is None:
                raise ValueError('storage has no verifiable volume grid')
            spacing = tuple(math.sqrt(sum(reg.affine.linear[k][i]**2 for k in range(3))) * self.volume.pixel_size_um for i in (2, 1, 0))
        if not all(math.isclose(v, self.estimated_resolution_um, rel_tol=1e-6) for v in spacing):
            raise ValueError(f'storage spacing {spacing} contradicts catalog estimate {self.estimated_resolution_um}')
        expected = self.volume.shape_zyx
        spatial_shape = tuple(reader.shape[reader.axes.index(a)] for a in ('z', 'y', 'x'))
        if expected is not None:
            expected = tuple(-(-n // 2**self.level) for n in expected)
            if spatial_shape != expected:
                raise ValueError(f'storage shape {spatial_shape} contradicts catalog shape {expected}')
        return replace(self, verified=True, selection=resolved.selection, spacing_um=spacing, shape=spatial_shape)

    def resolve(self, **options):
        with _select_level(self.source, self.level).open(**options) as reader:
            return self._verify(reader)

    def open(self, **options):
        reader = _select_level(self.source, self.level).open(**options)
        try:
            self._verify(reader)
            return reader
        except BaseException:
            reader.close()
            raise
