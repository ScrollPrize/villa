"""Metadata-only catalog adapters and indexed source queries."""
from __future__ import annotations

import copy
import hashlib
import json
from collections import OrderedDict, defaultdict
from dataclasses import replace
from threading import RLock
from urllib.parse import quote

from .affine import Affine
from .artifacts import ArtifactType as T
from .artifacts import pyramid_sublevel
from .catalog import (
    Catalog, InkLabelSet, Volume, VolumeLevel, _filter_ink_labels, _match_ink_labels,
)
from .errors import AmbiguousMatchError, NotFoundError
from .sources import (
    Companion, Frame, LevelRequest, QueryContext, Selection, Source, Storage,
    Surface, SurfaceRegistration, VolumeRegistration,
)


def _storage(artifact, *, format=None, pyramid=False):
    return Storage(
        location=artifact.url() or artifact.url(None), alternatives=artifact.urls(),
        format=format, pyramid=pyramid, grid_level=artifact.grid_level,
    )


def _sort_key(source):
    reg = source.registration
    return (source.sample_id or (reg.sample_id if reg else ""), reg.volume_id if reg else "", source.id)


class SourceCollection:
    """A workflow's immutable source snapshot, without opening any storage.

    User IDs use ``user:``; catalog IDs are fully qualified. Iteration includes
    unknown assets. Query results are new lists of immutable values. A bounded
    128-entry query cache avoids repeating adaptation in dataset item loops.
    """

    def __init__(self, catalog: Catalog, entries=()):
        if not isinstance(catalog, Catalog):
            raise TypeError("catalog must be a Catalog")
        self.catalog = catalog
        digest = hashlib.sha256()
        encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        for chunk in encoder.iterencode(catalog.raw):
            digest.update(chunk.encode())
        self.snapshot_sha256 = digest.hexdigest()
        self._cache = OrderedDict()
        self._cache_lock = RLock()
        self._sources = {}
        self._artifact_ids = {}
        self._ink = defaultdict(list)
        self._samples = {}
        self._by_frame = defaultdict(list)

        # Copy only relationship state used by queries; no duplicate raw catalog
        # or storage handles. Existing Sample.transform remains authoritative.
        for sample in catalog.samples.values():
            frozen_sample = copy.copy(sample)
            frozen_sample._transforms = dict(sample._transforms)
            frozen_sample.volumes = {}
            for volume in sample.volumes.values():
                frozen_volume = copy.copy(volume)
                frozen_volume.sample = frozen_sample
                frozen_sample.volumes[volume.id] = frozen_volume
            self._samples[sample.id] = frozen_sample

        owners = []
        for sample in sorted(catalog.samples.values(), key=lambda s: s.id):
            owners.append((sample, "sample", sample))
            owners.extend((sample, "volume", v) for v in sample.volume_list())
            owners.extend((sample, "segment", s) for s in sample.segment_list())
        for sample, kind, owner in owners:
            for index, artifact in enumerate(owner.artifacts):
                owner_key = "/".join(quote(v, safe="") for v in (sample.id, kind, owner.id))
                suffix = "id/" + quote(artifact.id, safe="") if artifact.id is not None else (
                    f"snapshot/{self.snapshot_sha256}/data/{index}"
                )
                identifier = f"catalog:{owner_key}/{suffix}"
                self._artifact_ids[artifact] = identifier

        for sample, kind, owner in owners:
            ink_sets = []
            for artifact in owner.artifacts:
                source = self._adapt(sample, kind, owner, artifact)
                self._add(source)
                if kind == "segment" and artifact.type == T.INK_LABELS:
                    ink_sets.append(InkLabelSet(owner, artifact))
                else:
                    self._index_source(source)
            if ink_sets:
                self._ink[sample.id].append((tuple(ink_sets), dict(owner.coverage)))

        for source in entries:
            if not isinstance(source, Source) or not source.id.startswith("user:"):
                raise ValueError("collection entries must be Sources in the user: namespace")
            self._add(source)
            self._index_source(source)
        self._validate_supersession()
        self._ordered = tuple(sorted(self._sources.values(), key=_sort_key))
        self._by_frame = {key: tuple(value) for key, value in self._by_frame.items()}

    def _add(self, source):
        previous = self._sources.get(source.id)
        if previous is not None:
            raise ValueError(f"duplicate source ID {source.id}: {previous.provenance!r} and {source.provenance!r}")
        self._sources[source.id] = source

    def _index_source(self, source):
        if source.registration is not None and source.role is not None:
            reg = source.registration
            self._by_frame[(reg.sample_id, reg.volume_id)].append(source)

    def _adapt(self, sample, kind, owner, artifact):
        identifier = self._artifact_ids[artifact]
        storage = _storage(artifact)
        role = axes = registration = None
        companions = ()
        derived = kind == "volume" and artifact.type != T.OME_ZARR
        if kind == "volume":
            registration = VolumeRegistration(sample.id, owner.id, origin="catalog")
            if artifact.type in (T.OME_ZARR, T.SURFACE_PREDICTION_ZARR, T.INK_DETECTION_3D_ZARR):
                storage = _storage(artifact, format="zarr", pyramid=True)
                role = "image" if artifact.type == T.OME_ZARR else "scalar"
                axes = ("z", "y", "x")
                g = artifact.grid_level
                registration = VolumeRegistration(
                    sample.id, owner.id, affine=Affine.scale(2**g) if g is not None else None,
                    declared_grid=Selection(level=0), origin="catalog-fallback",
                    volume_spacing_um=((owner.pixel_size_um,) * 3 if owner.pixel_size_um is not None else None),
                )
            elif artifact.type == T.NORMAL_GRIDS:
                role = "normal"  # Bundle layout and component frame stay unknown.
        elif kind == "segment" and artifact.target_volume_id:
            target = artifact.target_volume_id
            if artifact.type == T.INK_LABELS:
                labels = InkLabelSet(owner, artifact)
                meshes = owner.artifacts_of(T.TIFXYZ_TRANSFORMED, target_volume=target)
                if len(meshes) > 1:
                    raise AmbiguousMatchError(f"{identifier}: multiple canonical surface grids", meshes)
                mesh = meshes[0] if meshes else None
                target_volume = sample.volumes.get(target)
                px = target_volume.pixel_size_um if target_volume else None
                surface = Surface(
                    id=self._artifact_ids[mesh] if mesh else identifier + "/surface",
                    geometry=_storage(mesh, format="tifxyz") if mesh else Storage(),
                    sample_id=sample.id, volume_id=target, axes=("y", "x"),
                    component_order=("x", "y", "z"),
                    xyz_to_volume=Affine.identity() if mesh else None,
                    spacing_um=(px, px, px) if px is not None else None,
                )
                registration = SurfaceRegistration(surface, origin="catalog")
                root = storage.location
                storage = Storage(
                    location=labels.array_url() or (root.rstrip("/") + "/inklabels.zarr/" if root else None),
                    alternatives=tuple(u.rstrip("/") + "/inklabels.zarr/" for u in artifact.urls()),
                    format="zarr", pyramid=True, selection=Selection(level=0),
                )
                role, axes = "label", ("y", "x")
                companions = tuple(Companion(
                    name, Source(
                        id=identifier + "/" + name,
                        storage=Storage(
                            location=root.rstrip("/") + f"/{name}.zarr/" if root else None,
                            alternatives=tuple(u.rstrip("/") + f"/{name}.zarr/" for u in artifact.urls()),
                            format="zarr", pyramid=True, selection=Selection(level=0),
                        ), role="label", axes=axes,
                    ), same_grid=True,
                ) for name in ("supervision", "validation") if name in labels.arrays)
            else:
                # Target identity is useful even when array/geometry semantics
                # are not declared by an adapter.
                registration = VolumeRegistration(sample.id, target, origin="catalog")

        supersedes = ()
        if artifact.type != T.INK_LABELS:
            declared = artifact.properties.get("supersedes") or ()
            declared = (declared,) if isinstance(declared, str) else declared
            prefix = identifier.split("/id/")[0] if artifact.id is not None else identifier.split("/snapshot/")[0]
            supersedes = tuple(v if v.startswith(("catalog:", "user:")) else prefix + "/id/" + quote(v, safe="")
                               for v in declared)
        return Source(
            id=identifier, storage=storage, role=role, axes=axes, registration=registration,
            companions=companions, artifact_id=artifact.id, artifact_kind=artifact.type,
            sample_id=sample.id, owner_id=owner.id, model_id=artifact.model_id,
            label_set=artifact.label_set, supersedes=supersedes, derived=derived,
            coverage=tuple(owner.coverage.values()) if kind == "segment" else (),
            provenance={"owner_kind": kind, "artifact": artifact.raw},
        )

    def _validate_supersession(self):
        visited, active = set(), set()
        for root in self._sources:
            stack = [(root, False)]
            while stack:
                identifier, done = stack.pop()
                if done:
                    active.remove(identifier)
                    visited.add(identifier)
                elif identifier in active:
                    raise ValueError(f"supersession cycle at {identifier}")
                elif identifier not in visited and identifier in self._sources:
                    active.add(identifier)
                    stack.append((identifier, True))
                    stack.extend((older, False) for older in self._sources[identifier].supersedes)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_cache_lock')
        state['_cache'] = OrderedDict()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cache_lock = RLock()

    def __iter__(self):
        return iter(self._ordered)

    def __len__(self):
        return len(self._ordered)

    def get(self, source_id):
        try:
            return self._sources[source_id]
        except KeyError:
            raise NotFoundError(f"no source {source_id!r}") from None

    def _reference(self, volume):
        level = volume.level if isinstance(volume, VolumeLevel) else None
        volume = volume.volume if isinstance(volume, VolumeLevel) else volume
        if not isinstance(volume, Volume):
            raise TypeError("volume must be a Volume or VolumeLevel")
        sample = self.catalog.samples.get(volume.sample.id)
        if sample is None or sample.volumes.get(volume.id) is not volume:
            raise ValueError("volume must belong to this collection's catalog")
        reference = Frame(volume.sample.id, volume.id, level or 0)
        return self._samples[volume.sample.id].volumes[volume.id], level, reference

    def _remember(self, key, results):
        values = tuple(results)
        with self._cache_lock:
            self._cache[key] = values
            self._cache.move_to_end(key)
            if len(self._cache) > 128:
                self._cache.popitem(last=False)
        return list(values)

    def _cached(self, key):
        with self._cache_lock:
            values = self._cache.get(key)
            if values is not None:
                self._cache.move_to_end(key)
                return list(values)
        return None

    def _context(self, source, reference, transform, *, level=None, representation=False, coverage=None):
        reg = source.registration
        query = QueryContext(reference, "direct" if reg.volume_id == reference.volume_id else "transform", transform, coverage)
        storage = source.storage
        if representation and level is not None and storage.pyramid:
            sublevel = pyramid_sublevel(storage.grid_level, level) if reg.volume_id == reference.volume_id else None
            storage = replace(storage, selection=replace(
                storage.selection, array_path=None, level=None, corresponding=LevelRequest(reference, sublevel),
            ))
        return replace(source, query=query, storage=storage)

    def _query(self, volume, *, labels, mappable, latest_only, label_set=None):
        if type(mappable) is not bool or type(latest_only) is not bool:
            raise TypeError("mappable and latest_only must be booleans")
        if label_set is not None and not isinstance(label_set, str):
            raise TypeError("label_set must be a string or None")
        volume, level, reference = self._reference(volume)
        key = ("labels" if labels else "representations", reference, level, mappable, latest_only, label_set)
        cached = self._cached(key)
        if cached is not None:
            return cached
        results = []
        if labels:
            for sets, coverage in self._ink[volume.sample.id]:
                selected = _filter_ink_labels(sets, label_set=label_set, latest_only=latest_only)
                for match in _match_ink_labels(selected, volume, coverage.get(volume.id), mappable=mappable):
                    source = self._sources[self._artifact_ids[match.labels.artifact]]
                    results.append(self._context(source, reference, match.transform, coverage=match.coverage))

        generic = []
        for (sample_id, volume_id), candidates in self._by_frame.items():
            if sample_id != volume.sample.id or (not mappable and volume_id != volume.id):
                continue
            transform = volume.sample.transform(volume_id, volume.id)
            if transform is None:
                continue
            for source in candidates:
                if labels:
                    include = source.role == "label" and (label_set is None or source.label_set == label_set)
                else:
                    include = source.role in ("scalar", "vector", "normal", "coordinate") or (source.role == "image" and source.derived)
                if include:
                    coverage = next((c for c in source.coverage if c.volume_id == volume.id), None)
                    generic.append((source, transform, coverage))
        hidden = {old for source, _, _ in generic for old in source.supersedes} if latest_only else set()
        for source, transform, coverage in generic:
            if source.id not in hidden:
                results.append(self._context(source, reference, transform, level=level,
                                             representation=not labels, coverage=coverage))
        return self._remember(key, sorted(results, key=_sort_key))

    def labels(self, *, volume, mappable=True, latest_only=True, label_set=None) -> list[Source]:
        return self._query(volume, labels=True, mappable=mappable, latest_only=latest_only, label_set=label_set)

    def representations(self, *, volume, mappable=True, latest_only=True) -> list[Source]:
        return self._query(volume, labels=False, mappable=mappable, latest_only=latest_only)

    def registered_volumes(self, *, volume) -> list[Volume]:
        target, _, reference = self._reference(volume)
        key = ("volumes", reference.sample_id, reference.volume_id)
        cached = self._cached(key)
        if cached is not None:
            return cached
        original = self.catalog.samples[reference.sample_id].volumes
        return self._remember(key, (original[v.id] for v in target.sample.volume_list()
                                   if v.id != target.id and v.transform_to(target) is not None))
