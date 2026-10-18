"""One metadata-only index shared by every public contextual view."""
from __future__ import annotations
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, replace
from urllib.parse import quote
from .affine import Affine
from .artifacts import ArtifactType as T
from .sources import Companion, Source, Storage, Selection, Surface, SurfaceRegistration, VolumeRegistration


def _storage(artifact, *, format=None, pyramid=False):
    return Storage(location=artifact.url() or artifact.url(None), alternatives=artifact.urls(),
                   format=format, pyramid=pyramid, grid_level=artifact.grid_level)


@dataclass(frozen=True)
class Pairing:
    label_id: str
    render_id: str
    # Label base pixel XY -> render base pixel XY. None means inspect metadata.
    pixel_affine: tuple | None
    evidence: str


class RelationshipIndex:
    def __init__(self, catalog, overlays=()):
        self.catalog, self.overlays = catalog, tuple(overlays)
        if overlays:
            self.snapshot_sha256 = catalog._index.snapshot_sha256
        else:
            digest = hashlib.sha256()
            encoder = json.JSONEncoder(sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)
            for chunk in encoder.iterencode(catalog.raw):
                digest.update(chunk.encode())
            self.snapshot_sha256 = digest.hexdigest()
        self.sources, self._artifact_ids = {}, {}
        self.owners, self.kinds, self.volume_ids, self.segment_ids = {}, {}, {}, {}
        self.by_sample, self.by_volume, self.by_segment = defaultdict(list), defaultdict(list), defaultdict(list)
        self.pairs, self.reverse_pairs = defaultdict(list), defaultdict(list)
        self.hidden = set()
        if overlays:
            base = catalog._index
            for identifier, source in base.sources.items():
                self._add(source, base.kinds[identifier], base.owners[identifier], base.volume_ids[identifier], base.segment_ids[identifier])
        else:
            owners = []
            for sample in sorted(catalog._samples.values(), key=lambda s: s.id):
                owners.append((sample, 'sample', sample))
                owners.extend((sample, 'volume', v) for v in sample.volume_list())
                owners.extend((sample, 'segment', s) for s in sample.segment_list())
            for sample, kind, owner in owners:
                for position, artifact in enumerate(owner.artifacts):
                    owner_key = '/'.join(quote(v, safe='') for v in (sample.id, kind, owner.id))
                    suffix = 'id/' + quote(artifact.id, safe='') if artifact.id is not None else f'snapshot/{self.snapshot_sha256}/data/{position}'
                    self._artifact_ids[artifact] = f'catalog:{owner_key}/{suffix}'
            for sample, kind, owner in owners:
                for artifact in owner.artifacts:
                    source = self._adapt(sample, kind, owner, artifact)
                    asset_kind = ('volume' if kind == 'volume' and artifact.type == T.OME_ZARR else
                                  'label' if artifact.type == T.INK_LABELS else
                                  'render' if artifact.type in (T.LAYERS_ZARR, T.LAYERS_TIF, T.ALPHA_RENDER, T.ALPHA_RENDER_DOWNSAMPLED) else
                                  'geometry' if artifact.type.startswith(('tifxyz', 'obj')) or artifact.type in (T.UMBILICUS, T.FIBERLETS) else
                                  'representation' if (source.derived and source.role is not None) or artifact.type in (T.LASAGNA, T.INK_DETECTION, T.INK_DETECTION_DOWNSAMPLED, T.INK_3D_PREDICTION) else 'artifact')
                    volume = owner.id if kind == 'volume' else artifact.target_volume_id
                    segment = owner.id if kind == 'segment' else None
                    self._add(source, asset_kind, sample.id, volume, segment)
        for entry in self.overlays:
            source = Source.from_dict(entry['source'])
            self._add(source, entry['kind'], entry['sample'], entry['volume'], entry['segment'])
        # Explicit supersession is resolved globally before filtering any view.
        for source in self.sources.values():
            self.hidden.update(source.supersedes)
            if (source.provenance.get('artifact', {}).get('properties') or {}).get('superseded_by'):
                self.hidden.add(source.id)
            if source.artifact_kind == T.INK_LABELS:
                older = source.provenance.get('artifact', {}).get('properties') or {}
                older = older.get('supersedes')
                if older:
                    for identifier in self.by_segment[(source.sample_id, self.segment_ids[source.id])]:
                        candidate = self.sources[identifier]
                        if candidate.label_set == older and self.volume_ids[identifier] == self.volume_ids[source.id]:
                            self.hidden.add(identifier)
                            if identifier not in source.supersedes:
                                source = replace(source, supersedes=(*source.supersedes, identifier))
                                self.sources[source.id] = source
        self._validate_supersession()
        self._link_pairs()
        for table in (self.by_sample, self.by_volume, self.by_segment):
            for key in table:
                table[key] = tuple(sorted(table[key]))

    def _add(self, source, kind, sample, volume, segment):
        if source.id in self.sources:
            raise ValueError(f'duplicate source ID {source.id}')
        self.sources[source.id] = source
        self.kinds[source.id] = kind
        self.owners[source.id] = sample
        self.volume_ids[source.id], self.segment_ids[source.id] = volume, segment
        self.by_sample[sample].append(source.id)
        if volume is not None:
            self.by_volume[(sample, volume)].append(source.id)
        if segment is not None:
            self.by_segment[(sample, segment)].append(source.id)

    def _link_pairs(self):
        identity = ((1., 0., 0.), (0., 1., 0.))
        for identifier, source in self.sources.items():
            if self.kinds[identifier] != 'label' or not identifier.startswith('catalog:'):
                continue
            artifact = source.provenance['artifact']
            creation = artifact.get('creation_info') or {}
            explicit = (creation.get('target') or {}).get('render')
            for other in self.by_segment[(source.sample_id, self.segment_ids[identifier])]:
                render = self.sources[other]
                if self.kinds[other] != 'render' or self.volume_ids[other] != self.volume_ids[identifier]:
                    continue
                # Published labels are on the target segment canvas. An explicit
                # render reference narrows this convention, never shape matching.
                locations = (render.storage.location, *render.storage.alternatives)
                if explicit and explicit.rstrip('/') not in {v.rstrip('/') for v in locations if v}:
                    continue
                self._pair(Pairing(identifier, other, identity, 'published render reference' if explicit else 'segment/target-volume canvas convention'))
        for entry in self.overlays:
            for relationship in entry.get('pairs', ()):
                label, render = relationship['label'], relationship['render']
                if self.kinds.get(label) != 'label' or self.kinds.get(render) != 'render':
                    raise ValueError('pair relationships require known label and render IDs')
                if self.owners[label] != self.owners[render]:
                    raise ValueError('paired assets must belong to the same sample')
                self._pair(Pairing(label, render, tuple(tuple(r) for r in relationship['affine']), 'explicit same grid' if relationship.get('same_grid') else 'explicit user pixel mapping'))

    def _pair(self, relationship):
        self.pairs[relationship.label_id].append(relationship)
        self.reverse_pairs[relationship.render_id].append(relationship)

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
                    labels = artifact
                    meshes = owner.artifacts_of(T.TIFXYZ_TRANSFORMED, target_volume=target)
                    mesh = meshes[0] if len(meshes) == 1 else None
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
                        location=(root.rstrip("/") + "/inklabels.zarr/" if root else None),
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
                    ) for name in ("supervision", "validation", "mapping_validity")
                    if name in (labels.properties.get("arrays") or ()))
                elif artifact.type == T.LAYERS_ZARR:
                    storage = _storage(artifact, format="zarr", pyramid=True)
                    storage = replace(storage, selection=Selection(level=0))
                    role, axes = "image", ("z", "y", "x")
                    # Stored z enumerates sampled surface layers, not volume voxels.
                    registration = None
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
            for root in self.sources:
                stack = [(root, False)]
                while stack:
                    identifier, done = stack.pop()
                    if done:
                        active.remove(identifier)
                        visited.add(identifier)
                    elif identifier in active:
                        raise ValueError(f"supersession cycle at {identifier}")
                    elif identifier not in visited and identifier in self.sources:
                        active.add(identifier)
                        stack.append((identifier, True))
                        stack.extend((older, False) for older in self.sources[identifier].supersedes)
