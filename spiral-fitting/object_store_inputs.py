"""Fit inputs read from a vc_object_store database.

Patches, fibers and point collections (PCLs), and the relationships between
them, come from one object store. This module turns a selection of them into
the fitter's runtime inputs in three stages:

1. Snapshot extraction (read_snapshot): resolve object selectors to explicit
   UUID sets, read every selected object, its geometry revision and the
   relationships between selected objects in one read view, and filter the
   relationships by applicability, approval and semantics.
2. Frame conversion (ObjectFrame): every selected object declares its frame in
   ``meta.coordinates``; geometry and anchors convert once from stored XYZ to
   ZYX voxels of one reference volume, the only units the fitter uses.
3. Runtime materialization: patches, PCL and fiber constraints built from the
   frozen snapshot (materialize_* below).

The store's automatic detectors are authoritative: their relationships are
consumed as they are, never relinked by proximity here.

    python object_store_inputs.py STORE.vcobj [--volume V] \
        [--z-begin Z0 --z-end Z1 [--config FIT.json]]

prints the selection's report without training. Given a z window it also
materializes and derives the training views exactly as a fit with that
configuration would, failing on the same validation errors.
"""
import argparse
import dataclasses
import hashlib
import json
import sys
from collections import Counter

import numpy as np

KINDS = ('patch', 'point_collection', 'fiber')
KIND_LABELS = {'patch': 'patches', 'point_collection': 'point collections',
               'fiber': 'fibers'}

# Relationship families the fitter consumes, keyed by (edge kind, endpoint
# kinds in KINDS order). Patch overlaps and fiber-patch contacts share the
# detectors' 'same-surface' edge kind and differ by endpoint kinds.
FAMILIES = {
    ('point-patch', ('patch', 'point_collection')): 'pcl_patch',
    ('same-surface', ('patch', 'fiber')): 'fiber_patch',
    ('same-surface', ('patch', 'patch')): 'patch_overlap',
    ('fiber_branch', ('fiber', 'fiber')): 'fiber_branch',
}

# The coordinates the fitter works in, recorded with every provenance.
FIT_UNITS = 'reference-volume voxels (ZYX)'


class ObjectStoreInputError(ValueError):
    """A selection, frame or relationship the fitter cannot consume."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__('\n'.join(self.problems))


@dataclasses.dataclass(frozen=True)
class Selector:
    """Selects objects of one kind.

    ``uuids`` and ``tags`` both None selects every active object of the kind
    with a geometry revision (the default selection). Otherwise the selection
    is the union of the listed UUIDs, which must exist, be active and have
    geometry, and the active objects carrying any listed tag. An explicit
    selector that matches nothing selects nothing.
    """
    uuids: tuple = None
    tags: tuple = None

    @property
    def is_default(self):
        return self.uuids is None and self.tags is None

    def describe(self):
        if self.is_default:
            return 'all active'
        parts = []
        if self.uuids is not None:
            parts.append(f'{len(self.uuids)} UUID(s)')
        if self.tags is not None:
            parts.append(f'tags {list(self.tags)}')
        return ' + '.join(parts)


@dataclasses.dataclass(frozen=True)
class InputRequest:
    """What to read: per-kind selectors, the fit volume and the approval policy.

    ``fit_volume_uuid`` None infers the volume from the selected objects'
    declarations. ``approved_links_only`` consumes only approved
    relationships; otherwise pending and approved ones are consumed. Rejected
    relationships are never consumed.
    """
    selectors: dict = dataclasses.field(default_factory=dict)
    fit_volume_uuid: str = None
    approved_links_only: bool = False

    def selector(self, kind):
        return self.selectors.get(kind) or Selector()


@dataclasses.dataclass(frozen=True)
class ObjectFrame:
    """One object's frame: stored XYZ = reference-volume XYZ x scale_factor."""
    volume_uuid: str
    scale_factor: float
    # stored -> fit multiplier from the store's shared coordinate helper.
    to_fit: float

    def xyz_to_fit_zyx(self, xyz, invalid=None):
        """Stored XYZ (..., 3) -> float32 fit ZYX. Rows where ``invalid`` is
        True keep the -1 sentinel rather than becoming scaled coordinates."""
        xyz = np.asarray(xyz, dtype=np.float64)
        zyx = (xyz[..., ::-1] * self.to_fit).astype(np.float32)
        if invalid is not None:
            zyx[np.asarray(invalid, dtype=bool)] = -1.0
        return zyx

    def grid_scale_to_fit(self, scale):
        """Patch grid steps per stored unit -> grid steps per fit voxel."""
        return [float(value) / self.to_fit for value in scale]


@dataclasses.dataclass
class StoredObject:
    uuid: str
    kind: str
    name: str
    meta: dict
    tags: list
    object_revision: str
    geometry_revision: str
    frame: ObjectFrame = None

    def provenance(self):
        return {'uuid': self.uuid, 'kind': self.kind, 'name': self.name,
                'object_revision': self.object_revision,
                'geometry_revision': self.geometry_revision,
                'scale_factor': self.frame.scale_factor}


@dataclasses.dataclass
class StoredRelationship:
    """A consumed relationship with endpoints ordered by kind (KINDS order;
    UUID order between objects of one kind), not by stored order."""
    uuid: str
    revision: str
    kind: str
    family: str
    approval_state: str
    generation: int
    a: str
    b: str
    a_anchor: dict
    b_anchor: dict
    meta: dict

    def provenance(self):
        return {'uuid': self.uuid, 'revision': self.revision,
                'kind': self.kind, 'family': self.family,
                'approval_state': self.approval_state,
                'generation': self.generation}


@dataclasses.dataclass
class InputSnapshot:
    """The frozen selection: objects, their frames and consumed relationships."""
    store_path: str
    store_uuid: str
    volume_uuid: str
    volume_source: str
    request: InputRequest
    objects: dict
    relationships: list
    exclusions: Counter
    notes: list

    def objects_of(self, kind):
        return [obj for obj in self.objects.values() if obj.kind == kind]

    def relationships_of(self, family):
        return [rel for rel in self.relationships if rel.family == family]

    def manifest(self):
        """Complete input/frame provenance, in stable order."""
        return {
            'store': {'path': self.store_path, 'uuid': self.store_uuid},
            'volume_uuid': self.volume_uuid,
            'volume_source': self.volume_source,
            'units': FIT_UNITS,
            'approved_links_only': self.request.approved_links_only,
            'selection': {
                kind: {'selector': dataclasses.asdict(self.request.selector(kind)),
                       'count': len(self.objects_of(kind))}
                for kind in KINDS},
            'objects': [obj.provenance() for obj in self.objects.values()],
            'relationships': [rel.provenance() for rel in self.relationships],
            'excluded_relationships': dict(sorted(self.exclusions.items())),
        }

    def digest(self):
        """SHA-256 of the manifest without the store's local path."""
        manifest = self.manifest()
        manifest['store'] = {'uuid': self.store_uuid}
        text = json.dumps(manifest, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(text.encode()).hexdigest()


def resolve_selector(store, kind, selector):
    """Explicit sorted UUIDs selected by ``selector`` among active objects."""
    if selector.is_default:
        return sorted(record['uuid'] for record in store.select_objects(
            object_kind=kind, fields=['uuid']))
    selected = set()
    problems = []
    for uuid in selector.uuids or ():
        record = store.object(uuid)
        if record is None:
            problems.append(f'selected {kind} {uuid} does not exist')
        elif record['kind'] != kind:
            problems.append(f'selected {kind} {uuid} is a {record["kind"]}')
        elif record.get('retired'):
            problems.append(f'selected {kind} {uuid} is retired')
        else:
            selected.add(record['uuid'])
    if problems:
        raise ObjectStoreInputError(problems)
    for tag in selector.tags or ():
        selected.update(record['uuid'] for record in store.select_objects(
            object_kind=kind, tag=tag, fields=['uuid']))
    return sorted(selected)


def declared_coordinates(meta):
    """The normalized ``meta.coordinates`` declaration of an object."""
    from vc_object_store import geometry_coordinates
    return geometry_coordinates(meta.get('coordinates') or {})


def infer_fit_volume(records, configured=None):
    """Return ``(volume_uuid, 'inferred' | 'configured')``.

    Selected objects declaring different volumes are an error, even when one
    of them matches a configured volume: a configuration cannot override a
    conflict or silently drop objects. Without any declaration (including an
    empty selection) the volume must be configured.
    """
    declared = {}
    for record in records:
        volume = declared_coordinates(record['meta'])['volume_uuid']
        if volume is not None:
            declared.setdefault(volume, []).append(record['uuid'])
    if len(declared) > 1:
        raise ObjectStoreInputError([
            'selected objects declare different volumes: ' + '; '.join(
                f'{volume} ({len(uuids)} object(s), e.g. {uuids[0]})'
                for volume, uuids in sorted(declared.items()))])
    if configured is not None:
        if declared and configured not in declared:
            raise ObjectStoreInputError([
                f'fit volume {configured} does not match the volume '
                f'{next(iter(declared))} the selected objects declare'])
        return configured, 'configured'
    if not declared:
        raise ObjectStoreInputError([
            'no selected object declares a volume; configure fit_volume_uuid'])
    return next(iter(declared)), 'inferred'


def object_frame(meta, volume_uuid):
    """The object's ObjectFrame in ``volume_uuid``; raises ValueError when its
    frame is unknown, invalid or in another volume."""
    from vc_object_store import geometry_scale_to_frame, resolve_geometry_coordinates
    coordinates = resolve_geometry_coordinates(meta, volume_uuid)
    return ObjectFrame(volume_uuid, float(coordinates['scale_factor']),
                       float(geometry_scale_to_frame(meta, volume_uuid)))


def _relationship_family(record, kinds):
    endpoint_kinds = tuple(sorted((kinds[record['a']], kinds[record['b']]),
                                  key=KINDS.index))
    family = FAMILIES.get((record['kind'], endpoint_kinds))
    if family == 'fiber_branch' and (record['meta'] or {}).get('adjacent'):
        return 'adjacent_branch'
    return family


def _ordered_endpoints(record, kinds):
    """(a, a_anchor, b, b_anchor) ordered by kind, then UUID."""
    a, b = record['a'], record['b']
    key = lambda uuid: (KINDS.index(kinds[uuid]), uuid)
    if key(b) < key(a):
        return b, record['b_anchor'], a, record['a_anchor']
    return a, record['a_anchor'], b, record['b_anchor']


def filter_relationships(view_relationships, objects, approved_links_only):
    """Consumed relationships and exclusion counts.

    A relationship is consumed when both endpoints are selected, it applies
    to their selected revisions, it belongs to a family the fitter uses and
    its approval state is allowed (never 'rejected'; only 'approved' in
    approved-only mode). Adjacent fiber branches are excluded explicitly.
    """
    kinds = {uuid: obj.kind for uuid, obj in objects.items()}
    consumed, exclusions = [], Counter()
    for entry in view_relationships:
        record = entry['record']
        if record['a'] not in kinds or record['b'] not in kinds:
            exclusions['endpoint outside selection'] += 1
            continue
        if not entry['applicable']:
            exclusions['inapplicable'] += 1
            continue
        family = _relationship_family(record, kinds)
        if family is None:
            exclusions[f'unused kind {record["kind"]}'] += 1
            continue
        if family == 'adjacent_branch':
            exclusions['adjacent fiber branch'] += 1
            continue
        state = record['approval_state']
        if state == 'rejected':
            exclusions['rejected'] += 1
            continue
        if approved_links_only and state != 'approved':
            exclusions[f'{state} (approved-only)'] += 1
            continue
        a, a_anchor, b, b_anchor = _ordered_endpoints(record, kinds)
        consumed.append(StoredRelationship(
            uuid=record['uuid'], revision=entry['uuid'], kind=record['kind'],
            family=family, approval_state=state,
            generation=record['generation'], a=a, b=b, a_anchor=a_anchor,
            b_anchor=b_anchor, meta=record['meta'] or {}))
    consumed.sort(key=lambda rel: (rel.family, rel.a, rel.b, rel.uuid))
    return consumed, exclusions


def resolve_selection(store, request):
    """{kind: sorted selected UUIDs} of a request's selectors."""
    return {kind: resolve_selector(store, kind, request.selector(kind))
            for kind in KINDS}


def read_snapshot(store, request, uuids=None):
    """Resolve, validate and freeze one selection of the store.

    Selectors resolve to explicit UUID sets before any volume filter, so
    conflicting frame declarations among selected objects are seen and fail.
    ``uuids`` (resolve_selection's result) reuses a selection resolved
    earlier, so one startup reads exactly the objects it refreshed.
    Every selected object must declare a frame in the fit volume; all frame
    problems are reported together.
    """
    if uuids is None:
        uuids = resolve_selection(store, request)
    selection = {uuid: None for kind in KINDS for uuid in uuids[kind]}
    view = store.revision_read_view(selection)

    notes, problems = [], []
    revisions = {revision['object_uuid']: revision for revision in view['objects']}
    usable = []
    for kind in KINDS:
        without_geometry = []
        for uuid in uuids[kind]:
            revision = revisions[uuid]
            if revision['geometry_revision'] is None:
                without_geometry.append(uuid)
            else:
                usable.append(revision)
        if not without_geometry:
            continue
        if request.selector(kind).is_default:
            notes.append(f'skipped {len(without_geometry)} {KIND_LABELS[kind]} '
                         'without geometry')
        else:
            problems.extend(f'selected {kind} {uuid} has no geometry'
                            for uuid in without_geometry)
    if problems:
        raise ObjectStoreInputError(problems)

    volume_uuid, volume_source = infer_fit_volume(
        [revision['record'] for revision in usable], request.fit_volume_uuid)
    objects = {}
    for revision in usable:
        record = revision['record']
        try:
            frame = object_frame(record['meta'], volume_uuid)
        except ValueError as error:
            problems.append(f'{record["kind"]} {record["uuid"]} '
                            f'({record["name"]!r}): {error}')
            continue
        objects[record['uuid']] = StoredObject(
            uuid=record['uuid'], kind=record['kind'], name=record['name'],
            meta=record['meta'] or {}, tags=list(record.get('tags') or ()),
            object_revision=revision['uuid'],
            geometry_revision=revision['geometry_revision'], frame=frame)
    if problems:
        raise ObjectStoreInputError(problems)

    relationships, exclusions = filter_relationships(
        view['relationships'], objects, request.approved_links_only)
    return InputSnapshot(
        store_path=store.path, store_uuid=store.uuid, volume_uuid=volume_uuid,
        volume_source=volume_source, request=request, objects=objects,
        relationships=relationships, exclusions=exclusions, notes=notes)


# ---------------------------------------------------------------------------
# Geometry reads (pinned by geometry revision, so independent of later heads)
# ---------------------------------------------------------------------------

def read_geometry(store, geometry_revision, names):
    """``({name: ndarray}, attributes)`` of an immutable geometry revision."""
    descriptor = store.geometry_revision(geometry_revision)
    arrays = {array['name']: array for array in descriptor['arrays']}
    result = {}
    for name in names:
        if name not in arrays:
            raise ObjectStoreInputError(
                [f'geometry {geometry_revision} has no {name!r} array'])
        array = arrays[name]
        data = store.read_geometry_array(geometry_revision, name)
        result[name] = np.frombuffer(data, dtype=np.dtype(array['dtype'])).reshape(
            array['shape'])
    return result, descriptor['attributes'] or {}


def point_annotations(store, geometry_revision, point_ids):
    """{point id: annotation dict} aligned to the stored point order."""
    annotations = store.geometry_point_annotations(geometry_revision, 'points')
    if annotations is None:
        return {}
    if len(annotations) != len(point_ids):
        raise ObjectStoreInputError(
            [f'geometry {geometry_revision} annotations do not match its points'])
    return {int(point_id): annotation or {}
            for point_id, annotation in zip(point_ids, annotations)}


# ---------------------------------------------------------------------------
# Runtime materialization
# ---------------------------------------------------------------------------

# Deterministic default sampling groups (the stems of the conventional PCL
# documents, so existing pcl_sampling_weights keep applying). An object's
# ``meta.sampling_group`` overrides them.
ABSOLUTE_PCL_GROUP = 'abs_winding'
RELATIVE_PCL_GROUP = 'relative_windings'
SAME_WINDING_PCL_GROUP = 'same_windings'
# Patch-overlap pairs default to their own group, so a dense overlap graph
# cannot crowd out the PCL groups; an edge's ``meta.sampling_group`` overrides it.
PATCH_OVERLAP_GROUP = 'patch_overlaps'


@dataclasses.dataclass
class FitInputs:
    """Runtime inputs materialized from a frozen snapshot.

    ``patches`` maps patch UUIDs to fit-frame tifxyz.Patch objects in UUID
    order. ``point_collections`` maps resident collection ids to regular PCLs
    whose attached points carry ``on_patch`` ({'id', 'ij', 'edge_uuid'}).
    ``fibers`` maps fiber UUIDs to fiber collections (materialize_fibers) and
    ``fiber_links`` holds their resolved same-winding branch links.
    ``patch_overlaps`` maps resident ids to the two-point same-surface
    collections of materialize_patch_overlaps.
    ``dropped_attachments`` lists ``(edge UUID, reason)`` for consumed
    attachments whose quad the patch losses do not sample (on_retained_quad).
    """
    snapshot: InputSnapshot
    patches: dict
    point_collections: dict
    fibers: dict
    fiber_links: list
    counts: Counter
    dropped_attachments: list = dataclasses.field(default_factory=list)
    patch_overlaps: dict = dataclasses.field(default_factory=dict)
    # Complete input/frame provenance and its digest (load_fit_inputs).
    manifest: dict = None
    digest: str = None

    def report(self, max_named=5):
        lines = snapshot_report(self.snapshot) + [
            'constraints: ' + ', '.join(
                f'{name} {count}' for name, count in sorted(self.counts.items()))]
        if self.dropped_attachments:
            named = ', '.join(edge for edge, _ in self.dropped_attachments[:max_named])
            more = len(self.dropped_attachments) - max_named
            lines.append(
                f'WARNING: {len(self.dropped_attachments)} consumed attachment(s) '
                f'lie on patch quads the losses do not sample (eroded, outside '
                f'the z window or a detached island) and were dropped: {named}'
                + (f', and {more} more' if more > 0 else ''))
        return lines


def materialize_patches(store, snapshot, *, z_begin, z_end, z_margin,
                        erode_cells_default):
    """Fit-frame patches of the snapshot, keyed by UUID; and drop counts.

    Grid rows and columns are the stored ones: nothing is cropped or
    subsampled, so stored (row, col) anchors are runtime (i, j). Invalid
    vertices keep the -1 sentinel. A patch's own erosion request is
    ``meta.source.spiral_patch_erode_cells``, as in its tifxyz metadata.
    Each patch carries its loss sampling mask (``z_margin`` is the fit's
    patch_loss_z_margin), so attachments can be checked against it.
    """
    import torch
    from spiral_helpers import (patch_intersects_z_roi, patch_sampling_quad_mask,
                                prepare_patch_for_fit)
    from tifxyz import Patch

    patches, counts = {}, Counter()
    for obj in snapshot.objects_of('patch'):
        arrays, attributes = read_geometry(store, obj.geometry_revision, ['points'])
        points = arrays['points']
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ObjectStoreInputError(
                [f'patch {obj.uuid} points have shape {points.shape}'])
        invalid = np.all(points == -1, axis=-1)
        zyxs = obj.frame.xyz_to_fit_zyx(points, invalid)
        valid_quads = (~invalid[:-1, :-1] & ~invalid[1:, :-1]
                       & ~invalid[:-1, 1:] & ~invalid[1:, 1:])
        if not valid_quads.any():
            counts['patches without valid quads'] += 1
            continue
        erosion = (obj.meta.get('source') or {}).get('spiral_patch_erode_cells')
        patch = Patch(torch.from_numpy(zyxs),
                      torch.tensor(obj.frame.grid_scale_to_fit(attributes['scale'])),
                      None, None, obj.uuid, erosion)
        if not patch_intersects_z_roi(patch, z_begin, z_end):
            counts['patches outside z ROI'] += 1
            continue
        reason = prepare_patch_for_fit(patch, z_begin, z_end, erode_cells_default)
        if reason is not None:
            counts[f'patches dropped by {reason}'] += 1
            continue
        patch._sampling_valid_quad_mask_np = patch_sampling_quad_mask(
            patch, z_begin, z_end, z_margin)
        patches[obj.uuid] = patch
    counts['patches'] = len(patches)
    return patches, counts


def pcl_role(obj, relationships):
    """True for an absolute-winding PCL, from ``meta.source.winding_is_absolute``.

    A non-boolean value, or detector evidence recorded under the other role,
    is an error rather than a guess.
    """
    value = (obj.meta.get('source') or {}).get('winding_is_absolute', False)
    if not isinstance(value, bool):
        raise ObjectStoreInputError([
            f'point collection {obj.uuid} ({obj.name!r}) has a non-boolean '
            f'winding_is_absolute {value!r}'])
    for rel in relationships:
        recorded = (rel.meta.get('point_collection_patch') or {}).get(
            'winding_is_absolute')
        if recorded is not None and recorded != value:
            raise ObjectStoreInputError([
                f'point collection {obj.uuid} ({obj.name!r}) is '
                f'{"absolute" if value else "relative"}, but edge {rel.uuid} '
                f'was detected as {"absolute" if recorded else "relative"}; '
                'refresh its links'])
    return value


def _winding_annotation(obj, point_id, annotation):
    value = annotation.get('wind_a')
    if value is None:
        return float('nan')
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ObjectStoreInputError([
            f'point collection {obj.uuid} point {point_id} has a non-numeric '
            f'wind_a {value!r}'])
    return float(value)


def on_retained_quad(patch, ij):
    """Whether (row, col) ``ij`` lies on a quad the fitted patch's losses keep.

    The quads are the patch's sampling mask (spiral_helpers.
    patch_sampling_quad_mask, stored by materialize_patches exactly as the fit
    computes it): eroded, restricted to the z window and to the largest
    connected piece. The quad is the floor of ``ij``, clamped to the grid, as
    in losses._valid_patch_annotation. Stored attachments were detected on
    the whole un-eroded grid, so any of these steps can remove their quad.
    """
    mask = patch._sampling_valid_quad_mask_np
    row = min(max(int(ij[0]), 0), mask.shape[0] - 1)
    col = min(max(int(ij[1]), 0), mask.shape[1] - 1)
    return bool(mask[row, col])


def _anchor_index(anchor, array):
    points = anchor.get('points') or []
    if len(points) < 1 or points[0]['array'] != array or len(points[0]['index']) != 1:
        return None
    return int(points[0]['index'][0])


def _pcl_attachment(rel, patches):
    """``(point id, on_patch)`` of a pcl-patch edge, or (point id, None) when
    its patch is not fitted; raises when the edge cannot be materialized."""
    point_id = _anchor_index(rel.b_anchor, 'point_ids')
    ij = (rel.meta.get('point_collection_patch') or {}).get('ij')
    if point_id is None or not isinstance(ij, list) or len(ij) != 2:
        raise ObjectStoreInputError([
            f'point-patch edge {rel.uuid} lacks a point ID anchor or patch ij'])
    if rel.a not in patches:
        return point_id, None
    return point_id, {'id': rel.a, 'ij': [float(ij[0]), float(ij[1])],
                      'edge_uuid': rel.uuid}


def materialize_point_collections(store, snapshot, patches, first_id=0,
                                  dropped=None):
    """Regular PCLs of the snapshot, keyed by resident id in UUID order.

    Points are keyed by their stable IDs and carry fit-frame ZYX and their
    aligned ``wind_a`` annotation. Each consumed point-patch edge attaches its
    point to its patch at the recorded (row, col); edges to patches dropped
    from the fit are counted and ignored. An attachment on a quad the losses
    do not sample (on_retained_quad) leaves its point unattached and is appended to ``dropped`` as
    ``(edge UUID, reason)``.
    """
    edges_by_pcl = {}
    for rel in snapshot.relationships_of('pcl_patch'):
        edges_by_pcl.setdefault(rel.b, []).append(rel)
    collections, counts, problems = {}, Counter(), []
    for cid, obj in enumerate(snapshot.objects_of('point_collection'), start=first_id):
        arrays, _ = read_geometry(store, obj.geometry_revision, ['points', 'point_ids'])
        point_ids = [int(point_id) for point_id in arrays['point_ids']]
        if len(set(point_ids)) != len(point_ids):
            raise ObjectStoreInputError(
                [f'point collection {obj.uuid} repeats point IDs'])
        annotations = point_annotations(store, obj.geometry_revision, point_ids)
        edges = edges_by_pcl.get(obj.uuid, [])
        absolute = pcl_role(obj, edges)
        zyxs = obj.frame.xyz_to_fit_zyx(arrays['points'])
        points = {}
        for point_id, zyx in sorted(zip(point_ids, zyxs), key=lambda item: item[0]):
            points[point_id] = {
                'id': point_id, 'collectionId': cid, 'p': zyx[::-1].tolist(),
                'zyx': zyx,
                'winding_annotation': _winding_annotation(
                    obj, point_id, annotations.get(point_id, {})),
                'creation_time': 0,
            }
        for rel in edges:
            try:
                point_id, attachment = _pcl_attachment(rel, patches)
            except ObjectStoreInputError as error:
                problems.extend(error.problems)
                continue
            point = points.get(point_id)
            if point is None:
                problems.append(f'point-patch edge {rel.uuid} names point '
                                f'{point_id}, absent from {obj.uuid}')
            elif attachment is None:
                counts['pcl attachments to unfitted patches'] += 1
            elif not on_retained_quad(patches[attachment['id']], attachment['ij']):
                counts['pcl attachments off sampled quads'] += 1
                if dropped is not None:
                    dropped.append((rel.uuid, 'quad not sampled'))
            elif 'on_patch' in point:
                problems.append(
                    f'point {point_id} of point collection {obj.uuid} attaches '
                    f'to patches {point["on_patch"]["id"]} and {attachment["id"]}')
            else:
                point['on_patch'] = attachment
                counts['pcl attachments'] += 1
        annotated = any(np.isfinite(p['winding_annotation']) for p in points.values())
        group = obj.meta.get('sampling_group') or (
            ABSOLUTE_PCL_GROUP if absolute else
            RELATIVE_PCL_GROUP if annotated else SAME_WINDING_PCL_GROUP)
        collections[cid] = {
            'id': cid, 'name': obj.name, 'points': points, 'color': [0.0, 0.0, 0.0],
            'source_file': None, 'sampling_group': group,
            'metadata': {
                'winding_is_absolute': absolute,
                'input_role': 'absolute' if absolute else 'object_store',
                'resident_collection_id': cid,
                'object_uuid': obj.uuid,
                'object_revision': obj.object_revision,
                'geometry_revision': obj.geometry_revision,
            },
        }
        counts['absolute point collections' if absolute
               else 'relative point collections'] += 1
    if problems:
        raise ObjectStoreInputError(problems)
    return collections, counts


def _fiber_contact(rel):
    """``(raw line index, ij)`` of a fiber-patch edge's representative contact."""
    line_index = _anchor_index(rel.b_anchor, 'line_points')
    grid = (rel.meta.get('fiber_patch') or {}).get('patch_grid')
    if line_index is None or not isinstance(grid, list) or len(grid) != 2:
        raise ObjectStoreInputError([
            f'fiber-patch edge {rel.uuid} lacks a line-point anchor or patch_grid'])
    # patch_grid is (column, row); attachments use (row, column).
    return line_index, [float(grid[1]), float(grid[0])]


def materialize_fibers(store, snapshot, patches, *, min_point_spacing,
                       first_id=0, dropped=None):
    """Fit-frame fibers, their contacts and same-winding branch links.

    Returns ``(fiber_catalog, links, counts)``. The catalog maps fiber UUIDs
    (in UUID order) to fiber collections whose points are keyed by raw line
    index: trimming to the control span and min-spacing decimation never
    renumber them. Every consumed contact and branch endpoint is kept through
    both. A contact's point lists each patch it attaches to in
    ``point['attachments']``, so one physical point (one chain/theta node) can
    attach to several patches. A contact on a quad the losses do not sample
    is not attached, and is appended to ``dropped`` as ``(edge UUID, reason)``.
    ``links`` are resolved branch links keyed by resident collection id and
    raw line index.
    """
    from vc_object_store import fiber_branch_control_index
    from spiral_helpers import (_decimate_ordered_points_min_spacing,
                                fiber_control_span)

    fibers = snapshot.objects_of('fiber')
    resident_id = {obj.uuid: cid for cid, obj in enumerate(fibers, start=first_id)}
    contacts = {obj.uuid: {} for obj in fibers}
    counts, problems = Counter(), []
    for rel in snapshot.relationships_of('fiber_patch'):
        try:
            line_index, ij = _fiber_contact(rel)
        except ObjectStoreInputError as error:
            problems.extend(error.problems)
            continue
        if rel.a not in patches:
            counts['fiber contacts with unfitted patches'] += 1
            continue
        if not on_retained_quad(patches[rel.a], ij):
            counts['fiber contacts off sampled quads'] += 1
            if dropped is not None:
                dropped.append((rel.uuid, 'quad not sampled'))
            continue
        contacts[rel.b].setdefault(line_index, []).append(
            {'id': rel.a, 'ij': ij, 'edge_uuid': rel.uuid})

    geometry, endpoints = {}, {obj.uuid: {} for obj in fibers}
    for obj in fibers:
        arrays, _ = read_geometry(store, obj.geometry_revision,
                                  ['control_points', 'line_points', 'control_point_ids'])
        geometry[obj.uuid] = arrays
    for rel in snapshot.relationships_of('fiber_branch'):
        try:
            control = {
                uuid: fiber_branch_control_index(
                    [int(v) for v in geometry[uuid]['control_point_ids']], anchor)
                for uuid, anchor in ((rel.a, rel.a_anchor), (rel.b, rel.b_anchor))}
        except RuntimeError as error:
            problems.append(f'fiber branch {rel.uuid}: {error}')
            continue
        endpoints[rel.a][rel.uuid] = control[rel.a]
        endpoints[rel.b][rel.uuid] = control[rel.b]
    if problems:
        raise ObjectStoreInputError(problems)

    catalog, branch_points = {}, {}
    for obj in fibers:
        arrays = geometry[obj.uuid]
        # Fit-frame XYZ, in the stored point order.
        line = np.ascontiguousarray(
            obj.frame.xyz_to_fit_zyx(arrays['line_points'])[:, ::-1])
        controls = np.ascontiguousarray(
            obj.frame.xyz_to_fit_zyx(arrays['control_points'])[:, ::-1])
        if len(controls) < 1:
            raise ObjectStoreInputError([f'fiber {obj.uuid} has no control points'])
        own_contacts = contacts[obj.uuid]
        polyline, span_begin, control_line = fiber_control_span(controls, line)
        if polyline is controls:
            # No usable dense polyline: the controls are the fiber, and line
            # point contacts cannot be placed on it.
            if own_contacts:
                raise ObjectStoreInputError([
                    f'fiber {obj.uuid} has contacts but no usable line points'])
            line = controls
        span_end = span_begin + len(polyline) - 1
        outside = [i for i in own_contacts if not span_begin <= i <= span_end]
        if any(not 0 <= i < len(line) for i in outside):
            raise ObjectStoreInputError(
                [f'fiber {obj.uuid} contact outside its {len(line)} line points'])
        if outside:
            # The detector's control-span rule can differ by a line point at
            # each end; keep consumed contacts rather than dropping them.
            counts['fiber spans widened to keep contacts'] += 1
            span_begin = min(span_begin, *outside)
            span_end = max(span_end, *outside)
        branch_lines = {edge: int(control_line[index])
                        for edge, index in endpoints[obj.uuid].items()}
        keep = set(own_contacts) | set(branch_lines.values())
        span = np.arange(span_begin, span_end + 1)
        _, kept = _decimate_ordered_points_min_spacing(
            line[span], min_point_spacing,
            return_indices=True,
            force_keep={i - span_begin for i in keep})
        cid = resident_id[obj.uuid]
        points = {}
        for position in kept:
            raw = int(span[position])
            xyz = line[raw]
            point = {'id': raw, 'collectionId': cid, 'p': xyz.tolist(),
                     'zyx': np.asarray(xyz[::-1], dtype=np.float32),
                     'winding_annotation': float('nan'), 'creation_time': 0,
                     'orig_index': raw}
            if raw in own_contacts:
                point['attachments'] = own_contacts[raw]
                counts['fiber attachments'] += len(own_contacts[raw])
            points[raw] = point
        source = obj.meta.get('source') or {}
        catalog[obj.uuid] = {
            'id': cid, 'name': obj.name, 'points': points,
            'color': [0.0, 0.0, 0.0], 'source_file': None,
            'metadata': {
                'hv_classification': source.get('hv_classification') or {},
                'logical_input_revision': obj.object_revision,
                'object_uuid': obj.uuid,
                'geometry_revision': obj.geometry_revision,
                'fiber_num_control_points': len(controls),
            },
        }
        branch_points[obj.uuid] = branch_lines
        counts['fibers'] += 1

    links = []
    for rel in snapshot.relationships_of('fiber_branch'):
        links.append({
            'a_coll': resident_id[rel.a], 'a_point': branch_points[rel.a][rel.uuid],
            'b_coll': resident_id[rel.b], 'b_point': branch_points[rel.b][rel.uuid],
            'pending': rel.approval_state != 'approved', 'edge_uuid': rel.uuid,
        })
    counts['fiber branches'] = len(links)
    return catalog, links, counts


def _overlap_point(cid, index, patch_uuid, grid, xyz, edge_uuid):
    zyx = np.asarray(xyz, dtype=np.float64)[::-1].astype(np.float32)
    return {'id': index, 'collectionId': cid, 'p': zyx[::-1].tolist(),
            'zyx': zyx, 'winding_annotation': 0.0, 'creation_time': 0,
            'on_patch': {'id': patch_uuid,
                         'ij': [float(grid[0]), float(grid[1])],
                         'edge_uuid': edge_uuid}}


def materialize_patch_overlaps(store, snapshot, patches, *, pair_spacing,
                               max_pairs, first_id=0, dropped=None):
    """Same-surface pair collections of the snapshot's patch overlaps.

    The store materializes each consumed overlap edge's point pairs from its
    stored pose (``overlap_correspondences``); nothing is matched here. Pairs
    follow the stored endpoint order, with (row, col) grids and XYZ already in
    fit-volume voxels, so only the axes are reversed. Each pair becomes one
    collection of two points, the pair's point on either patch, with zero
    relative winding: the relative-winding loss then compares exactly that
    correspondence along its own two-node chain. Every pair of an edge
    carries ``sampling_member_weight`` 1 / (its retained pairs), so an edge
    weighs the same however many pairs it produces.

    A pair on a quad either patch's losses do not sample is dropped; an edge
    losing every pair is appended to ``dropped``. Edge errors, and pairs
    computed from geometry other than the snapshot's, are reported together.
    Returns ``(collections keyed by resident id, counts)``.
    """
    edges = {rel.uuid: rel for rel in snapshot.relationships_of('patch_overlap')}
    collections, counts, problems = {}, Counter(), []
    if not edges:
        return collections, counts
    results = store.overlap_correspondences(
        list(edges), snapshot.volume_uuid, float(pair_spacing), int(max_pairs))
    cid = first_id
    for result in results:
        rel = edges.pop(result['edge_uuid'], None)
        if rel is None:
            problems.append(f'patch overlap {result["edge_uuid"]} was not requested')
            continue
        if result['error'] is not None:
            problems.append(f'patch overlap {rel.uuid}: {result["error"]}')
            continue
        ends = ((result['a'], result['a_geometry_revision']),
                (result['b'], result['b_geometry_revision']))
        if sorted(uuid for uuid, _ in ends) != sorted((rel.a, rel.b)):
            problems.append(f'patch overlap {rel.uuid} was materialized between '
                            f'{result["a"]} and {result["b"]}')
            continue
        stale = [uuid for uuid, revision in ends
                 if snapshot.objects[uuid].geometry_revision != revision]
        if stale:
            problems.append(f'patch overlap {rel.uuid} was materialized from '
                            f'geometry of {", ".join(stale)} other than the snapshot\'s')
            continue
        a, b = result['a'], result['b']
        if a not in patches or b not in patches:
            counts['patch overlaps with unfitted patches'] += 1
            continue
        kept = [pair for pair in result['pairs']
                if on_retained_quad(patches[a], pair['a_grid'])
                and on_retained_quad(patches[b], pair['b_grid'])]
        if len(kept) < len(result['pairs']):
            counts['overlap pairs off sampled quads'] += len(result['pairs']) - len(kept)
        if not kept:
            counts['patch overlaps off sampled quads'] += 1
            if dropped is not None:
                dropped.append((rel.uuid, 'no pair on sampled quads'))
            continue
        group = rel.meta.get('sampling_group') or PATCH_OVERLAP_GROUP
        for index, pair in enumerate(kept):
            collections[cid] = {
                'id': cid, 'name': f'overlap {rel.uuid} pair {index}',
                'points': {
                    0: _overlap_point(cid, 0, a, pair['a_grid'], pair['a_xyz'], rel.uuid),
                    1: _overlap_point(cid, 1, b, pair['b_grid'], pair['b_xyz'], rel.uuid)},
                'color': [0.0, 0.0, 0.0], 'source_file': None,
                'sampling_group': group,
                'sampling_member_weight': 1.0 / len(kept),
                'metadata': {
                    'winding_is_absolute': False,
                    'input_role': 'patch_overlap',
                    'resident_collection_id': cid,
                    'edge_uuid': rel.uuid,
                    'edge_revision': rel.revision,
                    'pair_distance': float(pair['distance']),
                },
            }
            cid += 1
        counts['patch overlaps'] += 1
        counts['overlap pairs'] += len(kept)
    problems.extend(f'patch overlap {uuid} returned no result' for uuid in edges)
    if problems:
        raise ObjectStoreInputError(problems)
    return collections, counts


def materialize(store, snapshot, *, z_begin, z_end, z_margin,
                erode_cells_default, fiber_min_point_spacing=0.0,
                overlap_pair_spacing=8.0, overlap_max_pairs=16):
    """Every runtime input of a frozen snapshot.

    Fibers are decimated to ``fiber_min_point_spacing`` (the fitter's
    pcl_fiber_min_point_spacing); patch overlaps yield at most
    ``overlap_max_pairs`` pairs ``overlap_pair_spacing`` voxels apart.
    Resident ids number PCLs, then fibers, then overlap pairs.
    """
    patches, counts = materialize_patches(
        store, snapshot, z_begin=z_begin, z_end=z_end, z_margin=z_margin,
        erode_cells_default=erode_cells_default)
    dropped = []
    point_collections, pcl_counts = materialize_point_collections(
        store, snapshot, patches, dropped=dropped)
    counts.update(pcl_counts)
    fibers, fiber_links, fiber_counts = materialize_fibers(
        store, snapshot, patches, min_point_spacing=fiber_min_point_spacing,
        first_id=len(point_collections), dropped=dropped)
    counts.update(fiber_counts)
    patch_overlaps, overlap_counts = materialize_patch_overlaps(
        store, snapshot, patches, pair_spacing=overlap_pair_spacing,
        max_pairs=overlap_max_pairs,
        first_id=len(point_collections) + len(fibers), dropped=dropped)
    counts.update(overlap_counts)
    return FitInputs(snapshot, patches, point_collections, fibers,
                     fiber_links, counts, dropped_attachments=dropped,
                     patch_overlaps=patch_overlaps)


def materialize_from_config(store, snapshot, config):
    """materialize() with the fit configuration's window and settings."""
    return materialize(
        store, snapshot, z_begin=config['z_begin'], z_end=config['z_end'],
        z_margin=config['patch_loss_z_margin'],
        erode_cells_default=config['patch_erode_patches'],
        fiber_min_point_spacing=config['pcl_fiber_min_point_spacing'],
        overlap_pair_spacing=config['pcl_patch_overlap_pair_spacing'],
        overlap_max_pairs=config['pcl_patch_overlap_max_pairs'])


@dataclasses.dataclass
class PointViews:
    """The training views FitContext installs (FitContext._install_point_views)."""
    cross_patch: list
    strips: list
    strip_groups: list
    fiber_catalog: dict
    regular_pcl_catalog: dict
    resolved_links: list
    link_components: list

    def report(self):
        def by_group(groups):
            counts = Counter(str(group) for group in groups)
            return ', '.join(f'{group} {count}' for group, count in sorted(counts.items()))
        merged = [pcl for pcl in self.cross_patch if pcl.get('link_member_cids')]
        return [
            f'views: {len(self.cross_patch)} cross-patch ('
            + (by_group(pcl['sampling_group'] for pcl in self.cross_patch) or 'none')
            + f'), {len(self.strips)} strips ('
            + (by_group(self.strip_groups) or 'none') + ')',
            f'fiber links: {len(self.resolved_links)} in '
            f'{len(self.link_components)} component(s); {len(merged)} merged '
            'cross-patch fiber component(s)',
            f'patch overlaps: {len(self.overlap_edges)} edge(s) as '
            f'{self.overlap_pairs} pair view(s)',
        ]

    @property
    def overlap_edges(self):
        return {pcl['metadata']['edge_uuid'] for pcl in self.cross_patch
                if pcl['metadata'].get('input_role') == 'patch_overlap'}

    @property
    def overlap_pairs(self):
        return sum(1 for pcl in self.cross_patch
                   if pcl['metadata'].get('input_role') == 'patch_overlap')


def derive_point_views(inputs, patches, config):
    """Classify materialized inputs into training views; no linking.

    The store's relationships already attach points to patches, so the
    collections go straight to the shared classification: regular PCLs
    through regular_pcl_views, fibers through fiber_training_views, where
    every fiber with a usable in-ROI path keeps its strip. Patch-overlap
    pairs are cross-patch views as they are, after the regular and fiber ones.
    ``patches`` are the patches the fit uses; ``config`` the fit
    configuration. Raises the runtime validation errors (e.g. unannotated
    absolute points).
    """
    from pcl_views import (catalog_copy_of_pcl, fiber_training_views,
                           group_points_by_patch, prepare_fiber_catalog,
                           regular_pcl_views, vertical_fiber_radial_offset)
    from spiral_helpers import SequenceChain

    point_collections = inputs.point_collections
    regular_pcl_catalog = {
        cid: catalog_copy_of_pcl(pcl) for cid, pcl in point_collections.items()}
    for pcl in point_collections.values():
        pcl['chain'] = SequenceChain(pcl)
    cross_patch, strips, strip_groups = regular_pcl_views(
        point_collections, patches,
        z_begin=config['z_begin'], z_end=config['z_end'],
        z_margin=config['patch_loss_z_margin'],
        min_point_spacing=config['pcl_unattached_pcl_min_point_spacing'])
    fibers = prepare_fiber_catalog(inputs.fibers)
    resolved_links = (list(inputs.fiber_links)
                      if config['pcl_use_fiber_links'] else [])
    fiber_cross_patch, fiber_strips, fiber_groups, link_components = \
        fiber_training_views(
            fibers, resolved_links, patches,
            z_begin=config['z_begin'], z_end=config['z_end'],
            z_margin=config['patch_loss_z_margin'],
            min_point_spacing=config['pcl_unattached_pcl_min_point_spacing'],
            vertical_min_z_fraction=config['pcl_vertical_fiber_min_z_fraction'],
            vertical_min_auto_certainty=config['pcl_vertical_fiber_min_auto_certainty'],
            vertical_radial_offset=vertical_fiber_radial_offset(config),
            strip_every_fiber=True)
    strips.extend(fiber_strips)
    strip_groups.extend(fiber_groups)
    overlaps = []
    for pcl in inputs.patch_overlaps.values():
        pcl['chain'] = SequenceChain(pcl)
        group_points_by_patch(pcl, patches)
        if len(pcl['points_by_patch']) == 2:
            overlaps.append(pcl)
    return PointViews(
        list(cross_patch.values()) + fiber_cross_patch + overlaps, strips,
        strip_groups, inputs.fibers, regular_pcl_catalog, resolved_links,
        link_components)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def snapshot_report(snapshot):
    """Human-readable lines describing the selection and its relationships."""
    lines = [
        f'object store {snapshot.store_path} ({snapshot.store_uuid})',
        f'fit volume {snapshot.volume_uuid} ({snapshot.volume_source}); '
        f'units: {FIT_UNITS}',
    ]
    for kind in KINDS:
        objects = snapshot.objects_of(kind)
        scales = Counter(obj.frame.scale_factor for obj in objects)
        scale_text = ', '.join(f'{count} at scale {scale:g}'
                               for scale, count in sorted(scales.items()))
        lines.append(
            f'{KIND_LABELS[kind]}: {len(objects)} selected '
            f'({snapshot.request.selector(kind).describe()})'
            + (f'; {scale_text}' if scale_text else ''))
    families = Counter(rel.family for rel in snapshot.relationships)
    mode = 'approved only' if snapshot.request.approved_links_only else 'pending and approved'
    lines.append(f'relationships consumed ({mode}): ' + (', '.join(
        f'{family} {count}' for family, count in sorted(families.items())) or 'none'))
    if snapshot.exclusions:
        lines.append('relationships excluded: ' + ', '.join(
            f'{reason} {count}' for reason, count in sorted(snapshot.exclusions.items())))
    lines.extend(snapshot.notes)
    lines.extend(f'WARNING: {warning}' for warning in empty_family_warnings(snapshot))
    return lines


def empty_family_warnings(snapshot):
    """Constraint families the selection could hold but that consume nothing."""
    count = Counter(obj.kind for obj in snapshot.objects.values())
    expected = {
        'pcl_patch': count['patch'] and count['point_collection'],
        'fiber_patch': count['patch'] and count['fiber'],
        'patch_overlap': count['patch'] > 1,
    }
    families = Counter(rel.family for rel in snapshot.relationships)
    excluded = sum(snapshot.exclusions.values())
    return [
        f'no {family} relationships are consumed although the selection has '
        f'objects for them' + (f' ({excluded} relationship(s) excluded in all)'
                               if excluded else '')
        for family, possible in expected.items()
        if possible and not families[family]]


def constraints_report(store, snapshot, config):
    """Materialize and derive the runtime views as a fit would, and describe
    them; raises the same validation errors a fit would."""
    inputs = materialize_from_config(store, snapshot, config)
    views = derive_point_views(inputs, inputs.patches, config)
    return inputs.report() + views.report()


# ---------------------------------------------------------------------------
# Startup lifecycle: selection from the fit configuration, one link refresh
# on rank zero, then a frozen read every rank verifies against the others
# ---------------------------------------------------------------------------

class LinkRefreshError(ObjectStoreInputError):
    """Startup link refresh failed; the fit must not train."""


_SELECTOR_CONFIG = {'patch': 'patch', 'point_collection': 'pcl', 'fiber': 'fiber'}
_NONE = Selector(uuids=())


def request_from_config(config):
    """The InputRequest of a fit configuration.

    Each kind's ``object_store_<kind>_uuids``/``_tags`` select objects (both
    None: every active object). Switching a source's input toggle off
    selects none of it. The PCL role toggles cannot apply before the roles
    are read from metadata, so they must agree: all on, or all off.
    """
    selectors = {}
    for kind, name in _SELECTOR_CONFIG.items():
        uuids = config[f'object_store_{name}_uuids']
        tags = config[f'object_store_{name}_tags']
        selectors[kind] = Selector(None if uuids is None else tuple(uuids),
                                   None if tags is None else tuple(tags))
    if not config['input_use_verified_patches'] or config['input_disable_patches']:
        selectors['patch'] = _NONE
    if not config['input_use_fibers']:
        selectors['fiber'] = _NONE
    roles = {key: config[key] for key in (
        'input_use_pcl_absolute', 'input_use_pcl_relative', 'input_use_pcl_same_winding')}
    if len(set(roles.values())) > 1:
        raise ObjectStoreInputError([
            'object-store PCLs are selected by UUID or tag, not by role: the '
            f'PCL role toggles must all agree ({roles})'])
    if not any(roles.values()):
        selectors['point_collection'] = _NONE
    return InputRequest(selectors, config['fit_volume_uuid'],
                        bool(config['object_store_approved_links_only']))


@dataclasses.dataclass(frozen=True)
class NormalSource:
    """Strip normals for fiber-patch link refresh.

    ``kind`` is 'lasagna' (the fit's own normals), 'ct' (structure-tensor
    normals of a CT zarr) or 'none' (centerline-only gating).
    """
    kind: str
    ct_volume: str = ''
    parameters: tuple = ()
    note: str = None

    def describe(self):
        if self.kind == 'none':
            return 'fiber-patch normals: none (centerline-only linking)'
        values = dict(self.parameters)
        if self.kind == 'lasagna':
            return (f'fiber-patch normals: lasagna group {values["ct_level"]} at '
                    f'scale {values["ct_to_base_scale"]:g} '
                    f'({values["normal_x_volume"]}, {values["normal_y_volume"]})')
        return (f'fiber-patch normals: CT {self.ct_volume} level '
                f'{values["ct_level"]}, downsampling {values["ct_to_base_scale"]:g}')

    def provenance(self):
        return {'kind': self.kind, 'ct_volume': self.ct_volume or None,
                **dict(self.parameters)}


def fiber_normal_source(config, lasagna=None, volume_uuid=None):
    """The fiber-patch normal source: lasagna, else normal CT, else none.

    ``lasagna`` describes the fit's lasagna normals when it loads them:
    ``{'normal_x', 'normal_y', 'group', 'scale'}``, ``scale`` being the
    group's downsampling relative to the fit volume. The store decodes them
    as the fitter does. Lasagna and CT normals are exclusive; a configured
    CT is then ignored with a note.
    """
    ct_path = config['object_store_normal_ct_path']
    if lasagna is not None:
        try:
            level = int(lasagna['group'])
        except (TypeError, ValueError):
            raise ObjectStoreInputError([
                f'lasagna normal group {lasagna["group"]!r} is not an integer '
                'zarr group']) from None
        return NormalSource(
            'lasagna', '', (
                ('ct_level', level),
                ('ct_to_base_scale', float(lasagna['scale'])),
                ('normal_x_volume', str(lasagna['normal_x'])),
                ('normal_y_volume', str(lasagna['normal_y']))),
            note=(f'ignoring object_store_normal_ct_path {ct_path!r}: the fit '
                  'loads lasagna normals') if ct_path else None)
    if not ct_path:
        return NormalSource('none')
    ct_volume = config['object_store_normal_ct_volume_uuid']
    if ct_volume is None:
        raise ObjectStoreInputError([
            'object_store_normal_ct_path needs object_store_normal_ct_volume_uuid'])
    if volume_uuid is not None and ct_volume != volume_uuid:
        raise ObjectStoreInputError([
            f'the normal CT belongs to volume {ct_volume}, not the fit volume '
            f'{volume_uuid}'])
    return NormalSource('ct', str(ct_path), (
        ('ct_level', int(config['object_store_normal_ct_level'])),
        ('ct_to_base_scale', float(config['object_store_normal_ct_downsampling']))))


# Detectors the refresh runs, with the object kinds each needs selected.
DETECTORS = (
    ('pcl_patch', ('patch', 'point_collection')),
    ('fiber_patch', ('patch', 'fiber')),
    ('patch_overlap', ('patch',)),
)


@dataclasses.dataclass
class RefreshOutcome:
    detector: str
    skipped: str = None
    report: dict = None
    error: str = None

    @property
    def failed(self):
        return self.error is not None or bool(self.report and self.report['failed'])

    @property
    def committed(self):
        return bool(self.report and self.report['committed'])

    def lines(self, max_listed=5):
        if self.skipped is not None:
            return [f'link refresh {self.detector}: skipped ({self.skipped})']
        if self.error is not None:
            return [f'link refresh {self.detector}: FAILED: {self.error}']
        report = self.report
        lines = [
            f'link refresh {self.detector}: {report["edges_created"]} created, '
            f'{report["edges_updated"]} updated, '
            f'{report["edges_skipped_existing"]} unchanged, '
            f'{report["stale_removed"]} stale removed '
            f'({"committed" if report["committed"] else "nothing committed"})'
            + ('; FAILED' if report['failed'] else '')]
        failures = [issue for issue in report['issues'] if issue['severity'] == 'failure']
        for issue in failures:
            objects = issue['object_uuids']
            named = ', '.join(objects[:max_listed]) + (
                f' and {len(objects) - max_listed} more' if len(objects) > max_listed else '')
            lines.append(f'  failure at {issue["stage"]}: {issue["reason"]}'
                         + (f' ({named})' if objects else ''))
        diagnostics = [issue for issue in report['issues'] if issue['severity'] != 'failure']
        if diagnostics:
            lines.append(f'  {sum(issue["count"] or 1 for issue in diagnostics)} '
                         f'candidate diagnostic(s), e.g. {diagnostics[0]["reason"]}')
        return lines


def refresh_links(store, snapshot, config, normal_source, cache_directory=None):
    """Run each automatic detector over the snapshot's objects.

    A detector runs only when every object kind it needs is selected, with
    exactly the selected UUIDs, so its writes and stale-link removal stay
    within the selection. Exceptions and reported failures are collected
    for every detector; none stops the others.
    """
    selected = {kind: [obj.uuid for obj in snapshot.objects_of(kind)] for kind in KINDS}
    volume = snapshot.volume_uuid
    run = {
        'pcl_patch': lambda: store.link_point_collections_to_patches(
            volume_uuid=volume, collection_uuids=selected['point_collection'],
            patch_uuids=selected['patch'],
            tolerance=float(config['pcl_link_distance_tolerance']),
            window_points=int(config['pcl_link_window_points']),
            window_min_points=int(config['pcl_link_window_min_points'])),
        'fiber_patch': lambda: store.link_fibers_to_patches(
            normal_source.ct_volume, volume_uuid=volume,
            fiber_uuids=selected['fiber'], patch_uuids=selected['patch'],
            **dict(normal_source.parameters),
            **({'cache_directory': cache_directory} if cache_directory else {})),
        'patch_overlap': lambda: store.link_patch_overlaps(
            volume_uuid=volume, patch_uuids=selected['patch']),
    }
    outcomes = []
    for detector, kinds in DETECTORS:
        empty = [KIND_LABELS[kind] for kind in kinds if not selected[kind]]
        if empty:
            outcomes.append(RefreshOutcome(detector, skipped='no selected ' + ' or '.join(empty)))
            continue
        try:
            outcomes.append(RefreshOutcome(detector, report=run[detector]()))
        except Exception as error:  # noqa: BLE001 - every failure is reported
            outcomes.append(RefreshOutcome(detector, error=f'{type(error).__name__}: {error}'))
    return outcomes


def refresh_report(outcomes):
    """Report lines of a refresh, and its problems ([] on success)."""
    lines = [line for outcome in outcomes for line in outcome.lines()]
    failed = [outcome.detector for outcome in outcomes if outcome.failed]
    if not failed:
        return lines, []
    committed = [outcome.detector for outcome in outcomes if outcome.committed]
    problems = lines + [
        f'link refresh failed in {", ".join(failed)}; the fit will not train. '
        + (f'{", ".join(committed)} committed before the failure (detectors '
           'are not rolled back together; a rerun reconciles them)'
           if committed else 'No detector committed changes.')]
    return lines, problems


def _startup_phase(config, request, lasagna, cache_directory, log):
    """Rank zero's startup: resolve the selection once, validate it and its
    frames, then (unless refresh is off) run the detectors as
    object_store_author. Returns ``(selected UUIDs by kind, report lines)``;
    raises ObjectStoreInputError on any failure."""
    from vc_object_store import ObjectStore

    refresh = bool(config['object_store_refresh_links'])
    author = config['object_store_author']
    if refresh and not author:
        raise ObjectStoreInputError([
            'object_store_author is required to refresh links '
            '(or set object_store_refresh_links false)'])
    options = {'author': author} if refresh else {'read_only': True}
    with ObjectStore.open(config['object_store_path'], **options) as store:
        uuids = resolve_selection(store, request)
        snapshot = read_snapshot(store, request, uuids)
        normal = fiber_normal_source(config, lasagna, snapshot.volume_uuid)
        if not refresh:
            return uuids, ['link refresh: off; reading the store\'s current relationships']
        log(normal.describe())
        if normal.note:
            log(normal.note)
        outcomes = refresh_links(store, snapshot, config, normal, cache_directory)
    lines, problems = refresh_report(outcomes)
    if problems:
        raise LinkRefreshError(problems)
    return uuids, lines


def _broadcast(value, dist):
    if dist is None or not dist.is_distributed:
        return value
    import torch.distributed
    holder = [value]
    torch.distributed.broadcast_object_list(holder, src=0)
    return holder[0]


def _all_gather(value, dist):
    if dist is None or not dist.is_distributed:
        return [value]
    import torch.distributed
    gathered = [None] * dist.world_size
    torch.distributed.all_gather_object(gathered, value)
    return gathered


# Configuration that shapes the materialized constraints, recorded in the
# manifest so a resume can tell when they changed.
MATERIALIZATION_KEYS = (
    'z_begin', 'z_end', 'patch_loss_z_margin', 'patch_erode_patches',
    'pcl_fiber_min_point_spacing', 'pcl_patch_overlap_pair_spacing',
    'pcl_patch_overlap_max_pairs', 'pcl_unattached_pcl_min_point_spacing',
    'pcl_use_fiber_links', 'pcl_vertical_fiber_min_z_fraction',
    'pcl_vertical_fiber_min_auto_certainty',
)


def input_manifest(snapshot, config, normal_source, external_frames=None):
    """The snapshot manifest extended with materialization settings,
    external-input frames and the link-refresh mode."""
    manifest = snapshot.manifest()
    manifest['materialization'] = {key: config[key] for key in MATERIALIZATION_KEYS}
    manifest['external_frames'] = dict(sorted((external_frames or {}).items()))
    manifest['link_refresh'] = {
        'enabled': bool(config['object_store_refresh_links']),
        'fiber_patch_normals': normal_source.provenance(),
    }
    return manifest


def _compared(manifest):
    """The manifest without what does not define the inputs: the store's
    local path and how links were refreshed (their results are recorded)."""
    manifest = dict(manifest)
    manifest['store'] = {'uuid': (manifest.get('store') or {}).get('uuid')}
    manifest.pop('link_refresh', None)
    return manifest


def manifest_digest(manifest):
    text = json.dumps(_compared(manifest), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(text.encode()).hexdigest()


def manifest_differences(recorded, current):
    """Names of the manifest entries (``section`` or ``section.key``) that
    differ between two manifests."""
    recorded, current = _compared(recorded), _compared(current)
    differences = []
    for key in sorted(set(recorded) | set(current)):
        old, new = recorded.get(key), current.get(key)
        if old == new:
            continue
        if isinstance(old, dict) and isinstance(new, dict):
            differences.extend(f'{key}.{sub}' for sub in sorted(set(old) | set(new))
                               if old.get(sub) != new.get(sub))
        else:
            differences.append(key)
    return differences


def resume_problems(checkpoint_input_manifest, inputs, allow_changed=False, log=print):
    """Reasons a checkpoint may not resume against these frozen inputs.

    The checkpoint must record the same inputs, frames and materialization
    settings; ``allow_changed`` (object_store_allow_changed_inputs) resumes
    anyway, with a warning naming what changed.
    """
    recorded = (checkpoint_input_manifest or {}).get('object_store')
    if recorded is None:
        problem = 'checkpoint records no object-store inputs'
    else:
        differences = manifest_differences(recorded, inputs.manifest)
        if not differences:
            return []
        problem = ('checkpoint object-store inputs differ from this fit\'s in '
                   + ', '.join(differences))
    if allow_changed:
        log(f'WARNING: {problem}; resuming anyway (object_store_allow_changed_inputs)')
        return []
    return [f'{problem}; set object_store_allow_changed_inputs to resume anyway']


def load_fit_inputs(config, *, lasagna=None, external_frames=None, dist=None,
                    cache_directory=None, log=print):
    """Refresh, freeze and materialize a fit's object-store inputs.

    Rank zero (the only process when ``dist`` is None or single-rank) runs
    the one refresh phase unless object_store_refresh_links is off; the
    other ranks wait for its result, and every rank fails when it fails.
    Every rank then reads the now-immutable store read-only, materializes
    the same frozen snapshot and closes the store; their manifests must
    agree before anyone trains. Returns FitInputs carrying ``manifest`` and
    ``digest``.
    """
    from vc_object_store import ObjectStore

    if not config['object_store_path']:
        raise ObjectStoreInputError(['object_store_path is not configured'])
    main = dist is None or dist.is_main_process
    status = None
    if main:
        try:
            uuids, lines = _startup_phase(
                config, request_from_config(config), lasagna, cache_directory, log)
            status = {'ok': True, 'uuids': uuids, 'lines': lines}
        except ObjectStoreInputError as error:
            status = {'ok': False, 'problems': error.problems}
        except Exception as error:  # noqa: BLE001 - other ranks must hear of it
            status = {'ok': False, 'problems': [f'{type(error).__name__}: {error}']}
    status = _broadcast(status, dist)
    if not status['ok']:
        raise LinkRefreshError(status['problems'])
    if main:
        for line in status['lines']:
            log(line)

    request = request_from_config(config)
    with ObjectStore.open(config['object_store_path'], read_only=True) as store:
        snapshot = read_snapshot(store, request, status['uuids'])
        normal = fiber_normal_source(config, lasagna, snapshot.volume_uuid)
        inputs = materialize_from_config(store, snapshot, config)
    inputs.manifest = input_manifest(snapshot, config, normal, external_frames)
    inputs.digest = manifest_digest(inputs.manifest)
    digests = _all_gather(inputs.digest, dist)
    if len(set(digests)) > 1:
        raise ObjectStoreInputError([
            'ranks read different object-store inputs: '
            + ', '.join(f'rank {rank} {digest[:12]}' for rank, digest in enumerate(digests))])
    if main:
        log(f'object-store inputs frozen: digest {inputs.digest[:16]}')
    return inputs


def _report_config(args):
    from config import Config
    overrides = {}
    if args.config:
        with open(args.config) as stream:
            overrides = json.load(stream)
    config = Config(overrides).as_dict()
    config.update(z_begin=args.z_begin, z_end=args.z_end)
    return config


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('store')
    parser.add_argument('--volume', help='fit volume UUID (default: inferred)')
    parser.add_argument('--approved-only', action='store_true')
    for kind, flag in (('patch', 'patch'), ('point_collection', 'pcl'),
                       ('fiber', 'fiber')):
        parser.add_argument(f'--{flag}-uuid', action='append', dest=f'{kind}_uuids')
        parser.add_argument(f'--{flag}-tag', action='append', dest=f'{kind}_tags')
    parser.add_argument('--z-begin', type=int,
                        help='with --z-end, also materialize the constraints of this z window')
    parser.add_argument('--z-end', type=int)
    parser.add_argument('--config', help='fit configuration JSON overriding the defaults')
    parser.add_argument('--json', help='write the manifest and report here')
    return parser.parse_args(argv)


def main(argv=None):
    from vc_object_store import ObjectStore

    args = _parse_args(argv)
    selectors = {}
    for kind in KINDS:
        uuids, tags = getattr(args, f'{kind}_uuids'), getattr(args, f'{kind}_tags')
        selectors[kind] = Selector(
            tuple(uuids) if uuids is not None else None,
            tuple(tags) if tags is not None else None)
    request = InputRequest(selectors, args.volume, args.approved_only)
    with ObjectStore.open(args.store, read_only=True) as store:
        try:
            snapshot = read_snapshot(store, request)
            if args.z_begin is not None and args.z_end is not None:
                lines = constraints_report(store, snapshot, _report_config(args))
            else:
                lines = snapshot_report(snapshot)
        except ValueError as error:
            for problem in getattr(error, 'problems', None) or [str(error)]:
                print(f'error: {problem}', file=sys.stderr)
            return 1
    print('\n'.join(lines))
    if args.json:
        with open(args.json, 'w') as stream:
            json.dump({'manifest': snapshot.manifest(),
                       'digest': snapshot.digest(), 'report': lines},
                      stream, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
