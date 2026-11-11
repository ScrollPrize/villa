"""Fit inputs read from a vc_object_store database.

Patches, fibers, point collections (PCLs) and tracks, and the relationships
between them, come from one object store. This module turns a selection of them into
the fitter's runtime inputs in three stages:

1. Snapshot extraction (read_snapshot): resolve object selectors to explicit
   UUID sets, read every selected object, its geometry revision and the
   relationships between selected objects in one read view, and filter the
   relationships by applicability, approval and semantics.
2. Frame conversion (ObjectFrame): every selected object declares its frame in
   ``meta.coordinates``; geometry and anchors convert once from stored XYZ to
   ZYX voxels of one reference volume, the only units the fitter uses.
3. Runtime materialization: patches, PCL and fiber constraints and the track
   collection with its crossings, built from the frozen snapshot
   (materialize_* below).

The store's automatic detectors are authoritative: their relationships are
consumed as they are, never relinked by proximity here.

    python object_store_inputs.py STORE [--volume V] \
        [--z-begin Z0 --z-end Z1 [--config FIT.json]]

prints the selection's report without training. Given a z window it also
materializes and derives the training views exactly as a fit with that
configuration would, failing on the same validation errors.

STORE, like the fit configuration's object_store_location, is a store file
or the endpoint of the vc_object_server serving one (open_store): every
read, the link refresh and the pins work alike on both.
"""
import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import sys
import time
from collections import Counter

import numpy as np

KINDS = ('surface', 'point_collection', 'fiber', 'track')
# The tag of the scroll's outer shell: a surface the fit reads on its own
# (InputSnapshot.outer_shell, FitInputs.outer_shell), never as a patch.
OUTER_SHELL_TAG = 'spiral-outer-shell'
# The tag of a committed Spiral fit (service_commits): a fit result, never a
# fit input.
SPIRAL_FIT_TAG = 'spiral-fit'
KIND_LABELS = {'surface': 'surfaces', 'point_collection': 'point collections',
               'fiber': 'fibers', 'track': 'tracks'}

# Relationship families the fitter consumes, keyed by (edge kind, endpoint
# kinds in KINDS order). Surface overlaps and fiber-surface contacts share the
# detectors' 'same-surface' edge kind and differ by endpoint kinds. Family
# names are Spiral's roles and keep the fitter's "patch" vocabulary. Track
# crossings feed the track graph; track-surface edges ('same-surface' between
# a surface and a track) constrain nothing in the fit and are excluded.
FAMILIES = {
    ('point-surface', ('surface', 'point_collection')): 'pcl_patch',
    ('same-surface', ('surface', 'fiber')): 'fiber_patch',
    ('same-surface', ('surface', 'surface')): 'patch_overlap',
    ('fiber_branch', ('fiber', 'fiber')): 'fiber_branch',
    ('crossing', ('track', 'track')): 'track_crossing',
}

# The coordinates the fitter works in, recorded with every provenance.
FIT_UNITS = 'reference-volume voxels (ZYX)'

# Manifest fields that pin records without defining the inputs: an object's
# or relationship's numbered version moves with every change to it, those
# the adapter does not read included (a review note, a relationship of a
# selected object), so versions resolve the inputs exactly but are left out
# of the digest. Geometry revisions, record UUIDs, consumed content and
# approval states define them.
VERSION_FIELDS = {'objects': ('version', 'reference'), 'relationships': ('version',)}


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

    def provenance(self):
        """The selector as JSON holds it (lists, not tuples), so a recorded
        manifest compares equal to the live one."""
        return {key: None if value is None else list(value)
                for key, value in dataclasses.asdict(self).items()}

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
class InputPosition:
    """Identify a store state by UUID, history generation and commit sequence.

    Local and hosted readers use the same identifiers.
    """
    store: str
    generation: str
    commit_sequence: int

    def provenance(self):
        return {'store': self.store, 'generation': self.generation,
                'commit_sequence': self.commit_sequence}

    def describe(self):
        return f'commit {self.commit_sequence}'

    def same_store(self, status):
        """Whether a store status (store.status()) is of this position's
        store and generation."""
        return (str(status['store']) == self.store
                and str(status['generation']) == self.generation)

    @classmethod
    def of_status(cls, status):
        return cls(str(status['store']), str(status['generation']), int(status['commit_sequence']))

    @classmethod
    def from_provenance(cls, value):
        if value is None:
            return None
        if 'position' in value and 'commit_sequence' not in value:
            value = value['position']
        return cls(str(value['store']), str(value['generation']), int(value['commit_sequence']))


@dataclasses.dataclass(frozen=True)
class InputPins:
    """What a fit reads exactly: the selected objects (and the outer shell)
    at the numbered versions they had when the links were made, and the
    position (the links' commit) that state belongs to. Reads at pins use
    the store's numbered history. Inherited endpoints do not pin overlay
    membership; such selections cannot replay after an overlay commit.

    ``references`` may be empty (a position alone, e.g. a configuration's
    object_store_input_position): the versions are then those at the
    store's head when the head is that position, else read from the
    store's history at the position's publication (local stores).
    ``references`` retains each selected version's actual source owner, which
    may differ from the attached store. Legacy numeric pins are decoded at
    the provenance boundary with their recorded single-store identity."""
    position: InputPosition
    references: dict = dataclasses.field(default_factory=dict)  # {uuid: complete reference}
    selection: dict = dataclasses.field(default_factory=dict)  # {kind: sorted UUIDs}
    outer_shell: str = None

    @property
    def pinned(self):
        """Whether the object versions are known."""
        return bool(self.selection)

    def provenance(self):
        return {'position': self.position.provenance(),
                'selection': {kind: list(self.selection.get(kind, ())) for kind in KINDS},
                'outer_shell': self.outer_shell, 'references': self.references}

    @staticmethod
    def _decode_reference(position, uuid, reference, version=None):
        """Decode new or original single-store legacy pin evidence once."""
        from vc_object_store import object_version_from_json
        if reference is None:
            reference = dict(store=position.store, generation=position.generation, object=uuid, version=version)
        reference = object_version_from_json(json.dumps(reference))
        if reference['object'] != uuid or (version is not None and reference['version'] != version):
            raise ValueError(f'inconsistent exact input reference for {uuid}')
        return reference

    @classmethod
    def from_provenance(cls, value):
        if value is None:
            return None
        position = InputPosition.from_provenance(value)
        if 'objects' not in value and 'references' not in value:
            return cls(position)
        old = value.get('objects') or {}
        references = value.get('references') or {}
        references = {uuid: cls._decode_reference(position, uuid, references.get(uuid), old.get(uuid))
                      for uuid in dict.fromkeys([*old, *references])}
        return cls(position, references,
                   {kind: sorted(value.get('selection', {}).get(kind, ())) for kind in KINDS},
                   value.get('outer_shell'))

    @classmethod
    def from_manifest(cls, manifest):
        """The pins an input manifest (InputSnapshot.manifest) records, or
        None when it contains no position."""
        position = InputPosition.from_provenance((manifest or {}).get('position'))
        if position is None:
            return None
        shell = manifest.get('outer_shell') or None
        entries = manifest.get('objects') or ()
        references = {entry['uuid']: cls._decode_reference(position, entry['uuid'], entry.get('reference'),
                                                         entry.get('version'))
                      for entry in [*entries, *([shell] if shell else [])]}
        return cls(position, references,
                   {kind: sorted(entry['uuid'] for entry in entries if entry['kind'] == kind) for kind in KINDS},
                   shell['uuid'] if shell else None)

    def refs(self, uuids=None):
        """Numbered references of ``uuids`` (default: every pinned object)."""
        return [self.references[uuid] for uuid in (self.references if uuids is None else uuids)]


@dataclasses.dataclass(frozen=True)
class InputRequest:
    """What to read: per-kind selectors, the fit volume and the approval policy.

    ``fit_volume_uuid`` None infers the volume from the selected objects'
    declarations. ``approved_links_only`` consumes only approved
    relationships; otherwise pending and approved ones are consumed. Rejected
    relationships are never consumed. ``pins`` None reads the store as it is
    now; InputPins read the pinned object versions (selector membership
    included), whatever the store holds now.
    """
    selectors: dict = dataclasses.field(default_factory=dict)
    fit_volume_uuid: str = None
    approved_links_only: bool = False
    pins: InputPins = None
    # Whether the fit reads the outer shell: 'off' (input_use_outer_shell
    # false), 'optional' (it filters tracks when present) or 'required' (shell
    # losses, winding-model supervision; a store without one is refused).
    outer_shell: str = 'optional'
    # The pyramid level of the fit volume the fit works in (fit_volume_level):
    # a fit voxel is 2^level of the volume's voxels.
    fit_volume_level: int = 0

    @property
    def fit_scale(self):
        """Volume voxels per fit voxel."""
        return 1 << int(self.fit_volume_level)

    def __post_init__(self):
        # An unknown key would otherwise leave its kind unselected-by-default:
        # every active object.
        unknown = sorted(set(self.selectors) - set(KINDS))
        if unknown:
            raise ValueError(f'unknown object kinds {unknown}; expected {list(KINDS)}')

    def selector(self, kind):
        return self.selectors.get(kind) or Selector()

    @property
    def position(self):
        return None if self.pins is None else self.pins.position


@dataclasses.dataclass(frozen=True)
class ObjectFrame:
    """One object's frame: stored XYZ = reference-volume XYZ x scale_factor,
    and fit XYZ = stored XYZ x to_fit (the fit voxels are those of a pyramid
    level of the reference volume)."""
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


def content_sha256(value):
    """SHA-256 of a JSON value in canonical form."""
    text = json.dumps(value, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(text.encode()).hexdigest()


def adapter_metadata(kind, meta, fiber=None, track=None):
    """The metadata the adapter reads from one object of ``kind``.

    With the geometry revision (points, arrays and point annotations) it is
    everything materialization reads, so its hash identifies the object's
    input content in any copy of the store. Other metadata (review notes, ingest
    diagnostics) does not reach the fit and does not change the digest.

    This fit-protocol projection is mirrored by VC3D's ``adapterMetadata``;
    ``test_spiral_store_inputs`` checks parity for coordinates, source tuning,
    sampling group and fiber/track family fields. These are Spiral semantics,
    not store identity/placement rules, so they do not belong in vcobj's schema.
    """
    meta = meta or {}
    source = meta.get('source') or {}
    read = {'coordinates': meta.get('coordinates')}
    if kind == 'surface':
        read['spiral_patch_erode_cells'] = source.get('spiral_patch_erode_cells')
    elif kind == 'point_collection':
        read['winding_is_absolute'] = source.get('winding_is_absolute')
        read['sampling_group'] = meta.get('sampling_group')
    elif kind == 'fiber':
        read['hv_classification'] = source.get('hv_classification')
        read['family'] = (fiber or {}).get('family')
        read['manual_family'] = (fiber or {}).get('manual_family')
    elif kind == 'track':
        # The family gates crossings; the dataset and key order the tracks.
        for key in ('family', 'dataset_ref', 'external_key'):
            read[key] = (track or {}).get(key)
    return read


@dataclasses.dataclass
class StoredObject:
    kind: str
    name: str
    meta: dict
    tags: list
    reference: dict
    geometry_revision: str
    frame: ObjectFrame = None
    fiber: dict = None
    track: dict = None

    @property
    def uuid(self):
        return self.reference['object']

    @property
    def version(self):
        return self.reference['version']

    @property
    def metadata_sha256(self):
        return content_sha256(adapter_metadata(self.kind, self.meta, self.fiber, self.track))

    def provenance(self):
        """``metadata`` is what the adapter read, so a client compares it
        with an object's current metadata as values; ``metadata_sha256`` is
        its identity in the digest."""
        return {'uuid': self.uuid, 'kind': self.kind, 'name': self.name,
                'version': self.version, 'reference': self.reference,
                'geometry_revision': self.geometry_revision,
                'metadata': adapter_metadata(self.kind, self.meta, self.fiber, self.track),
                'metadata_sha256': self.metadata_sha256,
                'scale_factor': self.frame.scale_factor}


@dataclasses.dataclass
class StoredRelationship:
    """A consumed relationship with endpoints ordered by kind (KINDS order;
    UUID order between objects of one kind), not by stored order."""
    uuid: str
    version: int
    kind: str
    family: str
    approval_state: str
    a: str
    b: str
    a_anchor: dict
    b_anchor: dict
    meta: dict
    a_geometry_revision: str = None
    b_geometry_revision: str = None
    stored_order: tuple = None  # (a, b) as the record stores them

    def provenance(self):
        """Identity by content: the shared record, its endpoints' geometry
        and a hash of the anchors and metadata materialization reads.
        ``version`` (its numbered version) only pins it."""
        return {'uuid': self.uuid, 'version': self.version,
                'kind': self.kind, 'family': self.family,
                'approval_state': self.approval_state,
                'a': self.a, 'a_geometry_revision': self.a_geometry_revision,
                'b': self.b, 'b_geometry_revision': self.b_geometry_revision,
                'content_sha256': content_sha256(
                    {'a_anchor': self.a_anchor, 'b_anchor': self.b_anchor,
                     'meta': self.meta})}


@dataclasses.dataclass
class InputSnapshot:
    """The frozen selection: objects, their frames and consumed relationships."""
    store_location: str
    store_uuid: str
    store_generation: str
    volume_uuid: str
    volume_source: str
    request: InputRequest
    objects: dict
    relationships: list
    exclusions: Counter
    notes: list
    # The outer shell (StoredObject), read with the selection but not part
    # of it; None when the store has none.
    outer_shell: StoredObject = None

    @property
    def position(self):
        return self.request.position

    def pins(self):
        """The snapshot's objects at the versions it read, as the pins of
        ``position`` (the store's head when it read the current state)."""
        return InputPins(self.position,
                         {obj.uuid: obj.reference for obj in [*self.objects.values(), self.outer_shell]
                          if obj is not None},
                         {kind: sorted(obj.uuid for obj in self.objects_of(kind)) for kind in KINDS},
                         None if self.outer_shell is None else self.outer_shell.uuid)

    def objects_of(self, kind):
        return [obj for obj in self.objects.values() if obj.kind == kind]

    def relationships_of(self, family):
        return [rel for rel in self.relationships if rel.family == family]

    def manifest(self):
        """Complete input/frame provenance, in stable order."""
        return {
            'store': {'location': self.store_location, 'uuid': self.store_uuid,
                      'generation': self.store_generation},
            'position': self.position.provenance() if self.position else None,
            'volume_uuid': self.volume_uuid,
            'volume_source': self.volume_source,
            'units': FIT_UNITS,
            'approved_links_only': self.request.approved_links_only,
            'selection': {
                kind: {'selector': self.request.selector(kind).provenance(),
                       'count': len(self.objects_of(kind))}
                for kind in KINDS},
            'objects': [obj.provenance() for obj in self.objects.values()],
            'outer_shell': self.outer_shell.provenance() if self.outer_shell else None,
            'relationships': [rel.provenance() for rel in self.relationships],
            'excluded_relationships': dict(sorted(self.exclusions.items())),
        }

    def digest(self):
        """SHA-256 of the manifest without its version pins."""
        return manifest_digest(self.manifest())


def attached_data(record):
    """Whether an object is data attached to another (``meta.role``, e.g. fit
    results) rather than a fit input. The store's PCL-surface detector refuses
    such collections, so they are never selected as inputs."""
    return record['kind'] == 'point_collection' and 'role' in (record.get('meta') or {})


def outer_shell(record):
    """Whether an object is the scroll's outer shell (OUTER_SHELL_TAG): read
    on its own, never as a patch."""
    return record['kind'] == 'surface' and OUTER_SHELL_TAG in (record.get('tags') or ())


def spiral_fit(record):
    """Whether an object is a committed Spiral fit (SPIRAL_FIT_TAG). Taking
    one in would make a fit's own output its next input, and change the input
    digest the commit recorded."""
    return record['kind'] == 'surface' and SPIRAL_FIT_TAG in (record.get('tags') or ())


def fit_input(record):
    """Whether a default or tag selection may take this object in."""
    return not attached_data(record) and not outer_shell(record) and not spiral_fit(record)


def resolve_selector(store, kind, selector, records=None):
    """``{UUID: reference}`` of the objects ``selector`` selects among active
    objects, at the numbered versions read.

    Default and tag selections leave out attached data, the outer shell and
    committed Spiral fits; naming any of them by UUID is an error. ``records``
    ({UUID: record} of every pinned object, retired ones included) replaces
    the store's current objects."""
    def remember(record):
        from vc_object_store import object_reference
        return object_reference(record)

    def matching(tag=None):
        if records is not None:
            return {uuid: remember(record) for uuid, record in records.items()
                    if record['kind'] == kind and not record.get('retired')
                    and (tag is None or tag in (record.get('tags') or ()))
                    and fit_input(record)}
        query = {} if tag is None else {'tag': tag}
        return {record['uuid']: remember(record) for record in store.select_objects(
            object_kind=kind, fields=['uuid', 'kind', 'meta', 'tags'], **query)
            if fit_input(record)}

    if selector.is_default:
        return matching()
    selected = {}
    problems = []
    for uuid in selector.uuids or ():
        record = current_record(store, uuid) if records is None else records.get(uuid)
        if record is None:
            problems.append(f'selected {kind} {uuid} does not exist'
                            + ('' if records is None else ' at this position'))
        elif record['kind'] != kind:
            problems.append(f'selected {kind} {uuid} is a {record["kind"]}')
        elif record.get('retired'):
            problems.append(f'selected {kind} {uuid} is retired')
        elif attached_data(record):
            problems.append(f'selected {kind} {uuid} is attached data with role '
                            f'{record["meta"]["role"]!r}, not a fit input')
        elif outer_shell(record):
            problems.append(f'selected {kind} {uuid} is the outer shell (tagged '
                            f'{OUTER_SHELL_TAG}), not a patch')
        elif spiral_fit(record):
            problems.append(f'selected {kind} {uuid} is a committed Spiral fit (tagged '
                            f'{SPIRAL_FIT_TAG}), not a fit input')
        else:
            selected[record['uuid']] = remember(record)
    if problems:
        raise ObjectStoreInputError(problems)
    for tag in selector.tags or ():
        selected.update(matching(tag=tag))
    return selected


def current_record(store, uuid):
    """An object's current record with its numbered ``version``, or None
    (absent, or deleted)."""
    revision = store.current_object_revision(uuid)
    if revision is None or revision.get('deleted'):
        return None
    return {**revision['record'], 'version': revision['version']}


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


def object_frame(meta, volume_uuid, fit_scale=1):
    """The object's ObjectFrame in ``volume_uuid``, whose fit voxels are
    ``fit_scale`` of its voxels; raises ValueError when its frame is unknown,
    invalid or in another volume."""
    from vc_object_store import geometry_scale_to_frame, resolve_geometry_coordinates
    coordinates = resolve_geometry_coordinates(meta, volume_uuid)
    return ObjectFrame(volume_uuid, float(coordinates['scale_factor']),
                       float(geometry_scale_to_frame(meta, volume_uuid)) / fit_scale)


def _relationship_family(record, kinds):
    endpoint_kinds = tuple(sorted((kinds[record['a']], kinds[record['b']]),
                                  key=KINDS.index))
    return FAMILIES.get((record['kind'], endpoint_kinds))


def adjacent_winding_offset(fiber_a, fiber_b):
    """W_b - W_a an adjacent-winding branch between two fiber objects
    asserts: the vertical fiber lies one winding inside the horizontal one.
    None unless one is H and the other V by stored family (the effective
    manual-else-automatic tag VC3D signed the branch with; VC3D's
    adjacentWindingOffset)."""
    families = tuple((fiber.fiber or {}).get('family') for fiber in (fiber_a, fiber_b))
    return {('H', 'V'): -1, ('V', 'H'): 1}.get(families)


def _ordered_endpoints(entry, kinds):
    """(a, a_anchor, a geometry pin, b, b_anchor, b geometry pin) of a view
    relationship, ordered by kind, then UUID."""
    record = entry['record']
    ends = [(record[side], record[f'{side}_anchor'],
             entry.get(f'{side}_geometry_revision')) for side in ('a', 'b')]
    ends.sort(key=lambda end: (KINDS.index(kinds[end[0]]), end[0]))
    return (*ends[0], *ends[1])


def filter_relationships(view_relationships, objects, approved_links_only):
    """Consumed relationships and exclusion counts.

    A relationship is consumed when both endpoints are selected, it applies
    to their selected revisions, it belongs to a family the fitter uses and
    its approval state is allowed (never 'rejected'; only 'approved' in
    approved-only mode). Adjacent-winding fiber branches are consumed like
    ordinary ones; materialize_fibers signs them.
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
        state = record['approval_state']
        if state == 'rejected':
            exclusions['rejected'] += 1
            continue
        if approved_links_only and state != 'approved':
            exclusions[f'{state} (approved-only)'] += 1
            continue
        a, a_anchor, a_geometry, b, b_anchor, b_geometry = _ordered_endpoints(entry, kinds)
        consumed.append(StoredRelationship(
            uuid=record['uuid'], version=int(entry['version']), kind=record['kind'],
            family=family, approval_state=state,
            a=a, b=b, a_anchor=a_anchor,
            b_anchor=b_anchor, meta=record['meta'] or {},
            a_geometry_revision=a_geometry, b_geometry_revision=b_geometry,
            stored_order=(record['a'], record['b'])))
    consumed.sort(key=lambda rel: (rel.family, rel.a, rel.b, rel.uuid))
    return consumed, exclusions


@dataclasses.dataclass(frozen=True)
class SelectedObjects:
    """A resolved selection: the UUIDs per kind, every selected object's
    numbered version (the outer shell's included) and the store position
    (store.status()) the versions were read at. ``uuids`` is what callers
    index: ``selected['surface']``."""
    uuids: dict
    references: dict
    position: InputPosition
    outer_shell: str = None

    def __getitem__(self, kind):
        return self.uuids[kind]

    def pins(self):
        return InputPins(self.position, self.references,
                         {kind: list(self.uuids[kind]) for kind in KINDS}, self.outer_shell)


def check_position(store, position):
    """Refuse a position of another store or history generation."""
    status = store.status()
    if not position.same_store(status):
        raise ObjectStoreInputError([
            f'input position {position.describe()} belongs to store {position.store} '
            f'(generation {position.generation}); this store is {status["store"]} '
            f'(generation {status["generation"]})'])
    if position.commit_sequence > status['commit_sequence']:
        raise ObjectStoreInputError([
            f'input position {position.describe()} is after this store\'s head '
            f'(commit {status["commit_sequence"]})'])


def _records_at_position(store, position):
    """{UUID: record (with its version)} of every object live at a position,
    retired ones included, from the store's numbered history. Needs the
    publication time of the position's commit, so it reads the change feed
    for that one commit (changes_since)."""
    status = store.status()
    if not hasattr(store, 'versions_at'):
        raise ObjectStoreInputError([
            f'the state of a served store at {position.describe()} cannot be read without '
            'the object versions pinned there; update the fit\'s inputs to a current position'])
    cursor = {'store': position.store, 'generation': position.generation,
              'sequence': position.commit_sequence - 1}
    page = store.changes_since(cursor, max_changes=1)
    if not page['cursor_valid'] or not page['commits'] \
            or page['commits'][0]['sequence'] != position.commit_sequence:
        raise ObjectStoreInputError([
            f'the state of the store at {position.describe()} cannot be read: its change '
            f'feed starts after it (feed start {status.get("feed_start")}); update the fit\'s '
            'inputs to a current position'])
    published = page['commits'][0]['published_at_us']
    refs = store.versions_at(published, include_retired=True)
    records = {}
    for revision in store.objects_at_versions(refs):
        if revision is None or revision.get('deleted'):
            continue
        records[revision['object_uuid']] = {**revision['record'], 'version': revision['version']}
    return records


def _pinned_records(store, pins):
    """{UUID: record} of the pinned objects at their versions (retired or
    deleted ones left out), read exactly from the numbered history."""
    records = {}
    for ref, revision in zip(pins.refs(), store.objects_at_versions(pins.refs())):
        if revision is None:
            raise ObjectStoreInputError([
                f'pinned object {ref["object"]} has no version {ref["version"]} in this store'])
        if revision.get('deleted'):
            continue
        records[revision['object_uuid']] = {**revision['record'], 'version': revision['version']}
    return records


# How often a read of the store's current objects is repeated when a commit
# lands between its queries, before it reads the store's history instead.
CAPTURE_ATTEMPTS = 8


def _capture_current(store, request):
    """``(status, {kind: {UUID: reference}}, shell_references)`` of the request's
    selection among the store's current objects and the surfaces tagged
    OUTER_SHELL_TAG, all at the head ``status`` names. The queries are
    separate reads, so the head is read before and after them: a commit in
    between would label versions of a later commit with an earlier head,
    and the capture is repeated from the new head (CAPTURE_ATTEMPTS)."""
    for _ in range(CAPTURE_ATTEMPTS):
        from vc_object_store import object_reference
        status = store.status()
        selected = {kind: resolve_selector(store, kind, request.selector(kind)) for kind in KINDS}
        shells = []
        if request.outer_shell != 'off':
            for record in store.select_objects(object_kind='surface', tag=OUTER_SHELL_TAG, fields=['uuid']):
                shells.append(object_reference(record))
        after = store.status()
        if (after['generation'], after['commit_sequence']) == (status['generation'], status['commit_sequence']):
            return status, selected, shells
    raise ObjectStoreInputError([
        f'the store kept changing while its inputs were read ({CAPTURE_ATTEMPTS} attempts); retry'])


def _shells_among(request, records, uuid=None):
    """Complete references of the outer shells among ``records`` (the
    state at a position), or of the one pinned as ``uuid``."""
    if request.outer_shell == 'off':
        return []
    if uuid is not None:
        records = {uuid: records[uuid]} if uuid in records else {}
    from vc_object_store import object_reference
    return [object_reference(record) for record in records.values()
            if outer_shell(record) and not record.get('retired')]


def _one_shell(shells):
    if len(shells) > 1:
        raise ObjectStoreInputError([
            f'{len(shells)} surfaces are tagged {OUTER_SHELL_TAG} ({", ".join(sorted(s["object"] for s in shells))}); '
            'the outer shell must be one'])
    return shells[0] if shells else None


def resolve_selection(store, request):
    """The request's selection (SelectedObjects): resolved among the store's
    current objects, or, with pins, among the pinned objects at their
    versions. Pins without object versions (a position alone) resolve at
    the store's head when that is the position, else from the store's
    history at the position. The selection, the outer shell and the
    position are one capture: every version is of the position's commit."""
    def resolve(records):
        return {kind: resolve_selector(store, kind, request.selector(kind), records)
                for kind in KINDS}
    pins = request.pins
    if pins is None:
        status, selected, shells = _capture_current(store, request)
        position = InputPosition.of_status(status)
    elif pins.pinned:
        records = _pinned_records(store, pins)
        selected = resolve(records)
        for kind in KINDS:
            missing = sorted(set(pins.selection.get(kind, ())) - set(selected[kind]))
            if missing:
                raise ObjectStoreInputError([
                    f'pinned {kind} {uuid} is not a fit input at {pins.position.describe()}'
                    for uuid in missing])
        shells = [] if pins.outer_shell is None else _shells_among(request, records, pins.outer_shell)
        position = pins.position
    else:
        check_position(store, pins.position)
        position = pins.position
        status = store.status()
        captured = None
        if status['commit_sequence'] == position.commit_sequence:
            # The head is the position: its current objects, unless a commit
            # moves the head meanwhile (the capture then reads a later one).
            status, selected, shells = _capture_current(store, request)
            if status['commit_sequence'] == position.commit_sequence:
                captured = selected, shells
        if captured is None:
            records = _records_at_position(store, position)
            selected, shells = resolve(records), _shells_among(request, records)
    shell = _one_shell(shells)
    references = {uuid: ref for by_kind in selected.values() for uuid, ref in by_kind.items()}
    if shell is not None:
        references[shell['object']] = shell
    return SelectedObjects({kind: sorted(selected[kind]) for kind in KINDS},
                           references, position,
                           None if shell is None else shell['object'])


def read_snapshot(store, request, selected=None):
    """Resolve, validate and freeze one selection of the store.

    Selectors resolve to explicit UUID sets before any volume filter, so
    conflicting frame declarations among selected objects are seen and fail.
    ``selected`` (resolve_selection's result) reuses a selection resolved
    earlier, so one startup reads exactly the objects it refreshed. Every
    object is read at the version the selection resolved it at
    (objects_at_versions), and the relationships between those versions
    (relationships_at_versions) are guarded against moving overlay membership.
    Every selected object must declare a frame in the fit volume; all frame
    problems are reported together. The selected outer shell is
    read alongside, in the same volume, but is not among ``objects``.
    """
    if selected is None:
        selected = resolve_selection(store, request)
    shell = selected.outer_shell
    if shell is None and request.outer_shell == 'required':
        raise ObjectStoreInputError([
            f'the fit needs the outer shell, and no surface is tagged {OUTER_SHELL_TAG}'])
    position = selected.position
    pins = selected.pins()
    refs = pins.refs()
    refs_by_uuid = {ref['object']: ref for ref in refs}
    revisions = {}
    for ref, revision in zip(refs, store.objects_at_versions(refs)):
        if revision is None or revision.get('deleted'):
            raise ObjectStoreInputError([
                f'selected object {ref["object"]} is not at version {ref["version"]} of this store'])
        revisions[revision['object_uuid']] = revision
    inherited = any(ref['store'] != position.store for ref in refs)
    if inherited and InputPosition.of_status(store.status()) != position:
        raise ObjectStoreInputError(['Cannot replay inherited relationship membership after the store changed'])
    relationships_read = store.relationships_at_versions(refs)
    if inherited and InputPosition.of_status(store.status()) != position:
        raise ObjectStoreInputError(['The store changed while inherited relationships were read; retry'])

    notes, problems = [], []
    usable = []
    for kind in KINDS:
        without_geometry = []
        for uuid in selected[kind]:
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
    shell_revision = None if shell is None else revisions[shell]
    if shell_revision is not None and shell_revision['geometry_revision'] is None:
        problems.append(f'the outer shell {shell} has no geometry')
    if problems:
        raise ObjectStoreInputError(problems)

    volume_uuid, volume_source = infer_fit_volume(
        [revision['record'] for revision in usable]
        + ([shell_revision['record']] if shell_revision is not None else []),
        request.fit_volume_uuid)

    def stored(revision):
        record = revision['record']
        try:
            frame = object_frame(record['meta'], volume_uuid, request.fit_scale)
        except ValueError as error:
            problems.append(f'{record["kind"]} {record["uuid"]} '
                            f'({record["name"]!r}): {error}')
            return None
        return StoredObject(
            kind=record['kind'], name=record['name'],
            meta=record['meta'] or {}, tags=list(record.get('tags') or ()),
            geometry_revision=revision['geometry_revision'], frame=frame,
            fiber=record.get('fiber'), track=record.get('track'),
            reference=refs_by_uuid[record['uuid']])

    objects = {}
    for revision in usable:
        if (obj := stored(revision)) is not None:
            objects[obj.uuid] = obj
    shell_object = None if shell_revision is None else stored(shell_revision)
    if problems:
        raise ObjectStoreInputError(problems)

    relationships, exclusions = filter_relationships(
        relationships_read, objects, request.approved_links_only)
    pinned_request = request if request.pins is not None and request.pins.pinned \
        else dataclasses.replace(request, pins=selected.pins())
    return InputSnapshot(
        store_location=store_location(store), store_uuid=position.store,
        store_generation=position.generation, volume_uuid=volume_uuid,
        volume_source=volume_source, request=pinned_request, objects=objects,
        relationships=relationships, exclusions=exclusions, notes=notes,
        outer_shell=shell_object)


def store_location(store):
    """Where a store handle reads from: its file path, or its endpoint."""
    for attribute in ('path', 'endpoint'):
        location = getattr(store, attribute, None)
        if location:
            return str(location)
    return ''


def is_served(store):
    """Whether a store handle reads a store served by vc_object_server (the
    detectors of its link requests run there) rather than a store file."""
    return bool(getattr(store, 'endpoint', None))


# Under the fit's cache directory: the fiber strips a local link refresh
# keeps between runs, and a served store's persistent cache of the numbered
# records and geometry it read (HostedSessionOptions::cacheDirectory).
STRIP_CACHE_DIRECTORY = 'object_store_strips'
HOSTED_CACHE_DIRECTORY = 'object_store_hosted'


@contextlib.contextmanager
def open_store(config, *, read_only=False, cache_directory=None):
    """The store ``object_store_location`` names, for the block
    (vc_object_store.locations.open_store): a store file opened read-only
    or as ``object_store_author``, or a served store attached as that author
    with ``object_store_link_timeout_s`` for its link requests and, under
    ``cache_directory`` (the fit's cache), a persistent cache of what it
    reads. Both give the same reading and request methods."""
    from vc_object_store.locations import open_store as open_location
    location = config['object_store_location']
    if not location:
        raise ObjectStoreInputError(['object_store_location is not configured'])
    hosted_cache = os.path.join(cache_directory, HOSTED_CACHE_DIRECTORY) if cache_directory else None
    with open_location(location, author=config['object_store_author'], read_only=read_only,
                       link_timeout_s=int(config['object_store_link_timeout_s'] or 300),
                       cache_directory=hosted_cache) as store:
        yield store


def strip_cache_directory(store, cache_directory):
    """The fiber strip cache a link refresh of ``store`` uses: under the
    fit's ``cache_directory`` for a store file; none for a served store,
    whose server keeps its own (its --strip-cache)."""
    if not cache_directory or is_served(store):
        return None
    return os.path.join(cache_directory, STRIP_CACHE_DIRECTORY)


# ---------------------------------------------------------------------------
# Geometry reads (pinned by geometry revision, so independent of later heads)
# ---------------------------------------------------------------------------

def read_geometry(store, geometry_revision, names, optional=()):
    """``({name: ndarray}, attributes)`` of an immutable geometry revision.
    Arrays named in ``optional`` are returned only when stored."""
    descriptor = store.geometry_revision(geometry_revision)
    arrays = {array['name']: array for array in descriptor['arrays']}
    result = {}
    for name in [*names, *(name for name in optional if name in arrays)]:
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
    ``fiber_links`` holds their resolved branch links, each with the
    ``winding_offset`` it asserts (0 same-winding, +-1 adjacent).
    ``patch_overlaps`` maps resident ids to the two-point same-surface
    collections of materialize_patch_overlaps. ``tracks`` holds the track
    collection and its crossings (materialize_tracks), or None when no track
    is selected.
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
    # The outer shell as a fit-frame tifxyz.Patch (materialize_outer_shell),
    # or None.
    outer_shell: object = None
    tracks: 'TrackInputs' = None
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


def _surface_zyxs(store, obj, what):
    """``(zyxs, grid scale)`` of a stored surface in the fit frame; invalid
    vertices (and holes in its ``valid`` channel) carry the -1 sentinel."""
    arrays, attributes = read_geometry(store, obj.geometry_revision, ['points'],
                                       optional=['valid'])
    points = arrays['points']
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ObjectStoreInputError([f'{what} {obj.uuid} points have shape {points.shape}'])
    from vc_object_store import Surface
    validity = arrays.get('valid')
    surface = Surface.from_arrays(np.ascontiguousarray(points, dtype='<f4'), attributes['scale'],
                                  None if validity is None else np.ascontiguousarray(validity, dtype='u1'))
    invalid = ~surface.valid_mask()
    return (obj.frame.xyz_to_fit_zyx(points, invalid), invalid,
            obj.frame.grid_scale_to_fit(attributes['scale']))


def materialize_outer_shell(store, snapshot):
    """The snapshot's outer shell as a fit-frame tifxyz.Patch (whole: no z
    window or erosion), or None."""
    obj = snapshot.outer_shell
    if obj is None:
        return None
    import torch
    from tifxyz import Patch
    zyxs, _, scale = _surface_zyxs(store, obj, 'outer shell')
    return Patch(torch.from_numpy(zyxs), torch.tensor(scale), None, None, obj.uuid, None)


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
    for obj in snapshot.objects_of('surface'):
        zyxs, invalid, scale = _surface_zyxs(store, obj, 'patch')
        valid_quads = (~invalid[:-1, :-1] & ~invalid[1:, :-1]
                       & ~invalid[:-1, 1:] & ~invalid[1:, 1:])
        if not valid_quads.any():
            counts['patches without valid quads'] += 1
            continue
        erosion = (obj.meta.get('source') or {}).get('spiral_patch_erode_cells')
        patch = Patch(torch.from_numpy(zyxs), torch.tensor(scale),
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
        recorded = (rel.meta.get('point_collection_surface') or {}).get(
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
    ij = (rel.meta.get('point_collection_surface') or {}).get('ij')
    if point_id is None or not isinstance(ij, list) or len(ij) != 2:
        raise ObjectStoreInputError([
            f'point-surface edge {rel.uuid} lacks a point ID anchor or surface ij'])
    if rel.a not in patches:
        return point_id, None
    return point_id, {'id': rel.a, 'ij': [float(ij[0]), float(ij[1])],
                      'edge_uuid': rel.uuid}


def materialize_point_collections(store, snapshot, patches, first_id=0,
                                  dropped=None):
    """Regular PCLs of the snapshot, keyed by resident id in UUID order.

    Points are keyed by their stable IDs and carry fit-frame ZYX and their
    aligned ``wind_a`` annotation. Each consumed point-surface edge attaches its
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
                problems.append(f'point-surface edge {rel.uuid} refers to point '
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
                'object_version': obj.version,
                'geometry_revision': obj.geometry_revision,
            },
        }
        counts['absolute point collections' if absolute
               else 'relative point collections'] += 1
    if problems:
        raise ObjectStoreInputError(problems)
    return collections, counts


def _fiber_contact(rel):
    """``(raw line index, ij)`` of a fiber-surface edge's representative contact."""
    line_index = _anchor_index(rel.b_anchor, 'line_points')
    grid = (rel.meta.get('fiber_surface') or {}).get('surface_grid')
    if line_index is None or not isinstance(grid, list) or len(grid) != 2:
        raise ObjectStoreInputError([
            f'fiber-surface edge {rel.uuid} lacks a line-point anchor or surface_grid'])
    # surface_grid is (column, row); attachments use (row, column).
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
    raw line index; each carries the winding gap it asserts between its
    ends, ``winding_offset`` = W_b - W_a: 0 for a same-winding branch, -1 or
    +1 for an adjacent-winding one (adjacent_winding_offset). An adjacent
    branch whose fibers are not one H and one V asserts no direction and is
    left out.
    """
    from vc_object_store import fiber_branch_control_index, fiber_hv_classification
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
        if not len(controls) and not len(line):
            raise ObjectStoreInputError([f'fiber {obj.uuid} has no points'])
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
        hv = source.get('hv_classification') or {}
        if obj.fiber is not None:
            hv = fiber_hv_classification(
                hv, obj.fiber['family'], obj.fiber['manual_family'])
        catalog[obj.uuid] = {
            'id': cid, 'name': obj.name, 'points': points,
            'color': [0.0, 0.0, 0.0], 'source_file': None,
            'metadata': {
                'hv_classification': hv,
                'object_version': obj.version,
                'object_uuid': obj.uuid,
                'geometry_revision': obj.geometry_revision,
                'fiber_num_control_points': len(controls),
            },
        }
        branch_points[obj.uuid] = branch_lines
        counts['fibers'] += 1

    links = []
    for rel in snapshot.relationships_of('fiber_branch'):
        winding_offset = 0
        if rel.meta.get('adjacent'):
            winding_offset = adjacent_winding_offset(
                snapshot.objects[rel.a], snapshot.objects[rel.b])
            if winding_offset is None:
                counts['adjacent fiber branches not one H and one V'] += 1
                continue
        links.append({
            'a_coll': resident_id[rel.a], 'a_point': branch_points[rel.a][rel.uuid],
            'b_coll': resident_id[rel.b], 'b_point': branch_points[rel.b][rel.uuid],
            'pending': rel.approval_state != 'approved', 'edge_uuid': rel.uuid,
            'winding_offset': winding_offset,
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
    follow the stored endpoint order, with (row, col) grids and XYZ in the
    fit volume's own voxels, converted to the fit level's. Each pair becomes one
    collection of two points, the pair's point on either patch, with zero
    relative winding: the relative-winding loss then compares exactly that
    correspondence along its own two-node chain. Every pair of an edge
    carries ``sampling_member_weight`` 1 / (its retained pairs) and the
    edge's UUID as ``sampling_unit``: the sampler draws edges, then one pair
    of each, so an edge weighs the same however many pairs it produces.

    A pair on a quad either patch's losses do not sample is dropped; an edge
    losing every pair is appended to ``dropped``. Edge errors, and pairs
    computed from geometry other than the snapshot's, are reported together.
    Returns ``(collections keyed by resident id, counts)``.
    """
    edges = {rel.uuid: rel for rel in snapshot.relationships_of('patch_overlap')}
    collections, counts, problems = {}, Counter(), []
    if not edges:
        return collections, counts
    # From the snapshot's records (the edge's pose, its endpoints' frames and
    # geometry revisions), not the store's current state: a store that moved
    # past the snapshot's position must materialize what the digest names.
    def pinned(rel):
        a, b = rel.stored_order or (rel.a, rel.b)
        anchors = {rel.a: rel.a_anchor, rel.b: rel.b_anchor}
        return {'uuid': rel.uuid,
                'whole_geometry': all(bool((anchors[end] or {}).get('whole_geometry'))
                                      for end in (a, b)),
                'meta': json.dumps(rel.meta),
                'a_reference': snapshot.objects[a].reference,
                'b_reference': snapshot.objects[b].reference}
    fit_scale = snapshot.request.fit_scale
    results = store.overlap_correspondences_at(
        [pinned(rel) for rel in edges.values()], snapshot.volume_uuid,
        float(pair_spacing) * fit_scale, int(max_pairs))
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
                    0: _overlap_point(cid, 0, a, pair['a_grid'],
                                      np.asarray(pair['a_xyz']) / fit_scale, rel.uuid),
                    1: _overlap_point(cid, 1, b, pair['b_grid'],
                                      np.asarray(pair['b_xyz']) / fit_scale, rel.uuid)},
                'color': [0.0, 0.0, 0.0], 'source_file': None,
                'sampling_group': group,
                'sampling_member_weight': 1.0 / len(kept),
                'sampling_unit': rel.uuid,
                'metadata': {
                    'winding_is_absolute': False,
                    'input_role': 'patch_overlap',
                    'resident_collection_id': cid,
                    'edge_uuid': rel.uuid,
                    'edge_version': rel.version,
                    'pair_distance': float(pair['distance']) / fit_scale,
                },
            }
            cid += 1
        counts['patch overlaps'] += 1
        counts['overlap pairs'] += len(kept)
    problems.extend(f'patch overlap {uuid} returned no result' for uuid in edges)
    if problems:
        raise ObjectStoreInputError(problems)
    return collections, counts


@dataclasses.dataclass
class TrackInputs:
    """The selected tracks as the track losses read them.

    ``tracks`` is a fit-frame tracks.PackedTrackCollection of the tracks lying
    entirely inside the z window, ordered by dataset and source key (the
    order of the packed store they were ingested from); a track's source id is
    its row among every selected track. ``crossings`` is the exact-crossing
    CSR over every selected track (rows are source ids): partner rows,
    crossing point indices on both tracks, and the crossing's arclength
    position and clearance in fit voxels.
    """
    tracks: object
    crossings: dict


# Stored track family -> the crossing kernel's family code.
TRACK_FAMILY_CODES = {'horizontal': 0, 'vertical': 1}


def _track_order(obj):
    """Tracks sort by dataset, then numeric source key, then UUID."""
    track = obj.track or {}
    key = str(track.get('external_key') or '')
    return (str(track.get('dataset_ref') or ''), not key.isdigit(),
            int(key) if key.isdigit() else 0, key, obj.uuid)


def _track_crossing_csr(snapshot, tracks, lengths, problems):
    """The crossing CSR of the snapshot's track crossings over ``tracks``."""
    row = {obj.uuid: index for index, obj in enumerate(tracks)}
    records = []
    for rel in snapshot.relationships_of('track_crossing'):
        meta = rel.meta.get('track_crossing') or {}
        endpoints = meta.get('endpoints') or {}
        ends = []
        for uuid, anchor in ((rel.a, rel.a_anchor), (rel.b, rel.b_anchor)):
            local = _anchor_index(anchor, 'points')
            arclength = (endpoints.get(uuid) or {}).get('arclength_voxels')
            if local is None or arclength is None:
                problems.append(f'track crossing {rel.uuid} lacks a point anchor '
                                f'or arclength for {uuid}')
            elif not 0 <= local < lengths[row[uuid]]:
                problems.append(f'track crossing {rel.uuid} anchors point {local} '
                                f'outside the {lengths[row[uuid]]} points of {uuid}')
            else:
                ends.append((row[uuid], local,
                             float(arclength) * tracks[row[uuid]].frame.to_fit))
        clearance = meta.get('clearance_voxels')
        if clearance is None:
            problems.append(f'track crossing {rel.uuid} lacks its clearance')
        if len(ends) < 2 or clearance is None:
            continue
        clearance = float(clearance) * tracks[ends[0][0]].frame.to_fit
        (a, a_local, a_position), (b, b_local, b_position) = ends
        records.append((a, b, a_local, b_local, a_position, clearance))
        records.append((b, a, b_local, a_local, b_position, clearance))
    table = np.array(records, dtype=np.float64).reshape(-1, 6)
    table = table[np.lexsort((table[:, 1], table[:, 0]))]
    rows = table[:, 0].astype(np.int64)
    offsets = np.zeros(len(tracks) + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=len(tracks)), out=offsets[1:])
    return {
        'source_ids': np.arange(len(tracks), dtype=np.uint64),
        'offsets': offsets,
        'partners': table[:, 1].astype(np.int32),
        'self_local': table[:, 2].astype(np.int32),
        'partner_local': table[:, 3].astype(np.int32),
        'positions': table[:, 4].copy(),
        'clearances': table[:, 5].copy(),
    }


def materialize_tracks(store, snapshot, *, z_begin, z_end):
    """The snapshot's tracks and crossings (TrackInputs), or None without
    tracks; and counts.

    A track is kept when every point's fit-frame z lies in ``[z_begin,
    z_end)``. Arclengths and tortuosities are measured on the stored integer
    voxels, as the packed track store measures them, then converted to fit
    voxels.
    """
    from tracks import PackedTrackCollection

    tracks = sorted(snapshot.objects_of('track'), key=_track_order)
    counts = Counter()
    if not tracks:
        return None, counts
    problems = []
    lengths = np.zeros(len(tracks), dtype=np.int64)
    kept, stored, fit, families, scales = [], [], [], [], []
    for index, obj in enumerate(tracks):
        arrays, _ = read_geometry(store, obj.geometry_revision, ['points'])
        xyz = arrays['points']
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            problems.append(f'track {obj.uuid} points have shape {xyz.shape}')
            continue
        lengths[index] = len(xyz)
        if not len(xyz):
            continue
        z = xyz[:, 2].astype(np.float64) * obj.frame.to_fit
        if z.min() < z_begin or z.max() >= z_end:
            counts['tracks outside the z window'] += 1
            continue
        family = (obj.track or {}).get('family') or ''
        kept.append(index)
        stored.append(np.ascontiguousarray(xyz[:, ::-1], dtype=np.int32))
        fit.append(obj.frame.xyz_to_fit_zyx(xyz))
        families.append(TRACK_FAMILY_CODES.get(family, -1))
        scales.append(obj.frame.to_fit)
    crossings = _track_crossing_csr(snapshot, tracks, lengths, problems)
    if problems:
        raise ObjectStoreInputError(problems)

    track_lengths = np.array([len(points) for points in stored], dtype=np.int64)
    offsets = np.zeros(len(stored) + 1, dtype=np.int64)
    np.cumsum(track_lengths, out=offsets[1:])
    if stored:
        coordinates = np.concatenate(stored)
        starts = offsets[:-1]
        steps = np.empty(len(coordinates), dtype=np.float64)
        steps[0] = 0.0
        deltas = np.subtract(coordinates[1:], coordinates[:-1], dtype=np.float64)
        steps[1:] = np.linalg.norm(deltas, axis=1)
        steps[starts] = 0.0
        arclengths = np.add.reduceat(steps, starts)
        chords = np.linalg.norm(
            coordinates[offsets[1:] - 1].astype(np.float64)
            - coordinates[starts].astype(np.float64), axis=1)
        tortuosities = np.full(len(stored), np.inf, dtype=np.float64)
        np.divide(arclengths, chords, out=tortuosities, where=chords > 0)
        arclengths = arclengths * np.asarray(scales, dtype=np.float64)
        points = np.concatenate(fit)
    else:
        arclengths = tortuosities = np.empty(0, dtype=np.float64)
        points = np.empty((0, 3), dtype=np.float32)
    collection = PackedTrackCollection(
        points, offsets, np.asarray(kept, dtype=np.uint64),
        np.asarray(families, dtype=np.int8), arclengths, tortuosities)
    counts['tracks'] = len(kept)
    counts['track crossings'] = len(crossings['partners']) // 2
    return TrackInputs(collection, crossings), counts


def materialize(store, snapshot, *, z_begin, z_end, z_margin,
                erode_cells_default, fiber_min_point_spacing=0.0,
                overlap_pair_spacing=8.0, overlap_max_pairs=16, patch_overlaps=True):
    """Every runtime input of a frozen snapshot.

    Fibers are decimated to ``fiber_min_point_spacing`` (the fitter's
    pcl_fiber_min_point_spacing); patch overlaps yield at most
    ``overlap_max_pairs`` pairs ``overlap_pair_spacing`` voxels apart.
    Resident ids number PCLs, then fibers, then overlap pairs;
    ``patch_overlaps`` False (input_use_patch_overlaps off) makes none.
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
    if patch_overlaps:
        overlaps, overlap_counts = materialize_patch_overlaps(
            store, snapshot, patches, pair_spacing=overlap_pair_spacing,
            max_pairs=overlap_max_pairs,
            first_id=len(point_collections) + len(fibers), dropped=dropped)
        counts.update(overlap_counts)
    else:
        overlaps = {}
        counts['patch overlaps unused (input_use_patch_overlaps off)'] = len(
            snapshot.relationships_of('patch_overlap'))
    tracks, track_counts = materialize_tracks(store, snapshot, z_begin=z_begin, z_end=z_end)
    counts.update(track_counts)
    return FitInputs(snapshot, patches, point_collections, fibers,
                     fiber_links, counts, dropped_attachments=dropped,
                     patch_overlaps=overlaps,
                     outer_shell=materialize_outer_shell(store, snapshot),
                     tracks=tracks)


def materialize_from_config(store, snapshot, config):
    """materialize() with the fit configuration's window and settings."""
    return materialize(
        store, snapshot, z_begin=config['z_begin'], z_end=config['z_end'],
        z_margin=config['patch_loss_z_margin'],
        erode_cells_default=config['patch_erode_patches'],
        fiber_min_point_spacing=config['pcl_fiber_min_point_spacing'],
        overlap_pair_spacing=config['pcl_patch_overlap_pair_spacing'],
        overlap_max_pairs=config['pcl_patch_overlap_max_pairs'],
        patch_overlaps=bool(config['input_use_patch_overlaps']))


@dataclasses.dataclass
class PointViews:
    """The training views FitContext installs (FitContext._install_point_views)."""
    cross_patch: list
    strips: list
    strip_groups: list
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
    from pcl_views import (fiber_training_views, group_points_by_patch,
                           prepare_fiber_catalog, regular_pcl_views,
                           vertical_fiber_radial_offset)
    from spiral_helpers import SequenceChain

    point_collections = inputs.point_collections
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
            vertical_radial_offset=vertical_fiber_radial_offset(config))
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
        strip_groups, resolved_links, link_components)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def snapshot_summary(snapshot):
    """The selection a snapshot holds, as data; snapshot_report prints it.

    Per object its provenance (UUID, kind, name, revision pins, metadata
    hash, scale) and, for point collections, the role its metadata
    declares; then per-kind counts, consumed relationship families,
    exclusions and the fit volume. Reads nothing beyond the snapshot, so a
    service can describe a selection without refreshing or materializing.
    """
    selection = {}
    for kind in KINDS:
        objects = snapshot.objects_of(kind)
        selector = snapshot.request.selector(kind)
        selection[kind] = {
            'selector': dataclasses.asdict(selector),
            'description': selector.describe(),
            'count': len(objects),
            'scales': [[scale, count] for scale, count in sorted(
                Counter(obj.frame.scale_factor for obj in objects).items())],
        }
    objects = []
    for obj in snapshot.objects.values():
        entry = obj.provenance()
        entry['volume_uuid'] = obj.frame.volume_uuid
        if obj.kind == 'point_collection':
            absolute = (obj.meta.get('source') or {}).get('winding_is_absolute', False)
            entry['pcl_role'] = 'absolute' if absolute is True else 'relative'
        objects.append(entry)
    return {
        'store': {'location': snapshot.store_location, 'uuid': snapshot.store_uuid,
                  'generation': snapshot.store_generation},
        'volume_uuid': snapshot.volume_uuid,
        'volume_source': snapshot.volume_source,
        'units': FIT_UNITS,
        'approved_links_only': snapshot.request.approved_links_only,
        'selection': selection,
        'objects': objects,
        'outer_shell': (snapshot.outer_shell.provenance()
                        if snapshot.outer_shell else None),
        'relationships': dict(sorted(
            Counter(rel.family for rel in snapshot.relationships).items())),
        'excluded_relationships': dict(sorted(snapshot.exclusions.items())),
        'notes': list(snapshot.notes),
        'warnings': empty_family_warnings(snapshot),
    }


def snapshot_report(snapshot):
    """Human-readable lines describing the selection and its relationships."""
    summary = snapshot_summary(snapshot)
    lines = [
        f'object store {summary["store"]["location"]} ({summary["store"]["uuid"]}, '
        f'generation {summary["store"]["generation"]})',
        f'fit volume {summary["volume_uuid"]} ({summary["volume_source"]}); '
        f'units: {FIT_UNITS}',
    ]
    for kind in KINDS:
        selected = summary['selection'][kind]
        scale_text = ', '.join(f'{count} at scale {scale:g}'
                               for scale, count in selected['scales'])
        lines.append(
            f'{KIND_LABELS[kind]}: {selected["count"]} selected '
            f'({selected["description"]})'
            + (f'; {scale_text}' if scale_text else ''))
    shell = summary['outer_shell']
    lines.append(f'outer shell: {shell["name"]!r} ({shell["uuid"]})' if shell
                 else f'outer shell: none (no surface tagged {OUTER_SHELL_TAG})')
    mode = 'approved only' if summary['approved_links_only'] else 'pending and approved'
    lines.append(f'relationships consumed ({mode}): ' + (', '.join(
        f'{family} {count}' for family, count in summary['relationships'].items())
        or 'none'))
    if summary['excluded_relationships']:
        lines.append('relationships excluded: ' + ', '.join(
            f'{reason} {count}'
            for reason, count in summary['excluded_relationships'].items()))
    lines.extend(summary['notes'])
    lines.extend(f'WARNING: {warning}' for warning in summary['warnings'])
    return lines


def empty_family_warnings(snapshot):
    """Constraint families the selection could hold but that consume nothing."""
    count = Counter(obj.kind for obj in snapshot.objects.values())
    expected = {
        'pcl_patch': count['surface'] and count['point_collection'],
        'fiber_patch': count['surface'] and count['fiber'],
        'patch_overlap': count['surface'] > 1,
        'track_crossing': count['track'] > 1,
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


_SELECTOR_CONFIG = {'surface': 'patch', 'point_collection': 'pcl', 'fiber': 'fiber',
                    'track': 'track'}
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
        selectors['surface'] = _NONE
    if not config['input_use_fibers']:
        selectors['fiber'] = _NONE
    if not config['input_use_tracks']:
        selectors['track'] = _NONE
    roles = {key: config[key] for key in (
        'input_use_pcl_absolute', 'input_use_pcl_relative', 'input_use_pcl_same_winding')}
    if len(set(roles.values())) > 1:
        raise ObjectStoreInputError([
            'object-store PCLs are selected by UUID or tag, not by role: the '
            f'PCL role toggles must all agree ({roles})'])
    if not any(roles.values()):
        selectors['point_collection'] = _NONE
    from fit_session import input_source_enabled, outer_shell_required
    shell = ('required' if outer_shell_required(config)
             else 'optional' if input_source_enabled(config, 'outer_shell') else 'off')
    return InputRequest(selectors, config['fit_volume_uuid'],
                        bool(config['object_store_approved_links_only']),
                        outer_shell=shell,
                        fit_volume_level=int(config['fit_volume_level']))


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
            where = (f'pack {values["normal_pool"]}' if 'normal_pool' in values
                     else f'{values["normal_x_volume"]}, {values["normal_y_volume"]}')
            return (f'fiber-patch normals: lasagna at '
                    f'scale {values["ct_to_base_scale"]:g} ({where})')
        return (f'fiber-patch normals: CT {self.ct_volume}, downsampling '
                f'{values["ct_to_base_scale"]:g}')

    def provenance(self):
        return {'kind': self.kind, 'ct_volume': self.ct_volume or None,
                **dict(self.parameters)}


def fiber_normal_source(config, lasagna=None, volume_uuid=None):
    """The fiber-patch normal source: lasagna, else normal CT, else none.

    ``lasagna`` describes the fit's lasagna normals when they are configured
    (input_use_normals with a dataset whose normals pack is at hand, whether
    or not the dense-normal loss is on; fit_session.lasagna_normal_inputs):
    ``{'pool', 'scale'}``, the pack the fitter loads (lasagna_source) and its
    array's downsampling relative to the fit volume. The store reads the
    pack, never the dataset's arrays; a normal CT's level selects its group.
    Lasagna and CT normals are exclusive; a configured CT is then ignored
    with a note.
    """
    ct_path = config['object_store_normal_ct_path']
    if lasagna is not None:
        return NormalSource(
            'lasagna', '', (
                ('ct_level', 0),
                ('ct_to_base_scale', float(lasagna['scale'])),
                ('normal_pool', str(lasagna['pool']))),
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
    level = str(config['object_store_normal_ct_level'])
    return NormalSource('ct', os.path.join(str(ct_path), level), (
        ('ct_level', 0),
        ('ct_to_base_scale', float(config['object_store_normal_ct_downsampling']))))


# The store's detector identities (meta.detector.name) of the refresh's steps.
DETECTOR_NAMES = {'pcl_patch': 'point-collection-surface', 'fiber_patch': 'fiber-surface',
                  'patch_overlap': 'surface-overlap', 'track_crossing': 'track-track'}

# Detectors the refresh runs, with the object kinds each needs selected.
DETECTORS = (
    ('pcl_patch', ('surface', 'point_collection')),
    ('fiber_patch', ('surface', 'fiber')),
    ('patch_overlap', ('surface',)),
    ('track_crossing', ('track',)),
)


# Detectors that link incrementally (refresh_links ``new``), with the object
# kinds whose new objects each examines.
INCREMENTAL_KINDS = {
    'pcl_patch': ('point_collection', 'surface'),
    'fiber_patch': ('fiber', 'surface'),
    'patch_overlap': ('surface',),
}


@dataclasses.dataclass
class RefreshOutcome:
    """One detector of a refresh: skipped (and why), or run with the report
    the store recorded in the link marker."""
    detector: str
    skipped: str = None
    report: dict = None
    # 'full' or 'incremental over N new object(s)'; None when skipped.
    mode: str = None

    def lines(self, max_listed=3):
        if self.skipped is not None:
            return [f'link refresh {self.detector}: skipped ({self.skipped})']
        report = self.report
        lines = [
            f'link refresh {self.detector}: {report["edges_created"]} created, '
            f'{report["edges_updated"]} updated, '
            f'{report["edges_skipped_existing"]} unchanged, '
            f'{report["stale_removed"]} stale removed'
            + (f' [{self.mode}]' if self.mode else '')]
        problems = list(report.get('problems') or ())
        if problems:
            shown = '; '.join(problems[:max_listed])
            lines.append(f'  {len(problems)} note(s): {shown}'
                         + (f'; and {len(problems) - max_listed} more' if len(problems) > max_listed else ''))
        return lines


def _plan_progress(progress):
    """The link request's progress callback publishing through a
    ProgressReporter (spiral_progress): the detector (the stage's prefix)
    is the stage, its own stage and item the detail; a total of 0 is
    unknown."""
    if progress is None:
        return None
    current = {'stage': None}
    names = {detector: name for name, detector in DETECTOR_NAMES.items()}

    def report(stage, completed, total, detail):
        detector, separator, rest = stage.partition(': ')
        if not separator:
            detector, rest = 'marker', stage
        if detector != current['stage']:
            current['stage'] = detector
            progress.begin('linking', f'link refresh {names.get(detector, detector)}')
        progress.update(completed, total_steps=total,
                        detail=f'{rest}: {detail}' if detail else rest)
    return report


def link_plan(snapshot, config, normal_source, cache_directory=None, new=None, marker=None):
    """The ``link_objects`` plan refreshing the snapshot's selection: one
    step per automatic detector that runs, with the detector parameters the
    standalone calls take, and the marker content to record with the links
    (LinkPlanned: the plan, and the outcome of every detector including the
    skipped ones).

    A detector runs only when every object kind it needs is selected, with
    exactly the selected UUIDs, so its writes and stale-link removal stay
    within the selection. ``new`` ({kind: UUIDs}) makes the run incremental:
    each detector that supports it examines only pairs with a new object of
    its kinds (the store's new_uuids), and is skipped when none is new;
    track crossings are skipped (an incremental refresh requires unchanged
    tracks). ``cache_directory`` is the fiber strip cache of a local store;
    a served store keeps its own.
    """
    selected = {kind: [obj.uuid for obj in snapshot.objects_of(kind)] for kind in KINDS}
    volume = snapshot.volume_uuid

    def fresh(detector):
        return sorted(uuid for kind in INCREMENTAL_KINDS[detector] for uuid in new.get(kind, ()))

    def options(detector):
        return {'new_uuids': fresh(detector)} if new is not None else {}
    steps = {
        'pcl_patch': lambda: {
            'detector': 'point-collection-surface',
            'params': dict(
                volume_uuid=volume, collection_uuids=selected['point_collection'],
                surface_uuids=selected['surface'],
                tolerance=float(config['pcl_link_distance_tolerance']),
                window_points=int(config['pcl_link_window_points']),
                window_min_points=int(config['pcl_link_window_min_points']),
                **options('pcl_patch'))},
        'fiber_patch': lambda: {
            'detector': 'fiber-surface',
            **({'ct_volume': normal_source.ct_volume} if normal_source.ct_volume else {}),
            'params': dict(
                volume_uuid=volume, fiber_uuids=selected['fiber'], surface_uuids=selected['surface'],
                **dict(normal_source.parameters),
                **({'cache_directory': cache_directory} if cache_directory else {}),
                **options('fiber_patch'))},
        'patch_overlap': lambda: {
            'detector': 'surface-overlap',
            'params': dict(volume_uuid=volume, surface_uuids=selected['surface'],
                           **options('patch_overlap'))},
        # Crossings are exact shared voxels of the stored tracks, found with
        # the store's default angle, tangent and clustering parameters.
        'track_crossing': lambda: {
            'detector': 'track-track',
            'params': dict(track_uuids=selected['track'], workers=os.cpu_count() or 1)},
    }
    outcomes, detectors = [], []
    for detector, kinds in DETECTORS:
        empty = [KIND_LABELS[kind] for kind in kinds if not selected[kind]]
        if empty:
            outcome = RefreshOutcome(detector, skipped='no selected ' + ' or '.join(empty))
        elif detector == 'patch_overlap' and not config['input_use_patch_overlaps']:
            outcome = RefreshOutcome(detector, skipped='input_use_patch_overlaps is off')
        elif new is not None and detector not in INCREMENTAL_KINDS:
            outcome = RefreshOutcome(detector, skipped='incremental refresh: its inputs are unchanged')
        elif new is not None and not fresh(detector):
            outcome = RefreshOutcome(detector, skipped='incremental refresh: no new ' + ' or '.join(
                KIND_LABELS[kind] for kind in INCREMENTAL_KINDS[detector]))
        else:
            outcome = RefreshOutcome(detector, mode=(
                'full' if new is None else f'incremental over {len(fresh(detector))} new object(s)'))
            detectors.append(steps[detector]())
        outcomes.append(outcome)
    return LinkPlanned({'detectors': detectors, 'marker': dict(marker or {})}, outcomes)


@dataclasses.dataclass
class LinkPlanned:
    """A link_objects plan and the outcomes its detectors will have (the
    skipped ones already decided)."""
    plan: dict
    outcomes: list

    @property
    def runs(self):
        return [outcome for outcome in self.outcomes if outcome.skipped is None]


def link_request(status, plan, author=None, read_references=None):
    """The checked request submitting a link plan to the store at ``status``
    (store.status()): the detectors read under a snapshot and the links
    commit expecting every object they read at its version, so a commit in
    between conflicts and nothing is written. ``read_references`` (complete
    references of the selection as the caller read it before planning) are read
    dependencies of the commit, so an edit of a selected object between
    that read and the commit conflicts too, and the pins stay exact."""
    import uuid as uuid_module
    import vc_object_store as vos
    request = {'command_version': 2, 'request': str(uuid_module.uuid4()),
               'store': status['store'], 'generation': status['generation'],
               'expected_objects': {}, 'expected_relationships': {},
               'expected_object_references': list(read_references or ()),
               'read_dependencies': {},
               'expected_commit_sequence': None,
               'operations': [{'op': vos.LINK_OBJECTS_OPERATION, 'plan': plan}]}
    if author:
        request['author'] = author
    return request


def is_conflict(error):
    """Whether an exception a submit raised is a transaction conflict (a
    commit between the detectors' snapshot and their commit)."""
    import vc_object_store as vos
    conflict = getattr(vos, 'TransactionConflict', None)
    if conflict is not None and isinstance(error, conflict):
        return True
    return isinstance(error, RuntimeError) and 'conflict' in str(error).lower()


def submit_link_plan(store, planned, author=None, progress=None, log=None, read_references=None):
    """Run a plan (link_plan) on the store as one ``link_objects`` request:
    returns ``(receipt, outcomes)`` with each run detector's report from the
    link marker the commit wrote. Raises LinkRefreshError when a detector
    could not run (nothing is written), and lets a conflict propagate for
    the caller to relink from the new head (is_conflict). ``read_references``
    are read dependencies of the commit (link_request)."""
    import vc_object_store as vos
    runs = planned.runs
    if not runs:
        return None, planned.outcomes
    if log is not None:
        for outcome in runs:
            log(f'link refresh {outcome.detector}: running ({outcome.mode})')
    started = time.monotonic()
    request = link_request(store.status(), planned.plan, author, read_references)
    if progress is not None and is_served(store):
        # The detectors run on vc_object_server, which reports no progress
        # over the wire: one stage shows the request is running.
        progress.begin('linking', 'link refresh on the store\'s server',
                       detail='no progress is reported until it answers')
    try:
        receipt = store.submit(request, progress=_plan_progress(progress))
    except vos.PreparationFailed as error:
        raise LinkRefreshError([f'link refresh failed; nothing was written: {error}']) from error
    except ValueError as error:
        raise LinkRefreshError([f'link refresh refused; nothing was written: {error}']) from error
    finally:
        if progress is not None:
            progress.clear()
    # The marker as this commit wrote it (the receipt versions it): another
    # link run may have moved the current marker on already.
    marker = _link_marker_at(store, request, (receipt.get('objects') or {}).get(vos.link_marker_uuid()))
    reports = [step['report'] for step in (marker or {}).get('detectors') or ()]
    committed = receipt['commit_sequence']
    if marker is None:
        raise LinkRefreshError([
            f'the link marker of the links committed at commit {committed} cannot be read: '
            'the receipt contains no marker version'])
    if marker.get('commit_sequence') != committed:
        raise LinkRefreshError([
            f'the link marker referenced by the receipt records commit {marker.get("commit_sequence")}, '
            f'not the links committed at commit {committed}'])
    if len(reports) != len(runs):
        raise LinkRefreshError([
            f'the link marker of commit {committed} reports {len(reports)} detectors; '
            f'{len(runs)} ran'])
    for outcome, report in zip(runs, reports):
        outcome.report = report
    seconds = time.monotonic() - started
    runs[-1].mode = f'{runs[-1].mode}, {seconds:.1f} s in all'
    return receipt, planned.outcomes


def refresh_report(outcomes):
    """Report lines of a refresh."""
    return [line for outcome in outcomes for line in outcome.lines()]


# ---------------------------------------------------------------------------
# Link marker: skip the detectors when the links are current
# ---------------------------------------------------------------------------

LINK_MARKER_VERSION = 2
# Edge kinds the detectors write (link_plan runs them with their defaults):
# any change to one since a refresh (a manual link, an approval, a deletion)
# forces the next one to run in full.
DETECTOR_EDGE_KINDS = frozenset({'point-surface', 'same-surface', 'crossing'})
# Records that change how detectors map stored coordinates into the volume.
FRAME_RECORD_KINDS = frozenset({'volume_transform', 'volume_transform_choice'})
# Changes between the marker and the head that a plan examines one by one;
# beyond this many, the plan refreshes in full rather than classify them.
MAX_CLASSIFIED_CHANGES = 20000


def read_link_marker(store):
    """The store's link marker (the metadata of the object every
    link_objects commit publishes: the plan's marker content with the
    links' commit_sequence and each detector's report), or None."""
    from vc_object_store import link_marker_uuid
    record = current_record(store, link_marker_uuid())
    return None if record is None else dict(record.get('meta') or {})


def _link_marker_at(store, request, version):
    """The link marker's metadata at a numbered ``version`` of the store
    ``request`` (a checked request) names, or None when the version is
    None or absent."""
    from vc_object_store import link_marker_uuid
    if version is None:
        return None
    [revision] = store.objects_at_versions([{
        'store': request['store'], 'generation': request['generation'],
        'object': link_marker_uuid(), 'version': int(version)}])
    if revision is None or revision.get('deleted'):
        return None
    return dict(revision['record'].get('meta') or {})


def link_marker_key_parts(volume_uuid, config, normal_source):
    """What links depend on outside the store: the fit volume, the
    detector parameters and normal source (link_refresh_provenance), the
    store library's detector versions and which detectors run. A marker
    records them with their digest (``key``), so a mismatch reports what
    changed."""
    from vc_object_store import detector_versions
    return {
        'volume_uuid': volume_uuid,
        'link_refresh': link_refresh_provenance(config, normal_source),
        'detectors': detector_versions(),
        # Off leaves the overlap links unrefreshed: turning it on relinks.
        'patch_overlaps': bool(config['input_use_patch_overlaps']),
    }


def _flatten(value, prefix=''):
    if isinstance(value, dict) and value:
        return {path: leaf for key, item in value.items()
                for path, leaf in _flatten(item, f'{prefix}{key}.').items()}
    return {prefix[:-1]: value}


_UNRECORDED = object()


def key_changes(recorded, current, limit=3):
    """What differs between two link_marker_key_parts, as ``path: old -> new``
    (at most ``limit``, then a count)."""
    old, new = _flatten(recorded), _flatten(current)

    def shown(value):
        return 'not recorded' if value is _UNRECORDED else repr(value)
    changes = [f'{path}: {shown(old.get(path, _UNRECORDED))} -> {shown(new.get(path, _UNRECORDED))}'
               for path in sorted(set(old) | set(new))
               if old.get(path, _UNRECORDED) != new.get(path, _UNRECORDED)]
    if len(changes) > limit:
        changes = changes[:limit] + [f'{len(changes) - limit} more']
    return changes


def _uuids_sha256(uuids):
    digest = hashlib.sha256()
    for uuid in uuids:
        digest.update(uuid.encode())
        digest.update(b'\n')
    return digest.hexdigest()


def link_marker_selection(snapshot):
    """The snapshot's selection as a marker records it: per kind the count
    and digest of the sorted UUIDs, and the UUIDs themselves except for
    tracks (which only the digest and selector describe)."""
    selection = {}
    for kind in KINDS:
        uuids = sorted(obj.uuid for obj in snapshot.objects_of(kind))
        entry = {'count': len(uuids), 'sha256': _uuids_sha256(uuids)}
        if kind == 'track':
            entry['selector'] = snapshot.request.selector(kind).provenance()
        else:
            entry['uuids'] = uuids
        selection[kind] = entry
    return selection


@dataclasses.dataclass(frozen=True)
class StoreChange:
    """One record changed by a commit after a link marker: the record kind
    ('object' or 'relationship'), its UUID, the object kind or edge kind
    (None when it cannot be read: a deleted record) and the commit."""
    record_kind: str
    uuid: str
    kind: str
    commit: int
    deleted: bool = False
    version: int = 0


def changes_after(store, position, head):
    """The records changed by the commits after ``position`` (an
    InputPosition of this store) up to commit ``head``, from the store's
    change feed; None when the feed no longer reaches the position
    (another generation, or before its start), and a whole-store refresh
    is the only safe plan. Each change records its kind
    (StoreChange), read in batches."""
    cursor = {'store': position.store, 'generation': position.generation,
              'sequence': position.commit_sequence}
    raw = []
    while True:
        page = store.changes_since(cursor)
        if not page['cursor_valid']:
            return None
        for commit in page['commits']:
            if commit['sequence'] > head:
                break
            raw.extend(StoreChange(change['kind'], change['uuid'], None, commit['sequence'],
                                   bool(change.get('deleted')), int(change.get('version') or 0))
                       for change in commit['changes'])
        if not page['more'] or page['next']['sequence'] >= head:
            break
        cursor = page['next']
        if len(raw) > MAX_CLASSIFIED_CHANGES:
            break
    if len(raw) > MAX_CLASSIFIED_CHANGES:
        return raw  # unclassified: the plan refreshes in full
    kinds = {}
    objects = [change for change in raw if change.record_kind == 'object' and not change.deleted]
    for start in range(0, len(objects), 1000):
        batch = objects[start:start + 1000]
        refs = [{'store': position.store, 'generation': position.generation,
                 'object': change.uuid, 'version': change.version}
                for change in batch]
        for change, revision in zip(batch, store.objects_at_versions(refs)):
            if revision is not None:
                kinds[change.uuid] = revision['record']['kind']
    for change in raw:
        if change.record_kind == 'relationship' and not change.deleted:
            edge = store.edge(change.uuid)
            if edge is not None:
                kinds[change.uuid] = edge['kind']
    return [dataclasses.replace(change, kind=kinds.get(change.uuid)) for change in raw]


@dataclasses.dataclass(frozen=True)
class LinkPlan:
    """How a refresh runs: 'skip' (the links are current), 'incremental'
    (link ``new``, {kind: UUIDs}) or 'full', and why."""
    mode: str
    reason: str
    new: dict = None

    def line(self):
        return f'link refresh: {self.mode}: {self.reason}'


def plan_link_refresh(marker, position, head, changes, key_parts, selection, selected, *,
                      incremental):
    """Decide how to refresh (LinkPlan) from the store's link marker.

    Skip when the marker is this store's (``position``: the store and
    generation the refresh reads), was written by a full refresh (any
    refresh, for an input update: ``incremental``) with the same key
    (``key_parts``, link_marker_key_parts), the current ``selection`` is
    the marker's or a subset of it (tracks: the same), and every record
    changed since its commit (``changes(after)``: StoreChange list up to
    ``head``, or None when unknown) is an object outside ``selected`` (the
    current selection's UUIDs) and no detector relationship or volume
    registration. With ``incremental``, objects selected since the marker
    (created or newly selected) are linked incrementally, provided no other
    selected object changed. Anything else refreshes in full.
    """
    def full(reason):
        return LinkPlan('full', reason)
    if marker is None:
        return full('the store has no link marker')
    if marker.get('version') != LINK_MARKER_VERSION:
        return full('the link marker has another format')
    if marker.get('store') != position.store or marker.get('generation') != position.generation:
        return full('the link marker describes another store or history generation')
    if marker.get('key') != content_sha256(key_parts):
        recorded = marker.get('key_parts')
        if recorded is None:
            return full('the link inputs changed (the marker records no details)')
        return full('the link inputs changed: ' + '; '.join(key_changes(recorded, key_parts)))
    if not marker.get('exact') and not incremental:
        return full('the last refresh was incremental')
    linked = int(marker['commit_sequence'])
    if linked > head:
        return full(f'the link marker is at commit {linked}, after the head {head}')
    recorded = marker['selection']
    if selection['track']['sha256'] != recorded['track']['sha256']:
        return full('the track selection changed')
    new = {}
    for kind in KINDS:
        if kind == 'track':
            continue
        added = sorted(set(selection[kind]['uuids']) - set(recorded[kind]['uuids']))
        if added and not incremental:
            return full(f'{len(added)} {KIND_LABELS[kind]} selected since commit {linked}')
        if added:
            new[kind] = added
    fresh = {uuid for uuids in new.values() for uuid in uuids}
    changed = changes(linked)
    if changed is None:
        return full(f'the changes since commit {linked} cannot be read from the store')
    if len(changed) > MAX_CLASSIFIED_CHANGES:
        return full(f'more than {MAX_CLASSIFIED_CHANGES} records changed since commit {linked}')
    for change in changed:
        if change.record_kind == 'relationship':
            if change.kind is None or change.kind in DETECTOR_EDGE_KINDS:
                return full(f'a {change.kind or "deleted"} relationship changed at commit {change.commit}')
            continue
        if change.kind in FRAME_RECORD_KINDS:
            return full(f'a volume registration changed at commit {change.commit}')
        if change.uuid in selected and change.uuid not in fresh:
            return full(f'selected {change.kind or "object"} {change.uuid} changed at commit '
                        f'{change.commit}')
    if fresh:
        counts = ', '.join(f'{len(uuids)} {KIND_LABELS[kind]}' for kind, uuids in new.items())
        return LinkPlan('incremental', f'new since commit {linked}: {counts}', new)
    return LinkPlan('skip', f'unchanged since commit {linked}')


@dataclasses.dataclass
class LinkResult:
    """A refresh's report ``lines``, its LinkPlan, the receipt of the commit
    that published its links (None after a skip) and the pins
    (InputPins) of the selection after it: the objects at the versions the
    links' commit left them (a relationship change versions both endpoints),
    at the links' commit."""
    lines: list
    plan: LinkPlan
    receipt: dict
    pins: InputPins


def _refresh(store, snapshot, config, lasagna, cache_directory, log, *, head, incremental=False,
             progress=None, author=None):
    """Refresh the links of a snapshot's selection (LinkResult) as one
    link_objects request, or skip it when the store's link marker shows the
    links current (plan_link_refresh; ``incremental`` for an input update);
    object_store_full_link_refresh forces a full refresh. ``head`` is the
    store's commit the snapshot was read at. A conflict (a commit touching
    a read object or the marker between the detectors' snapshot and their
    commit) propagates for the caller to relink. ``cache_directory`` is the
    fit's cache (strip_cache_directory)."""
    normal = fiber_normal_source(config, lasagna, snapshot.volume_uuid)
    log(normal.describe())
    if normal.note:
        log(normal.note)
    if normal.kind != 'none' and is_served(store):
        # The detectors run on the server, which reads the source itself (a
        # pack directory or CT path readable there); an unavailable one
        # fails the refresh, and nothing is written.
        log('fiber-patch normals are read by the store\'s server (the source must be '
            'readable there; no fallback)')
    selection = link_marker_selection(snapshot)
    key_parts = link_marker_key_parts(snapshot.volume_uuid, config, normal)
    position = snapshot.position
    if config.get('object_store_full_link_refresh'):
        plan = LinkPlan('full', 'object_store_full_link_refresh is set')
    else:
        plan = plan_link_refresh(
            read_link_marker(store), position, head,
            lambda after: changes_after(store, dataclasses.replace(position, commit_sequence=after), head),
            key_parts, selection, snapshot.objects, incremental=incremental)
    log(plan.line())  # as decided, before any detector runs
    if plan.mode == 'skip':
        return LinkResult([f'link refresh: skipped every detector ({plan.reason})'], plan, None,
                          snapshot.pins())
    if (normal.kind == 'none' and snapshot.objects_of('fiber')
            and snapshot.objects_of('surface')):
        log('WARNING: fiber-patch links are refreshed centerline-only: the fit '
            'loads no lasagna normals (input_use_normals with a Lasagna dataset '
            'whose normals pack is in the cache) and object_store_normal_ct_path '
            'is unset')
    marker = {'version': LINK_MARKER_VERSION, 'store': position.store,
              'generation': position.generation, 'key': content_sha256(key_parts),
              'key_parts': key_parts, 'exact': plan.mode == 'full', 'selection': selection}
    planned = link_plan(snapshot, config, normal, strip_cache_directory(store, cache_directory),
                        new=plan.new if plan.mode == 'incremental' else None, marker=marker)
    started = time.monotonic()
    pins = snapshot.pins()
    receipt, outcomes = submit_link_plan(store, planned, author=author, progress=progress, log=log,
                                         read_references=pins.refs())
    lines = refresh_report(outcomes) + [
        f'link refresh: {plan.mode} refresh ({plan.reason}) took {time.monotonic() - started:.1f} s']
    if receipt is None:
        return LinkResult(lines, plan, None, pins)
    # The links' commit versions every endpoint it touched: the pins are
    # those versions, at the links' commit.
    references = dict(pins.references)
    from vc_object_store import receipt_object_reference
    for uuid, version in (receipt.get('objects') or {}).items():
        if uuid in references and version is not None:
            references[uuid] = receipt_object_reference(receipt, uuid)
    lines.append(f'links committed at commit {receipt["commit_sequence"]} (linked at {head})')
    return LinkResult(lines, plan, receipt, dataclasses.replace(
        pins, position=dataclasses.replace(position, commit_sequence=int(receipt['commit_sequence'])),
        references=references))


@contextlib.contextmanager
def reading_stage(progress, stage, detail=None):
    """A store read as a progress stage of the input preparation:
    ``progress`` (a spiral_progress.ProgressReporter, or None) shows
    ``stage`` while the block runs. The reads report no steps; the stage's
    elapsed time shows they are still running."""
    if progress is None:
        yield
        return
    progress.begin('preparing_inputs', stage, detail=detail)
    try:
        yield
    finally:
        progress.clear()


def resolve_pinned(store, request, pins, progress=None):
    """resolve_selection at pins, as a progress stage."""
    with reading_stage(progress, f'Reading the inputs at {pins.position.describe()}', 'selection'):
        return resolve_selection(store, dataclasses.replace(request, pins=pins))


@contextlib.contextmanager
def _console_progress():
    """Detector progress on stderr, for a refresh outside the service."""
    from spiral_progress import ProgressReporter
    reporter = ProgressReporter(stream=sys.stderr)
    try:
        yield reporter
    finally:
        reporter.close()


def _startup_phase(config, request, lasagna, cache_directory, log, minimum_position=None,
                   pins=None):
    """Rank zero's startup: link and pin a position of the store
    (prepare_position) as object_store_author, unless
    object_store_input_position (or ``pins``, the versions a resume or an
    update provides) specifies the position to read exactly. Returns
    ``(SelectedObjects, report lines, InputPins)``; raises
    ObjectStoreInputError on any failure."""
    author = config['object_store_author']
    pinned = pins if pins is not None else InputPins.from_provenance(config['object_store_input_position'])
    if pinned is not None:
        # Prepared already (the service linked and pinned it, or a checkpoint
        # recorded it): read it exactly.
        with open_store(config, read_only=True, cache_directory=cache_directory) as store:
            check_position(store, pinned.position)
            selected = resolve_selection(store, dataclasses.replace(request, pins=pinned))
        return selected, [f'reading pinned {pinned.position.describe()}'], selected.pins()
    refresh = bool(config['object_store_refresh_links'])
    if refresh and not author:
        raise ObjectStoreInputError([
            'object_store_author is required to refresh links '
            '(or set object_store_refresh_links false)'])
    with open_store(config, read_only=not refresh, cache_directory=cache_directory) as store, \
            _console_progress() as progress:
        prepared = prepare_position(store, config, request, lasagna=lasagna,
                                    minimum=minimum_position, cache_directory=cache_directory,
                                    log=log, progress=progress)
    return prepared.selected, prepared.lines, prepared.pins


class PositionError(ObjectStoreInputError):
    """No position could be linked and pinned; the active fit is kept."""


@dataclasses.dataclass
class PreparedPosition:
    """A linked, pinned position: ``pins`` (at N, the links' commit) hold
    the selected objects at the versions the links left them; ``head`` (H)
    is the commit the links were made at; ``minimum`` (M) the position the
    caller asked to include."""
    pins: InputPins
    head: int
    minimum: InputPosition
    attempts: int
    selected: SelectedObjects
    lines: list

    @property
    def position(self):
        return self.pins.position


def fit_volume_at(store, request, prepared, progress=None):
    """The fit volume of a prepared position: the volume ``read_snapshot``
    infers (or checks a configured one against) for the selection pinned
    there, so it is the one the fit will read."""
    pinned = dataclasses.replace(request, pins=prepared.pins)
    with reading_stage(progress, f'Reading the inputs at {prepared.position.describe()}',
                       'fit volume'):
        return read_snapshot(store, pinned, prepared.selected).volume_uuid


def prepare_position(store, config, request, *, lasagna=None, minimum=None,
                     cache_directory=None, log=print, attempts=4, incremental=False,
                     progress=None):
    """Link, then pin: read the store's head H (at least ``minimum``, the
    position a client's saved edits reached), refresh the links over the
    whole selection as one link_objects request, and pin N, the links'
    commit, with the selected objects at the versions it left them (every
    endpoint of a changed relationship is versioned), which later commits
    never change. The request's detectors read under one snapshot and its
    commit expects every object they read: a commit touching one of them
    (or the link marker) in between conflicts, and the loop reads the new
    head and relinks; ``attempts`` bounds it. A detector failure or the
    bound raises, writing nothing, and leaves the caller's active inputs
    alone. With object_store_refresh_links off, the head is the position.

    The store's link marker lets the refresh skip the detectors when the
    links at the head are current, and, with ``incremental`` (an input
    update), link only objects selected since it (_refresh). ``progress``
    (a spiral_progress.ProgressReporter) receives the detectors' progress.
    ``cache_directory`` is the fit's cache, where a local refresh keeps its
    fiber strips (strip_cache_directory). Works alike on a store file and a
    served store (open_store): the detectors run where the store is (in
    this process, or on vc_object_server, which reports no progress).
    """
    refresh = bool(config['object_store_refresh_links'])
    author = config['object_store_author']
    current = dataclasses.replace(request, pins=None)
    notes = []
    if minimum is not None:
        check_position(store, minimum)
    for attempt in range(1, attempts + 1):
        with reading_stage(progress, 'Reading the selected inputs', 'head'):
            selected = resolve_selection(store, current)
            head = selected.position.commit_sequence
        if minimum is not None and head < minimum.commit_sequence:
            raise PositionError([
                f'the store is at commit {head}, before the requested {minimum.describe()}; '
                'is this the store the edits were saved to?'])
        if not refresh:
            pins, lines = selected.pins(), [f'link refresh: off; pinning {selected.position.describe()}']
            break
        with reading_stage(progress, 'Reading the selected inputs', f'commit {head}'):
            snapshot = read_snapshot(store, current, selected)
        try:
            result = _refresh(store, snapshot, config, lasagna, cache_directory, log, head=head,
                              incremental=incremental, progress=progress, author=author)
        except Exception as error:  # noqa: BLE001 - a conflict relinks; the rest reports
            if not is_conflict(error):
                raise
            notes.append(f'attempt {attempt}: the store changed while linking at commit {head} '
                         f'({error}); relinking')
            continue
        pins, lines = result.pins, result.lines
        break
    else:
        raise PositionError(notes + [
            f'inputs kept changing: no position could be linked in {attempts} attempts; retry'])
    if minimum is not None and pins.position.commit_sequence > minimum.commit_sequence:
        lines.append(f'{pins.position.describe()} includes commits after the requested '
                     f'{minimum.describe()}')
    return PreparedPosition(
        pins=pins, head=head, minimum=minimum, attempts=attempt,
        selected=resolve_pinned(store, request, pins, progress),
        lines=notes + lines)


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
    'pcl_vertical_fiber_min_auto_certainty', 'fit_volume_level', 'input_use_patch_overlaps',
)


def input_manifest(snapshot, config, normal_source, external_frames=None):
    """The snapshot manifest extended with materialization settings,
    external-input frames and the link-refresh mode."""
    manifest = snapshot.manifest()
    manifest['materialization'] = {key: config[key] for key in MATERIALIZATION_KEYS}
    manifest['external_frames'] = dict(sorted((external_frames or {}).items()))
    manifest['link_refresh'] = link_refresh_provenance(config, normal_source)
    return manifest


# Detector parameters a link refresh runs with (refresh_links), recorded so a
# later change shows the fit's links as stale.
LINK_PARAMETER_KEYS = ('pcl_link_distance_tolerance', 'pcl_link_window_points',
                       'pcl_link_window_min_points')


def link_refresh_provenance(config, normal_source):
    """How links are refreshed: whether, with which detector parameters and
    which fiber-patch normal source (its identity: kind, volume paths, level
    and scale)."""
    return {
        'enabled': bool(config['object_store_refresh_links']),
        'parameters': {key: config[key] for key in LINK_PARAMETER_KEYS},
        'fiber_patch_normals': normal_source.provenance(),
    }


def stale_link_reasons(manifest, config, normal_source):
    """Why the links a manifest's inputs consumed would not be the ones a
    refresh makes now: changed detector parameters or normal source. Links
    are results of the refresh, so these never change the digest; the next
    refresh (an input update) relinks."""
    recorded = (manifest or {}).get('link_refresh') or {}
    current = link_refresh_provenance(config, normal_source)
    reasons = []
    if not recorded.get('enabled'):
        return reasons
    if 'parameters' not in recorded:
        reasons.append('the fit predates recorded link parameters')
    else:
        reasons.extend(f'{key} changed from {recorded["parameters"].get(key)!r} to {value!r}'
                       for key, value in current['parameters'].items()
                       if recorded['parameters'].get(key) != value)
    if recorded.get('fiber_patch_normals') != current['fiber_patch_normals']:
        reasons.append('the fiber-patch normal source changed')
    return reasons


def _compared(manifest):
    """The manifest without what does not define the inputs: the store's
    location and history generation, the version pins (VERSION_FIELDS), how
    links were refreshed (their results are recorded) and the position they
    were read at (equal inputs at two positions are equal). A copy of a
    store holding the same inputs compares equal."""
    manifest = dict(manifest)
    manifest['store'] = {'uuid': (manifest.get('store') or {}).get('uuid')}
    manifest.pop('link_refresh', None)
    manifest.pop('position', None)
    if manifest.get('outer_shell'):
        manifest['outer_shell'] = {key: value for key, value in manifest['outer_shell'].items()
                                   if key != 'reference'}
    # Schema-20 checkpoints also recorded a relationship generation. Its
    # content hash already identifies the consumed data; schema 21 has no
    # equivalent counter. Do not discard it from older, hash-less manifests:
    # those cannot establish unchanged relationships across the conversion.
    for section, fields in VERSION_FIELDS.items():
        if section in manifest:
            manifest[section] = [
                {key: value for key, value in entry.items()
                 if key not in fields and not (
                     section == 'relationships' and key == 'generation'
                     and 'content_sha256' in entry)}
                for entry in manifest[section] or ()]
    return manifest


# Identities a manifest recorded before objects carried metadata hashes and
# relationships their content: geometry revisions, relationship generations
# and approval states only.
_GEOMETRY_IDENTITY = {
    'objects': ('uuid', 'kind', 'name', 'geometry_revision', 'scale_factor'),
    'relationships': ('uuid', 'kind', 'family', 'approval_state', 'generation'),
}


def records_content_identity(manifest):
    """Whether a manifest identifies objects and relationships by content
    (metadata hashes, relationship content), as every manifest written since
    replica-independent digests does."""
    objects = (manifest or {}).get('objects') or ()
    return all('metadata_sha256' in entry for entry in objects)


def _geometry_identity(manifest):
    manifest = dict(manifest)
    for section, fields in _GEOMETRY_IDENTITY.items():
        if section in manifest:
            manifest[section] = [{key: entry.get(key) for key in fields}
                                 for entry in manifest[section] or ()]
    return manifest


def track_identity(manifest):
    """The tracks and track crossings a manifest records, by content
    (VERSION_FIELDS left out): a fit consumes them at build, so two
    manifests with different ones need a rebuild."""
    compared = _compared(manifest)
    return {
        'objects': [entry for entry in compared.get('objects') or ()
                    if entry.get('kind') == 'track'],
        'relationships': [entry for entry in compared.get('relationships') or ()
                          if entry.get('family') == 'track_crossing'],
    }


def input_summary(inputs):
    """A small description of materialized inputs for a status poll: their
    identity (digest, store, fit volume) and counts. The manifest holds the
    per-object detail."""
    manifest = inputs.manifest
    return {
        'digest': inputs.digest,
        'store_uuid': manifest['store']['uuid'],
        'volume_uuid': manifest['volume_uuid'],
        'selection': {kind: entry['count']
                      for kind, entry in manifest['selection'].items()},
        'relationships': dict(sorted(Counter(
            rel['family'] for rel in manifest['relationships']).items())),
        'excluded_relationships': manifest['excluded_relationships'],
        'constraints': dict(sorted(inputs.counts.items())),
        'link_refresh': manifest['link_refresh']['enabled'],
        # The fiber-patch normal source the links were refreshed with
        # (NormalSource.provenance: its kind and where it was read from).
        'fiber_patch_normals': manifest['link_refresh'].get('fiber_patch_normals'),
        'outer_shell': (manifest.get('outer_shell') or {}).get('uuid'),
    }


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
        current = inputs.manifest
        legacy = not records_content_identity(recorded)
        if legacy:
            # Written before inputs were identified by content: its object
            # revisions name another copy's (or this store's older) records,
            # and it holds no metadata hashes, so compare what it does hold.
            recorded, current = _geometry_identity(recorded), _geometry_identity(current)
        differences = manifest_differences(recorded, current)
        if not differences:
            if legacy:
                log('WARNING: the checkpoint predates content-identified object-store '
                    'inputs (metadata hashes, relationship content); compared on '
                    'geometry revisions and relationship generations only')
            return []
        problem = ('checkpoint object-store inputs differ from this fit\'s in '
                   + ', '.join(differences))
    if allow_changed:
        log(f'WARNING: {problem}; resuming anyway (object_store_allow_changed_inputs)')
        return []
    return [f'{problem}; set object_store_allow_changed_inputs to resume anyway']


def load_fit_inputs(config, *, lasagna=None, external_frames=None, dist=None,
                    cache_directory=None, log=print, minimum_position=None, pins=None):
    """Refresh, freeze and materialize a fit's object-store inputs.

    Rank zero (the only process when ``dist`` is None or single-rank) runs
    the one refresh phase: it links and pins a position of the store,
    including at least ``minimum_position`` (prepare_position), unless
    object_store_input_position (or ``pins``: the versions a resume's
    checkpoint or an input update provides) specifies the position to read. The
    other ranks wait for its result, and every rank fails when it fails.
    Every rank then reads the pinned versions read-only, materializes the
    same frozen snapshot and closes the store; their manifests must agree
    before anyone trains. Returns FitInputs carrying ``manifest`` and
    ``digest``. ``cache_directory`` is the fit's cache (the strip cache of
    a local link refresh and a served store's persistent cache live under
    it).
    """
    if not config['object_store_location']:
        raise ObjectStoreInputError(['object_store_location is not configured'])
    main = dist is None or dist.is_main_process
    status = None
    if main:
        try:
            selected, lines, pinned = _startup_phase(
                config, request_from_config(config), lasagna, cache_directory, log,
                minimum_position, pins)
            status = {'ok': True, 'pins': pinned.provenance(), 'lines': lines}
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

    # Every rank reads and materializes on its own; a failure on any rank is
    # gathered with the others' results, so all ranks fail together instead
    # of some entering a collective the failed rank never joins.
    inputs, outcome = None, None
    request = dataclasses.replace(
        request_from_config(config), pins=InputPins.from_provenance(status['pins']))
    try:
        with open_store(config, read_only=True, cache_directory=cache_directory) as store:
            snapshot = read_snapshot(store, request)
            normal = fiber_normal_source(config, lasagna, snapshot.volume_uuid)
            inputs = materialize_from_config(store, snapshot, config)
        inputs.manifest = input_manifest(snapshot, config, normal, external_frames)
        inputs.digest = manifest_digest(inputs.manifest)
        outcome = {'digest': inputs.digest}
    except ObjectStoreInputError as error:
        outcome = {'problems': error.problems}
    except Exception as error:  # noqa: BLE001 - other ranks must hear of it
        outcome = {'problems': [f'{type(error).__name__}: {error}']}
    outcomes = _all_gather(outcome, dist)
    failed = [(rank, result['problems']) for rank, result in enumerate(outcomes)
              if 'problems' in result]
    if failed:
        raise ObjectStoreInputError([
            (f'rank {rank}: {problem}' if len(outcomes) > 1 else problem)
            for rank, problems in failed for problem in problems])
    digests = [result['digest'] for result in outcomes]
    if len(set(digests)) > 1:
        raise ObjectStoreInputError([
            'ranks read different object-store inputs: '
            + ', '.join(f'rank {rank} {digest[:12]}' for rank, digest in enumerate(digests))])
    if main:
        log(f'object-store inputs frozen: digest {inputs.digest[:16]}')
    return inputs


def describe_selection(config):
    """snapshot_summary of the selection a fit configuration names, read
    from its store as it stands: read-only, without refreshing links or
    materializing anything, and without importing Torch. A selection the
    fitter would refuse reports its problems instead of raising."""
    try:
        request = request_from_config(config)
        with open_store(config, read_only=True) as store:
            return {'ok': True, **snapshot_summary(read_snapshot(store, request))}
    except ObjectStoreInputError as error:
        return {'ok': False, 'problems': error.problems}


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
    parser.add_argument('store', help='a store file, or the endpoint of the vc_object_server serving one')
    parser.add_argument('--volume', help='fit volume UUID (default: inferred)')
    parser.add_argument('--approved-only', action='store_true')
    for kind, flag in _SELECTOR_CONFIG.items():
        parser.add_argument(f'--{flag}-uuid', action='append', dest=f'{kind}_uuids')
        parser.add_argument(f'--{flag}-tag', action='append', dest=f'{kind}_tags')
    parser.add_argument('--z-begin', type=int,
                        help='with --z-end, also materialize the constraints of this z window')
    parser.add_argument('--z-end', type=int)
    parser.add_argument('--config', help='fit configuration JSON overriding the defaults')
    parser.add_argument('--json', help='write the manifest and report here')
    return parser.parse_args(argv)


def main(argv=None):
    from vc_object_store.locations import open_store as open_location

    args = _parse_args(argv)
    selectors = {}
    for kind in KINDS:
        uuids, tags = getattr(args, f'{kind}_uuids'), getattr(args, f'{kind}_tags')
        selectors[kind] = Selector(
            tuple(uuids) if uuids is not None else None,
            tuple(tags) if tags is not None else None)
    request = InputRequest(selectors, args.volume, args.approved_only)
    with open_location(args.store, read_only=True) as store:
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
