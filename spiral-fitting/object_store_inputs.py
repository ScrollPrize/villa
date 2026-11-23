"""Fit inputs read from a vc_object_store database.

Patches, tracks and the outer shell come from one object store, opened
read-only: the fit writes nothing to it and reads no relationship between
objects. This module turns a selection of them into the fitter's runtime
inputs in three stages:

1. Snapshot extraction (read_snapshot): resolve object selectors to explicit
   UUID sets and read every selected object and its geometry revision in one
   read view.
2. Frame conversion (ObjectFrame): every selected object declares its frame in
   ``meta.coordinates``; geometry and anchors convert once from stored XYZ to
   ZYX voxels of one reference volume, the only units the fitter uses.
3. Runtime materialization: patches, the outer shell and the track
   collection with its crossings, built from the frozen snapshot
   (materialize_* below). The fit computes track crossings from the
   selected tracks' stored points (materialize_tracks) and ignores any
   crossing relationship the store holds.

    python object_store_inputs.py STORE [--volume V] \
        [--z-begin Z0 --z-end Z1 [--config FIT.json]]

prints the selection's report without training. Given a z window it also
materializes the inputs exactly as a fit with that configuration would,
failing on the same validation errors.

STORE, like the fit configuration's object_store_location, is a store file
or the endpoint of the vc_object_server serving one (open_store): every
read and the pins work alike on both.
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
from vc_object_store.selected import SelectedObject, SelectedRecord

KINDS = ('surface', 'track')
# The tag of the scroll's outer shell: a surface the fit reads on its own
# (InputSnapshot.outer_shell, FitInputs.outer_shell), never as a patch.
OUTER_SHELL_TAG = 'spiral-outer-shell'
# The tag of a committed Spiral fit (service_commits): a fit result, never a
# fit input.
SPIRAL_FIT_TAG = 'spiral-fit'
KIND_LABELS = {'surface': 'surfaces', 'track': 'tracks'}

# The coordinates the fitter works in, recorded with every provenance.
FIT_UNITS = 'reference-volume voxels (ZYX)'

# Manifest fields that pin records without defining the inputs: an object's
# numbered version moves with every change to it, those the adapter does not
# read included (a review note, a relationship of a selected object), so
# versions resolve the inputs exactly but are left out of the digest.
# Geometry revisions, record UUIDs and consumed content define them.
VERSION_FIELDS = {'objects': ('version', 'reference')}


class ObjectStoreInputError(ValueError):
    """A selection or frame the fitter cannot consume."""

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
        from vc_object_store import store_position_from_json
        return cls.of_status(store_position_from_json(json.dumps(value)))


@dataclasses.dataclass(frozen=True)
class InputPins:
    """What a fit reads exactly: the selected objects (and the outer shell)
    at the numbered versions they had at the pinned position, and that
    position. Reads at pins use
    the store's numbered history. Inherited endpoints do not pin overlay
    membership; such selections cannot replay after an overlay commit.

    ``references`` may be empty (a position alone, e.g. a configuration's
    object_store_input_position): the versions are then those at the
    store's head when the head is that position, else read from the
    store's history at the position's publication (local stores).
    ``references`` retains each selected version's actual source owner, which
    may differ from the attached store. Unqualified numeric pins must be
    regenerated; the attached store cannot recover their original owner."""
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
    def _decode_reference(uuid, reference):
        """Validate a complete selected reference at the provenance boundary."""
        from vc_object_store import object_version_from_json
        if reference is None:
            raise ValueError(f'input {uuid} has no complete reference; regenerate the input pins')
        reference = object_version_from_json(json.dumps(reference))
        if reference['object'] != uuid:
            raise ValueError(f'inconsistent exact input reference for {uuid}')
        return reference

    @classmethod
    def from_provenance(cls, value):
        if value is None:
            return None
        position = InputPosition.from_provenance(value)
        # This application-owned saved schema has maps of native references and
        # per-kind UUID arrays. Do not normalize malformed containers to empties.
        if not isinstance(value.get('references', {}), dict) or not isinstance(value.get('selection', {}), dict):
            raise ValueError('input pin references and selection must be maps')
        if any(not isinstance(uuids, list) for uuids in value.get('selection', {}).values()):
            raise ValueError('input pin selection values must be arrays')
        if 'objects' in value:
            raise ValueError('numeric input pins are unsupported; regenerate the input pins')
        if 'references' not in value:
            if value.get('selection') or value.get('outer_shell'):
                raise ValueError('selected inputs have no complete references; regenerate the input pins')
            return cls(position)
        references = value['references']
        references = {uuid: cls._decode_reference(uuid, reference)
                      for uuid, reference in references.items()}
        selected = {uuid for values in value.get('selection', {}).values() for uuid in values}
        if value.get('outer_shell'):
            selected.add(value['outer_shell'])
        if not selected.issubset(references):
            raise ValueError('selected inputs have no complete references; regenerate the input pins')
        return cls(position, references,
                   {kind: sorted(value.get('selection', {}).get(kind, ())) for kind in KINDS},
                   value.get('outer_shell'))

    @classmethod
    def from_manifest(cls, manifest):
        """The pins an input manifest (InputSnapshot.manifest) records, or
        None when it contains no position."""
        references = validated_manifest_references(manifest or {})
        position = InputPosition.from_provenance((manifest or {}).get('position'))
        if position is None:
            return None
        shell = manifest.get('outer_shell') or None
        entries = manifest.get('objects') or ()
        return cls(position, references,
                   {kind: sorted(entry['uuid'] for entry in entries if entry['kind'] == kind) for kind in KINDS},
                   shell['uuid'] if shell else None)

    def refs(self, uuids=None):
        """Numbered references of ``uuids`` (default: every pinned object)."""
        return [self.references[uuid] for uuid in (self.references if uuids is None else uuids)]


@dataclasses.dataclass(frozen=True)
class InputRequest:
    """What to read: per-kind selectors and the fit volume.

    ``fit_volume_uuid`` None infers the volume from the selected objects'
    declarations. ``pins`` None reads the store as it is
    now; InputPins read the pinned object versions (selector membership
    included), whatever the store holds now.
    """
    selectors: dict = dataclasses.field(default_factory=dict)
    fit_volume_uuid: str = None
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


def adapter_metadata(kind, meta, track=None):
    """The metadata the adapter reads from one object of ``kind``.

    With the geometry revision (points, arrays and point annotations) it is
    everything materialization reads, so its hash identifies the object's
    input content in any copy of the store. Other metadata (review notes, ingest
    diagnostics) does not reach the fit and does not change the digest.

    This fit-protocol projection is mirrored by VC3D's ``adapterMetadata``;
    ``test_spiral_store_inputs`` checks parity for coordinates, source tuning
    and track family fields. These are Spiral semantics,
    not store identity/placement rules, so they do not belong in vcobj's schema.
    """
    meta = meta or {}
    source = meta.get('source') or {}
    read = {'coordinates': meta.get('coordinates')}
    if kind == 'surface':
        read['spiral_patch_erode_cells'] = source.get('spiral_patch_erode_cells')
    elif kind == 'track':
        # The family gates crossings; the dataset and key order the tracks.
        for key in ('family', 'dataset_ref', 'external_key'):
            read[key] = (track or {}).get(key)
    return read


@dataclasses.dataclass
class StoredObject(SelectedObject):
    frame: ObjectFrame = None
    # A track's dataset as its owning store resolves it (selected_inputs'
    # track_datasets): what its crossings are grouped by. Not an input of
    # its own: the track's reference and dataset_ref determine it.
    dataset: str = None

    @property
    def metadata_sha256(self):
        return content_sha256(adapter_metadata(self.kind, self.meta, self.track))

    def provenance(self):
        """``metadata`` is what the adapter read, so a client compares it
        with an object's current metadata as values; ``metadata_sha256`` is
        its identity in the digest."""
        return {'uuid': self.uuid, 'kind': self.kind, 'name': self.name,
                'version': self.version, 'reference': self.reference,
                'geometry_revision': self.geometry_revision,
                'metadata': adapter_metadata(self.kind, self.meta, self.track),
                'metadata_sha256': self.metadata_sha256,
                'scale_factor': self.frame.scale_factor}


@dataclasses.dataclass
class InputSnapshot:
    """The frozen selection: objects and their frames."""
    store_location: str
    store_uuid: str
    store_generation: str
    volume_uuid: str
    volume_source: str
    request: InputRequest
    objects: dict
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

    def manifest(self):
        """Complete input/frame provenance, in stable order."""
        return {
            'store': {'location': self.store_location, 'uuid': self.store_uuid,
                      'generation': self.store_generation},
            'position': self.position.provenance() if self.position else None,
            'volume_uuid': self.volume_uuid,
            'volume_source': self.volume_source,
            'units': FIT_UNITS,
            'selection': {
                kind: {'selector': self.request.selector(kind).provenance(),
                       'count': len(self.objects_of(kind))}
                for kind in KINDS},
            'objects': [obj.provenance() for obj in self.objects.values()],
            'outer_shell': self.outer_shell.provenance() if self.outer_shell else None,
        }

    def digest(self):
        """SHA-256 of the manifest without its version pins."""
        return manifest_digest(self.manifest())


def outer_shell(record):
    """Whether an object is the scroll's outer shell (OUTER_SHELL_TAG): read
    on its own, never as a patch."""
    return record.kind == 'surface' and OUTER_SHELL_TAG in record.tags


def spiral_fit(record):
    """Whether an object is a committed Spiral fit (SPIRAL_FIT_TAG). Taking
    one in would make a fit's own output its next input, and change the input
    digest the commit recorded."""
    return record.kind == 'surface' and SPIRAL_FIT_TAG in record.tags


def fit_input(record):
    """Whether a default or tag selection may take this object in."""
    return not outer_shell(record) and not spiral_fit(record)


def resolve_selector(store, kind, selector, records=None):
    """``{UUID: reference}`` of the objects ``selector`` selects among active
    objects, at the numbered versions read.

    Default and tag selections leave out the outer shell and committed
    Spiral fits; naming any of them by UUID is an error. ``records``
    ({UUID: record} of every pinned object, retired ones included) replaces
    the store's current objects."""
    def matching(tag=None):
        if records is not None:
            return {uuid: record.reference for uuid, record in records.items()
                    if record.kind == kind and not record.retired
                    and (tag is None or tag in record.tags)
                    and fit_input(record)}
        query = {} if tag is None else {'tag': tag}
        return {record.uuid: record.reference for record in map(SelectedRecord, store.select_objects(
            object_kind=kind, fields=['uuid', 'kind', 'meta', 'tags'], **query))
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
        elif record.kind != kind:
            problems.append(f'selected {kind} {uuid} is a {record.kind}')
        elif record.retired:
            problems.append(f'selected {kind} {uuid} is retired')
        elif outer_shell(record):
            problems.append(f'selected {kind} {uuid} is the outer shell (tagged '
                            f'{OUTER_SHELL_TAG}), not a patch')
        elif spiral_fit(record):
            problems.append(f'selected {kind} {uuid} is a committed Spiral fit (tagged '
                            f'{SPIRAL_FIT_TAG}), not a fit input')
        else:
            selected[record.uuid] = record.reference
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
    return SelectedObject(revision)


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
        records[revision['object_uuid']] = SelectedObject(revision)
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
        records[revision['object_uuid']] = SelectedObject(revision)
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
        status = store.status()
        selected = {kind: resolve_selector(store, kind, request.selector(kind)) for kind in KINDS}
        shells = []
        if request.outer_shell != 'off':
            for record in map(SelectedRecord, store.select_objects(object_kind='surface', tag=OUTER_SHELL_TAG, fields=['uuid'])):
                shells.append(record.reference)
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
    return [record.reference for record in records.values()
            if outer_shell(record) and not record.retired]


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
    earlier, so one startup reads exactly the objects it pinned. Every
    object is read at the version the selection resolved it at. One native
    read (selected_inputs) validates and returns every object, enumerates no
    relationship (``relationship_refs`` empty) and reports each track's
    dataset as the store resolves it.
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
    captured = store.selected_inputs(selected.pins().refs(), position.provenance(),
                                     relationship_refs=[])
    revisions = {revision['object_uuid']: revision for revision in captured['objects']}
    track_datasets = captured.get('track_datasets') or {}

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
        return StoredObject(revision, frame, track_datasets.get(record['uuid']))

    objects = {}
    for revision in usable:
        if (obj := stored(revision)) is not None:
            objects[obj.uuid] = obj
    shell_object = None if shell_revision is None else stored(shell_revision)
    if problems:
        raise ObjectStoreInputError(problems)

    pinned_request = request if request.pins is not None and request.pins.pinned \
        else dataclasses.replace(request, pins=selected.pins())
    return InputSnapshot(
        store_location=store_location(store), store_uuid=position.store,
        store_generation=position.generation, volume_uuid=volume_uuid,
        volume_source=volume_source, request=pinned_request, objects=objects,
        notes=notes, outer_shell=shell_object)


def store_location(store):
    """Where a store handle reads from: its file path, or its endpoint."""
    for attribute in ('path', 'endpoint'):
        location = getattr(store, attribute, None)
        if location:
            return str(location)
    return ''


def is_served(store):
    """Whether a store handle reads a store served by vc_object_server
    rather than a store file."""
    return bool(getattr(store, 'endpoint', None))


# Under the fit's cache directory: a served store's persistent cache of the
# numbered records and geometry it read (HostedSessionOptions::cacheDirectory).
HOSTED_CACHE_DIRECTORY = 'object_store_hosted'


@contextlib.contextmanager
def open_store(config, *, cache_directory=None):
    """The store ``object_store_location`` names, for the block
    (vc_object_store.locations.open_store): a store file opened read-only,
    or a served store attached with, under ``cache_directory`` (the fit's
    cache), a persistent cache of what it reads. Both give the same reading
    methods; the fit writes nothing through either."""
    from vc_object_store.locations import open_store as open_location
    location = config['object_store_location']
    if not location:
        raise ObjectStoreInputError(['object_store_location is not configured'])
    hosted_cache = os.path.join(cache_directory, HOSTED_CACHE_DIRECTORY) if cache_directory else None
    with open_location(location, read_only=True, cache_directory=hosted_cache) as store:
        yield store


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


# ---------------------------------------------------------------------------
# Runtime materialization
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class FitInputs:
    """Runtime inputs materialized from a frozen snapshot.

    ``patches`` maps patch UUIDs to fit-frame tifxyz.Patch objects in UUID
    order. ``tracks`` holds the track collection and its crossings
    (materialize_tracks), or None when no track is selected.
    """
    snapshot: InputSnapshot
    patches: dict
    counts: Counter
    # The outer shell as a fit-frame tifxyz.Patch (materialize_outer_shell),
    # or None.
    outer_shell: object = None
    tracks: 'TrackInputs' = None
    # Complete input/frame provenance and its digest (load_fit_inputs).
    manifest: dict = None
    digest: str = None

    def report(self):
        return snapshot_report(self.snapshot) + [
            'constraints: ' + ', '.join(
                f'{name} {count}' for name, count in sorted(self.counts.items()))]


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
                        erode_cells_default, exclude_uuids=()):
    """Fit-frame patches of the snapshot, keyed by UUID; and drop counts.

    Grid rows and columns are the stored ones: nothing is cropped or
    subsampled, so stored (row, col) anchors are runtime (i, j). Invalid
    vertices keep the -1 sentinel. A patch's own erosion request is
    ``meta.source.spiral_patch_erode_cells``, as in its tifxyz metadata.
    Each patch carries its loss sampling mask (``z_margin`` is the fit's
    patch_loss_z_margin) and its object name (``name``) for the name-keyed sampling weights. Patches in
    ``exclude_uuids`` are left out like patches outside the z window.
    """
    import torch
    from spiral_helpers import (patch_intersects_z_roi, patch_sampling_quad_mask,
                                prepare_patch_for_fit)
    from tifxyz import Patch

    patches, counts = {}, Counter()
    exclude_uuids = frozenset(exclude_uuids)
    for obj in snapshot.objects_of('surface'):
        if obj.uuid in exclude_uuids:
            counts['patches excluded'] += 1
            continue
        zyxs, invalid, scale = _surface_zyxs(store, obj, 'patch')
        valid_quads = (~invalid[:-1, :-1] & ~invalid[1:, :-1]
                       & ~invalid[:-1, 1:] & ~invalid[1:, 1:])
        if not valid_quads.any():
            counts['patches without valid quads'] += 1
            continue
        erosion = (obj.meta.get('source') or {}).get('spiral_patch_erode_cells')
        patch = Patch(torch.from_numpy(zyxs), torch.tensor(scale),
                      None, None, obj.uuid, erosion)
        patch.name = obj.name
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


@dataclasses.dataclass
class TrackInputs:
    """The selected tracks as the track losses read them.

    ``tracks`` is a fit-frame tracks.PackedTrackCollection of the tracks lying
    entirely inside the z window, ordered by dataset and source key (the
    order of the packed store they were ingested from); a track's source id is
    its row among every selected track. ``crossings`` is the exact-crossing
    CSR over every selected track (rows are source ids): partner rows,
    crossing point indices on both tracks, and the crossing's arclength
    position and clearance in fit voxels. ``derived`` says how the crossings
    were derived from the tracks (the manifest's
    ``derived_inputs.track_crossing``).
    """
    tracks: object
    crossings: dict
    derived: dict = None


# Stored track family -> the crossing kernel's family code.
TRACK_FAMILY_CODES = {'horizontal': 0, 'vertical': 1}
# Under the fit's cache directory: the crossings computed for a selection of
# track revisions (materialize_tracks), in stored voxels.
TRACK_CROSSING_CACHE_DIRECTORY = os.path.join('track_crossings', 'v2')
TRACK_CROSSING_CACHE_FORMAT = 2
# What reading or writing a crossing cache store raises, the store's native
# errors included: a miss, or a cache left unwritten; never a failed fit.
TRACK_CROSSING_STORE_ERRORS = (OSError, RuntimeError, ValueError, KeyError, TypeError,
                               MemoryError)  # a failed allocation, the store's included
# Stored track coordinates the crossing kernel can index: [0, 2**20).
TRACK_COORDINATE_LIMIT = 1 << 20


@dataclasses.dataclass
class _CrossingRows:
    """What the crossings of ordered tracks (_ordered_tracks) depend on
    besides their points: per row the kernel's family code, the numeric
    source key and the stored -> fit scale, and the groups (``(dataset,
    first row, end row)``) tracks pair within."""
    families: np.ndarray
    keys: np.ndarray
    to_fit: np.ndarray
    groups: list


def _ordered_tracks(tracks):
    """``(tracks in crossing order, their _CrossingRows)``.

    Tracks pair only within a dataset, and are consecutive rows of it in
    source-ID order. The grouping, the order and their validation are the
    store's (vc_object_store.group_track_datasets, which its track-track
    detector uses; the dataset of a track is the one the store resolved for
    the selection): a track without a dataset, an ``external_key`` that is
    no canonical decimal uint64 or repeats in its dataset and a family
    other than horizontal, vertical or blank are input errors. So is a
    dataset in more than one frame scale, which only a fit can tell."""
    from vc_object_store import group_track_datasets
    tracks = list(tracks)
    undeclared = [obj.uuid for obj in tracks
                  if not obj.dataset and not (obj.track or {}).get('dataset_ref')]
    if undeclared:
        raise ObjectStoreInputError([f'track {uuid} has no dataset_ref' for uuid in undeclared])
    unresolved = [obj.uuid for obj in tracks if not obj.dataset]
    if unresolved:
        raise ObjectStoreInputError([
            f'the store did not resolve the dataset of {len(unresolved)} selected track(s) '
            f'(first: {unresolved[0]}); a served store needs a vc_object_server of this version'])
    try:
        groups = group_track_datasets(
            [obj.uuid for obj in tracks], [obj.dataset for obj in tracks],
            [str((obj.track or {}).get('external_key') or '') for obj in tracks],
            [str((obj.track or {}).get('family') or '') for obj in tracks])
    except RuntimeError as error:
        raise ObjectStoreInputError([str(error)]) from error
    order = [np.frombuffer(group['rows'], dtype=np.int64) for group in groups]
    ordered = [tracks[row] for rows in order for row in rows.tolist()]
    families = np.concatenate(
        [np.frombuffer(group['families'], dtype=np.int8) for group in groups]
        or [np.empty(0, dtype=np.int8)]).astype(np.int8)
    keys = np.concatenate(
        [np.frombuffer(group['source_ids'], dtype=np.uint64) for group in groups]
        or [np.empty(0, dtype=np.uint64)]).astype(np.uint64)
    to_fit = np.array([obj.frame.to_fit for obj in ordered], dtype=np.float64)
    bounds, begin, problems = [], 0, []
    for group, rows in zip(groups, order):
        end = begin + len(rows)
        bounds.append((group['dataset'], begin, end))
        scales = sorted(set(to_fit[begin:end].tolist()))
        if len(scales) > 1:
            problems.append(f'the tracks of dataset {group["dataset"]!r} are in different frames '
                            f'(stored -> fit scales {scales}); their crossings need one')
        begin = end
    if problems:
        raise ObjectStoreInputError(problems)
    return ordered, _CrossingRows(families, keys, to_fit, bounds)


def _empty_crossings(rows):
    return {
        'source_ids': np.arange(rows, dtype=np.uint64),
        'offsets': np.zeros(rows + 1, dtype=np.int64),
        'partners': np.empty(0, dtype=np.int32),
        'self_local': np.empty(0, dtype=np.int32),
        'partner_local': np.empty(0, dtype=np.int32),
        'positions': np.empty(0, dtype=np.float64),
        'clearances': np.empty(0, dtype=np.float64),
    }


def _compute_track_crossings(tracks, points, rows, workers=None):
    """The exact-crossing CSR over ordered tracks in stored voxels, from
    their stored ZYX points (``points[row]``, int32), with Spiral's crossing
    builder (tracks._build_crossing_partner_csr on the native kernel, the
    implementation the store's track-track detector shares) run once per
    group. Rows are the tracks' rows; partners are sorted within a row; the
    point indices are the stored ones. ``points`` is emptied group by group."""
    from tracks import _build_crossing_partner_csr, _load_native_track_crossings
    if _load_native_track_crossings() is None:
        raise ObjectStoreInputError([
            'the crossings of object-store tracks need the native kernel '
            'vc_spiral.track_crossings, which is not available'])
    limit = np.iinfo(np.int32).max
    if len(tracks) > limit:
        raise ObjectStoreInputError([f'{len(tracks)} tracks exceed the crossing index ({limit})'])
    parts = []
    for dataset, begin, end in rows.groups:
        group = points[begin:end]
        points[begin:end] = [None] * (end - begin)
        families = rows.families[begin:end]
        offsets = np.zeros(end - begin + 1, dtype=np.int64)
        np.cumsum([len(track) for track in group], out=offsets[1:])
        if offsets[-1] > np.iinfo(np.uint32).max:
            raise ObjectStoreInputError([
                f'the tracks of dataset {dataset!r} have {offsets[-1]} points, more than '
                f'the crossing kernel indexes ({np.iinfo(np.uint32).max})'])
        # Crossings pair opposite families: without both there are none.
        if not (np.any(families == 0) and np.any(families == 1)):
            continue
        flat = np.concatenate(group)
        del group
        try:
            csr = _build_crossing_partner_csr(
                [flat[offsets[row]:offsets[row + 1]] for row in range(end - begin)],
                families, flat_points=flat, offsets=offsets,
                source_ids=rows.keys[begin:end], workers=workers)
        except (RuntimeError, ValueError) as error:
            raise ObjectStoreInputError([
                f'the crossings of the tracks of dataset {dataset!r} cannot be computed: '
                f'{error}']) from error
        del flat
        parts.append((begin, csr))
    # Groups are consecutive rows and the kernel orders each group's records
    # by row, then partner row: the merged CSR is their concatenation.
    merged = _empty_crossings(len(tracks))
    if not parts:
        return merged
    counts = np.zeros(len(tracks), dtype=np.int64)
    for begin, csr in parts:
        counts[begin:begin + len(csr['offsets']) - 1] = np.diff(csr['offsets'])
    np.cumsum(counts, out=merged['offsets'][1:])
    for name in ('partners', 'self_local', 'partner_local', 'positions', 'clearances'):
        columns = [csr[name] + begin if name == 'partners' and begin else csr[name]
                   for begin, csr in parts]
        merged[name] = (columns[0] if len(columns) == 1 else np.concatenate(columns)).astype(
            merged[name].dtype, copy=False)
    return merged


def _track_crossing_key(tracks, rows):
    """What the crossings of ordered tracks are a function of, as a digest:
    the kernel and its parameters, and per row the track, its geometry
    revision, its family code and its group. Neither the store's commit nor
    a frame scale: the cached crossings are in stored voxels."""
    from tracks import TRACK_CROSSING_ALGORITHM, TRACK_CROSSING_PARAMETERS
    digest = hashlib.sha256(json.dumps(
        {'format': TRACK_CROSSING_CACHE_FORMAT, 'algorithm': TRACK_CROSSING_ALGORITHM,
         'parameters': TRACK_CROSSING_PARAMETERS}, sort_keys=True).encode())
    group = np.zeros(len(tracks), dtype=np.int64)
    for index, (_, begin, end) in enumerate(rows.groups):
        group[begin:end] = index
    for row, obj in enumerate(tracks):
        digest.update(f'\n{obj.uuid} {obj.geometry_revision} {rows.families[row]} '
                      f'{group[row]}'.encode())
    return digest.hexdigest()


def _track_crossing_cache_path(cache_directory, key):
    return os.path.join(cache_directory, TRACK_CROSSING_CACHE_DIRECTORY, f'{key}.vcobj')


def _track_uuid_bytes(tracks):
    """The tracks' UUIDs as uint8[N, 16], by row."""
    return np.frombuffer(b''.join(bytes.fromhex(obj.uuid.replace('-', '')) for obj in tracks),
                         dtype=np.uint8).reshape(len(tracks), 16)


def _crossing_columns(csr, uuids):
    """One row per crossing pair of a CSR: (a, b UUID bytes, local_a, local_b,
    position_a, position_b, clearance), from the slots whose partner is the
    later row and their reverse slots. None when the CSR is not the kernel's
    symmetric table (each pair in both rows, one clearance), which a store
    of pairs could not give back."""
    count = len(uuids)
    rows = np.repeat(np.arange(count, dtype=np.int64), np.diff(csr['offsets']))
    partners = csr['partners'].astype(np.int64)
    forward = partners > rows
    key = rows * count + partners  # ascending: rows ascend, partners ascend within a row
    wanted = partners[forward] * count + rows[forward]
    reverse = np.searchsorted(key, wanted)
    if (2 * int(forward.sum()) != len(partners) or len(key) and (
            reverse.max(initial=0) >= len(key) or not np.array_equal(key[reverse], wanted))):
        return None
    columns = (uuids[rows[forward]], uuids[partners[forward]],
               csr['self_local'][forward], csr['partner_local'][forward],
               csr['positions'][forward], csr['positions'][reverse], csr['clearances'][forward])
    if (not np.array_equal(csr['self_local'][reverse], columns[3])
            or not np.array_equal(csr['partner_local'][reverse], columns[2])
            or not np.array_equal(csr['clearances'][reverse], columns[6])):
        return None
    return tuple(np.ascontiguousarray(column) for column in columns)


def _crossings_from_columns(columns, uuids):
    """The CSR over the rows of ``uuids`` that the pairs of a crossing store
    are: each pair in both its rows, partners ascending within a row. A pair
    naming a track that is not among ``uuids`` raises ValueError."""
    count = len(uuids)

    def halves(column):
        # A UUID as two big-endian integers: (high, low) orders like its bytes.
        pair = np.ascontiguousarray(column, dtype=np.uint8).reshape(-1, 16).view('>u8')
        return pair[:, 0].astype(np.uint64), pair[:, 1].astype(np.uint64)
    high, low = halves(uuids)
    # Every lookup is a binary search of whole UUIDs. Halves are ranked
    # among the tracks' own, so a (high, low) pair is one integer key that
    # orders like the UUID; UUIDs sharing a half cost nothing extra.
    highs, high_rank = np.unique(high, return_inverse=True)
    lows, low_rank = np.unique(low, return_inverse=True)
    names = high_rank.astype(np.int64) * len(lows) + low_rank
    order = np.argsort(names, kind='stable')
    names = names[order]

    def rank(values, wanted):
        found = np.searchsorted(values, wanted)
        known = found < len(values)
        known[known] = values[found[known]] == wanted[known]
        return found, known

    distinct = len(highs) == count  # the usual case: no two tracks share a high half
    high_order = np.argsort(high, kind='stable') if distinct else None

    def rows_of(column):
        wanted_high, wanted_low = halves(column)
        if distinct:  # the high half finds the row; the low half confirms it
            found, known = rank(highs, wanted_high)
            rows = high_order[np.minimum(found, count - 1)] if count else found
            if not (known.all() and np.array_equal(low[rows], wanted_low)):
                raise ValueError('it names a track that is not selected')
            return rows
        (high_rank, high_known), (low_rank, low_known) = rank(highs, wanted_high), rank(lows, wanted_low)
        known = high_known & low_known
        wanted = np.where(known, high_rank.astype(np.int64) * len(lows) + low_rank, -1)
        found, named = rank(names, wanted)
        if not (known & named).all():
            raise ValueError('it names a track that is not selected')
        return order[found]
    a, b = rows_of(columns['a']), rows_of(columns['b'])
    rows, partners = np.concatenate([a, b]), np.concatenate([b, a])
    slots = np.argsort(rows * count + partners, kind='stable')  # by row, then partner
    both = lambda first, second: np.concatenate([columns[first], columns[second]])[slots]
    return {
        'source_ids': np.arange(count, dtype=np.uint64),
        'offsets': np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=count))]).astype(np.int64),
        'partners': partners[slots].astype(np.int32),
        'self_local': both('local_a', 'local_b').astype(np.int32),
        'partner_local': both('local_b', 'local_a').astype(np.int32),
        'positions': both('position_a', 'position_b').astype(np.float64),
        'clearances': both('clearance', 'clearance').astype(np.float64),
    }


def _crossing_store_workers(workers, limit=32):
    return max(1, min(limit, int(workers or os.cpu_count() or 1)))


def _load_track_crossings(path, key, tracks, workers=None, *, warn=True):
    """``(CSR in stored voxels, its sha256)`` the crossing store at ``path``
    holds for ``key`` over ``tracks`` (ordered), or None. A missing store is
    a miss; so is one that cannot be read, is of another key, algorithm or
    track count, names a track that is not selected, or whose crossings are
    not the ones its checksum was taken of (the checksum of the CSR as it
    was computed, crossing_csr_sha256), with a warning: it is rebuilt."""
    from tracks import (TRACK_CROSSING_ALGORITHM, _validate_crossing_partner_csr,
                        crossing_csr_sha256)
    if not os.path.lexists(path):
        return None
    try:
        from vc_object_store import CollectionReader
        with CollectionReader(path) as reader:
            info = reader.track_crossing_info()
            if info['key'] != key or info['algorithm'] != TRACK_CROSSING_ALGORITHM:
                raise ValueError('it was built for other tracks')
            if info['tracks'] != len(tracks):
                raise ValueError(f'it holds {info["tracks"]} rows, not {len(tracks)}')
            columns = reader.track_crossing_columns(_crossing_store_workers(workers))
        csr = _crossings_from_columns(columns, _track_uuid_bytes(tracks))
        _validate_crossing_partner_csr(csr)
        sha256 = crossing_csr_sha256(csr)
        if sha256 != info['checksum']:
            raise ValueError('its crossings do not match its checksum')
    except TRACK_CROSSING_STORE_ERRORS as error:
        if warn:
            print(f'WARNING: ignoring invalid track crossing cache {path}: {error}')
        return None
    return csr, sha256


def _store_track_crossings(path, key, csr, sha256, tracks, position=None, *, replace=False,
                           workers=None):
    """Publishes the crossings as the sealed crossing store ``path``
    (vc_object_store.write_track_crossing_store: one edge and one compact
    row per pair, built with the sharded streaming importer), recording
    ``key``, the CSR's checksum and the store ``position`` the tracks were
    read at. A store already there is left alone, unless ``replace``: the
    caller holds the cache's lock and found that store unusable. It is then
    read once more, here, immediately before it would be deleted: a process
    that could not take the lock may have published a good one since, which
    is kept. Nothing a cache cannot do fails the fit: the computed crossings
    stay in use."""
    from tracks import TRACK_CROSSING_ALGORITHM, TRACK_CROSSING_PARAMETERS
    absent = False  # the destination was ours to write
    try:
        if os.path.lexists(path):
            if not replace or _load_track_crossings(path, key, tracks, workers, warn=False):
                return
            os.remove(path)
        absent = True
        columns = _crossing_columns(csr, _track_uuid_bytes(tracks))
        if columns is None:
            print(f'WARNING: not caching the track crossings at {path}: they are not one '
                  'symmetric table of pairs')
            return
        info = {'algorithm': TRACK_CROSSING_ALGORITHM, 'parameters': dict(TRACK_CROSSING_PARAMETERS),
                'key': key, 'checksum': sha256, 'tracks': len(tracks)}
        if position is not None:
            info['based_on'] = [{'store_uuid': position.store, 'generation': position.generation,
                                 'commit_sequence': position.commit_sequence}]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        from vc_object_store import write_track_crossing_store
        # Shards and writers: more than eight did not write 6.6M pairs faster.
        write_track_crossing_store(path, *columns, info,
                                   workers=_crossing_store_workers(workers, limit=8))
    except TRACK_CROSSING_STORE_ERRORS as error:
        if not (absent and os.path.lexists(path)):  # else another process published it meanwhile
            print(f'WARNING: cannot write the track crossing cache {path}: {error}')


@contextlib.contextmanager
def _track_crossing_lock(path):
    """Holds the cache store's lock (service_files.ExclusiveFileLock), and
    gives whether it is held: of the processes that miss one cache store at
    once (the ranks of a distributed fit), one computes the crossings and
    the others wait for its store instead of each holding the kernel's
    buffers. Where the cache cannot be written nothing is held (False) and
    each computes."""
    from service_files import ExclusiveFileLock, FileLockUnavailable
    lock = ExclusiveFileLock(f'{path}.lock')
    try:
        try:
            lock.acquire()
        except FileLockUnavailable:
            print('track crossings: waiting for another process that computes them')
            lock.acquire(timeout=None)
    except OSError:
        yield False
        return
    try:
        yield True
    finally:
        lock.release()


def _track_crossing_provenance(csr, sha256):
    """The manifest's ``derived_inputs.track_crossing``: the crossings are
    computed from the selected tracks; their count and the checksum of the
    CSR in stored voxels identify them."""
    from tracks import TRACK_CROSSING_ALGORITHM, TRACK_CROSSING_PARAMETERS
    return {'source': 'computed', 'algorithm': TRACK_CROSSING_ALGORITHM,
            'parameters': dict(TRACK_CROSSING_PARAMETERS),
            'count': len(csr['partners']) // 2, 'sha256': sha256}


def materialize_tracks(store, snapshot, *, z_begin, z_end,
                       cache_directory=None, workers=None):
    """The snapshot's tracks and crossings (TrackInputs), or None without
    tracks; and counts.

    A track is kept when every point's fit-frame z lies in ``[z_begin,
    z_end)``. Arclengths and tortuosities are measured on the stored integer
    voxels, as the packed track store measures them, then converted to fit
    voxels.

    The crossings are computed here, from the stored points of every
    selected track (those outside the z window included), and are a function
    of the selected track revisions alone: exact shared voxels of
    opposite-family tracks of one dataset (_ordered_tracks,
    _compute_track_crossings). Crossing relationships a store holds are not
    read. ``cache_directory`` (the fit's cache) keeps them, in stored
    voxels, for the next fit of the same track revisions, as a sealed store
    of the crossing pairs (_store_track_crossings); ``workers`` bounds the
    kernel's threads and the cache store's writers and readers. They are computed whatever the fit's crossing
    limits, so a fit's inputs do not depend on its run settings.
    """
    from tracks import PackedTrackCollection, crossing_csr_sha256

    counts = Counter()
    if not snapshot.objects_of('track'):
        return None, counts
    cached = cache_path = key = None
    tracks, rows = _ordered_tracks(snapshot.objects_of('track'))
    if cache_directory:
        key = _track_crossing_key(tracks, rows)
        cache_path = _track_crossing_cache_path(cache_directory, key)
        cached = _load_track_crossings(cache_path, key, tracks, workers)
    with contextlib.ExitStack() as computing:
        locked = False
        if cache_path is not None and cached is None:
            # One process computes a missing store; one that waited reads it.
            locked = computing.enter_context(_track_crossing_lock(cache_path))
            cached = _load_track_crossings(cache_path, key, tracks, workers)
        # Every selected track's stored points, when the crossings are computed.
        selection = [None] * len(tracks) if cached is None else None
        problems = []
        kept, stored, fit, families, scales = [], [], [], [], []
        for index, obj in enumerate(tracks):
            arrays, _ = read_geometry(store, obj.geometry_revision, ['points'])
            xyz = arrays['points']
            if xyz.ndim != 2 or xyz.shape[1] != 3:
                problems.append(f'track {obj.uuid} points have shape {xyz.shape}')
                continue
            zyx = np.ascontiguousarray(xyz[:, ::-1], dtype=np.int32)
            if selection is not None:
                if len(xyz) and not (xyz.min() >= 0 and xyz.max() < TRACK_COORDINATE_LIMIT):
                    problems.append(f'track {obj.uuid} has coordinates outside '
                                    f'[0, {TRACK_COORDINATE_LIMIT}), which crossings cannot index')
                if len(xyz) > np.iinfo(np.int32).max:
                    problems.append(f'track {obj.uuid} has {len(xyz)} points, more than '
                                    'crossings can index')
                selection[index] = zyx
            if not len(xyz):
                continue
            z = xyz[:, 2].astype(np.float64) * obj.frame.to_fit
            if z.min() < z_begin or z.max() >= z_end:
                counts['tracks outside the z window'] += 1
                continue
            family = (obj.track or {}).get('family') or ''
            kept.append(index)
            stored.append(zyx)
            fit.append(obj.frame.xyz_to_fit_zyx(xyz))
            families.append(TRACK_FAMILY_CODES.get(family, -1))
            scales.append(obj.frame.to_fit)
        if problems:
            raise ObjectStoreInputError(problems)
        if cached is None:
            started = time.monotonic()
            csr = _compute_track_crossings(tracks, selection, rows, workers)
            sha256 = crossing_csr_sha256(csr)
            if cache_path is not None:
                # A store still there failed both reads: under the lock it is replaced.
                _store_track_crossings(cache_path, key, csr, sha256, tracks, snapshot.position,
                                       replace=locked, workers=workers)
            print(f'track crossings: {len(csr["partners"]) // 2} computed from '
                  f'{len(tracks)} tracks in {time.monotonic() - started:.1f} s')
        else:
            csr, sha256 = cached
            print(f'track crossings: {len(csr["partners"]) // 2} read from the cache '
                  f'{cache_path}')
    del selection
    derived = _track_crossing_provenance(csr, sha256)
    # Stored voxels -> fit voxels, by each row's (its group's) frame.
    scale = np.repeat(rows.to_fit, np.diff(csr['offsets']))
    csr = {**csr, 'positions': csr['positions'] * scale,
           'clearances': csr['clearances'] * scale}

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
    counts['track crossings'] = derived['count']
    return TrackInputs(collection, csr, derived), counts


def materialize(store, snapshot, *, z_begin, z_end, z_margin,
                erode_cells_default, exclude_patch_uuids=(), cache_directory=None):
    """Every runtime input of a frozen snapshot. ``cache_directory`` is
    materialize_tracks' cache."""
    patches, counts = materialize_patches(
        store, snapshot, z_begin=z_begin, z_end=z_end, z_margin=z_margin,
        erode_cells_default=erode_cells_default, exclude_uuids=exclude_patch_uuids)
    tracks, track_counts = materialize_tracks(
        store, snapshot, z_begin=z_begin, z_end=z_end,
        cache_directory=cache_directory)
    counts.update(track_counts)
    return FitInputs(snapshot, patches, counts,
                     outer_shell=materialize_outer_shell(store, snapshot),
                     tracks=tracks)


def materialize_from_config(store, snapshot, config, cache_directory=None):
    """materialize() with the fit configuration's window and settings.
    ``cache_directory`` is the fit's cache."""
    return materialize(
        store, snapshot, z_begin=config['z_begin'], z_end=config['z_end'],
        z_margin=config['patch_loss_z_margin'],
        erode_cells_default=config['patch_erode_patches'],
        exclude_patch_uuids=config['object_store_patch_exclude_uuids'] or (),
        cache_directory=cache_directory)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def snapshot_summary(snapshot):
    """The selection a snapshot holds, as data; snapshot_report prints it.

    Per object its provenance (UUID, kind, name, revision pins, metadata
    hash, scale); then per-kind counts and the fit volume. Reads nothing
    beyond the snapshot, so a service can describe a selection without
    materializing it.
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
        objects.append(entry)
    return {
        'store': {'location': snapshot.store_location, 'uuid': snapshot.store_uuid,
                  'generation': snapshot.store_generation},
        'volume_uuid': snapshot.volume_uuid,
        'volume_source': snapshot.volume_source,
        'units': FIT_UNITS,
        'selection': selection,
        'objects': objects,
        'outer_shell': (snapshot.outer_shell.provenance()
                        if snapshot.outer_shell else None),
        'notes': list(snapshot.notes),
    }


def snapshot_report(snapshot):
    """Human-readable lines describing the selection."""
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
    lines.extend(summary['notes'])
    return lines


def constraints_report(store, snapshot, config):
    """Materialize the inputs as a fit would, and describe them; raises the
    same validation errors a fit would."""
    return materialize_from_config(store, snapshot, config).report()


# ---------------------------------------------------------------------------
# Startup lifecycle: selection from the fit configuration, a pinned position
# on rank zero, then a frozen read every rank verifies against the others
# ---------------------------------------------------------------------------

_SELECTOR_CONFIG = {'surface': 'patch', 'track': 'track'}
_NONE = Selector(uuids=())


def request_from_config(config):
    """The InputRequest of a fit configuration.

    Each kind's ``object_store_<kind>_uuids``/``_tags`` select objects (both
    None: every active object). Switching a source's input toggle off
    selects none of it.
    """
    selectors = {}
    for kind, name in _SELECTOR_CONFIG.items():
        uuids = config[f'object_store_{name}_uuids']
        tags = config[f'object_store_{name}_tags']
        selectors[kind] = Selector(None if uuids is None else tuple(uuids),
                                   None if tags is None else tuple(tags))
    if not config['input_use_verified_patches'] or config['input_disable_patches']:
        selectors['surface'] = _NONE
    if not config['input_use_tracks']:
        selectors['track'] = _NONE
    from fit_session import input_source_enabled, outer_shell_required
    shell = ('required' if outer_shell_required(config)
             else 'optional' if input_source_enabled(config, 'outer_shell') else 'off')
    return InputRequest(selectors, config['fit_volume_uuid'], outer_shell=shell,
                        fit_volume_level=int(config['fit_volume_level']))


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


def _startup_phase(config, request, cache_directory, minimum_position=None, pins=None):
    """Rank zero's startup: pin the store's head (prepare_position), unless
    object_store_input_position (or ``pins``, the versions a resume or an
    update provides) specifies the position to read exactly. Returns
    ``(SelectedObjects, report lines, InputPins)``; raises
    ObjectStoreInputError on any failure."""
    pinned = pins if pins is not None else InputPins.from_provenance(config['object_store_input_position'])
    with open_store(config, cache_directory=cache_directory) as store:
        if pinned is not None:
            # Prepared already (the service pinned it, or a checkpoint
            # recorded it): read it exactly.
            check_position(store, pinned.position)
            selected = resolve_selection(store, dataclasses.replace(request, pins=pinned))
            return selected, [f'reading pinned {pinned.position.describe()}'], selected.pins()
        prepared = prepare_position(store, request, minimum=minimum_position)
    return prepared.selected, prepared.lines, prepared.pins


class PositionError(ObjectStoreInputError):
    """No position could be pinned; the active fit is kept."""


@dataclasses.dataclass
class PreparedPosition:
    """A pinned position: ``pins`` hold the selected objects at the versions
    of the store's head (``head``, its commit); ``minimum`` is the position
    the caller asked to include."""
    pins: InputPins
    head: int
    minimum: InputPosition
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


def prepare_position(store, request, *, minimum=None, progress=None):
    """Pin the store's head H (at least ``minimum``, the position a client's
    saved edits reached): the selected objects at their versions there,
    which later commits never change."""
    if minimum is not None:
        check_position(store, minimum)
    with reading_stage(progress, 'Reading the selected inputs', 'head'):
        selected = resolve_selection(store, dataclasses.replace(request, pins=None))
    head = selected.position.commit_sequence
    if minimum is not None and head < minimum.commit_sequence:
        raise PositionError([
            f'the store is at commit {head}, before the requested {minimum.describe()}; '
            'is this the store the edits were saved to?'])
    pins = selected.pins()
    lines = [f'pinning {selected.position.describe()}']
    if minimum is not None and head > minimum.commit_sequence:
        lines.append(f'{pins.position.describe()} includes commits after the requested '
                     f'{minimum.describe()}')
    return PreparedPosition(pins=pins, head=head, minimum=minimum, selected=selected,
                            lines=lines)


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
    'fit_volume_level',
)


def input_manifest(snapshot, config, external_frames=None, tracks=None):
    """The snapshot manifest extended with materialization settings,
    external-input frames and, with ``tracks`` (the materialized
    TrackInputs), the crossings derived from them
    (``derived_inputs.track_crossing``)."""
    manifest = snapshot.manifest()
    if tracks is not None:
        manifest['derived_inputs'] = {'track_crossing': tracks.derived}
    manifest['materialization'] = {key: config[key] for key in MATERIALIZATION_KEYS}
    manifest['external_frames'] = dict(sorted((external_frames or {}).items()))
    return manifest


def validated_manifest_references(manifest):
    """Validate saved evidence once and retain the decoded object references."""
    from vc_object_store import manifest_references
    return manifest_references(manifest)['objects']


def _compared(manifest):
    """The manifest without what does not define the inputs: the store's
    location and history generation, the version pins (VERSION_FIELDS) and
    the position they were read at (equal inputs at two positions are
    equal). A copy of a store holding the same inputs compares equal."""
    validated_manifest_references(manifest)
    manifest = dict(manifest)
    manifest['store'] = {'uuid': (manifest.get('store') or {}).get('uuid')}
    manifest.pop('position', None)
    if manifest.get('outer_shell'):
        manifest['outer_shell'] = {key: value for key, value in manifest['outer_shell'].items()
                                   if key != 'reference'}
    for section, fields in VERSION_FIELDS.items():
        if section in manifest:
            manifest[section] = [
                {key: value for key, value in entry.items() if key not in fields}
                for entry in manifest[section] or ()]
    return manifest


def track_identity(manifest):
    """The tracks a manifest records, by content (VERSION_FIELDS left
    out), and the crossings derived from them
    (``derived_inputs.track_crossing``): a fit consumes them at build, so
    two manifests with different ones need a rebuild."""
    compared = _compared(manifest)
    return {
        'objects': [entry for entry in compared.get('objects') or ()
                    if entry.get('kind') == 'track'],
        'derived': (compared.get('derived_inputs') or {}).get('track_crossing'),
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
        'constraints': dict(sorted(inputs.counts.items())),
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
        differences = manifest_differences(recorded, current)
        if not differences:
            return []
        problem = ('checkpoint object-store inputs differ from this fit\'s in '
                   + ', '.join(differences))
    if allow_changed:
        log(f'WARNING: {problem}; resuming anyway (object_store_allow_changed_inputs)')
        return []
    return [f'{problem}; set object_store_allow_changed_inputs to resume anyway']


def load_fit_inputs(config, *, external_frames=None, dist=None,
                    cache_directory=None, log=print, minimum_position=None, pins=None):
    """Pin, freeze and materialize a fit's object-store inputs.

    Rank zero (the only process when ``dist`` is None or single-rank) pins a
    position of the store, the head including at least
    ``minimum_position`` (prepare_position), unless
    object_store_input_position (or ``pins``: the versions a resume's
    checkpoint or an input update provides) specifies the position to read. The
    other ranks wait for its result, and every rank fails when it fails.
    Every rank then reads the pinned versions, materializes the
    same frozen snapshot and closes the store; their manifests must agree
    before anyone trains. Returns FitInputs carrying ``manifest`` and
    ``digest``. ``cache_directory`` is the fit's cache (a served store's
    persistent cache and the track crossings live under it).
    """
    if not config['object_store_location']:
        raise ObjectStoreInputError(['object_store_location is not configured'])
    main = dist is None or dist.is_main_process
    status = None
    if main:
        try:
            selected, lines, pinned = _startup_phase(
                config, request_from_config(config), cache_directory,
                minimum_position, pins)
            status = {'ok': True, 'pins': pinned.provenance(), 'lines': lines}
        except ObjectStoreInputError as error:
            status = {'ok': False, 'problems': error.problems}
        except Exception as error:  # noqa: BLE001 - other ranks must hear of it
            status = {'ok': False, 'problems': [f'{type(error).__name__}: {error}']}
    status = _broadcast(status, dist)
    if not status['ok']:
        raise ObjectStoreInputError(status['problems'])
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
        with open_store(config, cache_directory=cache_directory) as store:
            snapshot = read_snapshot(store, request)
            inputs = materialize_from_config(store, snapshot, config, cache_directory)
        inputs.manifest = input_manifest(snapshot, config, external_frames, inputs.tracks)
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
    from its store as it stands, without materializing anything and
    without importing Torch. A selection the
    fitter would refuse reports its problems instead of raising."""
    try:
        request = request_from_config(config)
        with open_store(config) as store:
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
    request = InputRequest(selectors, args.volume)
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
