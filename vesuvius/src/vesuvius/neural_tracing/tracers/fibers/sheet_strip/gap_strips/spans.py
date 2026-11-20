"""The annotated part of a store fiber, and the pieces derived from it.

A fiber is the span between its outermost control points, without the interior of the spans tagged ``gap`` or
``damaged`` (``segment_to_next.tags`` in the annotation of the control point that starts the span). Line points
beyond the outermost control points (tails) are not part of it.

A derived fiber (piece or chain) records what it is in ``meta.gap_strips``: ``role`` ('piece' / 'chain'),
``source`` and ``source_reference`` (the source fiber and its version) and the role's own fields (pieces.py,
mirror.py). That metadata travels with the derived fiber's version, so later steps need no side files.
"""
from collections import Counter
import uuid as uuidlib

import numpy as np
from vc_object_store import canonical_uuid, geometry_scale_to_volume, object_reference

from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import (
    WRITE_BATCH, fiber_volume, open_store, read_polylines)
from vesuvius.neural_tracing.tracers.shared.geometry import dense_line

EXCLUDED_SPAN_TAGS = frozenset({'gap', 'damaged'})
META_KEY = 'gap_strips'


def segment_tags(annotations, control_count):
    """Per control point, the tags of the segment to the next one, from the control points' annotations (None when
    the fiber has none)."""
    annotations = annotations or [None]*control_count
    return [frozenset(((a or {}).get('segment_to_next') or {}).get('tags') or []) for a in annotations]


def control_runs(tags):
    """The valid pieces as control-index runs [(k0, k1), ...] (k1 inclusive, k1 > k0): maximal runs of consecutive
    control points joined by segments that are not tagged gap/damaged. A piece never bridges an excluded span; the
    control points bounding an excluded span end one piece and start the next."""
    runs, start = [], 0
    for k, segment in enumerate(tags[:-1]):
        if segment & EXCLUDED_SPAN_TAGS:
            if k > start:
                runs.append((start, k))
            start = k+1
    if len(tags)-1 > start:
        runs.append((start, len(tags)-1))
    return runs


def anchors(line, controls):
    """For every control point, the index of its nearest line point."""
    return ((line[None, :, :]-controls[:, None, :])**2).sum(-1).argmin(1)


def piece_points(line, anchor, k0, k1, spacing=None):
    """The line points from the anchor of control point k0 to that of k1 (inclusive, in control order), densified
    to ``spacing`` when given; None when that is no line."""
    lo, hi = sorted((int(anchor[k0]), int(anchor[k1])))
    points = line[lo:hi+1]
    if anchor[k0] > anchor[k1]:
        points = points[::-1]
    if len(points) < 2 or not np.linalg.norm(np.diff(points, axis=0), axis=1).sum() > 0:
        return None
    return dense_line(points, spacing) if spacing else np.ascontiguousarray(points)


def derived_id(source_id, label):
    """A stable UUID for a piece or chain of a source fiber."""
    return str(uuidlib.uuid5(uuidlib.UUID(source_id), label))


def read_only_list(path):
    """Fiber UUIDs or names, one per line (a trailing '.json' is ignored); None without a path."""
    if not path:
        return None
    return [line.strip().removesuffix('.json') for line in open(path) if line.strip()]


def _canonical(value):
    try:
        return canonical_uuid(str(value))
    except (ValueError, RuntimeError):
        return None


def select_records(records, only, where='the store'):
    """The records ``only`` names, each entry a fiber UUID or, when it is no UUID of the store, a fiber name. Raises
    for an entry that matches nothing and for a name several fibers carry (select those by UUID)."""
    by_id = {r['uuid']: r for r in records}
    names = Counter(r['name'] for r in records)
    by_name = {r['name']: r for r in records}
    chosen, missing, ambiguous = {}, [], []
    for entry in only:
        uuid = _canonical(entry)
        if uuid in by_id:
            chosen[uuid] = by_id[uuid]
        elif names[entry] == 1:
            chosen[by_name[entry]['uuid']] = by_name[entry]
        elif names[entry] > 1:
            ambiguous.append(entry)
        else:
            missing.append(entry)
    for what, entries in (('Not in', missing), ('Several fibers have this name (use UUIDs) in', ambiguous)):
        if entries:
            raise ValueError(f'{what} {where}: {", ".join(entries[:5])}{" ..." if len(entries) > 5 else ""}')
    return list(chosen.values())


def read_annotated(path, only=None, *, volume=None):
    """``(volume, fibers)``: the fibers of a store bound to ``volume`` (default: the store's one volume) with their
    annotation, in UUID order: dicts of id, name, family, reference, line (N, 3) and controls (K, 3) in volume
    voxels, and tags (per control point, the tags of its segment to the next). ``only``: UUIDs or names to keep
    (select_records). Fibers with fewer than 2 line points or 2 control points have no span and are left out."""
    volume = volume or fiber_volume(path)
    fibers = []
    with open_store(path) as store:
        records = store.select_objects(object_kind='fiber', volume=volume, fields=['name', 'fiber', 'meta'])
        if only is not None:
            records = select_records(records, only, str(path))
        records.sort(key=lambda r: r['uuid'])
        selected = []
        for start in range(0, len(records), WRITE_BATCH):
            # One pinned identity per object; the view holds the store's reader, so arrays are read after it closes.
            with store.graph_read(versions=[object_reference(r) for r in records[start:start+WRITE_BATCH]],
                                  edge_limit=0) as view:
                selected += zip(view.objects(['name', 'fiber', 'meta']), list(view.geometry_revisions))
        for record, revision in selected:
            arrays = {a['name']: a for a in store.geometry_revision(revision)['arrays']}
            scale = geometry_scale_to_volume(record['meta'], volume)
            read = lambda name: np.frombuffer(
                store.read_geometry_array(revision, name),
                dtype=arrays[name]['dtype']).reshape(arrays[name]['shape']).astype(np.float64)*scale
            line, controls = read('line_points'), read('control_points')
            if len(line) < 2 or len(controls) < 2:
                continue
            tags = segment_tags(store.geometry_point_annotations(revision, 'control_points'), len(controls))
            fibers.append(dict(id=record['uuid'], name=record['name'],
                               family=(record.get('fiber') or {}).get('family', ''),
                               reference=object_reference(record), line=line, controls=controls, tags=tags))
    return volume, fibers


def pieces_of(fiber, spacing):
    """The valid pieces of a fiber (read_annotated): dicts of id, name, family, source, piece, controls [k0, k1],
    pieces (the fiber's count) and points, each running from its first control point to its last; a run without a
    line between its anchors is left out. Raises when the control points are not in the order of the line."""
    runs = control_runs(fiber['tags'])
    anchor = anchors(fiber['line'], fiber['controls'])
    steps = np.diff(anchor)
    if (steps < 0).any() and (steps > 0).any():
        raise ValueError(f"Fiber {fiber['name']} ({fiber['id']}): its control points are not in the order of its "
                         'line points')
    pieces = []
    for p, (k0, k1) in enumerate(runs):
        points = piece_points(fiber['line'], anchor, k0, k1, spacing)
        if points is not None:
            pieces.append(dict(id=derived_id(fiber['id'], f'piece{p}'), name=f"{fiber['name']}#piece{p}",
                               family=fiber['family'], source=fiber['id'], piece=p, controls=[k0, k1],
                               pieces=len(runs), points=points))
    return pieces


def fiber_row(entry, role, reference, **fields):
    """The fiber_store.write_fibers row of a piece or chain, recording what it was derived from."""
    return dict(id=entry['id'], name=entry['name'], family=entry['family'],
                meta={META_KEY: dict(role=role, source=entry['source'], source_reference=reference, **fields)})


def read_derived(path, role, fiber_ids=None):
    """{UUID: (row, points in volume voxels)} of a derived store's fibers (all, or the given ones); row['gap_strips']
    is the fiber's ``meta.gap_strips``. Raises when a fiber is not a derived fiber of ``role``."""
    fibers = {}
    for row, points in read_polylines(path, fiber_ids):
        derived = (row.get('meta') or {}).get(META_KEY) or {}
        if derived.get('role') != role:
            raise ValueError(f"{path}: fiber {row['id']} is not a gap_strips {role}")
        fibers[row['id']] = (dict(row, gap_strips=derived), points)
    return fibers
