"""Manual scaffold links as winding arcs.

A ``fiber_branch`` edge asserts the winding relation of two annotated fibers
at its two anchor points. The relation follows VC3D's own winding solver
(``FiberNetworkLayout.cpp:adjacentWindingOffset``): an ordinary branch means
the same winding, an adjacent branch means the V fiber sits one winding
inside the H fiber, so ``W_b - W_a`` is ``-1`` for ``a`` H and ``b`` V and
``+1`` for ``a`` V and ``b`` H. The stored H/V tags decide; the umbilicus is
never used to orient a link. An adjacent link between two fibers of one
family is an annotation error and constrains nothing.

The anchor-level relation becomes a whole-fiber arc with the same seam and
connector transport the automatic constraints use.
"""
from collections import Counter

import numpy as np

DEFAULT_APPROVAL_STATES = ('approved',)


def link_settings(config):
    """Validated ``manual_links`` settings with explicit defaults."""
    settings = dict(config.get('manual_links') or {})
    unknown = set(settings) - {'approval_states', 'include_auto', 'max_anchor_distance'}
    if unknown:
        raise ValueError('Unknown manual_links settings: ' + ', '.join(sorted(unknown)))
    states = settings.get('approval_states', list(DEFAULT_APPROVAL_STATES))
    if isinstance(states, str) or not states or any(not isinstance(s, str) for s in states):
        raise ValueError('manual_links.approval_states must be a non-empty list of approval states')
    distance = float(settings.get('max_anchor_distance', 5.))
    if not np.isfinite(distance) or distance < 0:
        raise ValueError('manual_links.max_anchor_distance must be a finite non-negative voxel distance')
    return dict(approval_states=tuple(states), include_auto=bool(settings.get('include_auto', False)),
                max_anchor_distance=distance)


def adjacent_step(family_a, family_b):
    """``W_b - W_a`` asserted by an adjacent link, or None when not one H and one V."""
    if family_a == 0 and family_b == 1:
        return -1
    if family_a == 1 and family_b == 0:
        return 1
    return None


def project_arc(paths, fiber, point):
    """Arc length of the polyline point nearest ``point`` and its distance."""
    begin, end = int(paths.offset[fiber]), int(paths.offset[fiber + 1])
    xyz = np.asarray(paths.xyz[begin:end], dtype=float)
    arc = np.asarray(paths.keys['arc'][begin:end], dtype=float)
    point = np.asarray(point, dtype=float)
    if len(xyz) == 1:
        return float(arc[0]), float(np.linalg.norm(xyz[0] - point))
    segment = xyz[1:] - xyz[:-1]
    length = np.einsum('ij,ij->i', segment, segment)
    t = np.where(length > 0, np.einsum('ij,ij->i', point - xyz[:-1], segment) / np.maximum(length, 1e-300), 0.)
    t = np.clip(t, 0., 1.)
    nearest = xyz[:-1] + t[:, None] * segment
    distance = np.linalg.norm(nearest - point, axis=1)
    j = int(np.argmin(distance))
    return float(arc[j] + t[j] * (arc[j + 1] - arc[j])), float(distance[j])


def read_links(store_path, kind='fiber_branch'):
    from vc_object_store import CollectionReader
    with CollectionReader(str(store_path)) as store:
        return store.edges(kind)


def manual_link_arcs(edges, ids, family, paths, settings):
    """Whole-fiber integer arcs from stored links; counts every skipped link.

    ``ids`` are the solver's fiber UUIDs in node order (they must equal
    ``paths.ids``), ``family`` their H(0)/V(1) labels. Returns unaggregated
    unit-weight arcs (``a<b``) plus a report; callers apply the manual weight.
    """
    if not np.array_equal(np.asarray(ids), np.asarray(paths.ids)):
        raise ValueError('Manual links need the geometry in solver node order')
    index = {str(uuid): i for i, uuid in enumerate(ids)}
    report = Counter(links=len(edges))
    rows = []
    for edge in edges:
        meta = edge.get('meta') or {}
        if edge.get('approval_state') not in settings['approval_states']:
            report['skipped_approval_state'] += 1; continue
        if meta.get('auto', False) and not settings['include_auto']:
            report['skipped_automatic'] += 1; continue
        a, b = index.get(str(edge['a'])), index.get(str(edge['b']))
        if a is None or b is None:
            report['skipped_disabled_endpoint'] += 1; continue
        if a == b:
            report['skipped_self_link'] += 1; continue
        step = 0
        if meta.get('adjacent', False):
            step = adjacent_step(int(family[a]), int(family[b]))
            if step is None:
                report['skipped_adjacent_same_family'] += 1; continue
            report['adjacent'] += 1
        else:
            report['ordinary'] += 1
        try:
            pa = edge['a_anchor']['position']; pb = edge['b_anchor']['position']
        except (KeyError, TypeError):
            report['skipped_missing_anchor'] += 1; continue
        rows.append((a, b, step, pa, pb))
    if not rows:
        return dict(a=np.empty(0, np.int64), b=np.empty(0, np.int64), target=np.empty(0, np.int64),
                    weight=np.empty(0, float)), dict(report, used=0, max_anchor_distance=0.)
    a = np.array([r[0] for r in rows], np.int64); b = np.array([r[1] for r in rows], np.int64)
    step = np.array([r[2] for r in rows], np.int64)
    arc_a = np.empty(len(rows)); arc_b = np.empty(len(rows)); distance = np.empty(len(rows))
    for k, (ia, ib, _, pa, pb) in enumerate(rows):
        arc_a[k], da = project_arc(paths, ia, pa)
        arc_b[k], db = project_arc(paths, ib, pb)
        distance[k] = max(da, db)
    if np.any(distance > settings['max_anchor_distance']):
        raise ValueError('Manual link anchors lie %.2f voxels from their fiber polyline; check the coordinate frame '
                         'or raise manual_links.max_anchor_distance' % float(distance.max()))
    xa, ta = paths.at_indices(a, arc_a)
    xb, tb = paths.at_indices(b, arc_b)
    connector = np.asarray(paths.seam.segments(xa, xb), np.int64)
    # Whole-fiber step: the anchor relation plus the transport along both
    # fibers and across the connector, exactly as corrected_targets does.
    target = step + connector - np.asarray(tb, np.int64) + np.asarray(ta, np.int64)
    swap = a > b
    aa = np.where(swap, b, a); bb = np.where(swap, a, b); tt = np.where(swap, -target, target)
    report.update(used=len(rows), max_anchor_distance=float(distance.max()))
    return dict(a=aa, b=bb, target=tt, weight=np.ones(len(rows))), dict(report)
