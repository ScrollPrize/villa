"""Score a winding solve against the crossing measurements themselves.

One row per measured crossing: no fractional expansion, no aggregation by
fiber pair, no clamping of the measured separation. The relation a crossing
asserts is ``k_b - k_a = target`` where the target is the measured winding
separation, corrected for the H/V offset and the seam transport
(``prepare_winding_graph(event_targets=True)``). The residual is reported two
ways:

* unrounded - against the measured target, so a mean or median away from zero
  is a real displacement of one group of fibers against another;
* rounded - against the target rounded to a whole turn, which is the relation
  the L1 problem can actually state. Its median is the headline: a nonzero
  median anywhere means a systematic winding shift.

Crossings are grouped by the winding the solve gives one side (``focus``), so
a drift that only affects part of the scroll is visible. ``focus`` names the
fiber UUIDs of interest (a manual scaffold, say); with it the eval keeps only
crossings between a focus fiber and a non-focus fiber, which is the question
"does this group sit where the rest of the volume says it should".
"""
import json
from pathlib import Path

import numpy as np

DISTANCE_CUTS = (0, 1, 2)


def crossing_events(config, settings, distance_cut=max(DISTANCE_CUTS), workers=32):
    """Every perpendicular crossing within `distance_cut` windings, as measured."""
    from vc_object_store import CollectionReader
    from . import fiber_wgraph_geometry as adapter
    from .fiber_packed_priors import _geometry
    with CollectionReader(str(config['database'])) as source:
        pinned = source.source
    graph, paths, ids, family, sign, _ = adapter._read_winding_inputs(
        config, pinned, adapter._input_lineage(config, settings=settings), stage='winding', modes=settings)
    distance = np.asarray(graph.distance, float)
    keep = np.flatnonzero(~np.asarray(graph.parallel) & np.isfinite(distance)
                          & (np.abs(np.round(distance)) <= distance_cut))
    prepared = _geometry(paths).prepare_winding_graph(
        np.ascontiguousarray(graph.i[keep]), np.ascontiguousarray(graph.j[keep]),
        np.ascontiguousarray(paths.graph_index, dtype=np.int64), family, sign,
        np.ascontiguousarray(graph.arc_a[keep]), np.ascontiguousarray(graph.arc_b[keep]),
        np.ascontiguousarray(distance[keep]), np.ascontiguousarray(graph.parallel[keep]),
        np.ascontiguousarray(graph.normal_a[keep]), np.ascontiguousarray(graph.normal_b[keep]),
        np.ascontiguousarray(graph.connector[keep]), np.ascontiguousarray(graph.valid_a[keep]),
        np.ascontiguousarray(graph.valid_b[keep]), np.ascontiguousarray(graph.valid_connector[keep]),
        parallel_weight=0., perpendicular_weight=1., event_targets=True, workers=workers)
    return dict(ids=np.array([str(u) for u in ids]),
                a=np.asarray(prepared['event_a'], np.int64), b=np.asarray(prepared['event_b'], np.int64),
                target=np.asarray(prepared['event_measured'], float),
                distance=np.asarray(prepared['event_distance'], float),
                weight=np.asarray(prepared['event_weight'], float))


def metrics(step, target):
    """The four reported numbers for one group of crossings."""
    unrounded = step - target
    rounded = step - np.round(target)
    return dict(crossings=int(len(step)),
                unrounded_mean=float(unrounded.mean()), unrounded_median=float(np.median(unrounded)),
                rounded_median=int(np.median(rounded)), rounded_exact_percent=float(100.0 * (rounded == 0).mean()))


def evaluate(events, winding, component, focus=(), largest_component_only=True):
    """Metrics per distance cut, and per winding for the tightest cut.

    `winding` and `component` map fiber UUID to the solve's winding number and
    component. `focus` is the set of fiber UUIDs whose placement is in
    question; crossings inside it, and crossings that avoid it, are skipped.
    """
    ids, a, b = events['ids'], events['a'], events['b']
    focus = set(str(u) for u in focus)
    placed = np.array([u in winding for u in ids])
    marked = np.array([u in focus for u in ids])
    main = np.ones(len(ids), bool)
    if largest_component_only and component:
        sizes = {}
        for fiber in ids[placed]:
            sizes[component[fiber]] = sizes.get(component[fiber], 0) + 1
        largest = max(sizes, key=lambda k: sizes[k]) if sizes else None
        main = np.array([component.get(u, None) == largest for u in ids])
    keep = placed[a] & placed[b] & main[a] & main[b]
    if focus:
        keep &= marked[a] ^ marked[b]
    turns = np.array([winding.get(u, 0) for u in ids], float)
    step = turns[b] - turns[a]
    report = dict(cuts={}, per_winding={})
    for cut in DISTANCE_CUTS:
        rows = np.flatnonzero(keep & (np.abs(np.round(events['distance'])) <= cut))
        if len(rows):
            report['cuts'][cut] = metrics(step[rows], events['target'][rows])
    rows = np.flatnonzero(keep & (np.abs(np.round(events['distance'])) <= DISTANCE_CUTS[0]))
    side = np.where(marked[a[rows]], a[rows], b[rows]) if focus else a[rows]
    bucket = turns[side].astype(np.int64)
    for value in np.unique(bucket):
        at = rows[bucket == value]
        report['per_winding'][int(value)] = metrics(step[at], events['target'][at])
    return report


def tables(report, title='crossings'):
    """The summary table, then the expanded per-winding one."""
    head = ('| crossings | unrounded mean | unrounded median | rounded median | rounded exact % |')
    def row(m):
        return (f"| {m['crossings']} | {m['unrounded_mean']:+.3f} | {m['unrounded_median']:+.3f} | "
                f"{m['rounded_median']:+d} | {m['rounded_exact_percent']:.1f} |")
    lines = [f'### {title}', '', '| distance cut ' + head, '| --- ' + '| ---: ' * 5 + '|']
    for cut, m in sorted(report['cuts'].items()):
        lines.append(f'| d<={cut} ' + row(m))
    lines += ['', '<details><summary>per winding, tightest cut</summary>', '',
              '| winding ' + head, '| ---: ' + '| ---: ' * 5 + '|']
    for value, m in sorted(report['per_winding'].items()):
        lines.append(f'| {value} ' + row(m))
    lines += ['', '</details>']
    return '\n'.join(lines)


def run(settings):
    """CLI entry: evaluate one or more result stores against one graph store."""
    from .fiber_wgraph_stages import _selected_run, _fiber_columns
    from . import fiber_wgraph_geometry as adapter
    config = dict(settings['winding'], database=str(settings['graph_store']))
    modes = adapter.winding_settings(config)
    events = crossing_events(config, modes, workers=settings.get('workers', 32))
    focus = settings.get('focus') or ()
    if isinstance(focus, str):
        focus = json.loads(Path(focus).read_text())['ids']
    out = {}
    for label, store in settings['solves'].items():
        run_uuid = _selected_run(str(store), 'winding', None)
        columns = _fiber_columns(str(store), run_uuid, None, required=('selected_winding', 'component'))
        winding = {str(u): int(w) for u, w in zip(columns['ids'], columns['selected_winding'])}
        component = {str(u): int(k) for u, k in zip(columns['ids'], columns['component'])}
        out[label] = evaluate(events, winding, component, focus=focus,
                              largest_component_only=settings.get('largest_component_only', True))
    if settings.get('output'):
        Path(settings['output']).write_text(json.dumps(out, indent=1))
    return out
