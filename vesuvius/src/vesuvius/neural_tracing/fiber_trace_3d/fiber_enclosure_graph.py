"""Exact whole-fiber relations implied by signed perpendicular enclosures.

No Lasagna winding-distance calibration or fitted winding assignment is used.
The ordinary measurements are immutable; derived relations retain both witnesses.
"""
import numpy as np


def extract_relations(d, paths):
    a, ta = paths.ends('a'); b, tb = paths.ends('b')
    connector = paths.seam.segments(a, b)
    groups = {}
    for e, (i, j) in enumerate(zip(d['i'], d['j'])):
        if d['parallel'][e] or d['family'][i] == d['family'][j] or not d['sign'][e]:
            continue
        forward = d['family'][i] == 0
        h, v = (int(i), int(j)) if forward else (int(j), int(i))
        sign = int(d['sign'][e]) * (1 if forward else -1)
        shift = int(connector[e] - tb[e] + ta[e]) * (1 if forward else -1)
        # Outward H->V allows zero; inward H->V requires at least one turn.
        bound = shift if sign > 0 else shift - 1
        groups.setdefault((h, v), []).append((sign, bound, e))
    relations = []; rejected = []
    for (h, v), rows in sorted(groups.items()):
        lower = [(bound, e) for sign, bound, e in rows if sign > 0]
        upper = [(bound, e) for sign, bound, e in rows if sign < 0]
        if not lower or not upper:
            rejected.append(dict(h=h, v=v, reason='one_sided')); continue
        lo, le = max(lower); hi, ue = min(upper)
        if lo != hi:
            rejected.append(dict(h=h, v=v, lower=lo, upper=hi,
                                 reason='contradictory' if lo > hi else 'nonunique'))
            continue
        relations.append(dict(h=h, v=v, anchor_step=lo, local_step=0,
                              type='perpendicular_enclosure', lower_event=le,
                              upper_event=ue, supporting_events=len(rows)))
    return relations, rejected


def graph_output(ordinary, relations, mode='only'):
    """Select ordinary / additional / only derived tagged edges for a solver."""
    if mode not in ('ordinary', 'additional', 'only'):
        raise ValueError('Unknown enclosure extraction mode')
    out = {k: np.asarray(v).copy() for k, v in ordinary.items()}
    count = len(out['a'])
    out['constraint_type'] = np.full(count, 'ordinary', dtype='<U24')
    out['lower_event'] = np.full(count, -1, dtype=np.int64)
    out['upper_event'] = np.full(count, -1, dtype=np.int64)
    if mode == 'ordinary': return out
    if mode == 'only': out = {k: v[:0] for k, v in out.items()}
    added = dict(a=[r['h'] for r in relations], b=[r['v'] for r in relations],
                 target=[r['anchor_step'] for r in relations], weight=np.ones(len(relations)),
                 one_sided=np.zeros(len(relations), bool),
                 constraint_type=['perpendicular_enclosure'] * len(relations),
                 lower_event=[r['lower_event'] for r in relations],
                 upper_event=[r['upper_event'] for r in relations])
    return {k: np.concatenate((v, np.asarray(added[k], dtype=v.dtype))) for k, v in out.items()}


def components(n, relations):
    """Integer propagation is a consistency audit, not conflict suppression."""
    adjacency = [[] for _ in range(n)]
    for e, r in enumerate(relations):
        a, b, step = r['h'], r['v'], r['anchor_step']
        adjacency[a].append((b, step, e)); adjacency[b].append((a, -step, e))
    labels = np.full(n, -1, np.int64); potential = np.zeros(n, np.int64)
    groups = []; conflicts = set()
    for seed in range(n):
        if labels[seed] >= 0: continue
        cid = len(groups); labels[seed] = cid; queue = [seed]
        for a in queue:
            for b, step, e in adjacency[a]:
                value = potential[a] + step
                if labels[b] < 0:
                    labels[b] = cid; potential[b] = value; queue.append(b)
                elif potential[b] != value: conflicts.add(e)
        groups.append(queue)
    return labels, potential, groups, sorted(conflicts)
