"""Identity scoring of a trace in an anisotropic tube around its annotation, and per-source summaries.

Papyrus fibers are strips: wide across the sheet, thin along its normal, and neighbours can stack within a few
CT voxels along the normal. Each trace vertex's offset from the annotation (projected on the annotation segments
around its bounded match, ``seeds.matched_profile``) is split along the tracer's own recorded frame: ``u`` (the
CT structure direction, the sheet normal) and ``v`` (the strip width). The frame is the current decision's while
the trace is in the tube and stays frozen at the last in-tube decision during an excursion, so off-fiber points
are measured in the fiber's orientation. Nothing is derived from the annotation's own orientation.

Per trace (all lengths in trace voxels, along the path):

- in the tube: |u| <= TUBE_NORMAL and |v| <= TUBE_WIDTH.
- annotation end: matched progress within END_PROGRESS of the annotation end while within END_DISTANCE of the
  fiber, or the endpoint crossing of ``seeds.trace_events``, whichever comes first. It does not depend on the
  tube. Later length is unscored behind an untagged end; behind a tagged physical end it is an endpoint overrun.
- excursion: a run out of the tube spanning >= EXCURSION_LENGTH. Counted, and wrong length while out.
- loss (identity failure): a run out of the tube spanning >= LOSS_LENGTH after which the trace is never back in
  the tube for RETURN_LENGTH before the scored end. At most one per trace. A 'switch' when, during the loss, the
  trace runs nearer another annotated fiber than its own (within SWITCH_DISTANCE) for >= SWITCH_LENGTH;
  otherwise 'lost'; 'unlabelled' without neighbour fibers.
- premature stop: the model stopped the trace in the tube with >= PREMATURE_REMAINING of annotation left.

Rates are per verified length (in-tube length before the annotation end). The legacy first-departure scores
(``seeds.score_trace``) are unchanged and still reported next to these.
"""
from __future__ import annotations

import re

import numpy as np

from .seeds import sustained_onsets, trace_events

TUBE_NORMAL = 1.5  # trace voxels (about 3 CT voxels at 9.6 um): the closest neighbour stacking along the normal
TUBE_WIDTH = 3.0  # about half the strip width
EXCURSION_LENGTH = 3.0
LOSS_LENGTH = 16.0
RETURN_LENGTH = 32.0
END_PROGRESS, END_DISTANCE = 3.0, 6.0
PREMATURE_REMAINING = 16.0
SWITCH_DISTANCE, SWITCH_LENGTH = 2.0, 4.0
# Stops the evaluation imposes; every other reason is the tracer's own decision.
EVALUATION_STOPS = ('annotation_end', 'oracle', 'max_len')
LOCALIZATION_BINS = np.linspace(0., 4., 161)
VERSION = 1


def settings():
    return dict(version=VERSION, tube_normal=TUBE_NORMAL, tube_width=TUBE_WIDTH, excursion_length=EXCURSION_LENGTH,
                loss_length=LOSS_LENGTH, return_length=RETURN_LENGTH, end_progress=END_PROGRESS,
                end_distance=END_DISTANCE, premature_remaining=PREMATURE_REMAINING,
                switch_distance=SWITCH_DISTANCE, switch_length=SWITCH_LENGTH)


def annotation_offsets(points, s, path, arc):
    """Offset of each path vertex from the annotation, projected on the two segments around its matched arc."""
    points, path = np.asarray(points, np.float64), np.asarray(path, np.float64)
    index = np.clip(np.searchsorted(s, arc), 0, len(s)-1)
    best = np.full((len(path), 3), np.inf)
    for start in (index-1, index):
        start = np.clip(start, 0, len(s)-2)
        origin, edge = points[start], points[start+1]-points[start]
        t = np.clip(np.einsum('ni,ni->n', path-origin, edge)/np.maximum(np.einsum('ni,ni->n', edge, edge), 1e-12), 0, 1)
        offset = path-(origin+t[:, None]*edge)
        closer = np.linalg.norm(offset, axis=1) < np.linalg.norm(best, axis=1)
        best[closer] = offset[closer]
    return best


def tube_components(offsets, decision, frames):
    """|u|, |v| and in-tube flags, with the frame frozen at the last in-tube decision during excursions."""
    u, v = np.zeros(len(offsets)), np.zeros(len(offsets))
    inside = np.zeros(len(offsets), bool)
    frame, previous_inside = frames[0], True
    bounds = np.r_[0, np.flatnonzero(np.diff(decision))+1, len(offsets)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        if previous_inside:
            frame = frames[decision[a]]
        u[a:b], v[a:b] = np.abs(offsets[a:b] @ frame[:, 0]), np.abs(offsets[a:b] @ frame[:, 1])
        inside[a:b] = (u[a:b] <= TUBE_NORMAL) & (v[a:b] <= TUBE_WIDTH)
        previous_inside = inside[b-1]
    return u, v, inside


def truncate_at(path, length):
    """Path vertices up to travelled ``length`` (a replayed earlier stop)."""
    path = np.asarray(path, np.float64)
    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    return path[:max(1, int(np.searchsorted(lengths, length+1e-9, side='right')))]


def score_tube(path, fiber, t0, sign, decision_travelled, frames, reason, *, foreign=None, stop_at=None):
    """Tube scores of one trace (module docstring).

    ``decision_travelled`` and ``frames`` are the tracer's recorded decision lengths and frames (columns u, v, f).
    ``foreign(points)`` returns each point's distance to the nearest other annotated fiber (``NeighborFibers``).
    ``stop_at`` replays an earlier model stop at that travelled length.
    """
    if stop_at is not None:
        path, reason = truncate_at(path, stop_at), 'confidence'
    path = np.asarray(path, np.float64)
    lengths, d, arc, _, crossing = trace_events(path, fiber, t0, sign)
    segment = np.diff(lengths, prepend=0.)
    decision_travelled = np.asarray(decision_travelled, np.float64)
    decision = np.clip(np.searchsorted(decision_travelled, lengths+1e-9, side='right')-1, 0, len(decision_travelled)-1)
    offsets = annotation_offsets(fiber.points, fiber.s, path, arc)
    u, v, inside = tube_components(offsets, decision, np.asarray(frames, np.float64))
    available = float(fiber.length-t0 if sign > 0 else t0)
    progress = (arc-t0)*sign
    cuts = list(np.flatnonzero((d <= END_DISTANCE) & (progress >= available-END_PROGRESS))[:1])
    if crossing is not None:
        cuts.append(int(np.searchsorted(lengths, crossing)))
    reached = min(cuts) if cuts else None
    scored = np.arange(len(lengths)) < (reached if reached is not None else len(lengths))
    endpoint_known = bool(fiber.endpoint_stop[1 if sign > 0 else 0])
    out = ~inside & scored
    verified = inside & scored
    excursions = sustained_onsets(lengths, out, EXCURSION_LENGTH)
    # A loss: the first long out-of-tube run never followed by a return before the scored end.
    returns = sustained_onsets(lengths, verified, RETURN_LENGTH)
    last_return = returns[-1] if returns else -1
    loss = next((o for o in sustained_onsets(lengths, out, LOSS_LENGTH) if o > last_return), None)
    kind = None
    if loss is not None:
        kind = 'unlabelled'
        if foreign is not None:
            lost = np.flatnonzero(out & (np.arange(len(lengths)) >= loss))
            near = np.zeros(len(lengths), bool)
            near[lost] = foreign(path[lost]) < np.minimum(d[lost], SWITCH_DISTANCE+1e-9)
            kind = 'switch' if sustained_onsets(lengths, near, SWITCH_LENGTH) else 'lost'
    first = excursions[0] if excursions else len(lengths)
    remaining = available-float(progress[-1])
    model_stop = reason not in EVALUATION_STOPS
    if reached is not None:
        end = 'end'
    elif inside[-1]:
        end = 'premature' if model_stop and remaining >= PREMATURE_REMAINING else ('end' if model_stop else 'censored')
    else:
        end = 'lost' if loss is not None else 'off'
    # On-fiber coverage of the annotation's own arclength (merged across both directions in the summary).
    covered = np.flatnonzero(verified[1:] & verified[:-1]
                             & (np.abs(np.diff(arc)) <= segment[1:]+2.))
    intervals = merge_intervals([(min(arc[i], arc[i+1]), max(arc[i], arc[i+1])) for i in covered])
    after = ~scored
    return dict(
        version=VERSION, verified_length=float(segment[verified].sum()), wrong_length=float(segment[out].sum()),
        uninterrupted_length=float(segment[verified & (np.arange(len(lengths)) < first)].sum()),
        excursions=len(excursions), loss=loss is not None, loss_at=float(lengths[loss]) if loss is not None else None,
        continued_after_loss=float(lengths[-1]-lengths[loss]) if loss is not None else 0.,
        scored_travelled=float(lengths[scored][-1]) if scored.any() else 0.,
        outcome=('mistake' if loss is not None else 'stopped_early' if end == 'premature'
                 else 'reached_end' if end == 'end' else 'other'),
        loss_kind=kind, end=end, reason=reason, remaining=remaining, available=available,
        reached_end=reached is not None, endpoint_known=endpoint_known,
        unscored_length=0. if endpoint_known else float(segment[after].sum()),
        endpoint_overrun=float(segment[after].sum()) if endpoint_known else 0.,
        total_length=float(lengths[-1]),
        normal_hist=np.histogram(u[verified], LOCALIZATION_BINS)[0].tolist(),
        width_hist=np.histogram(v[verified], LOCALIZATION_BINS)[0].tolist(),
        coverage_intervals=intervals)


def merge_intervals(intervals):
    merged = []
    for lo, hi in sorted(intervals):
        if merged and lo <= merged[-1][1]+1e-9:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([float(lo), float(hi)])
    return merged


def um_per_voxel(source_config):
    """Physical size of one trace voxel: native voxel size (config, else the CT store name) times grid_scale."""
    native = source_config.get('native_voxel_size_um')
    if native is None:
        match = re.search(r'-(\d+(?:\.\d+)?)um-', str(source_config.get('ct', '')))
        native = float(match.group(1)) if match else None
    return None if native is None else float(native)*float(source_config.get('grid_scale', 1.))


# Per-fiber sums; every summary metric is a ratio of these, so a fiber bootstrap resamples rows.
FIBER_SUMS = ('traces', 'verified', 'wrong', 'excursions', 'losses', 'switches', 'premature', 'reached', 'unscored',
              'overrun', 'total', 'coverage', 'reached_clean', 'other_outcome', 'continued', 'excursion_seeds',
              'available')


def outcome(tube):
    """Per-seed outcome (``score_tube``'s field; derived for rows scored before it existed)."""
    if 'outcome' in tube:
        return tube['outcome']
    return ('mistake' if tube['loss'] else 'stopped_early' if tube['end'] == 'premature'
            else 'reached_end' if tube['end'] == 'end' else 'other')


def fiber_table(rows):
    """Per-fiber sums (FIBER_SUMS order), localization histograms and fiber keys for one source's rows."""
    groups = {}
    for row in rows:
        groups.setdefault(row['fiber'], []).append(row)
    sums, normal, width = [], [], []
    for fiber, members in sorted(groups.items()):
        tubes = [r['tube'] for r in members]
        length = members[0]['fiber_length']
        covered = sum(hi-lo for lo, hi in merge_intervals([i for t in tubes for i in t['coverage_intervals']]))
        sums.append([len(tubes), sum(t['verified_length'] for t in tubes), sum(t['wrong_length'] for t in tubes),
                     sum(t['excursions'] for t in tubes), sum(t['loss'] for t in tubes),
                     sum(t['loss_kind'] == 'switch' for t in tubes), sum(t['end'] == 'premature' for t in tubes),
                     sum(t['reached_end'] for t in tubes), sum(t['unscored_length'] for t in tubes),
                     sum(t['endpoint_overrun'] for t in tubes), sum(t['total_length'] for t in tubes),
                     min(1., covered/max(length, 1e-9)),
                     sum(outcome(t) == 'reached_end' for t in tubes), sum(outcome(t) == 'other' for t in tubes),
                     sum(t.get('continued_after_loss', 0.) for t in tubes), sum(t['excursions'] > 0 for t in tubes),
                     sum(t['available'] for t in tubes)])
        normal.append(np.sum([t['normal_hist'] for t in tubes], 0))
        width.append(np.sum([t['width_hist'] for t in tubes], 0))
    return np.asarray(sums, float), np.asarray(normal, float), np.asarray(width, float), sorted(groups)


def group_metrics(sums, normal, width, um):
    """Metrics of summed per-fiber rows. Rates are per 10 mm when ``um`` is known, else per 1000 trace voxels."""
    v = dict(zip(FIBER_SUMS, sums))
    fibers = max(len(normal), 1)
    unit = 1e4/um if um else 1e3
    exposure = max(v['verified'], 1e-9)

    def quantile(hist, q):
        cdf = np.cumsum(hist)/max(hist.sum(), 1)
        value = LOCALIZATION_BINS[1:][min(np.searchsorted(cdf, q), len(cdf)-1)]
        return float(value*um) if um else float(value)

    n, w = normal.sum(0), width.sum(0)
    return dict(
        traces=v['traces'], verified_length=v['verified'],
        losses=v['losses'], switches=v['switches'], excursions=v['excursions'], premature_stops=v['premature'],
        losses_rate=v['losses']/exposure*unit, switches_rate=v['switches']/exposure*unit,
        distance_per_mistake=exposure/max(v['losses'], 1)*(um/1e3 if um else 1.),
        excursions_rate=v['excursions']/exposure*unit, premature_stops_rate=v['premature']/exposure*unit,
        precision=v['verified']/max(v['verified']+v['wrong'], 1e-9),
        wrong_per_verified=v['wrong']/exposure,
        normal_p50=quantile(n, .5), normal_p90=quantile(n, .9), width_p50=quantile(w, .5), width_p90=quantile(w, .9),
        fiber_coverage=v['coverage']/fibers, reached_end=v['reached']/max(v['traces'], 1),
        # Per seed (one directed trace), one outcome each: a mistake (identity loss, the trace kept going) takes
        # precedence over stopping early on the fiber, reaching the annotation end, or anything else.
        seeds_mistake=v['losses']/max(v['traces'], 1), seeds_stopped_early=v['premature']/max(v['traces'], 1),
        seeds_reached_end=v['reached_clean']/max(v['traces'], 1), seeds_other=v['other_outcome']/max(v['traces'], 1),
        seeds_any_excursion=v['excursion_seeds']/max(v['traces'], 1),
        continued_after_mistake=v['continued']/max(v['losses'], 1)*(um/1e3 if um else 1.),
        annotation_per_seed=v['available']/max(v['traces'], 1)*(um/1e3 if um else 1.),
        unscored_share=v['unscored']/max(v['total'], 1e-9), endpoint_overrun=v['overrun'])


def mistake_free(rows, distances):
    """Kaplan-Meier probability that a seed has traced each distance (trace voxels) without a mistake.

    Event: the identity loss, dated where it began (travelled length). A trace that stops, reaches the annotation
    end or is ended by the evaluation is censored at its last scored length. Returns (probability, traces still
    at risk) per distance; None where fewer than 20 traces are still at risk.
    """
    tubes = [r['tube'] for r in rows]
    times = np.asarray([t['loss_at'] if t['loss'] else t.get('scored_travelled', t['total_length']-t['unscored_length'])
                        for t in tubes])
    events = np.asarray([t['loss'] for t in tubes])
    order = np.lexsort((~events, times))  # events before censoring at equal times
    times, events = times[order], events[order]
    survival, at_risk, curve = 1., len(times), []
    for t, e in zip(times, events):
        if e:
            survival *= 1-1/at_risk
        at_risk -= 1
        curve.append((t, survival))
    out = []
    for d in distances:
        risk = int((times >= d).sum())
        before = [s for t, s in curve if t <= d]
        out.append((before[-1] if before else 1., risk) if risk >= 20 else None)
    return out


def summarize_tube(rows, um=None, boot=1000, seed=0):
    """Per-source metrics with fiber-cluster bootstrap 95% intervals: {metric: [value, low, high]}."""
    sums, normal, width, _ = fiber_table(rows)
    point = group_metrics(sums.sum(0), normal, width, um)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(boot):
        pick = rng.integers(0, len(sums), len(sums))
        draws.append(group_metrics(sums[pick].sum(0), normal[pick], width[pick], um))
    mm = [1., 2., 5., 10., 20.]
    voxels = [d*1e3/um for d in mm] if um else [d*1e3 for d in mm]
    lost_at = [r['tube']['loss_at'] for r in rows if r['tube']['loss']]
    return dict(unit='per 10 mm' if um else 'per 1000 trace voxels', um_per_voxel=um, fibers=len(sums),
                mistake_free=dict(zip([f'{d:g}mm' if um else f'{d:g}k_voxels' for d in mm], mistake_free(rows, voxels))),
                mistake_at_median=float(np.median(lost_at))*(um/1e3 if um else 1.) if lost_at else None,
                metrics={k: [value, *np.nanpercentile([d[k] for d in draws], [2.5, 97.5]).tolist()] if boot else [value]
                         for k, value in point.items()})


def compare_tube(rows_a, rows_b, um=None, boot=1000, seed=0):
    """Paired difference b - a on the traces both runs share (same source, fiber, direction and seed point)."""
    key = lambda r: (r['fiber'], r['sign'], round(float(r['t0']), 6))
    a, b = {key(r): r for r in rows_a}, {key(r): r for r in rows_b}
    shared = sorted(a.keys() & b.keys())
    sa, na, wa, fa = fiber_table([a[k] for k in shared])
    sb, nb, wb, fb = fiber_table([b[k] for k in shared])
    assert fa == fb
    point = {k: group_metrics(sb.sum(0), nb, wb, um)[k]-va for k, va in group_metrics(sa.sum(0), na, wa, um).items()}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(boot):
        pick = rng.integers(0, len(sa), len(sa))
        ma, mb = group_metrics(sa[pick].sum(0), na[pick], wa[pick], um), group_metrics(sb[pick].sum(0), nb[pick], wb[pick], um)
        draws.append({k: mb[k]-ma[k] for k in ma})
    return dict(shared_traces=len(shared), fibers=len(sa),
                metrics={k: [value, *np.nanpercentile([d[k] for d in draws], [2.5, 97.5]).tolist()] for k, value in point.items()})


def replay_stop(decision_travelled, gate_confidence, threshold, final_was_stop):
    """Travelled length where a higher confidence threshold would have stopped the trace, or None.

    ``final_was_stop``: the run's last decision was its own confidence stop (not an evaluation stop).
    The gate confidence is the running minimum over the gate planes at each recorded decision. Exact when proposal
    selection does not depend on the threshold; a flow model that takes the first sample passing the threshold
    could find another passing proposal, so replay overstates its stops.
    """
    below = np.flatnonzero(np.asarray(gate_confidence) < threshold)
    if not len(below) or (final_was_stop and below[0] == len(gate_confidence)-1):
        return None  # the run already stopped there
    return float(decision_travelled[below[0]])
