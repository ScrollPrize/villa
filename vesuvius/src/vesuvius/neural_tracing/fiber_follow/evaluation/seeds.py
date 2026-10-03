"""Seeded tracing evaluation against held-out GT fibers."""

from __future__ import annotations


import numpy as np
import warnings
from scipy.spatial import cKDTree

from vesuvius.neural_tracing.fiber_follow.data.data import DATA_VERSION, TracedFiber
from vesuvius.neural_tracing.fiber_follow.shared.geometry import interp_at, tangent_at
from vesuvius.neural_tracing.fiber_follow.tracing.heading import oriented_seed_heading, SeedHeadingError, SEED_HEADING_POLICY
from vesuvius.neural_tracing.fiber_follow.tracing.crop_frames import trace_family_kwargs


def make_seeds(fibers: list[TracedFiber], vol, per_fiber: int = 3,
               margin: float = 32.0, seed: int = 0):
    """Annotated positions, CT/HV axes; one entry per (seed, direction).

    Annotation chooses only the scoring sign of the estimated axis. Missing
    CT orientation is reported and skipped, never replaced by a GT tangent.
    """
    rng = np.random.default_rng(seed)
    out = []
    for fi, f in enumerate(fibers):
        if f.length < 2 * margin:
            continue
        ts = np.arange(margin, f.length - margin, 4.0)
        good = ts
        if len(good) == 0:
            continue
        for t in rng.choice(good, size=min(per_fiber, len(good)), replace=False):
            p = interp_at(f.points, f.s, np.array([t]))[0]
            tau = tangent_at(f.points, f.s, t)
            try:
                ax = oriented_seed_heading(vol, p, f.tag, tau)
            except SeedHeadingError as error:
                warnings.warn(f'Skipping seed for {f.name} at {p.tolist()}: {error}', stacklevel=2)
                continue
            for sgn in (1.0, -1.0):
                h = sgn*ax
                out.append(dict(fiber=fi, t=float(t), sign=sgn, pos=p, heading=h,
                                family=f.tag, seed_heading_policy=SEED_HEADING_POLICY))
    return out


def directed_seed(fiber: TracedFiber, vol, rng, sign: float, margin: float = 32.0):
    """One annotated position and its CT/HV axis in one traversal direction.

    Returns None for a fiber too short for the margin. Missing CT orientation raises
    ``SeedHeadingError``; the caller skips that fiber with an explicit reason.
    """
    if fiber.length < 2*margin:
        return None
    ts = np.arange(margin, fiber.length-margin, 4.0)
    t = float(ts[int(rng.integers(len(ts)))])
    p = interp_at(fiber.points, fiber.s, np.array([t]))[0]
    axis = oriented_seed_heading(vol, p, fiber.tag, tangent_at(fiber.points, fiber.s, t))
    return dict(t=t, sign=float(sign), pos=p, heading=float(sign)*axis, family=fiber.tag,
                seed_heading_policy=SEED_HEADING_POLICY)


def trace_events(path, fiber, t0, sign, tol=3.0, patience=3, tree=None):
    """First sustained departure and first supported endpoint-plane crossing.

    Endpoint crossing is checked on segments, not just nearest-point indices,
    so the crossing segment can be partitioned exactly. Returns cumulative
    path lengths, GT distances/arcs, departure index, and endpoint path length.
    """
    path = np.asarray(path, dtype=np.float64).reshape(-1, 3)
    tree = tree if tree is not None else cKDTree(fiber.points)
    d, j = tree.query(path)
    arc = fiber.s[j]
    lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))] if len(path) else np.zeros(0)
    bad = d > tol
    run = np.convolve(bad.astype(int), np.ones(patience, int), mode="full")[:len(bad)] if len(bad) else np.zeros(0)
    fail = np.flatnonzero(run >= patience)
    end = max(int(fail[0]) - patience + 1, 0) if len(fail) else len(path)
    endpoint = fiber.points[-1 if sign > 0 else 0]
    tangent = tangent_at(fiber.points, fiber.s, fiber.length if sign > 0 else 0.0) * sign
    forward = (path - endpoint) @ tangent
    crossing = None
    # A seed at the endpoint has no annotated continuation.
    if len(path) and abs((fiber.length - t0) if sign > 0 else t0) < 1e-9 and d[0] <= tol:
        crossing = 0.0
    for i in np.flatnonzero((forward[:-1] < 0) & (forward[1:] >= 0)):
        if i + 1 > end:
            break  # identity was lost before reaching the endpoint
        # A nearby winding can cross the endpoint plane long before the GT
        # traversal reaches its end. Require plausible along-fiber progress.
        remaining = fiber.length - arc[i] if sign > 0 else arc[i]
        if remaining > lengths[i + 1] - lengths[i] + 2 * tol:
            continue
        alpha = -forward[i] / (forward[i + 1] - forward[i])
        hit = path[i] + alpha * (path[i + 1] - path[i])
        if np.linalg.norm(hit - endpoint) <= tol:
            crossing = float(lengths[i] + alpha * (lengths[i + 1] - lengths[i]))
            break
    return lengths, d, arc, end, crossing


def score_trace(path: np.ndarray, fiber: TracedFiber, t0: float, sign: float, tol: float = 3.0,
                patience: int = 3, tree: cKDTree | None = None):
    """Partition every segment into supported, wrong, or unknown length.

    An untagged annotation endpoint censors subsequent continuation. A tagged
    physical endpoint makes subsequent length an endpoint overrun. A departure
    before that boundary remains wrong even if the path later returns to GT.
    """
    lengths, d, arc, end, crossing = trace_events(path, fiber, t0, sign, tol, patience, tree)
    total = float(lengths[-1]) if len(lengths) else 0.0
    avail = float((fiber.length - t0) if sign > 0 else t0)
    prog = (arc[:end] - t0) * sign
    followed = min(float(max(prog.max(), 0.0)) if len(prog) else 0.0, avail)
    endpoint_known = fiber.endpoint_stop[0 if sign < 0 else 1]
    unknown = overrun = 0.0
    diverged = end < len(d) and crossing is None
    if crossing is not None:
        correct = crossing
        followed = avail
        if endpoint_known:
            overrun = total - correct
            offtrack = overrun
        else:
            unknown = total - correct
            offtrack = 0.0
    elif diverged:
        # Include the transition from the last supported point to the first bad point.
        correct = float(lengths[max(0, end - 1)])
        offtrack = total - correct
    else:
        correct, offtrack = total, 0.0
    err_mask = np.arange(len(d)) < end
    if crossing is not None:
        err_mask &= lengths <= crossing
    err = float(d[err_mask].mean()) if err_mask.any() else float("nan")
    return dict(followed=followed, avail=avail, coverage=followed / max(avail, 1e-6),
                correct=correct, offtrack=offtrack, unknown=unknown, endpoint_overrun=overrun,
                scored_length=correct + offtrack, length=total,
                precision=correct / max(correct + offtrack, 1e-6),
                verified_fraction=correct / max(total, 1e-6),
                unknown_fraction=unknown / max(total, 1e-6),
                reached_end=bool(followed >= avail - tol), crossed_endpoint=crossing is not None,
                endpoint_known=bool(endpoint_known), diverged=bool(diverged), err=err)


def monitor_coverage(row, max_len):
    """Normalize monitor coverage to the trace budget, shared by both trainers.

    Keep all precision, departure and endpoint scoring from score_trace.
    Calibration/final evaluation retains its full-annotation denominator.
    """
    if not np.isfinite(max_len) or max_len <= 0:
        raise ValueError('Monitor length must be finite and positive')
    row = dict(row)
    row['avail'] = min(row['avail'], max_len)
    row['followed'] = min(row['followed'], max_len)
    row['coverage'] = row['followed'] / max(row['avail'], 1e-6)
    return row


def evaluate(tracer, fibers, seeds, batch: int = 256, history_audit=None, on_trace=None,
             coverage_max_len=None):
    """Score rollouts; optionally apply monitor normalization or expose paths."""
    trees = {}
    rows = []
    for b in range(0, len(seeds), batch):
        chunk = seeds[b:b + batch]
        kwargs = trace_family_kwargs(tracer, [s.get('family', fibers[s['fiber']].tag) for s in chunk])
        if history_audit is not None:
            history_audit.start_batch(fibers, chunk)
            kwargs['on_decision'] = history_audit
        paths, reasons = tracer.trace(np.stack([s["pos"] for s in chunk]), np.stack([s["heading"] for s in chunk]), **kwargs)
        for position, (s, p, r) in enumerate(zip(chunk, paths, reasons)):
            f = fibers[s["fiber"]]
            if s["fiber"] not in trees:
                trees[s["fiber"]] = cKDTree(f.points)
            m = score_trace(p, f, s["t"], s["sign"], tree=trees[s["fiber"]])
            if coverage_max_len is not None:
                m = monitor_coverage(m, coverage_max_len)
            span = next((span for span in f.spans if span.start <= s["t"] <= span.end), None)
            m.update(reason=r, fiber=s["fiber"], fiber_name=f.name,
                     seed_span_mode=span.provenance.interp_mode if span else None,
                     t0=s["t"], sign=s["sign"])
            if history_audit is not None and hasattr(history_audit, 'outcomes'):
                m.update(history_audit.outcomes(position, p, r))
            rows.append(m)
            if on_trace is not None:
                on_trace(s, p, r)
    return rows, summarize(rows)


def summarize(rows):
    if not rows:
        raise ValueError("No traces to evaluate; check the controlled spans and seed selection")
    cov = np.array([r["coverage"] for r in rows])
    fol = np.array([r["followed"] for r in rows])
    avail = np.array([r["avail"] for r in rows])
    correct = sum(r["correct"] for r in rows)
    wrong = sum(r["offtrack"] for r in rows)
    unknown = sum(r["unknown"] for r in rows)
    total = sum(r["length"] for r in rows)
    errors = np.array([r["err"] for r in rows])
    return dict(
        evaluation_version=DATA_VERSION, n=len(rows),
        coverage_mean=float(cov.mean()), coverage_median=float(np.median(cov)),
        length_weighted_coverage=float(fol.sum() / max(avail.sum(), 1e-6)),
        reached_end=float(np.mean([r["reached_end"] for r in rows])),
        diverged=float(np.mean([r["diverged"] for r in rows])),
        followed_median=float(np.median(fol)),
        offtrack_mean=wrong / len(rows), wrong_len_mean=wrong / len(rows),
        err_mean=float(errors[np.isfinite(errors)].mean()) if np.isfinite(errors).any() else float("nan"),
        precision_mean=float(np.mean([r["precision"] for r in rows])),
        length_precision=correct / max(correct + wrong, 1e-6),
        correct_length=correct, wrong_length=wrong, unknown_length=unknown,
        total_length=total, scored_length=correct + wrong,
        verified_length_fraction=correct / max(total, 1e-6),
        unknown_length_fraction=unknown / max(total, 1e-6),
        endpoint_overrun_length=sum(r["endpoint_overrun"] for r in rows),
        known_endpoint_traces=sum(r["endpoint_known"] for r in rows),
    )


RETURN_DISTANCE = 2.0  # a geometric return: at most this far for RETURN_LENGTH voxels of travel
RETURN_LENGTH = 32.0
DISTANCE_EVENT = 6.0


def distance_profile(path, fiber, t0, sign, chunk=8):
    """Per-vertex distance to the original fiber under the shared bounded correspondence."""
    from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength
    from vesuvius.neural_tracing.fiber_follow.data.state_labels import TraceLabeler
    path = np.asarray(path, np.float64).reshape(-1, 3)
    labeler = TraceLabeler(fiber, t0, sign, tolerance=1., max_recovery_distance=1.)
    arc = arclength(path) if len(path) > 1 else np.zeros(len(path))
    distances = []
    labeler.observe(path[:1], 0.)
    distances.extend(labeler.vertex_distances)
    for start in range(0, len(path)-1, chunk):
        end = min(len(path)-1, start+chunk)
        labeler.observe(path[start:end+1], float(arc[end]))
        distances.extend(labeler.vertex_distances)
    return arc, np.asarray(distances)


def sustained_onsets(bad, patience=3):
    """Start indices of runs of at least ``patience`` consecutive True values."""
    onsets, run = [], 0
    for index, value in enumerate(bad):
        run = run+1 if value else 0
        if run == patience:
            onsets.append(index-patience+1)
    return onsets


def returned_after(arc, distance, start):
    """Travel at which the trace stays within RETURN_DISTANCE for RETURN_LENGTH voxels, or None."""
    begin = None
    for index in range(start, len(distance)):
        if distance[index] <= RETURN_DISTANCE:
            begin = index if begin is None else begin
            if arc[index]-arc[begin] >= RETURN_LENGTH:
                return float(arc[begin])
        else:
            begin = None
    return None


def geometric_outcomes(path, fiber, t0, sign, *, tolerance=3.0):
    """Excursions/returns, distance events and current geometric agreement of one trace.

    The strict first-departure metric stays in ``score_trace``; a later return never
    erases it. Distance events (sustained > DISTANCE_EVENT) are not identity failures.
    """
    arc, distance = distance_profile(path, fiber, t0, sign)
    segment = np.diff(arc, prepend=0.)
    finite = np.isfinite(distance)
    agreement = float(segment[finite & (distance <= tolerance)].sum())
    outcome = dict(geometric_agreement=agreement, excursions=0, excursion_returns=0,
                   distance_events=0, distance_event_returns=0, distance_events_ended=0)
    for name, threshold in (('excursion', tolerance), ('distance_event', DISTANCE_EVENT)):
        bad = ~(distance <= threshold)
        cursor = 0
        while True:
            onsets = [o for o in sustained_onsets(bad[cursor:]) if o >= 0]
            if not onsets:
                break
            onset = cursor+onsets[0]
            outcome[name+'s'] += 1
            back = returned_after(arc, distance, onset)
            if back is None:
                if name == 'distance_event' and arc[-1]-arc[onset] < RETURN_LENGTH:
                    outcome['distance_events_ended'] += 1
                break
            outcome[name+'_returns'] += 1
            cursor = int(np.searchsorted(arc, back+RETURN_LENGTH))
            if cursor >= len(arc):
                break
    return outcome


class EvaluationAudit:
    """Decision-level outcomes under the shared state contract, without affecting the trace.

    Counts rejected unsafe proposals, stops with a supported continuation, recovery commits
    that cross a certified foreign fiber, and confirmed switches with the length accepted
    after them. Neighbor-bank coverage is reported because absent contact is not proof
    that no foreign fiber exists.
    """
    def __init__(self, tracer, tolerance=1.5, detector=None):
        from vesuvius.neural_tracing.fiber_follow.data.data import SampleConfig
        cfg = tracer.model.cfg
        self.cfg = SampleConfig(crop=tracer.crop, n_history=tracer.n_history, recent_history_points=cfg.recent_history_points,
                                n_future=cfg.n_future, future_step=cfg.future_step, label_tolerance=tolerance,
                                max_recovery_distance=cfg.max_recovery_distance)
        self.tolerance, self.detector = tolerance, detector

    def start_batch(self, fibers, seeds):
        from vesuvius.neural_tracing.fiber_follow.data.state_labels import TraceLabeler
        self.seeds = list(seeds)
        self.labelers = [TraceLabeler(fibers[s['fiber']], s['t'], s['sign'], tolerance=self.tolerance,
                                      max_recovery_distance=self.cfg.max_recovery_distance, fiber_idx=s['fiber'],
                                      bank_detector=self.detector) for s in seeds]
        self.records = [dict(decisions=0, rejected_unsafe=0, rejected_safe=0, accepted_unsafe=0, premature_stop=0,
                             stop_unknown=0, terminal_stop=0, recovery_foreign_contacts=0, recovery_commits=0,
                             displaced_previous=False) for _ in seeds]

    def __call__(self, index, state):
        import torch
        from vesuvius.neural_tracing.fiber_follow.data.data import label_state
        from vesuvius.neural_tracing.fiber_follow.data.labels import prefix_labels
        from vesuvius.neural_tracing.fiber_follow.data.state_labels import FOLLOWING, RECOVERABLE, TERMINAL
        labeler, record = self.labelers[index], self.records[index]
        switched_before = labeler.switch is not None
        facts = labeler.observe(state['last_segment'], state['travelled'])
        if labeler.switch is not None and not switched_before and record['displaced_previous']:
            record['recovery_foreign_contacts'] += 1
        item = label_state(labeler.fiber, state['pos'], state['frame'], state['hist'], state['hmask'], self.cfg,
                           t=facts['t'], reverse=labeler.sign < 0, trace=facts)
        tensor = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)[None]
        batch = {k: tensor(item[k]) for k in ('dense_ab', 'dense_mask', 'endpoint_known', 'end_local', 'terminal',
                                              'confidence_valid')}
        labels, known, _ = prefix_labels(tensor(state['points']), batch, self.tolerance, self.cfg.max_recovery_distance)
        commit = int(state['n_commit'])
        index_ = max(0, commit-1)
        safe_known, safe = bool(known[0, 0]), bool(labels[0, 0])
        supported = item['geometry_valid'] and int(item['supervision']) in (FOLLOWING, RECOVERABLE)
        record['decisions'] += 1
        displaced = facts['match_distance'] > 3.
        if state['would_stop']:
            if int(item['supervision']) == TERMINAL:
                record['terminal_stop'] += 1
            elif supported:
                record['premature_stop'] += 1
            else:
                record['stop_unknown'] += 1
            if safe_known:
                record['rejected_safe' if safe else 'rejected_unsafe'] += 1
        else:
            if bool(known[0, index_]) and not bool(labels[0, index_]):
                record['accepted_unsafe'] += 1
            if displaced:
                record['recovery_commits'] += 1
        record['displaced_previous'] = displaced and not state['would_stop']

    def outcomes(self, index, path, reason):
        from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength
        labeler, record = self.labelers[index], dict(self.records[index])
        record.pop('displaced_previous')
        path = np.asarray(path, np.float64).reshape(-1, 3)
        arc = arclength(path) if len(path) > 1 else np.zeros(len(path))
        total = float(arc[-1]) if len(arc) else 0.
        tail = arc > labeler.last_travelled+1e-6
        if self.detector is not None and labeler.switch is None and tail.any() and reason != 'oracle':
            # The final commit had no following decision; check its contact too.
            first = int(np.flatnonzero(tail)[0])
            labeler.observe(path[max(0, first-1):], total)
        switch = labeler.switch
        record.update(confirmed_switch=switch is not None,
                      length_after_switch=max(0., total-switch['switch_distance']) if switch else 0.,
                      identity_coverage=bool(self.detector is not None and any(
                          bank.spatial_records(labeler.fiber_idx, np.asarray(path), radius=8.)
                          for bank in self.detector.banks)),
                      **geometric_outcomes(path, labeler.fiber, self.seeds[index]['t'], labeler.sign))
        return record


OUTCOME_COUNTS = ('decisions', 'rejected_unsafe', 'rejected_safe', 'accepted_unsafe', 'premature_stop', 'stop_unknown',
                  'terminal_stop', 'recovery_foreign_contacts', 'recovery_commits', 'confirmed_switch',
                  'identity_coverage', 'excursions', 'excursion_returns', 'distance_events', 'distance_event_returns',
                  'distance_events_ended')
OUTCOME_LENGTHS = ('length_after_switch', 'geometric_agreement')


def summarize_outcomes(rows):
    """Totals of the decision and geometric outcomes, next to the first-failure summary."""
    from vesuvius.neural_tracing.fiber_follow.shared.experiment import rollout_summary
    result = rollout_summary(rows)
    for key in OUTCOME_COUNTS+OUTCOME_LENGTHS:
        if rows and key in rows[0]:
            result[key] = float(sum(float(r[key]) for r in rows))
    return result
