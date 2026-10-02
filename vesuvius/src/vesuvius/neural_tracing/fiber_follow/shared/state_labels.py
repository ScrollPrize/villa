"""The single per-state supervision contract for every training and evaluation source.

Three concepts stay separate:

- Historical events of one directed trace on its original fiber: the first sustained
  geometric departure (``DEPARTURE_DISTANCE`` voxels for ``DEPARTURE_PATIENCE`` committed
  points), the first certified foreign-fiber contact and the annotation-boundary crossing.
  Their locations persist after a geometric return.
- Correspondence: matched progress and distance on the original fiber only, searched from
  ``MATCH_BEHIND`` voxels behind the previous match to the newly committed travel plus
  ``MATCH_AHEAD`` ahead. It keeps updating after a departure and records validity and
  ambiguity. It never matches another fiber or winding.
- Current supervision: following, recoverable, terminal or unknown, with a reason and
  explicit geometry/confidence availability.

A geometric departure can become following again. A confirmed committed switch stays
terminal for the rest of its episode. Missing evidence censors supervision; it never
creates a positive continuation or a negative stop target. Annotation and neighbor-bank
geometry are supervision/evaluation oracles only; deployment relies on learned confidence.
"""
from __future__ import annotations

import numpy as np

from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength, tangent_at

DEPARTURE_DISTANCE = 3.0  # strict first-departure and displacement threshold, trace voxels
DEPARTURE_PATIENCE = 3  # consecutive committed points beyond DEPARTURE_DISTANCE
MATCH_BEHIND = 8.0
MATCH_AHEAD = 32.0
# A displaced head whose best window match has a rival this close in distance, but more
# than AMBIGUITY_ARC voxels away along the fiber, has no unique correspondence.
AMBIGUITY_ARC = 8.0
AMBIGUITY_MARGIN = 1.0

SUPERVISION = ('following', 'recoverable', 'terminal', 'unknown')
FOLLOWING, RECOVERABLE, TERMINAL, UNKNOWN = range(len(SUPERVISION))
REASONS = ('following', 'recoverable', 'switch', 'endpoint', 'unreachable', 'no_correspondence',
           'ambiguous', 'unannotated', 'unsupported_connection', 'identity')
REASON = {name: index for index, name in enumerate(REASONS)}
# Mutually exclusive replay membership in precedence order.
REPLAY_CLASSES = ('terminal', 'premature_stop', 'recoverable', 'pre_excursion', 'ordinary')
REPLAY_CLASS = {name: index for index, name in enumerate(REPLAY_CLASSES)}
NO_REPLAY_CLASS = -1
# Descriptive displacement strata only; they are never label rules.
DISPLACEMENT_STRATA = (('d<=3', 0., 3.), ('3<d<=6', 3., 6.), ('d>6', 6., np.inf))


def displacement_stratum(distance):
    if not np.isfinite(distance):
        return 'unknown'
    return next(name for name, _, hi in DISPLACEMENT_STRATA if distance <= hi)


def facts(*, match_distance, window_distance=None, match_valid=True, match_ambiguous=False,
          switched=False, beyond_end=False, tolerance, max_recovery_distance, **events):
    """Trace-level facts for one state; ``events`` carries the historical event record."""
    if not np.isfinite(tolerance) or tolerance <= 0 or not np.isfinite(max_recovery_distance) or max_recovery_distance <= 0:
        raise ValueError('Label tolerance and recovery limit must be finite and positive')
    return dict(match_distance=float(match_distance),
                window_distance=float(match_distance if window_distance is None else window_distance),
                match_valid=bool(match_valid), match_ambiguous=bool(match_ambiguous),
                switched=bool(switched), beyond_end=bool(beyond_end),
                tolerance=float(tolerance), max_recovery_distance=float(max_recovery_distance),
                identity_observable=events.pop('identity_observable', None), **events)


def first_plane_reachable(ab, plane, tolerance, max_distance):
    """Can any first point within the commit limit lie within tolerance of ``ab``?"""
    reach = np.sqrt(max(0., max_distance**2-plane**2))
    return float(np.linalg.norm(ab))-tolerance <= reach


def connection_certified(ab, plane, max_distance):
    """The annotated first crossing itself satisfies the origin-to-first-point limit."""
    return float(np.hypot(np.linalg.norm(ab), plane)) <= max_distance+1e-9


def _result(supervision, reason, geometry, confidence):
    return dict(supervision=np.int8(supervision), supervision_reason=np.int8(REASON[reason]),
                geometry_valid=bool(geometry), confidence_valid=bool(confidence),
                terminal=float(supervision == TERMINAL))


def classify(targets, trace):
    """Current supervision from relabeled targets and trace facts.

    ``targets`` are ``continuation_targets`` in the decision frame. The first predicted
    plane must be reachable under the commit limit; the original-fiber continuation is a
    geometry target only when its own first connection is within that limit. Foreign
    contact on that connection is checked later against the neighbor raster.
    """
    tol, limit = trace['tolerance'], trace['max_recovery_distance']
    displaced = not trace['match_distance'] <= DEPARTURE_DISTANCE
    if trace['switched']:
        result = _result(TERMINAL, 'switch', False, True)
    elif not trace['match_valid']:
        return _result(UNKNOWN, 'no_correspondence', False, False)
    elif trace['beyond_end']:
        # Past the annotation's end: an explicit physical endpoint is terminal; an
        # unannotated continuation is censored, however far the head has gone.
        if not targets['endpoint_known']:
            return _result(UNKNOWN, 'unannotated', False, False)
        result = _result(TERMINAL, 'endpoint', False, True)
    elif trace['window_distance'] > limit+tol:
        result = _result(TERMINAL, 'unreachable', False, True)
    elif trace['match_ambiguous']:
        return _result(UNKNOWN, 'ambiguous', False, False)
    elif not targets['plane_mask'][0]:
        end = float(np.asarray(targets['end_local'])[2])
        if targets['endpoint_known'] and end < float(targets['planes'][0]):
            result = _result(TERMINAL, 'endpoint', False, True)
        elif not targets['fmask'][0]:
            return _result(UNKNOWN, 'unannotated', False, False)
        else:
            return _result(UNKNOWN, 'unsupported_connection', False, False)
    else:
        ab, plane = np.asarray(targets['plane_ab'][0], np.float64), float(targets['planes'][0])
        if not first_plane_reachable(ab, plane, tol, limit):
            result = _result(TERMINAL, 'unreachable', False, True)
        elif not connection_certified(ab, plane, limit):
            # Some proposal may still be safe: score proposals, but teach no geometry.
            result = _result(UNKNOWN, 'unsupported_connection', False, True)
        else:
            result = _result(RECOVERABLE if displaced else FOLLOWING,
                             'recoverable' if displaced else 'following', True, True)
    # A displaced head needs visible original-fiber evidence for any supervision.
    if displaced and trace.get('identity_observable') is False:
        return _result(UNKNOWN, 'identity', False, False)
    return result


def supervise(item):
    """(Re)compute an item's supervision from its targets and ``trace_facts``."""
    item.update(classify(item, item['trace_facts']))
    item['match_distance'] = np.float32(item['trace_facts']['match_distance'])
    return item


def replay_class(row):
    """Membership with precedence terminal, premature stop, recoverable, pre-excursion, ordinary.

    Unknown states without usable partial labels have no class and never fill a quota.
    """
    supervision = int(row['supervision'])
    if supervision == TERMINAL:
        return REPLAY_CLASS['terminal']
    if not row['confidence_valid']:
        return NO_REPLAY_CLASS
    if row['would_stop'] and row['geometry_valid'] and supervision in (FOLLOWING, RECOVERABLE):
        return REPLAY_CLASS['premature_stop']
    if supervision == RECOVERABLE or (supervision == UNKNOWN and row['match_distance'] > DEPARTURE_DISTANCE):
        return REPLAY_CLASS['recoverable']
    if row.get('pre_excursion', False):
        return REPLAY_CLASS['pre_excursion']
    return REPLAY_CLASS['ordinary'] if supervision == FOLLOWING or row['confidence_valid'] else NO_REPLAY_CLASS


class TraceLabeler:
    """Original-fiber correspondence and historical events for one directed trace.

    ``t`` is the original fiber's own arclength (not traversal arclength). Every committed
    segment updates the bounded correspondence, departure patience, boundary crossing and,
    with a detector, the first certified foreign contact. State round-trips through
    ``state_dict`` so live chains and replay restarts resume exactly.
    """

    STATE = ('t', 'last_travelled', 'bad_run', 'bad_run_start', 'started', 'departure_distance',
             'boundary_distance', 'switch')

    def __init__(self, fiber, t, sign, *, tolerance, max_recovery_distance, fiber_idx=None, bank_detector=None,
                 state=None):
        self.fiber, self.sign, self.fiber_idx = fiber, float(sign), fiber_idx
        self.tolerance, self.max_recovery_distance = float(tolerance), float(max_recovery_distance)
        self.bank_detector = bank_detector
        self.t = float(t)
        self.last_travelled = 0.
        self.bad_run, self.bad_run_start, self.started = 0, None, False
        self.departure_distance = self.boundary_distance = None
        self.switch = None
        if state is not None:
            for key in self.STATE:
                setattr(self, key, state[key])
            self.t = float(self.t)

    def state_dict(self):
        return {key: getattr(self, key) for key in self.STATE}

    @property
    def end_tagged(self):
        return bool(self.fiber.endpoint_stop[0 if self.sign < 0 else 1])

    def _crossings(self, segment, nearest_s, advance):
        f, sign = self.fiber, self.sign
        endpoint = f.points[-1 if sign > 0 else 0]
        end_arc = f.length if sign > 0 else 0.
        tangent = tangent_at(f.points, f.s, end_arc)*sign
        residual = segment-endpoint
        beyond = residual @ tangent >= 0
        lateral = np.linalg.norm(residual-(residual @ tangent)[:, None]*tangent, axis=-1)
        # Crossing only counts after progressing near the actual annotation end.
        remaining = (end_arc-nearest_s)*sign
        return beyond & (lateral <= DEPARTURE_DISTANCE) & (remaining <= max(8., advance+2))

    def observe(self, segment, travelled):
        """Process one committed polyline ending at the new head; return its facts.

        ``segment`` starts at the previous head (or is the seed alone) and ``travelled``
        is the trace length at its end. Per-vertex distances are kept in
        ``vertex_distances`` for evaluation profiles.
        """
        f, sign = self.fiber, self.sign
        segment = np.asarray(segment, np.float64).reshape(-1, 3)
        advance = float(travelled)-self.last_travelled
        progress = (f.s-self.t)*sign
        indices = np.flatnonzero((progress >= -MATCH_BEHIND) & (progress <= advance+MATCH_AHEAD))
        along = self.last_travelled+arclength(segment)
        first = 1 if self.started else 0
        self.started = True
        if not len(indices):
            self.vertex_distances = np.full(len(segment)-first, np.nan)
            self.last_travelled = float(travelled)
            return self.facts(np.inf, np.inf, valid=False, ambiguous=False)
        distances = np.linalg.norm(segment[:, None]-f.points[indices][None], axis=-1)
        nearest = indices[distances.argmin(-1)]
        vertex = distances.min(-1)
        self.vertex_distances = vertex[first:]
        head = distances[-1]
        best = int(head.argmin())
        distance = float(head[best])
        ambiguous = False
        if distance > DEPARTURE_DISTANCE:
            cut = indices[best] not in (0, len(f.s)-1) and best in (0, len(indices)-1)
            rivals = (np.abs(f.s[indices]-f.s[indices[best]]) > AMBIGUITY_ARC) & (head <= distance+AMBIGUITY_MARGIN)
            ambiguous = bool(cut or rivals.any())
        for k in range(first, len(segment)):
            if vertex[k] <= DEPARTURE_DISTANCE:
                self.bad_run, self.bad_run_start = 0, None
                continue
            if not self.bad_run:
                self.bad_run_start = float(along[k])
            self.bad_run += 1
            if self.bad_run >= DEPARTURE_PATIENCE and self.departure_distance is None:
                self.departure_distance = self.bad_run_start
        if self.boundary_distance is None:
            crossing = np.flatnonzero(self._crossings(segment, f.s[nearest], advance)[first:])
            if len(crossing):
                self.boundary_distance = float(along[first+int(crossing[0])])
        if self.bank_detector is not None and self.switch is None:
            event = self.bank_detector.first_contact(self.fiber_idx, self.t, segment)
            if event is not None:
                start = float(travelled)-float(arclength(segment)[-1])
                self.switch = dict(switch_distance=start+event['distance'], switch_pos=np.asarray(event['pos']),
                                   switch_bank_path=event['bank_path'], switch_bank_run=event['bank_run'])
        self.t = float(f.s[indices[best]])
        self.last_travelled = float(travelled)
        return self.facts(distance, distance, valid=True, ambiguous=ambiguous)

    def facts(self, distance, window_distance, *, valid, ambiguous):
        switch = self.switch or {}
        return facts(match_distance=distance, window_distance=window_distance, match_valid=valid,
                     match_ambiguous=ambiguous, switched=self.switch is not None,
                     beyond_end=self.boundary_distance is not None,
                     tolerance=self.tolerance, max_recovery_distance=self.max_recovery_distance,
                     t=self.t, excursion=self.bad_run >= DEPARTURE_PATIENCE, bad_run=int(self.bad_run),
                     bad_run_start=np.nan if self.bad_run_start is None else float(self.bad_run_start),
                     departure_distance=np.nan if self.departure_distance is None else float(self.departure_distance),
                     boundary_distance=np.nan if self.boundary_distance is None else float(self.boundary_distance),
                     switch_distance=float(switch.get('switch_distance', np.nan)))


def constructed_facts(fiber, t, reverse, pos, cfg, **flags):
    """Facts for a state built on a known traversal arc ``t`` (fresh and synthetic sources).

    The correspondence is exact by construction; the window minimum still bounds
    reachability exactly as the collector's search does.
    """
    from .geometry import interp_at
    p, s = (fiber.points[::-1], fiber.length-fiber.s[::-1]) if reverse else (fiber.points, fiber.s)
    matched = interp_at(p, s, np.array([t]))[0]
    window = p[(s >= t-MATCH_BEHIND) & (s <= t+MATCH_AHEAD)]
    window = np.concatenate((window, matched[None]))
    pos = np.asarray(pos, np.float64)
    return facts(match_distance=float(np.linalg.norm(pos-matched)),
                 window_distance=float(np.linalg.norm(window-pos, axis=1).min()),
                 tolerance=cfg.label_tolerance, max_recovery_distance=cfg.max_recovery_distance,
                 t=float(fiber.length-t if reverse else t), excursion=False, bad_run=0, bad_run_start=np.nan,
                 departure_distance=np.nan, boundary_distance=np.nan, switch_distance=np.nan, **flags)
