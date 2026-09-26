"""Versioned observed-path departure labels, independent of forecast confidence."""
from dataclasses import dataclass, asdict
import numpy as np
from .geometry import arclength, interp_at

EVENT_VERSION = 'observed_departure_v1'


@dataclass(frozen=True)
class EventConfig:
    grid: float = .5
    distance: float = 3.5
    sustain: float = 3.
    backward: float = 8.
    forward: float = 32.


class DepartureEvents:
    """Streaming segment processor. Grid phase and endpoint events survive commits."""
    def __init__(self, annotation, q0, physical_end=False, cfg=None):
        self.cfg = cfg or EventConfig()
        self.annotation = np.asarray(annotation, float)
        self.q = arclength(self.annotation)
        if len(self.q) < 2 or self.q[-1] <= 0 or not 0 <= q0 <= self.q[-1]:
            raise ValueError('Invalid annotation correspondence')
        self.h, self.s_good = float(q0), 0.
        self.physical_end = bool(physical_end)
        nonzero = np.flatnonzero(np.diff(self.q) > 0)
        self.tangent = np.diff(self.annotation, axis=0)[nonzero[-1]] / np.diff(self.q)[nonzero[-1]]
        self.last = None
        self.arc = 0.
        self.next_grid = 0
        self.run = self.onset = self.confirmation = self.censored = None
        self.bracket = None
        self.kind = None
        self.invalid_seed = False
        self.samples = []

    def correspondence(self, point, arc):
        lo, hi = max(0., self.h-self.cfg.backward), min(self.q[-1], self.h+arc-self.s_good+self.cfg.forward)
        a, b = np.maximum(self.q[:-1], lo), np.minimum(self.q[1:], hi)
        valid = (b >= a) & (np.diff(self.q) > 0)
        if not valid.any():
            return None
        direction = np.diff(self.annotation, axis=0)[valid] / np.diff(self.q)[valid, None]
        start = self.annotation[:-1][valid] + (a[valid]-self.q[:-1][valid])[:, None]*direction
        t = np.clip(np.sum((point-start)*direction, axis=1), 0, b[valid]-a[valid])
        distances = np.linalg.norm(point-start-t[:, None]*direction, axis=1)
        qs = a[valid]+t
        index = np.lexsort((qs, distances))[0]
        return float(qs[index]), float(distances[index])

    def _observe(self, point, arc):
        if self.censored is not None or self.invalid_seed:
            return
        matched = self.correspondence(point, arc)
        if matched is None:
            self.censored = arc
            return
        q, d = matched
        self.samples.append((float(arc), q, d))
        if d <= self.cfg.distance:
            self.h, self.s_good = max(self.h, q), arc
            if self.onset is None:
                self.run = None
        elif arc == 0:
            self.invalid_seed = True
        elif self.onset is None:
            if self.run is None:
                self.run = arc
            if arc-self.run >= self.cfg.sustain-1e-9:
                self.onset, self.confirmation = self.run, arc
                self.bracket = (max(0., self.run-self.cfg.grid), self.run)
                self.kind = 'departure'

    def append(self, points, valid=None):
        points = np.asarray(points, float)
        valid = np.ones(len(points), bool) if valid is None else np.asarray(valid, bool)
        for point, ok in zip(points, valid):
            if not ok or not np.isfinite(point).all():
                if self.censored is None and self.onset is None:
                    self.censored = self.arc
                continue
            if self.last is None:
                self.last = point.copy()
                self._observe(point, 0.)
                self.next_grid = 1
                continue
            length = float(np.linalg.norm(point-self.last))
            if length <= 1e-12:
                continue
            end = self.arc+length
            crossing = None
            a, b = (self.last-self.annotation[-1]) @ self.tangent, (point-self.annotation[-1]) @ self.tangent
            if a <= 0 < b:
                t = -a/(b-a)
                location = self.last+t*(point-self.last)
                crossing = self.arc+t*length
                if np.linalg.norm(location-self.annotation[-1]) > self.cfg.distance:
                    crossing = None
            while self.next_grid*self.cfg.grid <= end+1e-9:
                s = self.next_grid*self.cfg.grid
                if crossing is not None and crossing <= s+1e-9:
                    self._cross(crossing)
                    crossing = None
                self._observe(self.last+(point-self.last)*((s-self.arc)/length), s)
                self.next_grid += 1
            if crossing is not None:
                self._cross(crossing)
            self.arc, self.last = end, point.copy()
        return self

    def _cross(self, arc):
        if self.onset is not None or self.censored is not None or self.invalid_seed:
            return
        if self.q[-1]-self.h > arc-self.s_good+self.cfg.backward:
            return
        if self.physical_end:
            self.onset = self.confirmation = arc
            self.bracket, self.kind = (arc, arc), 'endpoint_overrun'
        else:
            self.censored = arc

    def labels(self, arcs):
        arcs = np.asarray(arcs, float)
        target, known = np.ones(arcs.shape, np.float32), np.ones(arcs.shape, bool)
        known &= (arcs >= 0) & (arcs <= self.arc+1e-7)
        if self.invalid_seed:
            known[:] = False
        elif self.onset is not None:
            target[arcs >= self.onset] = 0
            known[(arcs > self.bracket[0]) & (arcs < self.bracket[1])] = False
        else:
            if self.run is not None:
                known[arcs > max(0., self.run-self.cfg.grid)] = False
            if self.censored is not None:
                known[arcs >= self.censored] = False
        return target, known

    def partition(self, end=None):
        end = self.arc if end is None else min(float(end), self.arc)
        boundaries = sorted(set([0., end]+[float(v) for v in (
            self.onset, self.censored, *(self.bracket or ()),
            max(0., self.run-self.cfg.grid) if self.run is not None and self.onset is None else None)
            if v is not None and 0 < v < end]))
        result = dict(correct=0., wrong=0., unknown=0.)
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            y, k = self.labels([(a+b)/2])
            result['unknown' if not k[0] else 'correct' if y[0] else 'wrong'] += b-a
        return result

    def to_dict(self):
        return dict(version=EVENT_VERSION, config=asdict(self.cfg), onset_arc=self.onset,
                    confirmation_arc=self.confirmation, bracket=self.bracket, event_type=self.kind,
                    pending_arc=self.run if self.onset is None else None, censored_arc=self.censored,
                    invalid_seed=self.invalid_seed, length=self.arc)


def label_path(path, annotation, q0, physical_end=False, valid=None):
    return DepartureEvents(annotation, q0, physical_end).append(path, valid)


def truncate_path(path, boundary):
    path = np.asarray(path)
    arc = arclength(path)
    boundary = float(np.clip(boundary, 0, arc[-1]))
    return np.concatenate((path[arc < boundary-1e-8], interp_at(path, arc, np.array([boundary]))))
