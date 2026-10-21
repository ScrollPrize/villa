"""Piece topology for flow labels expressed in each original fiber-start chart."""
import numpy as np
from .fiber_uv_layout import split_bounds


class Pieces:
    def __init__(self, data, minimum_length=1000., minimum_perpendicular=4, protected_sources=()):
        if len(np.unique(data['source_id'])) != len(data['source_id']) or len(data['links']):
            raise ValueError('Expected whole-fiber input')
        if not np.isfinite(minimum_length) or minimum_length < 0 or minimum_perpendicular < 1:
            raise ValueError('Invalid split settings')
        self.data = data
        self.bounds = {}
        self.first = {}
        parents, starts, ends = [], [], []
        protected = set(map(int, protected_sources))
        for parent, source in enumerate(data['source_id']):
            direction = np.sign(data['node_ds'][parent, -1])
            if direction not in (-1, 1):
                raise ValueError('Missing resolved arc direction')
            start, end = data['start'][parent], data['end'][parent]
            arcs = []
            seen = set()
            for side, key in [('a', 'i'), ('b', 'j')]:
                rows = (data[key] == parent) & ~data['parallel']
                for row in np.flatnonzero(rows):
                    event = int(data['event_ids'][row]) if 'event_ids' in data else int(row)
                    if event not in seen:
                        arcs.append(start+direction*data['ds_'+side][row])
                        seen.add(event)
            if minimum_length == 0 or int(source) in protected:
                bounds = np.array([start, end])
            else:
                bounds = start+split_bounds(end-start, np.asarray(arcs)-start,
                                            minimum_length, minimum_perpendicular)
            self.bounds[int(source)] = bounds
            self.first[int(source)] = len(parents)
            parents.extend([parent]*(len(bounds)-1))
            starts.extend(bounds[:-1]); ends.extend(bounds[1:])
        self.parent = np.asarray(parents, np.int64)
        self.start, self.end = np.asarray(starts), np.asarray(ends)
        self.source_id = data['source_id'][self.parent]

    def locate(self, source, arc):
        source = int(source)
        bounds = self.bounds[source]
        tolerance = max(1e-4, 2e-7*bounds[-1])
        if arc < bounds[0]-tolerance or arc > bounds[-1]+tolerance:
            raise ValueError('Point outside original fiber')
        return self.first[source]+int(np.searchsorted(bounds[1:-1], arc, side='right'))

    def endpoints(self, side):
        d = self.data
        parent = d['i' if side == 'a' else 'j']
        arcs = d['start'][parent]+np.sign(d['node_ds'][parent, -1])*d['ds_'+side]
        return np.array([self.locate(s, x) for s, x in zip(d['source_id'][parent], arcs)], np.int64)

    def remap(self, graph, transport):
        """Targets keep full original-start seam transport; only endpoints change."""
        event = transport['flow_event_index']
        return dict(graph, a=self.endpoints('a')[event], b=self.endpoints('b')[event])
