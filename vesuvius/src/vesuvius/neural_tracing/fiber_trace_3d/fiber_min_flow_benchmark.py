"""Read-only AFV/cache adapter for the existing minimum-flow solver."""
import json
from pathlib import Path
import sqlite3

import numpy as np

from .fiber_min_flow import Seam, corrected_targets, integer_targets, fractional_targets, split_fractional_edges



class Paths:
    def __init__(self, config, data, angle=-np.pi/2):
        from .fiber_normal_sign_visualization import load_umbilicus_frame
        from vesuvius.neural_tracing.tracers.shared.annotations.afv import read_afv_polylines
        with sqlite3.connect(Path(config['afv']).resolve().as_uri()+'?mode=ro', uri=True) as db:
            root = json.loads(db.execute("SELECT value FROM metadata WHERE key='root'").fetchone()[0])
        self.seam = Seam(load_umbilicus_frame(config['umbilicus'], root), angle)
        self.paths = {}
        for row, points in read_afv_polylines(config['afv'], data['source_id']):
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
            transport = self.seam.polyline(points)
            self.paths[int(row['id'])] = (arc, points, transport)
        self.data = data
        if len(data['links']) or len(np.unique(data['source_id'])) != len(data['source_id']):
            raise ValueError('Benchmark expects whole unsplit fibers')
        if np.any(data['start'] != 0):
            raise ValueError('Benchmark anchors must be original fiber starts')

    def at(self, source, arc):
        s, points, transport = self.paths[int(source)]
        # Saved endpoint offsets are float32, while original polyline arc is f64.
        if arc < -1e-4 or arc > s[-1]+max(1e-4, 2e-7*s[-1]):
            raise ValueError(f'Endpoint outside fiber {source}: {arc}, {s[-1]}')
        arc = np.clip(arc, 0, s[-1])
        j = min(np.searchsorted(s, arc, side='right')-1, len(s)-2)
        p = points[j]+(points[j+1]-points[j])*(arc-s[j])/max(s[j+1]-s[j], 1e-300)
        return p, int(transport[j])+self.seam.segment(points[j], p)

    def ends(self, side):
        d = self.data
        picks = d['i' if side == 'a' else 'j']
        direction = np.sign(d['node_ds'][picks, -1])
        if np.any(direction == 0):
            raise ValueError('Missing aligned arc direction')
        arcs = d['start'][picks]+d['ds_'+side]*direction
        samples = [self.at(d['source_id'][p], arc) for p, arc in zip(picks, arcs)]
        return np.array([s[0] for s in samples]), np.array([s[1] for s in samples])


def graph(config, d, paths, sign_only=False, target_mode="fractional"):
    if target_mode not in ("rounded", "fractional"):
        raise ValueError("Unknown target mode: "+target_mode)
    a, ta = paths.ends('a')
    b, tb = paths.ends('b')
    c = np.array([paths.seam.segment(x, y) for x, y in zip(a, b)])
    fd = d['family'][d['j']]-d['family'][d['i']]
    target_function = fractional_targets if target_mode == 'fractional' and not sign_only else integer_targets
    measured, valid = target_function(d['distance']*config.get('lasagna_winding_scale', 1.),
        d['sign'], fd, d['parallel'], config.get('perpendicular_offset', .5))
    target = corrected_targets(measured, ta, tb, c)
    ia, ib = d['i'].copy(), d['j'].copy()
    if sign_only:
        valid = d['sign'] != 0
        margin = ((~d['parallel']) & (d['sign']*fd < 0)).astype(int)
        target = margin+d['sign']*(c-tb+ta)
        reverse = d['sign'] < 0
        ia[reverse], ib[reverse] = ib[reverse].copy(), ia[reverse].copy()
    weights = np.where(d['parallel'], 1., config.get('perpendicular_relative_weight', 1.))
    g = dict(a=ia[valid], b=ib[valid], target=target[valid], weight=weights[valid],
             one_sided=np.full(valid.sum(), sign_only))
    event_index = np.flatnonzero(valid)
    if target_mode == 'fractional' and not sign_only:
        g, source = split_fractional_edges(g['a'], g['b'], g['target'], g['weight'])
        event_index = event_index[source]
    return g, dict(valid=valid, fiber_a=ta, fiber_b=tb, connector=c,
                   measured_target=measured, corrected_target=target, flow_event_index=event_index)
