"""Read-only fiber store/cache adapter for the existing minimum-flow solver."""
import numpy as np

from .fiber_min_flow import Seam, corrected_targets, integer_targets, fractional_targets, split_fractional_edges


def subset_minimal(d, keep):
    """Induce a whole-fiber minimal graph without inventing solver geometry."""
    remap=np.cumsum(keep)-1;edge=keep[d['i']] & keep[d['j']]
    out={k:d[k][keep] for k in ('source_id','family','start','end','node_ds')}
    out.update({k:d[k][edge] for k in ('ds_a','ds_b','distance','sign','parallel')})
    out.update(i=remap[d['i'][edge]],j=remap[d['j'][edge]],links=np.empty((0,4)))
    return out



class Paths:
    def __init__(self, config, data, angle=-np.pi/2, *, curves=None, cache=None):
        from .fiber_normal_sign_visualization import load_umbilicus_frame
        from .fiber_graph_store import workflow_volume
        from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import read_polylines
        volume = workflow_volume(config)
        self.volume = volume
        self.seam = Seam(load_umbilicus_frame(config['umbilicus'], volume), angle)
        self.paths = {} if cache is None else cache
        if curves is None:
            curves = read_polylines(config['fibers'], [s for s in data['source_id'] if str(s) not in self.paths], volume=volume)
        for row, points in curves:
            if str(row['id']) in self.paths: continue
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
            transport = self.seam.polyline(points)
            self.paths[row['id']] = (arc, points, transport)
        self.data = data
        if len(data['links']) or len(np.unique(data['source_id'])) != len(data['source_id']):
            raise ValueError('Benchmark expects whole unsplit fibers')
        if np.any(data['start'] != 0):
            raise ValueError('Benchmark anchors must be original fiber starts')

    def at(self, source, arc):
        s, points, transport = self.paths[str(source)]
        # Saved endpoint offsets are float32, while original polyline arc is f64.
        if arc < -1e-4 or arc > s[-1]+max(1e-4, 2e-7*s[-1]):
            raise ValueError(f'Endpoint outside fiber {source}: {arc}, {s[-1]}')
        arc = np.clip(arc, 0, s[-1])
        j = min(np.searchsorted(s, arc, side='right')-1, len(s)-2)
        p = points[j]+(points[j+1]-points[j])*(arc-s[j])/max(s[j+1]-s[j], 1e-300)
        return p, int(transport[j])+self.seam.segment(points[j], p)

    def at_many(self, source, arc):
        source, arc = np.asarray(source), np.asarray(arc)
        result = np.empty((len(arc), 3)); turns = np.empty(len(arc), np.int64)
        order = np.argsort(source, kind='stable')
        ordered = source[order]
        for selected in np.split(order, np.flatnonzero(ordered[1:] != ordered[:-1])+1):
            if not len(selected): continue
            s, points, transport = self.paths[str(source[selected[0]])]
            values = arc[selected]
            if np.any(values < -1e-4) or np.any(values > s[-1]+max(1e-4, 2e-7*s[-1])):
                raise ValueError('Endpoint outside fiber')
            values = np.clip(values, 0, s[-1])
            j = np.minimum(np.searchsorted(s, values, side='right')-1, len(s)-2)
            p = points[j]+(points[j+1]-points[j])*((values-s[j])/np.maximum(s[j+1]-s[j], 1e-300))[:, None]
            result[selected] = p
            turns[selected] = transport[j]+self.seam.segments(points[j], p)
        return result, turns

    def ends(self, side):
        d = self.data
        picks = d['i' if side == 'a' else 'j']
        direction = np.sign(d['node_ds'][picks, -1])
        if np.any(direction == 0):
            raise ValueError('Missing aligned arc direction')
        arcs = d['start'][picks]+d['ds_'+side]*direction
        return self.at_many(d['source_id'][picks], arcs)


def graph(config, d, paths, sign_only=False, target_mode="fractional"):
    if target_mode not in ("rounded", "fractional"):
        raise ValueError("Unknown target mode: "+target_mode)
    a, ta = paths.ends('a')
    b, tb = paths.ends('b')
    c = paths.seam.segments(a, b)
    # Comparison exports may store H/V as uint8; V->H must be -1, not 255.
    family=np.asarray(d['family'],dtype=np.int64)
    fd = family[d['j']]-family[d['i']]
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
