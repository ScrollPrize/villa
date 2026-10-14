"""Frozen crossing topology, offset V observations and component UV translations."""
import json
from pathlib import Path
import numpy as np
import torch
from opt_loss_fixed_uv import SlidingUVTargets


def source_components(source, pairs):
    """Union source fibers, not just individual crossing nodes."""
    ids = np.unique(source)
    parent = np.arange(len(ids))
    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in np.asarray(pairs, dtype=np.int64).reshape(-1, 2):
        ia, ib = np.searchsorted(ids, [a, b])
        if ia >= len(ids) or ib >= len(ids) or ids[ia] != a or ids[ib] != b:
            raise ValueError('Crossing references an absent source')
        parent[root(ib)] = root(ia)
    labels = np.array([root(i) for i in range(len(ids))])
    _, labels = np.unique(labels, return_inverse=True)
    return labels[np.searchsorted(ids, source)]


def interpolate_offsets(arc, crossing_arc, offsets, fallback):
    """Linear signed offsets along arc, constant end extension and global fallback."""
    a, o = np.asarray(crossing_arc), np.asarray(offsets)
    if not len(a):
        return np.full(len(arc), fallback, dtype=np.float64)
    order = np.argsort(a, kind='stable'); a, o = a[order], o[order]
    # Multiple H observations at the same source arc have equal measurement weight.
    unique, inverse = np.unique(a, return_inverse=True)
    mean = np.bincount(inverse, weights=o)/np.bincount(inverse)
    return np.interp(arc, unique, mean)


def selected_crossings(saved, source):
    """Select whole graph components, never silently sever a crossing edge."""
    ns = saved['v_offset_source']
    a, b = saved['v_offset_a'], saved['v_offset_b']
    ka, kb = np.isin(ns[a], source), np.isin(ns[b], source)
    if np.any(ka != kb):
        raise ValueError('Selection cuts a saved crossing component')
    return a[ka], b[ka]


def filter_components(curves, saved, minimum=3):
    """Keep mixed H/V components by distinct source count, not piece count."""
    if isinstance(minimum, bool) or int(minimum) != minimum or minimum < 0:
        raise ValueError('component_min_fibers must be a nonnegative integer')
    source = np.array([c['source'] for c in curves], dtype=np.int64)
    family = np.array([c['family'] for c in curves], dtype=np.int64)
    ids, first, inverse = np.unique(source, return_index=True, return_inverse=True)
    families = family[first]
    if np.any(family != families[inverse]) or np.any((family != 0) & (family != 1)):
        raise ValueError('Expected one consistent H/V family per source fiber')
    a, b = selected_crossings(saved, ids)
    ns = saved['v_offset_source']
    labels = source_components(ids, np.column_stack([ns[a], ns[b]]))
    counts = np.bincount(labels)
    h = np.bincount(labels, weights=families == 0, minlength=len(counts))
    v = np.bincount(labels, weights=families == 1, minlength=len(counts))
    eligible = (counts >= minimum) & (h > 0) & (v > 0)
    # Explicit zero disables filtering for historical replay.
    if minimum == 0:
        eligible[:] = True
    keep_source = eligible[labels]
    kept = [c for c, keep in zip(curves, keep_source[inverse]) if keep]
    edges = np.unique(np.sort(np.column_stack([a, b]), axis=1), axis=0)
    info = dict(minimum_fibers=int(minimum), input_sources=len(ids),
                retained_sources=int(keep_source.sum()), input_pieces=len(curves),
                retained_pieces=len(kept), input_components=len(counts),
                retained_components=int(eligible.sum()),
                input_crossings=len(edges),
                retained_crossings=int(np.isin(ns[edges[:, 0]], ids[keep_source]).sum()),
                retained_h_sources=int((keep_source & (families == 0)).sum()),
                retained_v_sources=int((keep_source & (families == 1)).sum()),
                retained_source_ids=ids[keep_source].tolist())
    if not kept:
        raise ValueError('No fiber components satisfy the size and H/V filter')
    return kept, info


class ConnectedUVTargets(SlidingUVTargets):
    """Crossings have only shared component motion; interior nodes remain flexible."""
    def __init__(self, *args, component, crossing, family, v_weight=.1, **kwargs):
        super().__init__(*args, **kwargs)
        if not np.isfinite(v_weight) or not 0 < v_weight < 1:
            raise ValueError('V data weight must be finite and strictly between zero and one')
        device = self.xyz.device
        component = np.asarray(component)
        if component.shape != (len(self.xyz),) or np.any(component < 0):
            raise ValueError('Invalid per-attachment component IDs')
        self.component = torch.as_tensor(component, dtype=torch.long, device=device)
        self.crossing = torch.as_tensor(crossing, dtype=torch.bool, device=device)
        self.family = torch.as_tensor(family, dtype=torch.long, device=device)
        if self.crossing.shape != self.component.shape or self.family.shape != self.component.shape:
            raise ValueError('Crossing/family shape mismatch')
        if not bool(((self.family == 0) | (self.family == 1)).all()) or not bool((self.family == 0).any()):
            raise ValueError('Expected H/V families and H support')
        self.component_delta = torch.nn.Parameter(torch.zeros((int(component.max())+1, 2), device=device))
        # Separate family means: adding V support never dilutes the H data term.
        self.data_weights = torch.zeros_like(self.arc_weights)
        for f, factor in [(0, 1.), (1, v_weight)]:
            mask = self.family == f
            if bool(mask.any()):
                self.data_weights[mask] = factor*self.arc_weights[mask]/self.arc_weights[mask].sum()

    def current_uv(self):
        return self.initial_uv+self.component_delta[self.component]+self.delta*(~self.crossing)[:, None]

    def optimizer_groups(self):
        return [dict(params=[self.delta, self.component_delta], lr=self.uv_lr)]

    def attachment_state(self):
        return dict(**super().attachment_state(),
                    component_delta=self.component_delta.detach().cpu().numpy(),
                    local_delta=self.delta.detach().cpu().numpy())

    @torch.no_grad()
    def statistics(self, xyz):
        result = super().statistics(xyz)
        motion = self.current_uv()-self.initial_uv
        expected = self.component_delta[self.component]
        result['crossing_lock_error_base'] = float((motion[self.crossing]-expected[self.crossing]).abs().max()*self.step) if bool(self.crossing.any()) else 0.
        result['components'] = len(self.component_delta)
        position, normal = self.sample(xyz, self.current_uv())
        error = ((position-self.xyz)*normal).sum(-1)
        tangent = position-self.xyz-error[:, None]*normal
        for f, name in [(0, 'h'), (1, 'v_offset')]:
            mask = self.family == f
            if bool(mask.any()):
                w = self.arc_weights[mask]; w = w/w.sum()
                result[name+'_normal_rms_base'] = float((error[mask].square()*w).sum().sqrt())
                result[name+'_tangent_rms_base'] = float((tangent[mask].square().sum(-1)*w).sum().sqrt())
                result[name+'_position_rms_base'] = float(((position[mask]-self.xyz[mask]).square().sum(-1)*w).sum().sqrt())
        return result


def prepare_metadata(curves, prepared, settings):
    """Read saved graph; no new nearest-neighbor matching or upstream optimization."""
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_surface import CurveBank
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_normal_convention import inward_solution
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_layout import sha256_file
    root = Path(settings['deformation_root'])
    cfg = json.loads((Path(settings['run_root'])/'config.json').read_text())
    with np.load(root/'solution.npz') as f:
        saved = {k: f[k] for k in f.files}
    with np.load(cfg['stage1']) as f:
        sign = inward_solution({k: f[k] for k in f.files})
    normal_root = settings['normal_root']
    source, arc = prepared[9], prepared[7]
    piece = prepared[6]
    families = {c['piece']: c['family'] for c in curves}
    family = np.array([families[int(p)] for p in piece])
    ns, na = saved['v_offset_source'], saved['v_offset_arc']
    a, b = selected_crossings(saved, source)
    pairs = np.sort(np.column_stack([a, b]), axis=1)
    _, unique = np.unique(pairs, axis=0, return_index=True)
    a, b = a[np.sort(unique)], b[np.sort(unique)]
    # All saved endpoints must belong to retained source spans. Never silently
    # discard crossings because a crop or selection happened to omit a neighbor.
    crossing = np.zeros(len(arc), bool)
    node_attachment = np.full(len(ns), -1, dtype=np.int64)
    active_nodes = np.unique(np.r_[a, b])
    for s in np.unique(ns[active_nodes]):
        nodes = active_nodes[ns[active_nodes] == s]
        samples = np.flatnonzero(source == s)
        if not len(samples):
            raise ValueError(f'Crossing source {s} not in selected curves')
        order = np.argsort(arc[samples], kind='stable'); samples = samples[order]
        for n in nodes:
            lo = np.searchsorted(arc[samples], na[n]-1e-7)
            hi = np.searchsorted(arc[samples], na[n]+1e-7, side='right')
            hits = samples[lo:hi]
            if not len(hits):
                raise ValueError(f'Missing exact crossing attachment source={s} arc={na[n]}')
            crossing[hits] = True
            node_attachment[n] = hits[0]
    mismatch = np.linalg.norm(prepared[3][node_attachment[a]]-prepared[3][node_attachment[b]], axis=1)*settings.get('grid_step', 100.)
    if len(mismatch) and mismatch.max() > .01:
        raise ValueError(f'Deformed crossing UV mismatch {mismatch.max()} base voxels')
    if np.any(saved['v_offset_family'][a] == saved['v_offset_family'][b]):
        raise ValueError('Saved crossing must connect H and V')
    bank = CurveBank(cfg['afv'], source, normal_root, sign)
    raw_xyz, normals, valid = bank.sample(np.searchsorted(bank.ids, source), arc)
    if not valid[family == 1].all():
        raise ValueError('Invalid aligned V normals: cannot construct offset targets')
    if not np.allclose(raw_xyz, prepared[4], atol=.05, rtol=0):
        raise ValueError('Sampled normal/source positions disagree with attachments')
    hnode = np.where(saved['v_offset_family'][a] == 0, a, b)
    vnode = np.where(saved['v_offset_family'][a] == 1, a, b)
    h, v = node_attachment[hnode], node_attachment[vnode]
    offsets = ((raw_xyz[h]-raw_xyz[v])*normals[v]).sum(-1)
    if not len(offsets) or not np.isfinite(offsets).all():
        raise ValueError('No finite H/V offsets to establish global fallback')
    fallback = float(offsets.mean())
    target_offset = np.zeros(len(source))
    unconnected = []
    for s in np.unique(source[family == 1]):
        at = np.flatnonzero((source == s) & (family == 1))
        obs = source[v] == s
        if not obs.any():
            unconnected.append(int(s))
        target_offset[at] = interpolate_offsets(arc[at], arc[v[obs]], offsets[obs], fallback)
    target = prepared[4].copy()
    is_v = family == 1
    target[is_v] += target_offset[is_v, None]*normals[is_v]
    component = source_components(source, np.column_stack([ns[a], ns[b]]))
    meta = dict(component=component, crossing=crossing, family=family,
                target_offset=target_offset, normals=normals, raw_xyz=prepared[4],
                crossing_a=h, crossing_b=v)
    info = dict(crossings=len(a), components=int(component.max()+1),
                crossing_samples=int(crossing.sum()), unconnected_v_sources=unconnected,
                unconnected_v_count=len(unconnected),
                global_signed_offset_base=fallback, offset_p0_p50_p100=np.quantile(offsets, [0,.5,1]).tolist(),
                input_crossing_uv_max_error_base=float(mismatch.max(initial=0)),
                normal_root=str(normal_root), normal_sign_solution=cfg['stage1'],
                normal_index_sha256=sha256_file(Path(normal_root)/'index.json'),
                normal_sign_sha256=sha256_file(cfg['stage1']))
    return target, meta, info
