"""H-only ruled initialization and fixed-UV fitting via the existing optimizer.

UV and XYZ are base-voxel units. U is the column axis; V is the row axis.
No nearest-3D correspondence, volume reads, normal sampling, or V supervision.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from opt_loss_fixed_uv import FixedUVTargets, ramp_weight


def prepare_sliding(curves, step=100., sample_spacing=50., **unused):
    """Sample immutable source-arc locations, including initial UV grid crossings."""
    if unused.get('uv_bounds') is not None:
        raise ValueError('Sliding preparation currently requires the full selected input, not uv_bounds')
    selected, uv, xyz, weights, owners, arcs, scales, sources = [], [], [], [], [], [], [], []
    for c in curves:
        points = np.column_stack((c['u'], np.broadcast_to(c['v'], np.shape(c['u']))))
        arc = np.asarray(c['arc'])
        fractions, segments = [], []
        for k, (a, b) in enumerate(zip(points[:-1], points[1:])):
            length = abs(arc[k+1]-arc[k])
            if length <= 1e-8:
                continue
            ts = [0., 1., *np.linspace(0, 1, max(2, int(np.ceil(length/sample_spacing))+1))]
            for axis in range(2):
                if abs(b[axis]-a[axis]) > 1e-8:
                    edges = np.arange(np.ceil(min(a[axis], b[axis])/step),
                                      np.floor(max(a[axis], b[axis])/step)+1)*step
                    ts.extend(((edges-a[axis])/(b[axis]-a[axis])).tolist())
            ts = np.unique(np.clip(ts, 0, 1))
            if k < len(points)-2:
                ts = ts[:-1]
            fractions.extend(ts); segments.extend([k]*len(ts))
        t = np.asarray(fractions); k = np.asarray(segments)
        p = points[k]*(1-t[:, None])+points[k+1]*t[:, None]
        a = arc[k]*(1-t)+arc[k+1]*t
        q = c['xyz'][k]*(1-t[:, None])+c['xyz'][k+1]*t[:, None]
        delta = np.abs(np.diff(a)); weight = (np.r_[0., delta]+np.r_[delta, 0.])*.5
        valid = weight > 1e-8
        selected.append(c)
        uv.append(p[valid]); xyz.append(q[valid]); weights.append(weight[valid]); arcs.append(a[valid])
        owners.append(np.full(valid.sum(), c['piece'], np.int64))
        sources.append(np.full(valid.sum(), c['source'], np.int64))
        scales.append(np.tile([c['winding_width']*.1, 1000.], (valid.sum(), 1)))
    uv, scales = np.concatenate(uv), np.concatenate(scales)
    # Padding is larger than the nominal soft displacement allowance.
    low = np.floor((uv.min(0)-1.5*scales.max(0))/step)*step
    high = np.ceil((uv.max(0)+1.5*scales.max(0))/step)*step
    x, y = [np.arange(low[j], high[j]+step*.5, step) for j in range(2)]
    if len(x)*len(y) > unused.get('max_vertices', 4_000_000):
        raise ValueError('Sliding mesh exceeds max_vertices')
    return (selected, x, y, (uv-low)/step, np.concatenate(xyz), np.concatenate(weights),
            np.concatenate(owners), np.concatenate(arcs), scales, np.concatenate(sources))


def curve_xyz(curve, u):
    return np.stack([np.interp(u, curve['u'], curve['xyz'][:, k]) for k in range(3)], axis=-1)


def prepare(curves, *, step=100., uv_bounds=None, sample_spacing=50., max_vertices=4_000_000):
    """Clip curves to explicit UV domain; retain knots, endpoints and grid crossings."""
    if not curves or not np.isfinite([step, sample_spacing]).all() or min(step, sample_spacing) <= 0:
        raise ValueError('Nonempty curves and finite positive sampling/grid steps required')
    for c in curves:
        if (len(c['u']) < 2 or np.asarray(c['xyz']).shape != (len(c['u']), 3)
                or not np.isfinite(c['xyz']).all() or not np.isfinite(c['u']).all()
                or not np.isfinite(c['v']) or not (np.diff(c['u']) > 0).all()):
            raise ValueError('Expected finite, increasing-U H polylines')
    if uv_bounds is None:
        uv_bounds = [min(c['u'][0] for c in curves), max(c['u'][-1] for c in curves),
                     min(c['v'] for c in curves), max(c['v'] for c in curves)]
    u0, u1, v0, v1 = map(float, uv_bounds)
    if not np.isfinite(uv_bounds).all() or u1 <= u0 or v1 <= v0:
        raise ValueError('uv_bounds must be finite [umin, umax, vmin, vmax] with positive area')
    x = np.arange(np.floor(u0/step), np.ceil(u1/step)+1)*step
    y = np.arange(np.floor(v0/step), np.ceil(v1/step)+1)*step
    if min(len(x), len(y)) < 3 or len(x)*len(y) > max_vertices:
        raise ValueError('Use a >=3x3 grid within max_vertices; specify a smaller pilot UV domain if needed')
    selected, targets_uv, targets_xyz, lengths, owners = [], [], [], [], []
    for c in curves:
        a, b = max(u0, c['u'][0]), min(u1, c['u'][-1])
        if b <= a or not v0 <= c['v'] <= v1:
            continue
        # Include curve vertices: coarse quads must not replace curved fibers by chords.
        q = np.unique(np.r_[a, b, c['u'][(c['u'] > a) & (c['u'] < b)],
                            x[(x > a) & (x < b)],
                            np.linspace(a, b, max(2, int(np.ceil((b-a)/sample_spacing))+1))])
        points = curve_xyz(c, q)
        selected.append({**c, 'u': q, 'xyz': points})
        targets_uv.append(np.column_stack(((q-x[0])/step, np.full(len(q), (c['v']-y[0])/step))))
        targets_xyz.append(points)
        delta = np.diff(q)
        lengths.append((np.r_[0., delta]+np.r_[delta, 0.])*.5)
        owners.append(np.full(len(q), c['piece'], np.int64))
    if not selected:
        raise ValueError('No H support inside requested UV bounds')
    return (selected, x, y, np.concatenate(targets_uv), np.concatenate(targets_xyz),
            np.concatenate(lengths), np.concatenate(owners))


def ruled_init(curves, x, y, *, max_extrapolation_slope=1.):
    """Ruled across V, linear continuation outside support; finite rectangular domain.

    Prefer genuine bracketing curves. At V boundaries continue the nearest pair.
    A column with one H borrows dXYZ/dV from the nearest multi-H column. Columns
    without any H are interpolated/linearly extended from supported columns.
    This is explicit extrapolation, not a geometry guarantee; expose the support
    classes in the output instead of disguising extrapolated values as evidence.
    """
    if not np.isfinite(max_extrapolation_slope) or max_extrapolation_slope <= 0:
        raise ValueError('max_extrapolation_slope must be finite and positive')
    def bounded(slope):
        norm = np.linalg.norm(slope, axis=-1, keepdims=True)
        return slope * np.minimum(1., max_extrapolation_slope/np.maximum(norm, 1e-12))
    xyz = np.full((len(y), len(x), 3), np.nan, np.float64)
    support = np.full((len(y), len(x)), 3, np.uint8)
    singles, slopes, filled = {}, {}, []
    for col, u in enumerate(x):
        active = []
        for c in curves:
            if np.ndim(c['v']) == 0:
                if c['u'][0] <= u <= c['u'][-1]:
                    active.append(c)
            else:
                # All crossings, including locally reversed U in a deformed map.
                for k in np.flatnonzero((np.minimum(c['u'][:-1], c['u'][1:]) <= u) &
                                        (np.maximum(c['u'][:-1], c['u'][1:]) > u)):
                    t = (u-c['u'][k])/(c['u'][k+1]-c['u'][k])
                    active.append(dict(piece=c['piece'], v=(1-t)*c['v'][k]+t*c['v'][k+1],
                                       u=np.array([u]), xyz=np.array([(1-t)*c['xyz'][k]+t*c['xyz'][k+1]])))
        active.sort(key=lambda c: (c['v'], c['piece']))
        if not active:
            continue
        # Same V cannot define a transverse slope; deterministic piece-ID tie break.
        vals, idx = np.unique([c['v'] for c in active], return_index=True)
        active = [active[i] for i in idx]
        points = np.asarray([curve_xyz(c, u) for c in active])
        if len(active) == 1:
            singles[col] = (vals[0], points[0])
            continue
        right = np.searchsorted(vals, y, side='right').clip(1, len(vals)-1)
        left = right-1
        t = (y-vals[left])/(vals[right]-vals[left])
        xyz[:, col] = points[left]*(1-t[:, None])+points[right]*t[:, None]
        slopes[col] = bounded((points[right]-points[left])/(vals[right]-vals[left])[:, None])
        # Near-coincident fitted V coordinates can have a huge raw secant.
        # Preserve supported interpolation; only extrapolation has bounded speed.
        below, above = y < vals[0], y > vals[-1]
        xyz[below, col] = points[0]+(y[below]-vals[0])[:, None]*slopes[col][below]
        xyz[above, col] = points[-1]+(y[above]-vals[-1])[:, None]*slopes[col][above]
        support[:, col] = ((y < vals[0]) | (y > vals[-1])).astype(np.uint8)
        filled.append(col)
    if not filled:
        raise ValueError('Need at least one U column with two distinct H heights to initialize transverse geometry')
    for col, (v, point) in singles.items():
        nearest = min(filled, key=lambda k: (abs(k-col), k))
        # Integrate bounded donor increments, then translate to the observed H.
        # Multiplying (y-v) by a row-dependent slope would introduce jumps at
        # every donor bracket boundary.
        dy = np.diff(y)[:, None]
        delta = bounded(np.diff(xyz[:, nearest], axis=0)/dy)*dy
        profile = np.vstack((np.zeros((1, 3)), np.cumsum(delta, axis=0)))
        anchor = np.array([np.interp(v, y, profile[:, axis]) for axis in range(3)])
        xyz[:, col] = point+profile-anchor
        support[:, col] = 2
    known = np.array(sorted(filled+list(singles)), dtype=int)
    if len(known) < 2:
        raise ValueError('Need two supported U columns to extrapolate along U; use a finer grid')
    for col in np.flatnonzero(~np.isfinite(xyz).all(axis=(0, 2))):
        r = np.searchsorted(known, col).clip(1, len(known)-1)
        a, b = known[r-1], known[r]
        t = (x[col]-x[a])/(x[b]-x[a])
        if 0 <= t <= 1:
            xyz[:, col] = (1-t)*xyz[:, a]+t*xyz[:, b]
        else:
            anchor = a if t < 0 else b
            slope = bounded((xyz[:, b]-xyz[:, a])/(x[b]-x[a]))
            xyz[:, col] = xyz[:, anchor]+(x[col]-x[anchor])*slope
        support[:, col] = 3 if 0 <= t <= 1 else 4
    if not np.isfinite(xyz).all():
        raise ValueError('Nonfinite ruled initializer')
    return xyz.astype(np.float32), support


def geometry_masks(xyz):
    """Shared local fold/degeneracy definition for solve and post-hoc audits."""
    p, q, r, s = xyz[:-1, :-1], xyz[:-1, 1:], xyz[1:, :-1], xyz[1:, 1:]
    a, b = np.cross(q-p, r-p), np.cross(r-s, q-s)
    area = np.linalg.norm(a, axis=-1)+np.linalg.norm(b, axis=-1)
    return area < 1e-6, (a*b).sum(-1) <= 0


def geometry_stats(xyz):
    """Local diagnostics only: not a global self-intersection audit."""
    degenerate, folded = geometry_masks(xyz)
    return dict(vertices=int(np.prod(xyz.shape[:2])), finite=bool(np.isfinite(xyz).all()),
                degenerate_quads=int(degenerate.sum()),
                opposing_triangle_quads=int(folded.sum()),
                bbox_min=xyz.min(axis=(0, 1)).tolist(), bbox_max=xyz.max(axis=(0, 1)).tolist())


def summarize_runs(settings):
    """Audit completed surfaces without optimizing or allocating a GPU."""
    import tifffile
    rows, cohorts = [], []
    for name in settings['summarize_runs']:
        root = Path(name)
        summary = json.loads((root/'summary.json').read_text())
        xyz = np.stack([tifffile.imread(root/'surface.tifxyz'/f'{axis}.tif') for axis in 'xyz'], -1)
        degenerate, folded = geometry_masks(xyz)
        with np.load(root/'attachments.npz') as f:
            guv = f['final_grid_uv']
            cohorts.append(np.unique(f['piece']))
            occupied = np.zeros(folded.shape, bool)
            col = np.floor(guv[:, 0]).astype(int).clip(0, folded.shape[1]-1)
            row = np.floor(guv[:, 1]).astype(int).clip(0, folded.shape[0]-1)
            occupied[row, col] = True
        with np.load(root/'targets.npz') as f:
            support = f['support_class']
            bracketed = ((support[:-1, :-1] == 0) & (support[1:, :-1] == 0) &
                         (support[:-1, 1:] == 0) & (support[1:, 1:] == 0))
        assert int(folded.sum()) == summary['after']['geometry']['opposing_triangle_quads']
        rows.append(dict(run=str(root), fit=summary['after']['fit'],
                         solve_seconds=summary['solve_seconds'],
                         folds=int(folded.sum()), degenerate=int(degenerate.sum()),
                         folds_with_attachments=int((folded & occupied).sum()),
                         attachment_cells=int(occupied.sum()),
                         folds_without_attachments=int((folded & ~occupied).sum()),
                         folds_initially_bracketed=int((folded & bracketed).sum())))
    if any(not np.array_equal(cohorts[0], c) for c in cohorts[1:]):
        raise ValueError('Surface comparison cohorts differ')
    out = Path(settings['report_path'])
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2)+'\n')
    print(json.dumps(rows, indent=2), flush=True)
    return rows


def default_stages(steps=600, lr=5.):
    # Geometry and data terms are dimensionless, divided by mesh_step**2.
    # No data-free stage; preserve the ruled geometry while gradually fitting it.
    return dict(base=dict(normal=0., fixed_uv=1., smooth=.5, step=.5, bend=5.),
                stages=[dict(name='fiber_surface', steps=steps, lr=lr, params=['mesh_ms'],
                             min_scaledown=0, args=dict(fixed_uv_ramp=[1., 20.], status_interval=50))])


def fit(initial, targets, *, step, stages, progress_fn=None):
    import model
    import fit_data
    import optimizer
    device = targets.xyz.device
    tensor = torch.as_tensor(initial, device=device)
    mdl = model.Model3D.from_tifxyz_crop(tensor, torch.ones(tensor.shape[:2], dtype=torch.bool, device=device),
                                       device=device, mesh_step=int(step), subsample_mesh=1, subsample_winding=1)
    # No synthetic volume: inactive data terms never request any samples.
    data = fit_data.FitData3D(cos=None, grad_mag=None, nx=None, ny=None, pred_dt=None,
                             corr_points=None, winding_volume=None, origin_fullres=(0., 0., 0.), spacing=(1., 1., 1.))
    parsed = optimizer.load_stages_cfg(stages)
    previous_weight = 0.
    for stage in parsed:
        opt = stage.global_opt
        if opt is None or opt.eff['fixed_uv'] <= 0 or opt.params != ['mesh_ms']:
            raise ValueError('Surface stages must optimize mesh_ms with positive fixed_uv; no grow/map stages')
        allowed = {'fixed_uv', 'smooth', 'step', 'bend', 'smooth_step', 'avg_step'}
        if any(value != 0 and name not in allowed and not name.startswith('map_') for name, value in opt.eff.items()):
            raise ValueError('Fiber surface stage must not enable volume/normal/other data losses')
        ramp = (opt.args or {}).get('fixed_uv_ramp', [opt.eff['fixed_uv']]*2)
        first = ramp_weight(ramp, 0, opt.steps)
        if first < previous_weight:
            raise ValueError('Fiber surface data weight must not decrease between stages')
        previous_weight = ramp_weight(ramp, opt.steps, opt.steps)
    optimizer.optimize(model=mdl, data=data, stages=parsed, snapshot_interval=0, snapshot_fn=lambda **_: None,
                       progress_fn=progress_fn, fixed_uv_targets=targets)
    with torch.no_grad():
        final = mdl._grid_xyz()
        return final[0].cpu().numpy(), targets.statistics(final)


def solve(curves, settings, provenance):
    from vesuvius.tifxyz.types import Tifxyz
    from vesuvius.tifxyz.writer import write_tifxyz
    out = Path(settings['output_root'])
    if out.exists():
        raise FileExistsError(out)
    device = settings.get('device', 'cpu')
    torch.set_num_threads(int(settings.get('cpu_threads', 8)))
    started = time.perf_counter()
    step = float(settings.get('grid_step', 100.))
    if step != int(step):
        raise ValueError('Lasagna mesh_step must be integer base voxels')
    sliding = settings.get('sliding_uv', False)
    connected = settings.get('connected_hv', False)
    if connected and not sliding:
        raise ValueError('Connected H/V mode requires sliding_uv')
    selection_info = {}
    if connected:
        from fiber_surface_connected import filter_components
        with np.load(Path(settings['deformation_root'])/'solution.npz') as saved:
            curves, selection_info = filter_components(
                curves, saved, settings.get('component_min_fibers', 3))
    prepared = (prepare_sliding if sliding else prepare)(
        curves, step=step, uv_bounds=settings.get('uv_bounds'),
        sample_spacing=settings.get('sample_spacing', step/2),
        max_vertices=settings.get('max_vertices', 4_000_000))
    selected, x, y, guv, target, weights, owners = prepared[:7]
    initial, support = ruled_init([c for c in selected if c.get('family', 0) == 0], x, y,
                                 max_extrapolation_slope=settings.get('max_extrapolation_slope', 1.))
    connected_meta, connected_info = {}, {}
    if connected:
        from fiber_surface_connected import prepare_metadata
        target, connected_meta, connected_info = prepare_metadata(curves, prepared, settings)
        connected_info['selection'] = selection_info
    if sliding:
        from opt_loss_fixed_uv import SlidingUVTargets
        target_class, extra = SlidingUVTargets, {}
        if connected:
            from fiber_surface_connected import ConnectedUVTargets
            target_class = ConnectedUVTargets
            extra = {k: connected_meta[k] for k in ('component', 'crossing', 'family')}
            extra['v_weight'] = settings.get('v_data_weight', .1)
        targets = target_class(guv, target, weights, initial.shape[:2], device=device,
                                   arc=prepared[7], scales=prepared[8], source=prepared[9],
                                   step=step, piece=owners, **extra, **settings.get('sliding_settings', {}))
    else:
        targets = FixedUVTargets(guv, target, weights, initial.shape[:2], device=device)
    barrier_meta = {}
    if settings.get('winding_barrier') is not None:
        if not connected:
            raise ValueError('Winding barriers require connected H/V aligned-normal metadata')
        from fiber_surface_barrier import prepare_barriers, attach_barrier
        with np.load(Path(settings['run_root'])/'solution.npz') as solution:
            widths = solution['widths']
        barrier_meta = prepare_barriers(guv, connected_meta['raw_xyz'], connected_meta['normals'],
            weights, widths, np.array([x[0], y[0]]), step, initial.shape[:2],
            **settings['winding_barrier'])
        attach_barrier(targets, barrier_meta)
    out.mkdir(parents=True)
    stages = settings.get('stages_config', default_stages())
    (out/'config.json').write_text(json.dumps(dict(settings=settings, stages=stages, provenance=provenance), indent=2)+'\n')
    np.savez(out/'targets.npz', grid_uv=guv, xyz=target, arc_weight=weights, piece=owners,
             support_class=support, uv_origin=[x[0], y[0]], grid_step=step, **connected_meta, **barrier_meta)
    def save(name, xyz):
        valid = np.isfinite(xyz).all(-1)
        write_tifxyz(out/(name+'.tifxyz'), Tifxyz(xyz[..., 0], xyz[..., 1], xyz[..., 2],
                      uuid=out.name+'-'+name, _scale=(1/step, 1/step), _mask=valid,
                      extra=dict(uv_origin=[float(x[0]), float(y[0])],
                                 method='Connected H/V offset sliding UV' if connected else ('H-only joint sliding UV' if sliding else 'H-only fixed UV'), provenance=provenance)),
                      compression='lzw')
    save('initial', initial)
    before = dict(fit=targets.statistics(torch.as_tensor(initial, device=device)[None]), geometry=geometry_stats(initial))
    init_seconds = time.perf_counter()-started
    history = []
    def progress_fn(**row):
        history.append(row)
    solve_start = time.perf_counter()
    final, final_fit = fit(initial, targets, step=step, stages=stages, progress_fn=progress_fn)
    solve_seconds = time.perf_counter()-solve_start
    save('surface', final)
    if sliding:
        saved_connected = dict(connected_meta)
        if connected:
            saved_connected['v_data_weight'] = np.array(settings.get('v_data_weight', .1))
        np.savez(out/'attachments.npz', initial_grid_uv=guv,
                 xyz=target, piece=owners, source=prepared[9], arc=prepared[7], scales=prepared[8],
                 **saved_connected, **targets.attachment_state())
    report = dict(curves=len(selected), shape=list(initial.shape), before=before,
                  after=dict(fit=final_fit, geometry=geometry_stats(final)),
                  support_percent={str(k): float(100*(support == k).mean()) for k in range(5)},
                  initialization_seconds=init_seconds, solve_seconds=solve_seconds,
                  total_seconds=time.perf_counter()-started, output_root=str(out), connected_hv=connected_info)
    (out/'progress.json').write_text(json.dumps(history, indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    return report
