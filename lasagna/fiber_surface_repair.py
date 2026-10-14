"""Boundary-frozen sparse patch reconstruction; no change to source fiber data."""
from pathlib import Path
import json
import time
import hashlib

import numpy as np
import torch
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve

from fiber_surface import geometry_masks, geometry_stats, fit
from opt_loss_fixed_uv import SlidingUVTargets
from opt_loss_surface_guard import SurfaceGuard


def restore_targets(data, attachments, shape, step, device, options=None):
    options = dict(options or {})
    options.setdefault('geometry_tangent_weight', float(attachments.get('geometry_tangent_weight', 0.)))
    for key, other in [('grid_uv','initial_grid_uv'), ('xyz','xyz'), ('piece','piece')]:
        if not np.array_equal(data[key], attachments[other]):
            raise ValueError(f'Inconsistent saved attachments: {key}')
    cls, extra = SlidingUVTargets, {}
    if 'component' in attachments:
        from fiber_surface_connected import ConnectedUVTargets
        cls = ConnectedUVTargets
        for key in ('component', 'crossing', 'family'):
            if not np.array_equal(data[key], attachments[key]):
                raise ValueError('Inconsistent saved connected topology: '+key)
            extra[key] = attachments[key]
        extra['v_weight'] = float(attachments['v_data_weight'])
    t = cls(attachments['initial_grid_uv'], data['xyz'], data['arc_weight'], shape,
        arc=attachments['arc'], scales=attachments['scales'], source=attachments['source'],
        piece=attachments['piece'], step=step, device=device, **extra, **options)
    with torch.no_grad():
        if 'component' in attachments:
            t.delta.copy_(torch.as_tensor(attachments['local_delta'], device=device))
            t.component_delta.copy_(torch.as_tensor(attachments['component_delta'], device=device))
            if not np.allclose(t.current_uv().cpu().numpy(), attachments['final_grid_uv'], atol=1e-5, rtol=0):
                raise ValueError('Restored component motion differs from saved final UV')
        else:
            t.delta.copy_(torch.as_tensor(attachments['final_grid_uv']-attachments['initial_grid_uv'], device=device))
    from fiber_surface_barrier import attach_barrier
    attach_barrier(t, data)
    return t


def crop_winding_domain(initial, data, attachments, prior, bounds):
    """Restrict fitted U to a declared bucket band; never let outliers enlarge it."""
    if 'barrier_grid_uv' in data:
        raise ValueError('Legacy cropping cannot rebind saved winding barriers')
    if 'component' in attachments:
        raise ValueError('Connected H/V targets cannot use the legacy per-sample crop; preserve whole components')
    root=Path(prior['settings']['run_root'])
    with np.load(root/'solution.npz') as f: widths=f['widths'].astype(np.float64)
    lo,hi=map(int,bounds)
    if not 0<=lo<hi<=len(widths): raise ValueError('Expected [first,exclusive last] winding bounds')
    cumulative=np.r_[0.,np.cumsum(widths)]
    step=float(data['grid_step']); origin=data['uv_origin'].copy()
    a=max(0,int(np.ceil((cumulative[lo]-origin[0])/step)))
    b=min(initial.shape[1],int(np.floor((cumulative[hi]-origin[0])/step))+1)
    if b-a<3: raise ValueError('Selected winding domain does not intersect saved mesh')
    keep=np.ones(len(attachments['piece']),bool)
    for key in ['initial_grid_uv','final_grid_uv']:
        u=attachments[key][:,0];keep&=(u>=a)&(u<=b-1)
    if not keep.any():raise ValueError('No fiber observations in selected winding band')
    ids=np.flatnonzero(keep); parent=attachments['piece'][keep]
    # Gaps outside the domain must not acquire a fictitious smoothing link.
    pieces=np.cumsum(np.r_[True,(np.diff(ids)!=1)|(np.diff(parent)!=0)])-1
    reduced={k:(v[keep].copy() if np.ndim(v) and len(v) == len(keep) else np.array(v).copy())
             for k,v in attachments.items()}
    reduced['source_piece']=parent.copy();reduced['piece']=pieces
    for key in ['initial_grid_uv','final_grid_uv']: reduced[key][:,0]-=a
    data=dict(data,grid_uv=reduced['initial_grid_uv'],xyz=reduced['xyz'],piece=pieces,
              arc_weight=data['arc_weight'][keep],support_class=data['support_class'][:,a:b],
              uv_origin=origin+np.array([a*step,0]))
    info=dict(winding_bounds=[lo,hi],column_range=[a,b],original_shape=list(initial.shape),
              cropped_shape=list(initial[:,a:b].shape),retained_samples=int(keep.sum()),
              excluded_samples=int((~keep).sum()),retained_source_pieces=int(len(np.unique(parent))),
              uv_origin=data['uv_origin'].tolist(),u_end=float(data['uv_origin'][0]+(b-a-1)*step))
    return initial[:,a:b].copy(),data,reduced,info


def defects(x, step):
    p, q, r, s = x[:-1, :-1], x[:-1, 1:], x[1:, :-1], x[1:, 1:]
    a, b = np.cross(q-p, r-p), np.cross(r-s, q-s)
    n = a+b
    horizontal = (n[:, 1:]*n[:, :-1]).sum(-1) < 0
    vertical = (n[1:]*n[:-1]).sum(-1) < 0
    corners = [np.cross(u,v) for u,v in [(q-p,r-p), (q-p,s-q), (s-r,r-p), (s-r,s-q)]]
    bad = geometry_masks(x)[1]
    for corner in corners:
        bad |= (corner*n).sum(-1)<=0
        bad |= np.linalg.norm(corner, axis=-1)<step**2*1e-4
    return bad, horizontal, vertical


def defect_stats(x, step):
    masks = defects(x, step)
    return dict(**geometry_stats(x), near_or_folded=int(masks[0].sum()),
                neighbor_reversals=int(masks[1].sum()+masks[2].sum()))


def bilinear_matrix(uv, shape):
    h, w = shape
    col = np.floor(uv[:, 0]).astype(int).clip(0, w-2)
    row = np.floor(uv[:, 1]).astype(int).clip(0, h-2)
    u, v = uv[:, 0]-col, uv[:, 1]-row
    base = row*w+col
    idx = np.stack([base, base+1, base+w, base+w+1], -1)
    weights = np.stack([(1-u)*(1-v), u*(1-v), (1-u)*v, u*v], -1)
    return sparse.csr_matrix((weights.ravel(), (np.repeat(np.arange(len(uv)), 4), idx.ravel())),
                             shape=(len(uv), h*w))


def repair(xyz, uv, target, arc_weight, step, halos=(2, 4, 8), guide_weights=(0., .1, 1.),
           largest_first=False, seam_weight=0., blend_fractions=(1.,)):
    x = xyz.copy()
    h, w = x.shape[:2]
    interpolation = bilinear_matrix(uv, (h, w))
    original_error = np.linalg.norm(interpolation@x.reshape(-1,3)-target, axis=-1)
    global_limit = np.average(original_error, weights=arc_weight)+1
    records = []
    for halo in halos:
        masks = defects(x, step)
        seeds = masks[0].copy()
        seeds[:, 1:] |= masks[1]; seeds[:, :-1] |= masks[1]
        seeds[1:] |= masks[2]; seeds[:-1] |= masks[2]
        labels, count = ndimage.label(ndimage.binary_dilation(seeds, iterations=halo), np.ones((3, 3)))
        order = np.arange(1,count+1)
        if largest_first:
            areas = np.bincount(labels.ravel(), minlength=count+1)
            order = order[np.argsort(-areas[order], kind='stable')]
        for sequence, label in enumerate(order):
            label = int(label)
            if sequence%20 == 0:
                print(f'repair halo={halo} patch={sequence+1}/{count} label={label}', flush=True)
            cells = labels == label
            vertices = np.zeros((h, w), bool)
            for dy, dx in ((0,0), (0,1), (1,0), (1,1)):
                vertices[dy:dy+h-1, dx:dx+w-1] |= cells
            interior = ndimage.binary_erosion(vertices, structure=np.ones((3, 3)), border_value=0)
            ids = np.flatnonzero(interior)
            if not len(ids):
                continue
            # Harmonic interpolation equations at every free vertex. Boundary XYZ
            # enters the RHS; original defective interior is not a target.
            neighbors = np.stack([ids, ids-1, ids+1, ids-w, ids+w], -1)
            lap = sparse.csr_matrix((np.tile([1., -.25, -.25, -.25, -.25], len(ids)),
                                     (np.repeat(np.arange(len(ids)), 5), neighbors.ravel())), shape=(len(ids), h*w))
            free = lap[:, ids]
            flat = x.reshape(-1, 3).astype(np.float64)
            rhs = -(lap@flat-free@flat[ids])
            if seam_weight:
                # Laplacian changes at the frozen boundary measure changes in
                # its incoming slope. Preserve these while smoothing the inside.
                boundary = ndimage.binary_dilation(interior) & ~interior
                boundary[[0,-1]] = False; boundary[:,[0,-1]] = False
                bid = np.flatnonzero(boundary)
                nb = np.stack([bid,bid-1,bid+1,bid-w,bid+w],-1)
                edge_lap = sparse.csr_matrix((np.tile([1.,-.25,-.25,-.25,-.25],len(bid)),
                    (np.repeat(np.arange(len(bid)),5),nb.ravel())),shape=(len(bid),h*w))[:,ids]
                free = sparse.vstack([free,np.sqrt(seam_weight)*edge_lap],format='csr')
                rhs = np.vstack([rhs,np.sqrt(seam_weight)*(edge_lap@flat[ids])])
            guides = interpolation[:, ids]
            active = np.asarray(abs(guides).sum(1)).ravel() > 0
            guides = guides[active]
            desired = target[active]-(interpolation[active]@flat-guides@flat[ids])
            weights = np.sqrt(arc_weight[active]/step)
            guides = sparse.diags(weights)@guides
            desired *= weights[:, None]
            # Only incident cells and their neighboring cells can change. Include
            # a two-vertex halo so seam neighbor-orientation tests are complete.
            rr, cc = np.unravel_index(ids, (h,w))
            region = (slice(max(0,rr.min()-2), min(h,rr.max()+3)),
                      slice(max(0,cc.min()-2), min(w,cc.max()+3)))
            before_masks = defects(x[region], step)
            before_error = np.linalg.norm(interpolation@flat-target, axis=-1)
            best = None
            for strength in guide_weights:
                lhs = (free.T@free+strength*(guides.T@guides)).tocsc()
                result = spsolve(lhs, free.T@rhs+strength*(guides.T@desired))
                for fraction in blend_fractions:
                    candidate = flat.copy(); candidate[ids] += fraction*(result-flat[ids])
                    candidate = candidate.reshape(x.shape).astype(x.dtype)
                    after_masks = defects(candidate[region], step)
                    reduction = sum(int(a.sum()-b.sum()) for a,b in zip(before_masks, after_masks))
                    fresh = sum(int((b & ~a).sum()) for a,b in zip(before_masks, after_masks))
                    error = np.linalg.norm(interpolation@candidate.reshape(-1,3)-target, axis=-1)
                    local_ok = not active.any() or (np.average(error[active], weights=arc_weight[active]) <=
                        1.25*np.average(before_error[active], weights=arc_weight[active])+5)
                    global_ok = np.average(error, weights=arc_weight) <= global_limit
                    accepted = np.isfinite(candidate).all() and reduction>0 and fresh==0 and local_ok and global_ok
                    records.append(dict(halo=int(halo), patch=label, free_vertices=len(ids), guide_weight=strength,
                                        fraction=fraction, reduction=reduction, new_defects=fresh, local_fit_ok=bool(local_ok),
                                        global_fit_ok=bool(global_ok), eligible=bool(accepted), selected=False))
                    if accepted and (best is None or reduction>best[0]):
                        best = reduction, candidate, len(records)-1
            if best is not None:
                x = best[1]; records[best[2]]['selected'] = True
        print(f'patch repair halo={halo} stats={defect_stats(x, step)}', flush=True)
    return x, records


def run(settings):
    import tifffile
    from vesuvius.tifxyz.types import Tifxyz
    from vesuvius.tifxyz.writer import write_tifxyz
    root, out = Path(settings['repair_from']), Path(settings['output_root'])
    if out.exists():
        raise FileExistsError(out)
    start = time.perf_counter()
    torch.set_num_threads(settings.get('cpu_threads', 8))
    prior = json.loads((root/'config.json').read_text())
    provenance = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                  [root/'config.json', root/'targets.npz', root/'attachments.npz', Path(__file__),
                   Path(__file__).with_name('opt_loss_surface_guard.py'),
                   Path(__file__).with_name('opt_loss_fixed_uv.py'),
                   Path(__file__).with_name('fiber_surface.py'),
                   Path(__file__).with_name('optimizer.py'),
                   *[root/'surface.tifxyz'/f'{c}.tif' for c in 'xyz']]}
    with np.load(root/'targets.npz') as f:
        data = {k: f[k] for k in f.files}
    with np.load(root/'attachments.npz') as f:
        attachments = {k: f[k] for k in f.files}
    initial = np.stack([tifffile.imread(root/'surface.tifxyz'/f'{c}.tif') for c in 'xyz'], -1)
    crop_info=None
    if settings.get('winding_bounds') is not None:
        initial,data,attachments,crop_info=crop_winding_domain(initial,data,attachments,prior,settings['winding_bounds'])
    step = float(data['grid_step'])
    device = settings.get('device', 'cpu')
    def make_targets():
        return restore_targets(data, attachments, initial.shape[:2], step, device,
                               prior['settings'].get('sliding_settings', {}))
    targets = make_targets()
    out.mkdir(parents=True)
    if crop_info is not None:
        (out/'domain.json').write_text(json.dumps(crop_info,indent=2)+'\n')
    np.savez(out/'checkpoint_targets.npz',**data)
    np.savez(out/'checkpoint_attachments.npz',**attachments)
    orientation = None
    if settings.get('untangle_settings') is not None:
        from fiber_surface_untangle import normal_field, area_stats
        field,info = normal_field(prior,attachments,initial,settings['normal_curves'])
        orientation = torch.as_tensor(field,device=device)
        np.savez(out/'orientation.npz',normals=field)
        (out/'orientation.json').write_text(json.dumps(info,indent=2)+'\n')
    def save(name, x, t):
        write_tifxyz(out/f'{name}.tifxyz', Tifxyz(x[...,0], x[...,1], x[...,2], uuid=out.name+'-'+name,
            _scale=(1/step, 1/step), _mask=np.isfinite(x).all(-1), extra=dict(uv_origin=data['uv_origin'].tolist())), compression='lzw')
        metrics=dict(fit=t.statistics(torch.as_tensor(x, device=device)[None]), geometry=defect_stats(x, step))
        if orientation is not None:
            metrics['signed_area']=area_stats(torch.as_tensor(x,device=device),orientation,step)
        return metrics
    before = save('before', initial, targets)
    if orientation is not None:
        from fiber_surface_untangle import untangle
        repaired,records=untangle(initial,targets,orientation,step,**settings['untangle_settings'])
    else:
        guide_weights = targets.data_weights.detach().cpu().numpy() if hasattr(targets, 'data_weights') else data['arc_weight']
        repaired, records = repair(initial, attachments['final_grid_uv'], data['xyz'], guide_weights, step,
                                   **settings.get('repair_settings', {}))
    after = save('repaired', repaired, targets)
    repair_seconds = time.perf_counter()-start
    stages = prior['stages']
    # Continue at the prior final data weight; do not restart its weak ramp.
    if len(stages['stages']) != 1:
        raise ValueError('Repair continuation currently supports one saved stage')
    final_weight = stages['stages'][0]['args']['fixed_uv_ramp'][-1]
    stages['stages'][0]['args']['fixed_uv_ramp'] = [final_weight, final_weight]
    stages['stages'][0]['steps'] = settings.get('continuation_steps', 600)
    stages['stages'][0]['lr'] = settings.get('continuation_lr', 1.)
    output = dict(before=before, repaired=after, repair_seconds=repair_seconds, attempts=records)
    for name, mesh in [('control', initial), ('continued', repaired)]:
        t = make_targets()
        if orientation is not None:
            from fiber_surface_untangle import FeasibleStep
            t.step_guard=FeasibleStep(torch.as_tensor(mesh,device=device),orientation,step)
        else:
            t.geometry_guard = SurfaceGuard(torch.as_tensor(mesh, device=device), settings.get('guard_weight',100.), step)
        history = []
        begin = time.perf_counter()
        final, _ = fit(mesh, t, step=step, stages=stages, progress_fn=lambda **r: history.append(r))
        output[name] = save(name, final, t)
        if orientation is not None: output[name]['step_guard']=t.step_guard.statistics()
        output[name]['seconds'] = time.perf_counter()-begin
        np.savez(out/f'{name}_attachments.npz', **{**attachments, **t.attachment_state()})
        (out/f'{name}_progress.json').write_text(json.dumps(history, indent=2)+'\n')
    output['total_seconds'] = time.perf_counter()-start
    (out/'summary.json').write_text(json.dumps(output, indent=2)+'\n')
    (out/'config.json').write_text(json.dumps(dict(settings=settings, stages=stages, inputs=provenance), indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k!='attempts'}, indent=2), flush=True)
    return output
