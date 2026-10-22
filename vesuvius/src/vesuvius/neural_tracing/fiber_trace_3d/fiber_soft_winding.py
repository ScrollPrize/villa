"""Continuous N/U/V layout with optional fiber rejection; no bucket hypotheses."""
import argparse
import math
import json
from pathlib import Path
import time

import numpy as np
import torch
from .fiber_geometry_cache import width_boundaries

from lasagna.inference_provenance import atomic_write, sha256_file
from . import fiber_uv_layout as uv


def validate(config):
    if config.get('width_prior', 0.) != 0:
        raise ValueError('width_prior was removed: initialization widths are not evidence')
    for name, default in [('steps', 1000), ('batch_size', 32768), ('log_every', 100),
                          ('angular_sample_degrees', 45.), ('distance_sigma', .5),
                          ('v_pair_scale', 1000.), ('v_prior_scale', 1000.),
                          ('continuation_scale', 128.), ('lr', .07348469228349533)]:
        value = float(config.get(name, default))
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
        if name in ('steps', 'batch_size', 'log_every') and value != int(value):
            raise ValueError(f'{name} must be integral')
    if not np.isfinite(float(config.get('width_lr',2.))) or float(config.get('width_lr',2.))<0:
        raise ValueError('width_lr must be finite and nonnegative (zero freezes widths)')
    for name in ('phase_weight', 'distance_weight', 'metric_weight', 'rejection_cost',
                 'entropy_weight', 'prior_weight', 'v_prior_weight',
                 'width_smooth', 'continuation_weight', 'perpendicular_offset'):
        value = float(config.get(name, 0.))
        if not np.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if len(str(config.get('devices', '0')).split(',')) != 1 and not config.get('pair_data_parallel'):
        raise ValueError('Multiple devices require pair_data_parallel')


def circular(x):
    """Signed shortest circular difference, differentiable away from half turns."""
    return torch.remainder(x + .5, 1.) - .5


def constraint_batch_indices(count, batch_size, device):
    """Use the complete graph when it fits; preserve stochastic smaller batches."""
    if count <= 0 or batch_size <= 0:
        raise ValueError('Constraint count and batch size must be positive')
    if batch_size >= count:
        return torch.arange(count, device=device)
    return torch.randint(count, (batch_size,), device=device)


def correct_coordinates(q, phase, vertical, widths, offsets):
    """Apply physical U/V corrections, preserving the original hypothesis label."""
    if offsets is None: return q, phase, vertical
    du,dv=offsets
    while du.ndim<q.ndim:du=du.unsqueeze(-1);dv=dv.unsqueeze(-1)
    boundaries=width_boundaries(widths)
    corrected=uv.u_to_q(uv.q_to_u(q,widths,boundaries)+du,widths,boundaries)
    return corrected,phase+corrected-q,vertical+dv


def points(state, family, ds, widths, offsets=None, physical_u=False):
    n, phase, vertical, _ = state.unbind(-1)
    boundaries = width_boundaries(widths)
    anchor = n + phase
    q = uv.u_to_q(uv.q_to_u(anchor, widths, boundaries)
                  + (ds if physical_u else (family == 0) * ds), widths, boundaries)
    return correct_coordinates(q, q - n, vertical if physical_u else vertical + (family == 1) * ds,widths,offsets)


def pair_errors(sa, sb, fa, fb, da, db, widths, sign, distance, parallel,
                perpendicular_offset=.5, physical_u=False):
    qa, ua, va = points(sa, fa, da, widths,physical_u=physical_u)
    qb, ub, vb = points(sb, fb, db, widths,physical_u=physical_u)
    return coordinate_pair_errors(sa,sb,fa,fb,ua,ub,va,vb,sign,distance,parallel,perpendicular_offset)


def coordinate_pair_errors(sa,sb,fa,fb,ua,ub,va,vb,sign,distance,parallel,perpendicular_offset=.5):
    # Continuous diagnostic residual. Never round a learned endpoint position.
    separation = sb[:, 0] - sa[:, 0] + ub - ua
    calibration = (~parallel) * (fb - fa) * perpendicular_offset
    predicted = torch.where(sign != 0, sign * (separation + calibration),
                            (separation + calibration).abs())
    return predicted - distance, circular(ub - ua), vb - va


def integer_separation_distribution(sign, distance, calibration, sigma, dtype=None):
    """Shared measured distribution over integer relative winding targets."""
    radius=max(2,math.ceil(6*float(sigma)))
    center=torch.where(sign!=0,sign*distance,distance)-calibration
    other=-distance-calibration
    offsets=torch.arange(-radius,radius+1,device=distance.device,dtype=distance.dtype if dtype is None else dtype)
    m=torch.stack((torch.floor(center+.5),torch.floor(other+.5)),1)[...,None]+offsets
    measured=m+calibration[:,None,None]
    predicted=torch.where(sign[:,None,None]!=0,sign[:,None,None]*measured,measured.abs())
    evidence=-((predicted-distance[:,None,None])/sigma).square()
    # For unsigned measurements include both signs, counting overlapping integer
    # targets once. All masks depend only on recorded measurements, not the solve.
    duplicate=(m[:,1,:,None]==m[:,0,None,:]).any(-1)
    valid_second=(sign==0)[:,None]&~duplicate
    evidence=torch.stack((evidence[:,0],evidence[:,1].masked_fill(~valid_second,-torch.inf)),1)
    m=m.flatten(1);evidence=evidence.flatten(1)
    evidence=evidence-torch.logsumexp(evidence,1,keepdim=True)
    return m,evidence


def integer_separation_energy(delta, sign, distance, calibration, sigma):
    """Smooth integer-target mixture, with no rounding of optimized geometry."""
    m,evidence=integer_separation_distribution(sign,distance,calibration,sigma,delta.dtype)
    shape=(len(delta),)+(1,)*(delta.ndim-1)+(m.shape[1],)
    terms=evidence.reshape(shape)-((delta[...,None]-m.reshape(shape))/sigma).square()
    loglike=torch.logsumexp(terms,-1)
    same=(torch.exp(terms-loglike[...,None])*(m==0).reshape(shape)).sum(-1)
    return -loglike,same


def pair_integer_energy(sa,sb,fa,fb,da,db,widths,sign,distance,parallel,
                        perpendicular_offset=.5,sigma=.5,physical_u=False):
    qa,_,_=points(sa,fa,da,widths,physical_u=physical_u)
    qb,_,_=points(sb,fb,db,widths,physical_u=physical_u)
    calibration=(~parallel)*(fb-fa)*perpendicular_offset
    return integer_separation_energy(qb-qa,sign,distance,calibration,sigma)[0]


def decode_widths(raw, config):
    """One width parameterization; direct values require optimizer bounds."""
    mode=config.get('width_parameterization','softplus')
    if mode=='direct':return raw
    if mode=='monotone_additive':return raw.cumsum(0)
    if mode=='monotone_relative':
        scale=raw[0]
        if config.get('soft_minimum_width',False):
            # Identity above the numerical safeguard; C1 positive tail below.
            # The penalty acts on raw scale, so tail gradients cannot trap it.
            tiny=float(config.get('width_positive_guard',.001))
            if not np.isfinite(tiny) or tiny<=0:raise ValueError('Invalid width_positive_guard')
            scale=torch.where(scale>=tiny,scale,tiny/(2.-scale.clamp(max=tiny)/tiny))
        return scale*(1+torch.cat((raw.new_zeros(1),raw[1:].cumsum(0))))
    if mode!='softplus':raise ValueError('Unknown width_parameterization: '+str(mode))
    return torch.nn.functional.softplus(raw)+1e-4


def encode_widths(widths, config):
    mode=config.get('width_parameterization','softplus')
    if mode=='direct':return widths.clone()
    if mode in ('monotone_additive','monotone_relative'):
        minimum=1e-12 if config.get('soft_minimum_width',False) else float(config.get('minimum_winding_width',1000.))
        if not np.isfinite(minimum) or minimum<=0:raise ValueError('minimum_winding_width must be positive')
        increments=torch.diff(widths)
        # GPU cumulative sums can differ by a few ULPs on equal-width
        # plateaus. Only remove that representation roundoff on reload;
        # physically decreasing widths must still fail, not be projected.
        roundoff=8*torch.finfo(widths.dtype).eps*torch.maximum(widths[:-1].abs(),widths[1:].abs())
        if not bool(torch.isfinite(widths).all()) or bool((widths[0]<minimum) | (increments < -roundoff).any()):
            raise ValueError('Monotone widths require nondecreasing initialization above the minimum; no implicit projection')
        increments=increments.clamp_min(0)
        if mode=='monotone_relative':increments=increments/widths[0]
        scale=widths[:1]
        if config.get('soft_minimum_width',False):
            if mode!='monotone_relative':raise ValueError('Soft minimum requires relative widths')
            tiny=float(config.get('width_positive_guard',.001))
            if not np.isfinite(tiny) or tiny<=0:raise ValueError('Invalid width_positive_guard')
            scale=torch.where(scale>=tiny,scale,tiny*(2.-tiny/scale))
        return torch.cat((scale,increments))
    if mode!='softplus':raise ValueError('Unknown width_parameterization: '+str(mode))
    w=widths-1e-4
    return w+torch.log(-torch.expm1(-w))


def objective(state, raw_widths, initial_widths, d, samples, config, edges=None, pair_cost=None, extension=None):
    from .fiber_geometry_cache import geometry_evaluation
    with geometry_evaluation(config.get('reuse_objective_geometry',False),config.get('profile_solver',False)):
        return _objective(state,raw_widths,initial_widths,d,samples,config,edges,pair_cost,extension)


def _objective(state, raw_widths, initial_widths, d, samples, config, edges=None, pair_cost=None, extension=None):
    if config.get('shared_curve_geometry') and getattr(extension,'curve',None) is None:
        raise ValueError('Shared-curve model requires its curve; straight geometry is not a fallback')
    widths = decode_widths(raw_widths,config)
    edges = torch.arange(len(d['i']), device=state.device) if edges is None else edges
    i, j = d['i'][edges], d['j'][edges]
    p = state[:, 3].sigmoid()
    if pair_cost is None or not config.get('reuse_constraint_work'):
        wind, phase, vertical = pair_errors(state[i], state[j], d['family'][i],
            d['family'][j], d['ds_a'][edges], d['ds_b'][edges], widths,
            d['sign'][edges], d['distance'][edges], d['parallel'][edges],
            config.get('perpendicular_offset', .5))
    else:
        # These three terms are replaced below by pair_cost, not added to it.
        wind = phase = vertical = state.new_zeros(())
    gate = d['class_weight'][edges] * p[i] * p[j]
    winding_energy=(pair_integer_energy(state[i],state[j],d['family'][i],d['family'][j],
        d['ds_a'][edges],d['ds_b'][edges],widths,d['sign'][edges],d['distance'][edges],
        d['parallel'][edges],config.get('perpendicular_offset',.5),config.get('distance_sigma',.5))
        if pair_cost is None else wind*0)
    terms = {
        'winding': config.get('distance_weight', .25) *
            (gate * winding_energy).mean(),
        'phase': config.get('phase_weight', 1.) * (gate * phase.square()).mean(),
        'vertical': config.get('metric_weight', .02) *
            (gate * (vertical / config.get('v_pair_scale', 1000.)).square()).mean(),
        'rejection': (d.get('rejection_costs',config.get('rejection_cost',20.)) * (1 - p)).mean(),
        'entropy': config.get('entropy_weight', .025) *
            torch.nn.functional.binary_cross_entropy_with_logits(state[:, 3], p, reduction='mean'),
    }
    if pair_cost is not None:
        kwargs={}
        if extension is not None and hasattr(extension, 'pair_kwargs'):
            kwargs.update(extension.pair_kwargs(d, edges))
        if extension is not None and config.get('offset_application')=='all_constraints':
            kwargs['offsets_a']=extension.edge_offsets(edges,'a')
            kwargs['offsets_b']=extension.edge_offsets(edges,'b')
        sa,sb,da,db=(extension.pair_states(state,widths,d,edges)
                     if extension is not None and hasattr(extension,'pair_states') else
                     (state[i],state[j],d['ds_a'][edges],d['ds_b'][edges]))
        if extension is not None and hasattr(extension,'pair_geometry'):
            kwargs['geometry']=extension.pair_geometry(state,widths,edges)
        wind_cost, phase_cost, vertical_cost = pair_cost(sa, sb,
            d['family'][i], d['family'][j], da, db,
            widths, d['sign'][edges], d['distance'][edges], d['parallel'][edges], config,**kwargs)
        terms.update(winding=config.get('distance_weight', .25) * (gate * wind_cost).mean(),
            phase=config.get('phase_weight', 1.) * (gate * phase_cost).mean(),
            vertical=config.get('metric_weight', .02) * (gate * vertical_cost).mean())
    if 'reference_owner' in d:
        if pair_cost is None: raise ValueError('Fixed references require density pair loss')
        ri=d['reference_owner']; count=len(d['i'])+len(ri)
        kwargs = (extension.reference_kwargs(d) if extension is not None and
                  hasattr(extension, 'reference_kwargs') else {})
        costs=pair_cost(state[ri],d['reference_state'],d['family'][ri],d['reference_family'],
            d['reference_ds'],torch.zeros_like(d['reference_ds']),widths,
            d['reference_sign'],d['reference_distance'],d['reference_parallel'],config,fixed_b=True,**kwargs)
        for name,weight,cost in zip(('winding','phase','vertical'),
                (config.get('distance_weight',.25),config.get('phase_weight',1.),config.get('metric_weight',.02)),costs):
            terms[name]=terms[name]*len(d['i'])/count
            terms['reference_'+name]=weight*(d['reference_class_weight']*p[ri]*cost).sum()/count
    if config.get('_diagnostic_pair_only',False):
        terms={k:terms[k] for k in ('winding','phase','vertical')}
        return sum(terms.values()),terms
    ids = samples['owner']
    def offsets(key):
        return extension.field_offsets(key) if extension is not None and config.get('offset_application')=='all_constraints' else None
    winding, phase, vertical = (extension.field_points(state,widths,'unary')
        if getattr(extension,'curve',None) is not None else
        points(state[ids], d['family'][ids], samples['ds'], widths,offsets('unary')))
    # N centers a distribution over INTEGER labels; it is not geometric phase.
    # A realized endpoint is k+U, whose phase is U modulo one, never N+U.
    phase_error=(extension.phase_prior_error(state,widths,samples['phase'])
        if getattr(extension,'curve',None) is not None and hasattr(extension,'phase_prior_error') else
        circular(phase - samples['phase']).square())
    unary = config.get('prior_weight', .000625) * phase_error
    # Equal weight per piece, independent of length/event density/sample count.
    terms['anchor'] = (p[ids] * unary * samples['weight']).sum() / len(state)
    vi = samples['v_owner']
    _, _, vv = (extension.field_points(state,widths,'v_unary')
        if getattr(extension,'curve',None) is not None else
        points(state[vi], d['family'][vi], samples['v_ds'], widths,offsets('v_unary')))
    terms['vertical_anchor'] = config.get('v_prior_weight', .01) * (p[vi] *
        ((vv - samples['v_z']) / config.get('v_prior_scale', 1000.)).square() *
        samples['v_weight']).sum() / len(state)
    # Radial circumference is an initialization, not a target for the unwrap.
    # Explicit historical recipes may still request their old prior for replay.
    if config.get('soft_minimum_width',False):
        threshold=float(config.get('minimum_winding_width',1000.))
        terms['width_minimum']=config.get('width_minimum_weight',10.)*torch.relu(1.-raw_widths[0]/threshold).square()
    terms['width_smooth'] = config.get('width_smooth', .01) * (
        torch.diff(widths) / initial_widths.mean()).square().mean()
    links = d['links']
    terms['continuation'] = state.sum() * 0
    if len(links):
        a, b = links[:, 0].long(), links[:, 1].long()
        if getattr(extension,'curve',None) is not None:
            from .fiber_intermediate_uv import SharedPairGeometry
            if pair_cost is None:raise ValueError('Shared continuation requires the density pair objective')
            geometry=SharedPairGeometry(extension,state,widths,torch.arange(len(links),device=state.device),('link_a','link_b'))
            sigma=getattr(getattr(extension,'cavity',None),'values',None)
            zero=state.new_zeros(len(links))
            winding,phase,vertical=pair_cost(state[a],state[b],d['family'][a],d['family'][b],
                links[:,2],links[:,3],widths,zero,zero,torch.zeros_like(a,dtype=torch.bool),config,
                sigma_a=None if sigma is None else sigma[a],sigma_b=None if sigma is None else sigma[b],
                geometry=geometry,exact_endpoint_target=zero)
            cost=(config.get('distance_weight',.25)*winding+
                  config.get('phase_weight',1.)*phase+config.get('metric_weight',.02)*vertical)
        else:
            qa, ua, va = points(state[a], d['family'][a], links[:, 2], widths,offsets('link_a'))
            qb, ub, vb = points(state[b], d['family'][b], links[:, 3], widths,offsets('link_b'))
            cost = (qb - qa).square() + circular(ub - ua).square() + (
                (vb - va) / config.get('continuation_scale', 128.)).square()
        terms['continuation'] = config.get('continuation_weight', 10.) * (p[a] * p[b] * cost).mean()
    if extension is not None:
        terms.update(extension.terms(state, widths, d, edges))
    return sum(terms.values()), terms


@torch.no_grad()
def diagnostic_objective(state,raw,widths,data,nodes,config,pair_cost,extension):
    """Exact full-edge mean with bounded pair intermediates; globals once."""
    batch=int(config.get('diagnostic_batch_size',len(data['i'])))
    if batch<=0:raise ValueError('diagnostic_batch_size must be positive')
    if len(data['i'])<=batch:
        return objective(state,raw,widths,data,nodes,config,pair_cost=pair_cost,extension=extension)
    if 'reference_owner' in data:raise ValueError('Batched diagnostics with fixed references not supported')
    terms=None;pair_totals={}
    for start in range(0,len(data['i']),batch):
        edge=torch.arange(start,min(start+batch,len(data['i'])),device=state.device)
        _,part=objective(state,raw,widths,data,nodes,
            dict(config,_diagnostic_pair_only=True) if terms is not None else config,
            edge,pair_cost,extension)
        if terms is None:terms=dict(part)
        for key in ('winding','phase','vertical'):
            pair_totals[key]=pair_totals.get(key,0)+part[key]*(len(edge)/len(data['i']))
    terms.update(pair_totals)
    return sum(terms.values()),terms


def sample_arcs(arc, angle, start, end, step):
    """Sample observed angular travel, including reversals, and exact piece ends."""
    x = np.r_[start, arc[(arc > start) & (arc < end)], end]
    a = np.interp(x, arc, angle)
    travel = np.r_[0., np.cumsum(abs(np.diff(a)))]
    keep = np.r_[True, np.diff(travel) > 0]
    interior = np.interp(np.arange(step, travel[-1], step), travel[keep], x[keep])
    return np.unique(np.r_[start, interior, end])


def prepare_samples(config, d, existing=None):
    """Reuse AFV reader and resolved curve geometry, not old three-node sampling."""
    from .fiber_normal_sign_visualization import load_umbilicus_frame
    from vesuvius.neural_tracing.tracers.shared.annotations.afv import read_afv_polylines
    with uv._ro(config['afv']) as db:
        meta = {k: json.loads(v) for k, v in db.execute('SELECT key,value FROM metadata')}
    axis = load_umbilicus_frame(config['umbilicus'], meta['root'])
    sources = np.unique(d['source_id'])
    spacing=config.get('arc_prior_spacing_vx')
    if spacing is not None and (not np.isfinite(spacing) or spacing<=0):
        raise ValueError('arc_prior_spacing_vx must be finite and positive')
    rows = []; vrows = []
    node_positions=None
    if config.get('shared_node_priors_only',False):
        if not config.get('shared_curve_geometry') or len(sources)!=len(d['source_id']):
            raise ValueError('Node-only priors require whole-fiber shared curves')
        from .fiber_v_endpoint_offsets import topology
        mask=(~np.asarray(d['parallel'],dtype=bool))&(d['family'][d['i']]!=d['family'][d['j']])
        top=topology(d,0.,event_mask=mask)
        node_positions={int(source):np.empty(0) for source in sources}
        ids,starts=np.unique(top['v_offset_source'],return_index=True)
        node_positions.update(zip(map(int,ids),np.split(top['v_offset_arc'],starts[1:])))
    event_owner = np.r_[d['i'], d['j']]
    event_ds = np.r_[d['ds_a'], d['ds_b']]
    event_order = np.argsort(event_owner, kind='stable')
    offsets = np.searchsorted(event_owner[event_order], np.arange(len(d['family']) + 1))
    # Group once, avoiding a full piece scan for every source.
    order = np.argsort(d['source_id'], kind='stable')
    grouped = np.split(order, np.flatnonzero(np.diff(d['source_id'][order])) + 1)
    groups = dict(zip(sources, grouped))
    for row, xyz in read_afv_polylines(config['afv'], sources):
        ids = groups[row['id']]
        family = int(d['family'][ids[0]])
        arc, _, angle, _ = uv.fiber_geometry(xyz, family, axis)
        angle *= 1 if config.get('h_convention', 'ccw') == 'ccw' else -1
        for idx in ids:
            start, end = d['start'][idx], d['end'][idx]
            direction = np.sign(d['node_ds'][idx, -1])
            if direction not in (-1, 1):
                raise ValueError('Soft model requires resolved signed-arc cache')
            positions = (node_positions[int(row['id'])] if node_positions is not None else
                (sample_arcs(arc, angle, start, end,
                config.get('angular_sample_degrees', 45.) / 360.) if family == 0 else np.array([start, end])))
            if existing is not None:
                positions=start+direction*existing['ds'][existing['owner']==idx]
            # Add measured supervision, never interpolate targets from the old
            # sparse prior. Keep each family's total prior weight unchanged.
            grid=np.r_[np.arange(start,end,spacing),end] if spacing is not None else None
            if grid is not None and family==0:
                positions=np.unique(np.r_[positions,grid])
            for pos in positions:
                rows.append((idx, direction * (pos - start), np.interp(pos, arc, angle),
                             np.interp(pos, arc, xyz[:, 2]), 1 / len(positions)))
            eds = (direction*(positions-start) if node_positions is not None else
                   event_ds[event_order[offsets[idx]:offsets[idx + 1]]])
            if existing is not None:
                eds=existing['v_ds'][existing['v_owner']==idx]
            if grid is not None and family==1:
                eds=np.unique(np.r_[eds,direction*(grid-start)])
            if node_positions is None and not len(eds):
                if config.get('shared_curve_geometry'):
                    raise ValueError('Shared-curve input contains an unsupported fiber without constraint nodes')
                eds = direction * (np.array([start, end]) - start)
            for ds in eds:
                vrows.append((idx, ds, np.interp(start + direction * ds, arc, xyz[:, 2]), 1 / len(eds)))
    a = np.asarray(rows).reshape(-1,5)
    v = np.asarray(vrows).reshape(-1,4)
    return dict(owner=a[:, 0].astype(np.int64), ds=a[:, 1], phase=a[:, 2], z=a[:, 3], weight=a[:, 4],
                v_owner=v[:, 0].astype(np.int64), v_ds=v[:, 1], v_z=v[:, 2], v_weight=v[:, 3])


def constraint_family_compatible(d):
    parallel = np.asarray(d['parallel'], dtype=bool)
    fa, fb = d['family'][d['i']], d['family'][d['j']]
    valid = np.isin(fa, [0, 1]) & np.isin(fb, [0, 1])
    return valid & (parallel == (fa == fb))


def constraint_family_filter(d, enabled):
    """Filter only edge arrays; retain original piece identities and continuations."""
    parallel = np.asarray(d['parallel'], dtype=bool)
    compatible = constraint_family_compatible(d)
    keep = compatible if enabled else np.ones(len(parallel), dtype=bool)
    stats = {}
    for name, group in [('all', np.ones(len(parallel), dtype=bool)),
                        ('parallel', parallel), ('perpendicular', ~parallel)]:
        total = int(group.sum())
        removed = int((group & ~keep).sum())
        stats[name] = dict(total=total, incompatible=int((group & ~compatible).sum()),
                           removed=removed, retained=total-removed,
                           removed_percent=100*removed/total if total else 0.)
    result = dict(d)
    for key in ('i', 'j', 'ds_a', 'ds_b', 'sign', 'distance', 'parallel',
                'event_ids', 'phase_a', 'phase_b', 'phase_delta'):
        if key in d: result[key] = d[key][keep]
    if not keep.any():
        raise ValueError('Family filtering leaves no constraints')
    return result, stats


def tensor_inputs(d, samples, config, device):
    d, _ = constraint_family_filter(d, config.get('filter_constraint_families', False))
    data = {k: torch.as_tensor(d[k], device=device, dtype=(torch.long if k in ('i', 'j', 'family')
        else torch.bool if k == 'parallel' else torch.float32))
        for k in ('i', 'j', 'family', 'ds_a', 'ds_b', 'sign', 'distance', 'parallel', 'links')}
    data['class_weight'] = torch.as_tensor(uv.constraint_class_weights(d['parallel'],
        dict(config, constraint_class_balance=True)), device=device)
    from .fiber_piece_split import rejection_costs
    data['rejection_costs']=torch.as_tensor(rejection_costs(d,config),device=device,dtype=torch.float32)
    if config.get('fixed_reference_events'):
        with np.load(config['fixed_reference_events']) as src:
            for key in src.files:
                data['reference_'+key]=torch.as_tensor(src[key],device=device,
                    dtype=torch.long if key in ('owner','family','source') else torch.bool if key=='parallel' else torch.float32)
        compatible=data['reference_parallel']==(data['family'][data['reference_owner']]==data['reference_family'])
        if not bool(compatible.all()): raise ValueError('Incompatible fixed reference families')
        joint=np.r_[d['parallel'],data['reference_parallel'].cpu().numpy()]
        weights=uv.constraint_class_weights(joint,dict(config,constraint_class_balance=True))
        data['class_weight']=torch.as_tensor(weights[:len(d['parallel'])],device=device)
        data['reference_class_weight']=torch.as_tensor(weights[len(d['parallel']):],device=device)
    nodes = {k: torch.as_tensor(v, device=device, dtype=torch.long if k.endswith('owner') else torch.float32)
             for k, v in samples.items()}
    return data, nodes


def solve(config, *, device=None, pair_cost=None, model='continuous_soft_winding_v1', diagnostics=None, extension_factory=None):
    """Shared Adam with optional pair data parallelism; CPU for explicit tests."""
    if config.get('width_parameterization','softplus')!='softplus' and config.get('optimizer_kind')!='scaled_lbfgs':
        raise ValueError('Bounded width parameterizations require the bounded solver')
    validate(config)
    device = torch.device('cuda:0' if device is None else device)
    parallel = None
    if config.get('pair_data_parallel'):
        if pair_cost is None or device.type != 'cuda':
            raise ValueError('Pair data parallelism requires CUDA and a pair cost')
        from .fiber_pair_parallel import ParallelPairs
        device_count = len(str(config.get('devices', '0')).split(','))
        if torch.cuda.device_count() < device_count:
            raise ValueError('Manager granted fewer visible GPUs than requested')
        parallel = ParallelPairs(pair_cost, [f'cuda:{i}' for i in range(device_count)])
        pair_cost = parallel
    try:
        from .fiber_grouped_gather import width_gather_mode
        mode=config.get('width_gather_mode','original')
        if parallel is not None and mode!='original':
            raise ValueError('Width gather experiment currently requires single GPU')
        scope=config.get('width_gather_scope','all')
        if scope not in ('all','pairs'):raise ValueError('Unknown width gather scope')
        if scope=='pairs':
            if pair_cost is None:raise ValueError('Pair-scoped width gather requires a pair cost')
            if config.get('indexing_audit'):
                raise ValueError('Indexing audit requires all scope; it tests pair scope internally')
            original_pair=pair_cost
            pair_mode=mode
            def mapped_pair(*args,**kwargs):
                with width_gather_mode(pair_mode):return original_pair(*args,**kwargs)
            pair_cost=mapped_pair
            mode='original'
        with width_gather_mode(mode):
            return _solve(config, device, pair_cost, model, diagnostics, extension_factory, parallel)
    finally:
        if parallel is not None: parallel.close()


def _solve(config, device, pair_cost, model, diagnostics, extension_factory, parallel):
    for key, digest in [('tensor_cache', 'tensor_sha256'), ('soft_samples', 'samples_sha256')]:
        if digest in config and sha256_file(config[key]) != config[digest]:
            raise ValueError(f'Input changed after preparation: {key}')
    torch.manual_seed(int(config.get('seed', 0)))
    with np.load(config['tensor_cache']) as src:
        d = {k: src[k] for k in src.files}
    with np.load(config['soft_samples']) as src:
        samples = {k: src[k] for k in src.files}
    if not len(d['i']) or not len(d['family']):
        raise ValueError('Soft solver requires a nonempty constraint graph')
    centers, initial = uv.radial_initial_geometry(d, config)
    data, nodes = tensor_inputs(d, samples, config, device)
    _, filter_stats = constraint_family_filter(d, config.get('filter_constraint_families', False))
    print(json.dumps(dict(constraint_filter=filter_stats)), flush=True)
    solver_dtype=torch.float64 if config.get('optimizer_kind')=='scaled_lbfgs' else torch.float32
    state = torch.nn.Parameter(torch.as_tensor(np.column_stack((centers, d['phase'], d['z'],
        np.full(len(centers), config.get('initial_enabled_logit', 4.)))), device=device, dtype=solver_dtype))
    widths = torch.as_tensor(initial, device=device, dtype=solver_dtype)
    raw = torch.nn.Parameter(encode_widths(widths,config))
    if config.get('initial_solution'):
        with np.load(config['initial_solution']) as saved:
            for key in ('source_id', 'start', 'end', 'family'):
                if not np.array_equal(saved[key], d[key]):
                    raise ValueError(f'Initialization identity mismatch: {key}')
            with torch.no_grad():
                state.copy_(torch.as_tensor(saved['state'], device=device))
                w = torch.as_tensor(saved['widths'], device=device, dtype=solver_dtype)
                raw.copy_(encode_widths(w,config))
    if config.get('force_keep_fibers', False):
        # Trusted whole curves: no rejection degree of freedom. Float32
        # sigmoid(100) is exactly one; mask gradients including entropy.
        with torch.no_grad(): state[:, 3].fill_(100.)
        state.register_hook(lambda grad: torch.cat((grad[:, :3], torch.zeros_like(grad[:, 3:])), dim=1))
    if config.get('fixed_assignment_offset_refinement', False):
        from .fiber_v_endpoint_offsets import refine_fixed
        return refine_fixed(d, state.detach(), (torch.nn.functional.softplus(raw)+1e-4).detach(), config)
    extension = extension_factory(d,config,device) if extension_factory is not None else None
    if extension is not None and (config.get('offset_application')=='all_constraints' or config.get('shared_curve_geometry')):
        extension.prepare_fields(d,samples)
    groups = [{'params': [state], 'lr': config.get('lr', .07348469228349533)},
              {'params': [raw], 'lr': config.get('width_lr', 2.)}]
    if extension is not None:
        if config.get('initial_solution'):
            with np.load(config['initial_solution']) as saved:
                if hasattr(extension, 'restore'): extension.restore(saved)
                if 'v_offsets' in saved:
                    for key,val in extension.saved.items():
                        if not np.array_equal(saved[key],val): raise ValueError('V-offset initialization topology differs')
                    with torch.no_grad(): extension.values.copy_(torch.as_tensor(saved['v_offsets'],device=device))
                    if 'u_offsets' in saved:
                        if extension.u_values is None: raise ValueError('Cannot silently drop saved U offsets')
                        with torch.no_grad(): extension.u_values.copy_(torch.as_tensor(saved['u_offsets'],device=device))
        custom=extension.optimizer_groups() if hasattr(extension,'optimizer_groups') else None
        parameters = list(extension.parameters())
        if custom is not None:groups.extend(custom)
        elif parameters:
            groups.append({'params': parameters,'lr':config['bucket_uv_lr'] if config.get('bucket_uv_hypotheses') else config.get('sigma_lr', .01)
                           if config.get('learned_winding_sigma') else config['v_offset_lr']})
    if config.get('optimizer_kind')=='scaled_lbfgs':
        from .fiber_scaled_lbfgs import managed_solve
        return managed_solve(config,state,raw,widths,data,nodes,d,extension,pair_cost,diagnostics)
    fused_adam=config.get('fused_adam',False)
    if fused_adam=='auto':fused_adam=device.type=='cuda'
    optimizer = torch.optim.Adam(groups, **({'fused':True} if fused_adam else {}))
    if extension is not None and hasattr(extension,'bind_optimizer'):
        extension.bind_optimizer(optimizer)
    if config.get('initial_optimizer_state'):
        resumed=torch.load(config['initial_optimizer_state'],map_location='cpu',weights_only=True)
        optimizer.load_state_dict(resumed['optimizer'])
        torch.set_rng_state(resumed['cpu_rng'])
        if device.type=='cuda' and resumed.get('cuda_rng') is not None:
            torch.cuda.set_rng_state(resumed['cuda_rng'],device)
    root = Path(config['output_root']); root.mkdir(parents=True, exist_ok=True)
    steps = int(config.get('steps', 1000)); batch = int(config.get('batch_size', 32768))
    started = time.monotonic(); history = []
    winding_history=None
    if config.get('record_winding_history',False):
        from .fiber_winding_history import WindingHistory
        winding_history=WindingHistory(config.get('winding_history_transfer_interval',100))
    first_update_seconds=None
    plateau=None
    convergence=None; converged=False
    if config.get('convergence_monitor'):
        from .fiber_convergence import ConvergenceMonitor
        if config.get('plateau_lr_patience') is not None:
            raise ValueError('Convergence-only monitoring must not change learning rates')
        convergence=ConvergenceMonitor(config['convergence_monitor'])
    if config.get('plateau_lr_patience') is not None:
        from .fiber_plateau_lr import PlateauLR
        plateau=PlateauLR(optimizer,config['plateau_lr_patience'])
    profiler=None
    profile_start=int(config.get('profile_start',5));profile_end=profile_start+int(config.get('profile_updates',3))
    if device.type=='cuda':
        for i in range(len(parallel.devices) if parallel is not None else 1):
            torch.cuda.reset_peak_memory_stats(i)
    effective_batch = min(batch, len(data['i']))
    full_edges = constraint_batch_indices(len(data['i']), batch, device) if batch >= len(data['i']) else None
    for iteration in range(steps + 1):
        if config.get('profile_solver') and iteration==profile_start:
            profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA] if device.type=='cuda' else [torch.profiler.ProfilerActivity.CPU])
            profiler.start()
        if profiler is not None and iteration==profile_end:
            profiler.stop()
            atomic_write(root/'profile.txt',profiler.key_averages().table(sort_by='self_cuda_time_total',row_limit=40),durable=False)
            atomic_write(root/'profile_cpu.txt',profiler.key_averages().table(sort_by='self_cpu_time_total',row_limit=60),durable=False)
            profiler.export_chrome_trace(str(root/'profile_trace.json'))
            profiler=None
        if extension is not None and hasattr(extension, 'update'):
            with torch.profiler.record_function('solver/window_and_cavity'):
                extension.update(state, (torch.nn.functional.softplus(raw)+1e-4).detach(), data, iteration)
        if config.get('indexing_audit'):
            raise ValueError('The experimental indexing audit is not part of the reusable solver')
        iteration_config=(dict(config,_optimization_iteration=iteration)
                          if config.get('pair_weight_schedule') else config)
        plateau_action=None
        if plateau is not None:
            with torch.no_grad():
                loss,terms=diagnostic_objective(state,raw,widths,data,nodes,iteration_config,pair_cost,extension)
            plateau_action=plateau.observe(iteration,float(loss))
        if iteration == 0 or iteration == steps or plateau_action or iteration % int(config.get('log_every', 100)) == 0:
            with torch.no_grad():
                if plateau is None:
                    loss, terms = diagnostic_objective(state, raw, widths, data, nodes, iteration_config, pair_cost, extension)
                p = state[:, 3].sigmoid()
                record = dict(iteration=iteration, loss=float(loss),
                    enabled_percent=float(100 * (p >= .5).float().mean()),
                    mean_enabled_probability=float(p.mean()),
                    width_factor=float(((torch.nn.functional.softplus(raw) + 1e-4) / widths).mean()),
                    terms={k: float(v) for k, v in terms.items()}, seconds=time.monotonic() - started)
                if diagnostics is not None:
                    record.update(diagnostics(state, iteration_config))
                if extension is not None:
                    record.update(extension.diagnostics(state) if config.get('learned_winding_sigma') or config.get('cavity_winding_sigma') else extension.diagnostics())
                if plateau is not None:
                    record.update(lr_phase=plateau.phase,plateau_action=plateau_action,
                                  learning_rates=[g['lr'] for g in optimizer.param_groups])
            history.append(record); print(json.dumps(record), flush=True)
            atomic_write(root / 'history.json', history, durable=False)
            if convergence is not None:
                converged=convergence.observe(record)
                atomic_write(root/'convergence.json',dict(blocks=convergence.blocks,
                    converged=converged,reason=convergence.reason),durable=False)
            checkpoint_every=int(config.get('optimizer_checkpoint_every',0))
            if checkpoint_every and (iteration%checkpoint_every==0 or converged or iteration==steps):
                np.savez(root/'latest.npz',state=state.detach().cpu().numpy(),
                    **(extension.arrays() if extension is not None else {}),
                    widths=(torch.nn.functional.softplus(raw)+1e-4).detach().cpu().numpy(),
                    **{k:d[k] for k in ('source_id','start','end','family')})
                torch.save(dict(optimizer=optimizer.state_dict(),cpu_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None,
                    iteration=iteration),root/'optimizer.pt')
            if plateau is not None:
                atomic_write(root/'plateau_history.json',dict(rows=plateau.rows,events=plateau.events),durable=False)
            if config.get('save_diagnostic_checkpoints',False):
                checkpoints=root/'checkpoints';checkpoints.mkdir(exist_ok=True)
                np.savez(checkpoints/f'{iteration:06d}.npz',state=state.detach().cpu().numpy(),
                    **(extension.curve.arrays() if getattr(extension,'curve',None) is not None else {}),
                    widths=(torch.nn.functional.softplus(raw)+1e-4).detach().cpu().numpy())
        if winding_history is not None:
            winding_history.capture(iteration,state,extension,(torch.nn.functional.softplus(raw)+1e-4).detach())
        if iteration == steps or plateau_action=='stop' or converged: break
        edges = full_edges if full_edges is not None else constraint_batch_indices(len(data['i']), batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.profiler.record_function('solver/objective'):
            loss, _ = objective(state, raw, widths, data, nodes, iteration_config, edges, pair_cost, extension)
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite soft-winding loss')
        with torch.profiler.record_function('solver/backward'):
            loss.backward()
        with torch.profiler.record_function('solver/adam'):
            optimizer.step()
        if iteration==0:
            if device.type=='cuda':torch.cuda.synchronize(device)
            first_update_seconds=time.monotonic()-started
    if winding_history is not None:
        winding_history.save(root/'winding_history.npz',d['source_id'])
    np.savez(root / 'solution.npz', state=state.detach().cpu().numpy(),
        **(extension.arrays() if extension is not None else {}),
        widths=(torch.nn.functional.softplus(raw) + 1e-4).detach().cpu().numpy(),
        **{k: d[k] for k in ('source_id', 'start', 'end', 'family')})
    summary = dict(model=model, config=config, final=history[-1],
                   stopping_reason=('stationary' if converged else 'plateau' if plateau is not None and plateau_action=='stop' else 'step_limit'),
                   effective_batch_size=effective_batch,
                   constraint_sampling=('full_graph' if full_edges is not None else 'with_replacement_minibatch'),
                   angle_prior_coordinate='endpoint_phase_U_modulo_one',
                   pair_device_count=len(parallel.devices) if parallel is not None else 1,
                   cuda_peak_allocated_per_device=[torch.cuda.max_memory_allocated(i) for i in range(len(parallel.devices) if parallel is not None else 1)] if device.type=='cuda' else [],
                   constraint_filter=filter_stats,
                   cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
                   first_update_seconds=first_update_seconds,
                   subsequent_update_seconds=history[-1]['seconds']-first_update_seconds if first_update_seconds is not None else None,
                   solve_wall_seconds=time.monotonic() - started)
    atomic_write(root / 'summary.json', summary, durable=False)
    return summary


def run(config):
    config = dict(config, layout_mode='shared_anchor')
    validate(config)
    if Path(config['output_root'], 'summary.json').exists():
        raise FileExistsError('Completed output exists')
    uv.prepare(config)
    return run_prepared(config)


def run_prepared(config):
    """Launch on an explicitly prepared graph without rebuilding its selection."""
    config = dict(config, layout_mode='shared_anchor')
    validate(config)
    if Path(config['output_root'], 'summary.json').exists():
        raise FileExistsError('Completed output exists')
    with np.load(config['tensor_cache']) as src:
        d = {k: src[k] for k in src.files}
    root = Path(config['output_root']); root.mkdir(parents=True, exist_ok=True)
    samples = root / 'soft_samples.npz'
    np.savez(samples, **prepare_samples(config, d))
    config.update(soft_samples=str(samples), tensor_sha256=sha256_file(config['tensor_cache']),
                  samples_sha256=sha256_file(samples))
    return uv.launch_tensor_solve(config, __name__, Path(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--unmanaged', action='store_true')
    parser.add_argument('--devices')
    args = parser.parse_args()
    if not args.unmanaged:
        config = json.loads(args.config.read_text())
        if args.devices:
            config['devices'] = args.devices
        run(config)
        raise SystemExit(0)
    solve(json.loads(args.config.read_text()))
