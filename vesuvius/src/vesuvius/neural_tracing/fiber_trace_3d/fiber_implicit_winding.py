"""Implicit discrete Gaussian winding density, with no learned bucket ladder."""
import argparse
import json
import math
import functools
from pathlib import Path

import torch
from lasagna.inference_provenance import sha256_file
from . import fiber_soft_winding as soft


def hypotheses(state, sigma, maximum_sigma=None, radius_limit=None):
    if torch.is_tensor(sigma):
        if maximum_sigma is None:
            raise ValueError('Tensor sigma requires a fixed support bound')
        sigma = sigma.reshape(-1, 1)
        support_sigma = maximum_sigma
    else:
        support_sigma = sigma if maximum_sigma is None else maximum_sigma
    if not math.isfinite(support_sigma) or support_sigma <= 0:
        raise ValueError('winding_sigma must be finite and positive')
    radius = max(2, math.ceil(6 * support_sigma))
    k = torch.floor(state[:, :1].detach() + .5) + torch.arange(
        -radius, radius + 1, device=state.device, dtype=state.dtype)
    logp = torch.log_softmax(-.5 * ((k - state[:, :1]) / sigma).square(), dim=1)
    if radius_limit is not None and radius_limit<radius:
        k=k[:,radius-radius_limit:radius+radius_limit+1]
        logp=logp[:,radius-radius_limit:radius+radius_limit+1]
    expanded = state[:, None, :].expand(-1, k.shape[1], -1)
    values = torch.cat((k[..., None], expanded[..., 1:]), dim=-1)
    return values, logp


def map_state(state):
    """Integer Gaussian mode, preserving U/V; lower label wins exact ties.

    N=5.5,U=2.3 has tied realizations 7.3 and 8.3, never position 7.8.
    Expected losses still retain both hypotheses.
    """
    return torch.cat((torch.ceil(state[:, :1]-.5),state[:, 1:]),dim=1)


def saved_sigma(state, saved, config):
    """Shared readout of the distribution actually used by a saved solution."""
    if config.get('learned_winding_sigma') or config.get('cavity_winding_sigma'):
        sigma=torch.as_tensor(saved['winding_sigma'],dtype=state.dtype,device=state.device)
        if sigma.shape != state.shape[:1] or not torch.isfinite(sigma).all() or not (sigma>0).all():
            raise ValueError('Invalid saved per-fiber winding sigma')
        return sigma, config.get('sigma_max',1.)
    return float(config['winding_sigma']), config.get('sigma_support_max')


class LearnedSigma(torch.nn.Module):
    """Independent bounded uncertainty; geometry state retains its old schema."""
    def __init__(self, d, config, device):
        super().__init__()
        self.config = config
        self.low = float(config.get('sigma_min', .15))
        self.high = float(config.get('sigma_max', 1.))
        self.initial = float(config['winding_sigma'])
        self.weight = float(config.get('sigma_prior_weight', .01))
        if not (0 < self.low < self.initial < self.high and math.isfinite(self.high)
                and math.isfinite(self.weight) and self.weight >= 0):
            raise ValueError('Invalid learned sigma bounds/prior')
        t = (self.initial-self.low)/(self.high-self.low)
        self.logits = torch.nn.Parameter(torch.full((len(d['family']),), math.log(t/(1-t)), device=device))

    def sigma(self):
        return self.low+(self.high-self.low)*self.logits.sigmoid()

    def pair_kwargs(self, d, edges):
        s = self.sigma()
        return dict(sigma_a=s[d['i'][edges]], sigma_b=s[d['j'][edges]])

    def terms(self, state, widths, d, edges):
        return dict(sigma_prior=self.weight*(self.sigma()/self.initial).log().square().mean())

    def diagnostics(self, state=None):
        s = self.sigma()
        result = dict(sigma_mean=float(s.mean()), sigma_min=float(s.min()), sigma_max=float(s.max()),
                      sigma_p05=float(s.quantile(.05)), sigma_p50=float(s.quantile(.5)), sigma_p95=float(s.quantile(.95)))
        if state is not None:
            result.update(diagnostics(state, self.config, s))
        return result

    def arrays(self):
        return dict(winding_sigma=self.sigma().detach().cpu().numpy(),
                    winding_sigma_logits=self.logits.detach().cpu().numpy())

    def restore(self, saved):
        if 'winding_sigma_logits' in saved:
            with torch.no_grad():
                self.logits.copy_(torch.as_tensor(saved['winding_sigma_logits'], device=self.logits.device))
            if not torch.allclose(self.sigma(), torch.as_tensor(saved['winding_sigma'], device=self.logits.device)):
                raise ValueError('Saved sigma bounds differ from current configuration')


def hypothesis_points(s, family, ds, widths, physical_u=False):
    shape = s.shape[:2]
    vals = soft.points(s.reshape(-1, 4), family[:, None].expand(shape).reshape(-1),
        ds[:, None].expand(shape).reshape(-1), widths,physical_u=physical_u)
    return tuple(x.reshape(shape) for x in vals)


def expected_points(state,family,ds,widths,sigma=.5,maximum_sigma=None,physical_u=False):
    """Continuous guidance from integer geometry, not geometry at mean N."""
    h,lp=hypotheses(state,sigma,maximum_sigma)
    return tuple((lp.exp()*x).sum(1) for x in hypothesis_points(h,family,ds,widths,physical_u))


def transverse_weight(config):
    if 'parallel_transverse_weight' in config and 'disable_parallel_transverse_match' in config:
        raise ValueError('Specify parallel_transverse_weight only, not both transverse settings')
    weight = float(config.get('parallel_transverse_weight',
        0. if config.get('disable_parallel_transverse_match', False) else 1.))
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('parallel_transverse_weight must be finite and nonnegative')
    return weight


def sign_weight(config):
    weight = float(config.get('sign_violation_weight', 0.))
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('sign_violation_weight must be finite and nonnegative')
    return weight


def quadratic_blend_alpha(sigma_a, sigma_b, config):
    """Detached uncertainty controller; retain a density gradient at wide sigma."""
    low = float(config.get('quadratic_blend_sigma_low', .25))
    high = float(config.get('quadratic_blend_sigma_high', 1.))
    cap = float(config.get('quadratic_blend_max', .9))
    if not (math.isfinite(low) and math.isfinite(high) and 0 <= low < high
            and math.isfinite(cap) and 0 <= cap < 1):
        raise ValueError('Invalid adaptive quadratic blend thresholds/cap')
    uncertainty = ((sigma_a.detach().square() + sigma_b.detach().square()) * .5).sqrt()
    return cap * ((uncertainty - low) / (high - low)).clamp(0, 1)


def additive_quadratic_weight(config):
    weight = float(config.get('additive_quadratic_weight', 0.))
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('additive_quadratic_weight must be finite and nonnegative')
    if weight and config.get('adaptive_quadratic_blend'):
        raise ValueError('Additive quadratic and adaptive blend are separate experiments')
    return weight


def full_pair_weights(config):
    soft_weight=float(config.get('soft_pair_weight',0))
    probability_weight=float(config.get('probability_pair_weight',1))
    schedule=config.get('pair_weight_schedule')
    if schedule is not None:
        if 'soft_pair_weight' in config or 'probability_pair_weight' in config:
            raise ValueError('Specify fixed pair weights or schedule, not both')
        values=[float(schedule[k]) for k in ('soft_start','soft_end','probability_start','probability_end')]
        if any(not math.isfinite(v) or v<0 for v in values):raise ValueError('Invalid pair schedule weights')
        progress=min(1.,max(0.,float(config.get('_optimization_iteration',0))/max(1,int(config.get('steps',1000)))))
        soft_progress=probability_progress=progress
        if 'probability_ramp_fraction' in schedule:
            fraction=float(schedule['probability_ramp_fraction'])
            if not math.isfinite(fraction) or not 0<fraction<1:
                raise ValueError('Probability ramp fraction must be strictly between zero and one')
            probability_progress=min(1.,progress/fraction)
            soft_progress=max(0.,(progress-fraction)/(1-fraction))
        soft_weight=values[0]+soft_progress*(values[1]-values[0])
        probability_weight=values[2]+probability_progress*(values[3]-values[2])
    if any(not math.isfinite(v) or v<0 for v in (soft_weight,probability_weight)):
        raise ValueError('Pair weights must be finite and nonnegative')
    if soft_weight+probability_weight<=0:raise ValueError('At least one pair objective must be active')
    active=any(k in config for k in ('soft_pair_weight','probability_pair_weight','pair_weight_schedule'))
    if active and (config.get('additive_quadratic_weight',0) or config.get('adaptive_quadratic_blend')):
        raise ValueError('Full pair combination is separate from distance-only guidance')
    return soft_weight,probability_weight


def order_pair_terms(margin, la, lb, sign, config):
    """Shared sign-only/full order objective; measured distance is never used.

    margin_probability preserves the historical restricted-GT objective.
    one_sided_distance uses zero target distance to the allowed half-line,
    with the same quadratic / negative-log mixture structure as distance fitting.
    No thresholded wrong-order probability charge is part of the objective.
    """
    joint=la[:,:,None]+lb[:,None,:]
    mass=la.exp()[:,:,None]*lb.exp()[:,None,:];known=(sign!=0)
    error=(-margin).clamp_min(0)
    mode=config.get('unified_order_loss','margin_probability')
    if mode=='margin_probability':
        quadratic=sign_weight(config)*(mass*error.square()).sum((1,2))*known
        probability=quadratic
    elif mode=='one_sided_distance':
        sigma=float(config.get('distance_sigma',.5))
        if not math.isfinite(sigma) or sigma<=0:raise ValueError('distance_sigma must be positive')
        energy=(error/sigma).square()
        quadratic=sign_weight(config)*(mass*energy).sum((1,2))*known
        probability=-sign_weight(config)*torch.logsumexp((joint-energy).flatten(1),dim=1)*known
    else:
        raise ValueError('Unknown unified_order_loss')
    return quadratic,probability


def restricted_gt_pair_cost(sa,sb,fa,fb,da,db,widths,sign,target,parallel,config,sigma_a,sigma_b,geometry):
    """Explicit GT diagnostic: exact endpoint targets OR only geometric order.

    No Lasagna calibration, measured-distance likelihood, circular phase loss,
    or pairwise V matching. Both weighted branches marginalize over integer
    winding hypotheses using continuous violation magnitude.
    """
    from .fiber_winding_order import order_margin
    mode=config['gt_pair_mode']
    if mode not in ('enclosure','sign_only'):raise ValueError('Unknown GT pair mode')
    def coords(a,b):
        if geometry is not None:
            x,y=geometry.points_both(a,b)
            return x[0],y[0]
        physical=config.get('shared_curve_hypothesis_mapping',False)
        return hypothesis_points(a,fa,da,widths,physical)[0],hypothesis_points(b,fb,db,widths,physical)[0]
    bound=config.get('sigma_max',1.) if sigma_a is not None else config.get('sigma_support_max')
    a,la=hypotheses(sa,config['winding_sigma'] if sigma_a is None else sigma_a,bound)
    b,lb=hypotheses(sb,config['winding_sigma'] if sigma_b is None else sigma_b,bound)
    qa,qb=coords(a,b);delta=qb[:,None,:]-qa[:,:,None]
    mass=la.exp()[:,:,None]*lb.exp()[:,None,:]
    if mode=='enclosure':
        # A single fixed target; distribution uncertainty is in the fibers,
        # not a mixture of possible measured winding-step targets.
        per_pair=((delta-target[:,None,None])/config.get('distance_sigma',.5)).square()
        probability=(mass*per_pair).sum((1,2))
        soft_cost=probability
    else:
        margin=order_margin(delta,sign[:,None,None],(fb-fa)[:,None,None],parallel[:,None,None],config.get('sign_order_leeway',.5))
        # Marginalize the magnitude penalty too: neither a fractional N nor a
        # hard MAP switch is a differentiable realized position for training.
        soft_cost,probability=order_pair_terms(margin,la,lb,sign,config)
    sw,pw=full_pair_weights(config)
    cost=sw*soft_cost+pw*probability
    return cost,cost*0,cost*0


def density_energy_cost(energy,la,lb):
    """Shared probability-branch reduction for measured and exact pair targets."""
    return -torch.logsumexp((la[:,:,None]+lb[:,None,:]-energy).flatten(1),dim=1)


def exact_endpoint_pair_cost(sa,sb,fa,fb,da,db,widths,target,config,sigma_a=None,sigma_b=None,geometry=None):
    """Known endpoint winding step and V equality, over the full joint density.

    Continuation is an exact zero-step observation, not a Lasagna measurement
    or ordering inequality. Use the ordinary hypotheses and geometry mapping;
    evaluate the probability branch before marginalizing endpoint hypotheses.
    Retain the ordinary soft-guidance branch alongside that density objective.
    """
    bound=config.get('sigma_max',1.) if sigma_a is not None else config.get('sigma_support_max')
    a,la=hypotheses(sa,config['winding_sigma'] if sigma_a is None else sigma_a,bound)
    b,lb=hypotheses(sb,config['winding_sigma'] if sigma_b is None else sigma_b,bound)
    if geometry is not None:
        (qa,ua,va),(qb,ub,vb)=geometry.points_both(a,b)
    else:
        physical=config.get('shared_curve_hypothesis_mapping',False)
        qa,ua,va=hypothesis_points(a,fa,da,widths,physical)
        qb,ub,vb=hypothesis_points(b,fb,db,widths,physical)
    mass=la.exp()[:,:,None]*lb.exp()[:,None,:]
    energy=((qb[:,None,:]-qa[:,:,None]-target[:,None,None])/config.get('distance_sigma',.5)).square()
    winding=density_energy_cost(energy,la,lb)
    vertical=(mass*((vb[:,None,:]-va[:,:,None])/config.get('v_pair_scale',1000.)).square()).sum((1,2))
    sw,pw=full_pair_weights(config)
    # Same continuous guidance as ordinary integer-geometry pairs, with a known
    # zero-step target instead of measured-distance hypotheses. This is NOT a
    # replacement for the density branch above.
    delta=sb[:,0]-sa[:,0]+(lb.exp()*ub).sum(1)-(la.exp()*ua).sum(1)
    soft_winding=((delta-target)/config.get('distance_sigma',.5)).square()
    soft_vertical=(((lb.exp()*vb).sum(1)-(la.exp()*va).sum(1))/config.get('v_pair_scale',1000.)).square()
    return sw*soft_winding+pw*winding,winding*0,sw*soft_vertical+pw*vertical


def pair_cost(sa, sb, fa, fb, da, db, widths, sign, distance, parallel, config,offsets_a=None,offsets_b=None,sigma_a=None,sigma_b=None,fixed_b=False,diagnostics=None,geometry=None,exact_endpoint_target=None):
    if exact_endpoint_target is not None:
        if fixed_b or offsets_a is not None or offsets_b is not None:
            raise ValueError('Exact endpoint links require ordinary hypothesis pairs')
        return exact_endpoint_pair_cost(sa,sb,fa,fb,da,db,widths,exact_endpoint_target,
            config,sigma_a,sigma_b,geometry)
    if config.get('unified_order_loss'):
        if fixed_b or offsets_a is not None or offsets_b is not None:
            raise ValueError('Unified order experiment requires ordinary fiber pairs')
        if config.get('gt_pair_mode') not in (None,'sign_only'):
            raise ValueError('Unified order is incompatible with enclosure targets')
        # One common order function for both modes. Full mode ONLY adds the
        # ordinary measured-distance/phase/V terms, with its old sign cost off.
        order=restricted_gt_pair_cost(sa,sb,fa,fb,da,db,widths,sign,distance,parallel,
            dict(config,gt_pair_mode='sign_only'),sigma_a,sigma_b,geometry)
        if config.get('gt_pair_mode')=='sign_only':return order
        additional=pair_cost(sa,sb,fa,fb,da,db,widths,sign,distance,parallel,
            dict(config,unified_order_loss=None,gt_pair_mode=None,sign_violation_weight=0.),
            sigma_a=sigma_a,sigma_b=sigma_b,diagnostics=diagnostics,geometry=geometry)
        return tuple(a+b for a,b in zip(order,additional))
    if config.get('gt_pair_mode'):
        if fixed_b or offsets_a is not None or offsets_b is not None:raise ValueError('GT diagnostic requires ordinary whole-fiber pairs')
        return restricted_gt_pair_cost(sa,sb,fa,fb,da,db,widths,sign,distance,parallel,config,sigma_a,sigma_b,geometry)
    if config.get('pair_sigma_groups') and sigma_a is not None and sigma_b is not None and not fixed_b and offsets_a is None and offsets_b is None and diagnostics is None:
        result=grouped_pair_cost(sa,sb,fa,fb,da,db,widths,sign,distance,parallel,config,sigma_a,sigma_b,geometry)
        if geometry is not None and config.get('pair_audit'):raise ValueError('Use intermediate geometry regression tests for gradient audit')
        if config.get('pair_audit') and torch.is_grad_enabled():
            path=Path(config['output_root'])/'pair_audit.json'
            if not path.exists():
                aa=sa.detach().clone().requires_grad_();bb=sb.detach().clone().requires_grad_()
                ww=widths.detach().clone().requires_grad_()
                full=pair_cost(aa,bb,fa,fb,da,db,ww,sign,distance,parallel,
                    dict(config,pair_sigma_groups=False,pair_compile=False,pair_audit=False),
                    sigma_a=sigma_a,sigma_b=sigma_b)
                candidate=grouped_pair_cost(aa,bb,fa,fb,da,db,ww,sign,distance,parallel,
                    dict(config,pair_audit=False),sigma_a,sigma_b)
                params=(aa,bb,ww)
                a=torch.autograd.grad(sum(x.sum() for x in full),params)
                b=torch.autograd.grad(sum(x.sum() for x in candidate),params)
                from lasagna.inference_provenance import atomic_write
                atomic_write(path,dict(events=len(sa),cost_max_abs=[float((x-y).abs().max().detach()) for x,y in zip(full,candidate)],
                    gradient_max_abs=[float((x-y).abs().max()) for x,y in zip(a,b)],
                    gradient_max_relative=[float((x-y).abs().max()/x.abs().max().clamp_min(1e-20)) for x,y in zip(a,b)]),durable=False)
        return result
    soft_weight,probability_weight=full_pair_weights(config)
    if any(k in config for k in ('soft_pair_weight','probability_pair_weight','pair_weight_schedule')) and (offsets_a is not None or offsets_b is not None):
        raise ValueError('Full pair combination requires ordinary pairs without offsets')
    blend = config.get('adaptive_quadratic_blend', False)
    additive = additive_quadratic_weight(config)
    if (blend or additive) and (fixed_b or offsets_a is not None or offsets_b is not None):
        raise ValueError('Continuous guidance currently requires ordinary pairs without endpoint offsets')
    bound = config.get('sigma_max', 1.) if sigma_a is not None else config.get('sigma_support_max')
    prepared_hypotheses=getattr(geometry,'hypotheses',None)
    if prepared_hypotheses is not None:
        a,la,b,lb=prepared_hypotheses
    else:
        a, la = hypotheses(sa, float(config['winding_sigma']) if sigma_a is None else sigma_a, bound,config.get('_pair_radius'))
        if fixed_b:
            # Reference state is already evaluated at this endpoint with frozen GT
            # widths: one exact hypothesis, not a trainable Gaussian or enabled logit.
            b, lb = sb[:, None, :], sb.new_zeros((len(sb), 1))
        else:
            b, lb = hypotheses(sb, float(config['winding_sigma']) if sigma_b is None else sigma_b, bound,config.get('_pair_radius'))
    physical_u=config.get('shared_curve_hypothesis_mapping',False)
    continuous_geometry=None
    integer_geometry=getattr(geometry,'integer_geometry',False)
    if geometry is not None:
        if fixed_b or offsets_a is not None or offsets_b is not None or blend or additive:
            raise ValueError('Intermediate geometry requires ordinary soft/probability pairs')
        if config.get('reuse_objective_geometry') and (soft_weight or probability_weight!=1) and not integer_geometry:
            # Evaluate the continuous hypothesis alongside the discrete ones:
            # same mapping/interpolation, half as many endpoint dispatches.
            ha=torch.cat((a,sa[:,None,:]),1);hb=torch.cat((b,sb[:,None,:]),1)
            if config.get('batch_geometry_sides',False):
                (qa,ua,va),(qb,ub,vb)=geometry.points_both(ha,hb)
            else:
                qa,ua,va=geometry.points('a',ha)
                qb,ub,vb=geometry.points('b',hb)
            continuous_geometry=(ua[:,-1],ub[:,-1],va[:,-1],vb[:,-1])
            qa,ua,va=(x[:,:-1] for x in (qa,ua,va))
            qb,ub,vb=(x[:,:-1] for x in (qb,ub,vb))
        else:
            qa,ua,va=geometry.points('a',a)
            qb,ub,vb=geometry.points('b',b)
        if integer_geometry:
            # Continuous guidance is a surrogate on expectations, not a request
            # to place the physical curve at a fractional label N.
            continuous_geometry=tuple((lp.exp()*value).sum(1) for lp,value in
                ((la,ua),(lb,ub),(la,va),(lb,vb)))
    else:
        qa, ua, va = hypothesis_points(a, fa, da, widths,physical_u)
        qb, ub, vb = hypothesis_points(b, fb, db, widths,physical_u)
    if fixed_b:
        if bool(torch.any(db != 0)): raise ValueError('Fixed reference must be endpoint-evaluated')
        qb, ub, vb = b[..., 0]+b[..., 1], b[..., 1], b[..., 2]
    phase = ub[:, None, :] - ua[:, :, None]
    if offsets_a is not None:
        _,ua,va=soft.correct_coordinates(qa,ua,va,widths,offsets_a)
        _,ub,vb=soft.correct_coordinates(qb,ub,vb,widths,offsets_b)
        phase=ub[:,None,:]-ua[:,:,None]
    separation=b[:,None,:,0]-a[:,:,None,0]+phase
    # Calibrate measured Lasagna distance once, after recursive/group dispatch.
    # Geometric ordering below remains independent of this calibration.
    lasagna_scale=float(config.get('lasagna_winding_scale',1.))
    if not math.isfinite(lasagna_scale) or lasagna_scale<=0:
        raise ValueError('lasagna_winding_scale must be finite and positive')
    distance=distance*lasagna_scale
    calibration = (~parallel) * (fb - fa) * config.get('perpendicular_offset', .5)
    pred = separation + calibration[:, None, None]
    pred = torch.where(sign[:, None, None] != 0, sign[:, None, None] * pred, pred.abs())
    multiplier = float(config.get('same_sheet_match_multiplier', 1.))
    if not math.isfinite(multiplier) or multiplier < 0:
        raise ValueError('same_sheet_match_multiplier must be finite and >=0')
    reducer=compiled_pair_cost() if config.get('pair_compile') and torch.is_grad_enabled() else pair_reductions
    from .fiber_winding_order import order_margin
    order = order_margin(qb[:,None,:]-qa[:,:,None], sign[:,None,None],
        (fb-fa)[:,None,None], parallel[:,None,None], config.get('sign_order_leeway',.5))
    reduction_args=(pred,distance,la,lb,sign,
        separation,(~parallel)&(fa!=fb),phase,va,vb,
        config.get('distance_sigma',.5),config.get('v_pair_scale',1000.),multiplier,order,calibration)
    if config.get('pair_compile') and config.get('geometry_kernel_audit') and torch.is_grad_enabled():
        from .fiber_geometry_kernels import audit_pair_reduction
        audit_pair_reduction(reduction_args,config['output_root'])
    winding,phase_cost,vertical_cost,wrong_probability=reducer(*reduction_args)
    if blend or additive:
        qa,_,_=soft.points(sa,fa,da,widths,physical_u=physical_u)
        qb,_,_=soft.points(sb,fb,db,widths,physical_u=physical_u)
        quadratic,_=soft.integer_separation_energy(qb-qa,sign,distance,calibration,config.get('distance_sigma',.5))
        if blend:
            siga = sa.new_full((len(sa),), config['winding_sigma']) if sigma_a is None else sigma_a
            sigb = sb.new_full((len(sb),), config['winding_sigma']) if sigma_b is None else sigma_b
            alpha = quadratic_blend_alpha(siga, sigb, config)
            winding = alpha * quadratic + (1 - alpha) * winding
            if diagnostics is not None:
                diagnostics['quadratic_blend_alpha'] = alpha
        else:
            winding = winding + additive * quadratic
        if diagnostics is not None:
            diagnostics['continuous_quadratic'] = quadratic
    if diagnostics is not None:
        # A hard sign count is diagnostic only, never an optimized charge.
        diagnostics['wrong_probability'] = wrong_probability.detach()
    weight = transverse_weight(config)
    if weight != 1:
        phase_cost = phase_cost * torch.where(parallel & (fa == 1) & (fb == 1), weight, 1.)
        vertical_cost = vertical_cost * torch.where(parallel & (fa == 0) & (fb == 0), weight, 1.)
    if soft_weight or probability_weight!=1:
        if geometry is None:
            _,ua,va=soft.points(sa,fa,da,widths,physical_u=physical_u)
            _,ub,vb=soft.points(sb,fb,db,widths,physical_u=physical_u)
            continuous_geometry=(ua,ub,va,vb)
        else:
            if continuous_geometry is None:
                _,ua,va=geometry.points('a',sa[:,None,:]);_,ub,vb=geometry.points('b',sb[:,None,:])
                continuous_geometry=(ua[:,0],ub[:,0],va[:,0],vb[:,0])
        ua,ub,va,vb=continuous_geometry
        delta=sb[:,0]-sa[:,0]+ub-ua
        energy,_=soft.integer_separation_energy(delta,sign,distance,calibration,config.get('distance_sigma',.5))
        phase=soft.circular(ub-ua);vertical=vb-va
        continuous=(energy,phase.square(),
            (vertical/config.get('v_pair_scale',1000.)).square())
        return tuple(soft_weight*c+probability_weight*p for c,p in
            zip(continuous,(winding,phase_cost,vertical_cost)))
    return winding, phase_cost, vertical_cost


@functools.lru_cache(maxsize=1)
def compiled_pair_cost():
    """Compile the same eager algebra; explicit opt-in, fail rather than hide fallback."""
    return torch.compile(pair_reductions,fullgraph=True,dynamic=True,
                         options={'compile_threads':4,'use_fast_math':False})


def pair_reductions(pred,distance,la,lb,sign,separation,perpendicular,phase,va,vb,
                    distance_sigma,v_scale,multiplier,order,calibration=None):
    """Shared pure tensor reduction algebra for eager and fused paths."""
    if calibration is None:calibration=torch.zeros_like(distance)
    energy,same=soft.integer_separation_energy(separation,sign,distance,calibration,distance_sigma)
    joint=la[:,:,None]+lb[:,None,:]
    winding=density_energy_cost(energy,la,lb)
    probability=joint.exp()
    wrong=(probability*((order<0)&(sign[:,None,None]!=0))).sum((1,2))
    if multiplier!=1:
        same=same*perpendicular[:,None,None]
        probability=probability*(1+(multiplier-1)*same)
    phase_cost=(probability*soft.circular(phase).square()).sum((1,2))
    vertical=(vb[:,None,:]-va[:,:,None])/v_scale
    return winding,phase_cost,(probability*vertical.square()).sum((1,2)),wrong


def sigma_group_plan(radius,bounds,batched=False):
    """Stable group partition: preserve original within-group edge order."""
    if batched:
        boundary=radius.new_tensor(bounds)
        group=torch.bucketize(radius,boundary)
        order=torch.argsort(group,stable=True)
        counts=radius.new_zeros(len(bounds)).scatter_add_(0,group,torch.ones_like(group)).tolist()
        result=[];start=0
        for upper,count in zip(bounds,counts):
            if count:result.append((upper,order[start:start+count]))
            start+=count
        return result
    result=[];previous=0
    for upper in bounds:
        index=torch.where((radius>previous)&(radius<=upper))[0];previous=upper
        if len(index):result.append((upper,index))
    return result


def grouped_pair_cost(sa,sb,fa,fb,da,db,widths,sign,distance,parallel,config,sigma_a,sigma_b,geometry=None):
    """Uncertainty-based six-sigma supports, independent of current residual.

    Original full-support normalization is retained. This is a tail approximation,
    not an algebraically exact rewrite of the density likelihood.
    Legacy pair_residual_full_support is opt-in: its hard residual threshold can
    jump the objective and break line searches. Do not enable for smooth solves.
    """
    full=max(2,math.ceil(6*config['sigma_max']))
    with torch.no_grad():
        sigma=torch.maximum(sigma_a,sigma_b)
        radius=torch.ceil(6*sigma+.5).long().clamp(2,full)
        if config.get('pair_residual_full_support',False):
            radius=residual_support_radius(sa,sb,fa,fb,da,db,widths,sign,distance,
                parallel,config,sigma,radius,full)
        bounds=sorted(set([min(full,x) for x in (4,8,16,32,full)]))
    return grouped_pair_cost_with_radius(sa,sb,fa,fb,da,db,widths,sign,distance,
        parallel,config,sigma_a,sigma_b,geometry,radius,bounds)


def residual_support_radius(sa,sb,fa,fb,da,db,widths,sign,distance,parallel,config,sigma,radius,full):
    """Legacy discontinuous residual-triggered support expansion; opt-in only."""
    with torch.no_grad():
        if config.get('reuse_prior_conversion',False):
            _,p,v=soft.points(torch.cat((sa,sb)),torch.cat((fa,fb)),torch.cat((da,db)),widths,
                             physical_u=config.get('shared_curve_hypothesis_mapping',False))
            pa,pb=p.chunk(2);va,vb=v.chunk(2)
            error,_,_=soft.coordinate_pair_errors(sa,sb,fa,fb,pa,pb,va,vb,sign,distance,parallel,
                                                  config.get('perpendicular_offset',.5))
        else:
            error,_,_=soft.pair_errors(sa,sb,fa,fb,da,db,widths,sign,distance,parallel,
                                      config.get('perpendicular_offset',.5),
                                      physical_u=config.get('shared_curve_hypothesis_mapping',False))
        return torch.where(error.abs()>2*sigma,full,radius)


def grouped_pair_cost_with_radius(sa,sb,fa,fb,da,db,widths,sign,distance,parallel,
                                 config,sigma_a,sigma_b,geometry,radius,bounds):
    full=max(2,math.ceil(6*config['sigma_max']))
    result=[sa.new_zeros(len(sa)) for _ in range(3)]
    groups=sigma_group_plan(radius,bounds,config.get('batch_sigma_partition',False))
    prepared=None
    if geometry is not None and config.get('batch_sigma_geometry',False):
        requests=[];group_hypotheses=[]
        soft_weight,probability_weight=full_pair_weights(config)
        continuous=bool(soft_weight or probability_weight!=1) and not getattr(geometry,'integer_geometry',False)
        full_hypotheses=None
        if config.get('batch_full_hypotheses',False):
            # The normalizer already uses the full support in EVERY group.
            # Evaluate that identical expression once for the complete batch,
            # then retain each group's original conservative central slice.
            full_hypotheses=(hypotheses(sa,sigma_a,config['sigma_max']),
                             hypotheses(sb,sigma_b,config['sigma_max']))
        for upper,index in groups:
            if full_hypotheses is None:
                a,la=hypotheses(sa[index],sigma_a[index],config['sigma_max'],upper)
                b,lb=hypotheses(sb[index],sigma_b[index],config['sigma_max'],upper)
            else:
                (aa,lla),(bb,llb)=full_hypotheses
                section=slice(full-upper,full+upper+1)
                a,la=aa[index,section],lla[index,section]
                b,lb=bb[index,section],llb[index,section]
            group_hypotheses.append((a,la,b,lb))
            if continuous:
                a=torch.cat((a,sa[index,None,:]),1);b=torch.cat((b,sb[index,None,:]),1)
            requests.append((index,a,b))
        prepared=geometry.pack_groups(requests)
        if config.get('reuse_packed_hypotheses',False) or full_hypotheses is not None:
            for view,h in zip(prepared,group_hypotheses):view.hypotheses=h
    grouped_values=[]
    for group,(upper,index) in enumerate(groups):
        local=dict(config,pair_sigma_groups=False,_pair_radius=upper)
        values=pair_cost(sa[index],sb[index],fa[index],fb[index],da[index],db[index],
            widths,sign[index],distance[index],parallel[index],local,
            sigma_a=sigma_a[index],sigma_b=sigma_b[index],
            geometry=(prepared[group] if prepared is not None else
                      None if geometry is None else geometry.subset(index)))
        if config.get('batch_group_results',False):
            grouped_values.append(torch.stack(values,-1))
        else:
            result=[out.index_copy(0,index,value) for out,value in zip(result,values)]
    if grouped_values:
        result=sa.new_zeros((len(sa),3)).index_copy(0,torch.cat([i for _,i in groups]),
                                                 torch.cat(grouped_values)).unbind(-1)
    return tuple(result)


GEOMETRY_PERFORMANCE_DEFAULTS = {
    'width_gather_mode':'packed',
    'reuse_constraint_work':True,
    'reuse_objective_geometry':True,
    'affine_reuse_remap':True,
    'reuse_mapping_context':True,
    'batch_geometry_sides':True,
    'packed_endpoint_gathers':True,
    'fixed_shared_queries':True,
    'batch_sigma_geometry':True,
    'batch_layer_regularizers':True,
    'reuse_window_geometry':True,
    'fused_adam':'auto',
    'batch_full_hypotheses':True,
    'batch_group_results':True,
    'batched_interpolation_backward':False,
    'batched_affine_backward':False,
    'batch_sigma_partition':False,
    'reuse_prior_conversion':False,
    'reuse_packed_hypotheses':False,
    'fused_geometry':False,
    'fused_affine_geometry':False,
    'pair_compile':False,
}


def performance_defaults(config):
    """Measured execution policy for the additive three-layer model only.

    Keep explicit overrides and all scientific settings. Historical replay can
    disable injection with geometry_performance_defaults=false. Resolved options
    are saved by the ordinary run provenance machinery.
    """
    config=dict(config)
    if config.get('geometry_performance_defaults',True) and all(config.get(k) for k in
            ('fiber_affine_geometry','shared_curve_geometry','intermediate_uv_geometry')):
        for key,value in GEOMETRY_PERFORMANCE_DEFAULTS.items():config.setdefault(key,value)
    return config


def run(config):
    config=performance_defaults(config)
    soft.validate(config)
    if config.get('bucket_uv_hypotheses'):
        if any(config.get(k) for k in
                ('v_endpoint_offsets','fixed_reference_events','exact_fixed_deformation','learned_winding_sigma')):
            raise ValueError('Bucket UV requires fixed/cavity uncertainty without incompatible geometry extensions')
    full_pair_weights(config)
    if any(k in config for k in ('soft_pair_weight','probability_pair_weight','pair_weight_schedule')) and config.get('v_endpoint_offsets'):
        raise ValueError('Full pair combination does not support fixed references or offsets')
    additive = additive_quadratic_weight(config)
    if additive and (config.get('fixed_reference_events') or config.get('v_endpoint_offsets')):
        raise ValueError('Additive guidance does not support fixed references or offsets')
    if config.get('adaptive_quadratic_blend'):
        quadratic_blend_alpha(torch.ones(1), torch.ones(1), config)
        if config.get('fixed_reference_events') or config.get('v_endpoint_offsets'):
            raise ValueError('Adaptive blend does not support fixed references or offsets')
    sign_weight(config)
    if config.get('cavity_winding_sigma'):
        if any(config.get(k) for k in ('learned_winding_sigma','v_endpoint_offsets',
                                      'fixed_assignment_offset_refinement','exact_fixed_deformation')):
            raise ValueError('Cavity sigma is a separate whole-fiber regular-model experiment')
    if config.get('learned_winding_sigma'):
        if config.get('v_endpoint_offsets') or config.get('fixed_assignment_offset_refinement') or config.get('exact_fixed_deformation'):
            raise ValueError('Learned sigma is a separate regular-model experiment, not offset refinement')
        LearnedSigma({'family':[0]}, config, 'cpu')
        if not math.isfinite(config.get('sigma_lr', .01)) or config.get('sigma_lr', .01) <= 0:
            raise ValueError('sigma_lr must be positive and finite')
    if config.get('offset_application','added_only') not in ('added_only','all_constraints'):
        raise ValueError('Unknown offset_application')
    if config.get('offset_application')=='all_constraints' and not config.get('v_endpoint_offsets'):
        raise ValueError('all_constraints requires endpoint offset extension')
    if config.get('u_endpoint_offsets') and not config.get('v_endpoint_offsets'):
        raise ValueError('U endpoint offsets require the shared V endpoint extension')
    transverse_weight(config)
    hypotheses(torch.zeros(1, 4), float(config['winding_sigma']))
    root = Path(config['output_root'])
    if (root / 'summary.json').exists():
        raise FileExistsError('Completed output exists')
    source = Path(config['initial_solution']).parent
    prepared = json.loads((source / 'config.json').read_text())
    if sha256_file(config['tensor_cache']) != prepared['tensor_sha256']:
        raise ValueError('Initialization and trial use different constraint tensors')
    config = dict(config, soft_samples=prepared['soft_samples'],
        tensor_sha256=sha256_file(config['tensor_cache']),
        samples_sha256=sha256_file(prepared['soft_samples']),
        initial_solution_sha256=sha256_file(config['initial_solution']),
        shared_solver_sha256=sha256_file(soft.__file__))
    if config.get('fixed_reference_events'):
        config['fixed_reference_sha256']=sha256_file(config['fixed_reference_events'])
    if config.get('cavity_winding_sigma'):
        from . import fiber_cavity_sigma as cavity
        config['cavity_solver_sha256']=sha256_file(cavity.__file__)
    if config.get('bucket_uv_hypotheses'):
        from . import fiber_bucket_uv
        config['bucket_uv_solver_sha256']=sha256_file(fiber_bucket_uv.__file__)
    if config.get('shared_curve_geometry'):
        from . import fiber_shared_curve
        if not config.get('shared_curve_hypothesis_mapping'):
            raise ValueError('Shared curve solves require per-hypothesis width mapping; old mapping is audit-only')
        if not config.get('bucket_uv_hypotheses'):raise ValueError('Shared curve requires bucket UV')
        if config.get('pair_data_parallel'):raise ValueError('Shared curve pilot uses single-device pairs')
        if config.get('fixed_reference_events'):raise ValueError('Shared curve pilot has no fixed reference training')
        config['shared_curve_sha256']=sha256_file(fiber_shared_curve.__file__)
    if config.get('pair_data_parallel'):
        from . import fiber_pair_parallel
        config['pair_parallel_sha256']=sha256_file(fiber_pair_parallel.__file__)
    from . import fiber_grouped_gather
    config['width_gather_sha256']=sha256_file(fiber_grouped_gather.__file__)
    config['width_layout_sha256']=sha256_file(soft.uv.__file__)
    if config.get('indexing_audit'):
        raise ValueError('The experimental indexing audit is not part of the reusable solver')
    if config.get('v_endpoint_offsets',False):
        from . import fiber_v_endpoint_offsets as offsets
        config['offset_solver_sha256']=sha256_file(offsets.__file__)
    if config.get('exact_fixed_deformation',False):
        from .fiber_v_endpoint_offsets import run_exact
        return run_exact(config)
    if config.get('estimate_fixed_length_vram',False):
        if not (config.get('straight_fixed_length_geometry') and
                config.get('packed_optimizer')=='torch_lbfgs' and
                0<float(config.get('winding_sigma',.5))<=1. and
                not config.get('learned_winding_sigma') and
                not config.get('cavity_winding_sigma') and
                not config.get('pair_residual_full_support')):
            raise ValueError('VRAM estimate requires fixed-length L-BFGS with fixed sigma in (0,1]')
        import numpy as np
        from .fiber_scaled_lbfgs import estimated_fixed_length_vram
        with np.load(config['tensor_cache']) as src:
            count=len(src['i'])
        budget=estimated_fixed_length_vram(count,config.get('lbfgs_edge_batch_size',count),config.get('winding_sigma',.5))
        config['gpu_manager']=dict(config['gpu_manager'],vram_bytes_per_gpu=budget)
        print(json.dumps(dict(estimated_vram_bytes=budget,edges=count,
                              edge_batch_size=config.get('lbfgs_edge_batch_size',count))),flush=True)
    return soft.uv.launch_tensor_solve(config, __name__, Path(__file__))


def diagnostics(state, config, sigma=None):
    _, logp = hypotheses(state, float(config['winding_sigma']) if sigma is None else sigma,
                         config.get('sigma_max', 1.) if sigma is not None else config.get('sigma_support_max'))
    confidence = logp.exp().amax(1)
    keep = state[:, 3] >= 0
    return dict(soft_pair_weight=full_pair_weights(config)[0],probability_pair_weight=full_pair_weights(config)[1],mean_MAP_probability=float(confidence.mean()),
        enabled_mean_MAP_probability=float(confidence[keep].mean()) if keep.any() else None,
        joint_MAP_above_half_percent=float(100 * (confidence * state[:, 3].sigmoid() >= .5).float().mean()))


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
    config = json.loads(args.config.read_text())
    if sha256_file(config['initial_solution']) != config['initial_solution_sha256']:
        raise ValueError('Initialization changed after preparation')
    if config.get('fixed_reference_events') and sha256_file(config['fixed_reference_events'])!=config['fixed_reference_sha256']:
        raise ValueError('Fixed reference supervision changed after preparation')
    if config.get('shared_solver_sha256') and sha256_file(soft.__file__) != config['shared_solver_sha256']:
        raise ValueError('Shared optimizer changed while waiting for resources')
    config.update(executed_shared_solver_sha256=sha256_file(soft.__file__),
                  executed_density_solver_sha256=sha256_file(__file__))
    if config.get('pair_data_parallel'):
        from . import fiber_pair_parallel
        if sha256_file(fiber_pair_parallel.__file__) != config['pair_parallel_sha256']:
            raise ValueError('Parallel implementation changed while waiting')
    extension_factory = None
    if config.get('shared_curve_geometry'):
        from . import fiber_shared_curve
        if sha256_file(fiber_shared_curve.__file__)!=config['shared_curve_sha256']:
            raise ValueError('Shared curve implementation changed while waiting')
    from . import fiber_grouped_gather
    for key,path in [('width_gather_sha256',fiber_grouped_gather.__file__),('width_layout_sha256',soft.uv.__file__)]:
        if key in config and sha256_file(path)!=config[key]:raise ValueError('Width implementation changed while waiting')

    if config.get('cavity_winding_sigma'):
        from . import fiber_cavity_sigma as cavity
        if sha256_file(cavity.__file__) != config['cavity_solver_sha256']:
            raise ValueError('Cavity implementation changed while waiting for resources')
        extension_factory = cavity.CavitySigma
    if config.get('learned_winding_sigma'):
        extension_factory = LearnedSigma
    if config.get('bucket_uv_hypotheses'):
        from . import fiber_bucket_uv
        if sha256_file(fiber_bucket_uv.__file__)!=config['bucket_uv_solver_sha256']:
            raise ValueError('Bucket UV implementation changed while waiting')
        extension_factory=fiber_bucket_uv.BucketUV
    if config.get('v_endpoint_offsets',False):
        from . import fiber_v_endpoint_offsets as offsets
        if sha256_file(offsets.__file__) != config['offset_solver_sha256']:
            raise ValueError('V-offset implementation changed while waiting for resources')
        extension_factory = offsets.EndpointOffsets
    soft.solve(config, pair_cost=pair_cost, model='implicit_discrete_winding_v1', diagnostics=diagnostics,
               extension_factory=extension_factory)
