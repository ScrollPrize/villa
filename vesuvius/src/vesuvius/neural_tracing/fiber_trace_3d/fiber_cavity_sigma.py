"""Detached, pair-grouped cavity uncertainty (an approximate loopy controller)."""
import math
import hashlib
import json
import numpy as np
import torch

from . import fiber_soft_winding as soft


def combine(total, prior_mean, prior_var, cap=8.):
    """Precision moments: count, precision, precision*mean, precision*mean².

    The cap reduces precision only. Disagreement is not averaged away by degree.
    """
    count, precision, first, second = total.unbind(-1)
    safe = precision.clamp_min(1e-20)
    mean = first / safe
    disagreement = (second / safe - mean.square()).clamp_min(0)
    variance = (count / cap).clamp_min(1) / safe + disagreement
    return (torch.where(count > .5, mean, prior_mean),
            torch.where(count > .5, variance, prior_var), disagreement)


def grouped_pairs(i, j, delta, variance, n):
    """One observation per unordered pair, including within-pair disagreement."""
    lo, hi = torch.minimum(i, j), torch.maximum(i, j)
    keys, inverse, count = torch.unique(lo*n+hi, return_inverse=True, return_counts=True)
    delta = torch.where(i == lo, delta, -delta)
    sums = delta.new_zeros((len(keys), 3))
    sums.index_add_(0, inverse, torch.stack((delta, delta.square(), variance), -1))
    mean = sums[:, 0]/count
    var = sums[:, 2]/count + (sums[:, 1]/count-mean.square()).clamp_min(0)
    return keys//n, keys % n, mean, var, count


class FixedSigma(torch.nn.Module):
    """Fixed uncertainty with the geometry extension's existing sigma interface."""
    def __init__(self,d,config,device):
        super().__init__()
        sigma=float(config['winding_sigma'])
        if not math.isfinite(sigma) or not 0 < sigma <= config['sigma_max']:
            raise ValueError('Fixed sigma must be positive and within sigma_max')
        self.values=torch.full((len(d['family']),),sigma,device=device)

    def update(self,*args):pass
    def terms(self,*args):return {}
    def pair_kwargs(self,d,edges):
        return dict(sigma_a=self.values[d['i'][edges]],sigma_b=self.values[d['j'][edges]])
    def reference_kwargs(self,d):return dict(sigma_a=self.values[d['reference_owner']])
    def diagnostics(self,state=None):
        return dict(sigma_mode='fixed',sigma_mean=float(self.values.mean()),cavity_rounds=0)
    def arrays(self):return dict(winding_sigma=self.values.cpu().numpy())
    def restore(self,saved):
        if 'winding_sigma' not in saved:return  # Fresh geometry-only initialization.
        if not torch.allclose(self.values,torch.as_tensor(saved['winding_sigma'],device=self.values.device)):
            raise ValueError('Saved uncertainty differs from configured fixed sigma')


class CavitySigma(torch.nn.Module):
    def __init__(self, d, config, device):
        super().__init__()
        _,first,inverse=np.unique(d['source_id'],return_index=True,return_inverse=True)
        if len(first) != len(d['family']) and not config.get('cavity_group_source_pieces',False):
            raise ValueError('Cavity pilot requires whole distinct source fibers')
        self.source_first=torch.as_tensor(first,device=device)
        self.piece_source=torch.as_tensor(inverse,device=device)
        self.config = config
        self.n = len(first)
        self.low = float(config.get('sigma_min', .15))
        self.high = float(config.get('sigma_max', 1.))
        self.prior = float(config.get('cavity_prior_sigma', 1.))
        self.noise = float(config.get('cavity_constraint_sigma', .5))
        self.cap = float(config.get('cavity_neighbor_cap', 8.))
        self.damping = float(config.get('cavity_damping', .5))
        self.interval = int(config.get('cavity_update_every', 10))
        if not (all(math.isfinite(x) for x in (self.low,self.high,self.prior,self.noise,self.cap,self.damping))
                and 0 < self.low < self.high and self.prior > 0 and self.noise > 0
                and self.cap >= 1 and 0 < self.damping <= 1 and self.interval >= 1):
            raise ValueError('Invalid cavity parameters')
        self.ready = False
        self.pending = None
        self.rounds = 0
        filtered,_=soft.constraint_family_filter(d,config.get('filter_constraint_families',False))
        digest=hashlib.sha256()
        digest.update(b'live-geometry-prior-v2')
        for key in ('source_id','family','i','j','ds_a','ds_b','sign','distance','parallel'):
            if key in filtered:
                a=np.asarray(filtered[key])
                digest.update(key.encode());digest.update(str((a.shape,a.dtype.str)).encode());digest.update(a.tobytes())
        digest.update(json.dumps([self.low,self.high,self.prior,self.noise,self.cap,self.damping,
                                 self.interval,config.get('perpendicular_offset',.5)]).encode())
        if config.get('cavity_group_source_pieces',False):digest.update(b'source-grouped-v1')
        if config.get('shared_curve_geometry') and config.get('shared_curve_integer_geometry',True):
            digest.update(b'integer-geometry-v1')
        if config.get('fixed_reference_events'):
            from lasagna.inference_provenance import sha256_file
            digest.update(sha256_file(config['fixed_reference_events']).encode())
        self.fingerprint=digest.hexdigest()

    def geometry_error(self,a,b,fa,fb,da,db,widths,sign,distance,parallel):
        if self.config.get('shared_curve_geometry') and self.config.get('shared_curve_integer_geometry',True):
            from .fiber_implicit_winding import expected_points
            sigma=self.config.get('winding_sigma',.5)
            _,pa,va=expected_points(a,fa,da,widths,sigma)
            _,pb,vb=expected_points(b,fb,db,widths,sigma)
            return soft.coordinate_pair_errors(a,b,fa,fb,pa,pb,va,vb,sign,distance,parallel,
                self.config.get('perpendicular_offset',.5))
        return soft.pair_errors(a,b,fa,fb,da,db,widths,sign,distance,parallel,self.config.get('perpendicular_offset',.5))

    @torch.no_grad()
    def update(self, state, widths, d, iteration, pair_states=None):
        refresh = not self.ready or iteration % self.interval == 0
        previous = (self.mean,self.var,self.shift) if self.ready and refresh else None
        if refresh:
            translation=state[:,0]-state[self.source_first[self.piece_source],0]
            keep = (d['sign'] != 0) & (self.piece_source[d['i']] != self.piece_source[d['j']])
            self.excluded = int((~keep).sum())
            i,j = d['i'][keep],d['j'][keep]
            # Translations and fallback centers follow CURRENT geometry. The
            # controller remains detached and never directly supervises N.
            if pair_states is None:
                astate,bstate,da,db=state[i],state[j],d['ds_a'][keep],d['ds_b'][keep]
            else:
                astate,bstate,da,db=pair_states(state,widths,d,torch.where(keep)[0])
            error,_,_ = self.geometry_error(astate,bstate,d['family'][i],d['family'][j],
                da,db,widths,d['sign'][keep],d['distance'][keep],
                d['parallel'][keep])
            delta = state[j,0]-state[i,0]-error/d['sign'][keep]
            delta=delta+translation[i]-translation[j]
            a,b,shift,noise,count = grouped_pairs(self.piece_source[i],self.piece_source[j],delta.double(),
                torch.full_like(delta,self.noise**2).double(),self.n)
            self.src,self.dst = torch.cat((a,b)),torch.cat((b,a))
            self.shift,self.noise_var = torch.cat((shift,-shift)),torch.cat((noise,noise))
            p = len(a)
            self.reverse = torch.cat((torch.arange(p,2*p,device=state.device),torch.arange(p,device=state.device)))
            self.prior_mean = state[self.source_first,0].detach().double().clone()
            self.prior_var = torch.full_like(self.prior_mean,self.prior**2)
            self.reference_total = self.prior_mean.new_zeros((self.n, 4))
            self.reference_pairs = 0
            if 'reference_owner' in d:
                valid = d['reference_sign'] != 0
                owner = d['reference_owner'][valid]
                source = d['reference_source'][valid]
                error, _, _ = self.geometry_error(state[owner], d['reference_state'][valid],
                    d['family'][owner], d['reference_family'][valid], d['reference_ds'][valid],
                    torch.zeros_like(d['reference_ds'][valid]), widths, d['reference_sign'][valid],
                    d['reference_distance'][valid], d['reference_parallel'][valid])
                target = (state[owner, 0] + error/d['reference_sign'][valid]-translation[owner]).double()
                pairs, inverse, counts = torch.unique(torch.stack((self.piece_source[owner], source), 1),
                    dim=0, return_inverse=True, return_counts=True)
                sums = target.new_zeros((len(pairs), 2)).index_add_(0, inverse,
                    torch.stack((target, target.square()), 1)) / counts[:, None]
                variance = self.noise**2 + (sums[:, 1]-sums[:, 0].square()).clamp_min(0)
                precision = 1/variance
                moments = torch.stack((torch.ones_like(precision), precision,
                    precision*sums[:, 0], precision*sums[:, 0].square()), 1)
                self.reference_total.index_add_(0, pairs[:, 0], moments)
                self.reference_pairs = len(pairs)
            self.mean = self.prior_mean[self.src]+self.shift
            self.var = self.prior_var[self.src]+self.noise_var
            self.event_counts = count
            self.ready = True
            if self.pending is not None:
                for key in ('src','dst','event_counts'):
                    value=torch.as_tensor(self.pending['cavity_'+key],device=state.device)
                    if not torch.equal(getattr(self,key),value):
                        raise ValueError('Cavity restore graph differs')
                self.mean=torch.as_tensor(self.pending['cavity_mean'],device=state.device)
                self.var=torch.as_tensor(self.pending['cavity_var'],device=state.device)
                self.rounds=int(self.pending['cavity_rounds'])
                self.pending=None
            elif previous is not None:
                # Preserve cavity history, but rebase its outgoing translation
                # so old endpoint geometry is not retained as fresh evidence.
                self.mean=previous[0]+self.shift-previous[2]
                self.var=previous[1]
        if iteration % self.interval == 0:
            total, moments = self.moments()
            cavity = total[self.src]-moments[self.reverse]
            cm,cv,_ = combine(cavity,self.prior_mean[self.src],self.prior_var[self.src],self.cap)
            new_mean,new_var = cm+self.shift,cv+self.noise_var
            rate=self.damping
            # Mixture moment damping retains the disagreement between old/new means.
            self.var=(1-rate)*self.var+rate*new_var+rate*(1-rate)*(self.mean-new_mean).square()
            self.mean=(1-rate)*self.mean+rate*new_mean
            self.rounds+=1
        total,_=self.moments()
        self.node_mean,self.node_var,self.disagreement=combine(total,self.prior_mean,self.prior_var,self.cap)
        self.degree=total[:,0]
        self.source_values=self.node_var.sqrt().clamp(self.low,self.high).float()
        self.values=self.source_values[self.piece_source]

    def moments(self):
        precision=1/self.var.clamp_min(self.low**2)
        moments=torch.stack((torch.ones_like(precision),precision,precision*self.mean,precision*self.mean.square()),-1)
        total=moments.new_zeros((self.n,4)).index_add_(0,self.dst,moments)
        total = total + self.reference_total
        return total,moments

    def reference_kwargs(self, d):
        return dict(sigma_a=self.values[d['reference_owner']])

    def pair_kwargs(self,d,edges):
        return dict(sigma_a=self.values[d['i'][edges]],sigma_b=self.values[d['j'][edges]])

    def terms(self,*args):
        return {}

    def diagnostics(self,state=None):
        from .fiber_implicit_winding import diagnostics
        s=self.source_values
        result=dict(sigma_mean=float(s.mean()),sigma_p05=float(s.quantile(.05)),
            sigma_p50=float(s.quantile(.5)),sigma_p95=float(s.quantile(.95)),
            cavity_rounds=self.rounds,cavity_pairs=len(self.src)//2,
            cavity_reference_pairs=self.reference_pairs,
            cavity_excluded_events=self.excluded,cavity_degree_mean=float(self.degree.mean()),
            cavity_isolated=int((self.degree==0).sum()),
            cavity_sigma_upper_percent=float(100*(self.node_var.sqrt()>=self.high).double().mean()),
            cavity_raw_sigma_max=float(self.node_var.sqrt().max()),
            cavity_disagreement_mean=float(self.disagreement.mean()))
        result['cavity_by_degree']={}
        for name,mask in [('zero',self.degree==0),('one',self.degree==1),('two_four',(self.degree>=2)&(self.degree<=4)),('five_plus',self.degree>=5)]:
            result['cavity_by_degree'][name]=dict(count=int(mask.sum()),
                sigma_mean=float(s[mask].mean()) if mask.any() else None,
                disagreement_mean=float(self.disagreement[mask].mean()) if mask.any() else None)
        if state is not None:result.update(diagnostics(state,self.config,self.values))
        return result

    def arrays(self):
        result={ 'cavity_'+key:getattr(self,key).detach().cpu().numpy() for key in
            ('src','dst','shift','noise_var','prior_mean','mean','var','node_mean','node_var','degree','disagreement','event_counts','reference_total') }
        result.update(winding_sigma=self.values.cpu().numpy(),cavity_rounds=np.asarray(self.rounds),
                      cavity_fingerprint=np.asarray(self.fingerprint))
        if self.config.get('cavity_group_source_pieces',False):
            result.update(cavity_piece_source=self.piece_source.cpu().numpy(),
                          cavity_source_first=self.source_first.cpu().numpy())
        return result

    def restore(self,saved):
        if 'cavity_mean' in saved:
            if str(saved['cavity_fingerprint'])!=self.fingerprint:
                raise ValueError('Cavity restore inputs/configuration differ')
            self.pending={k:saved[k].copy() for k in saved.files if k.startswith('cavity_')}
