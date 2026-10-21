"""Opt-in live, source-grouped, reverse-excluded uncertainty for split pieces."""
import hashlib
import json
import numpy as np
import torch
from .fiber_cavity_sigma import FixedSigma, grouped_pairs


def evidence_moments(mean, variance, reliability):
    precision=reliability/variance.clamp_min(1e-12)
    return torch.stack((reliability,precision,precision*mean,precision*mean.square(),
                        precision*variance,precision.square()*variance),-1)


def combine_evidence(total, center, broad_variance, cap):
    count,p,first,second,pv,p2v=total.unbind(-1)
    safe=p.clamp_min(1e-20)
    mean=first/safe
    scatter=(second/safe-mean.square()).clamp_min(0)
    expected_scatter=(pv/safe-p2v/safe.square()).clamp_min(0)
    disagreement=(scatter-expected_scatter).clamp_min(0)
    support=(count/cap).clamp_min(1)/safe
    valid=p>1e-10
    return (torch.where(valid,mean,center),
            torch.where(valid,support+disagreement,torch.full_like(center,broad_variance)),
            disagreement,count)


def source_moments(dst, source, mean, variance, reliability, n):
    """One observation per recipient/neighbor-source, including sibling scatter."""
    keys,inverse=torch.unique(dst*n+source,return_inverse=True)
    sums=mean.new_zeros((len(keys),4))
    sums.index_add_(0,inverse,torch.stack((reliability,reliability*mean,
        reliability*(variance+mean.square()),torch.ones_like(mean)),-1))
    den=sums[:,0].clamp_min(1e-20)
    m=sums[:,1]/den;v=(sums[:,2]/den-m.square()).clamp_min(1e-12)
    r=mean.new_zeros(len(keys)).scatter_reduce_(0,inverse,reliability,reduce='amax',include_self=True)
    moments=evidence_moments(m,v,r)
    total=mean.new_zeros((n,6)).index_add_(0,keys//n,moments)
    return total,keys,moments


class AgreementSigma(FixedSigma):
    """Means are controller messages only; no loss pulls N to message centers."""
    def __init__(self,d,config,device):
        super().__init__(d,config,device)
        self.config=config;self.n=len(d['family'])
        self.sources=torch.as_tensor(np.unique(d['source_id'],return_inverse=True)[1],device=device)
        self.low=float(config.get('sigma_min',.15));self.high=float(config['sigma_max'])
        self.broad=float(config.get('cavity_prior_sigma',1.))
        self.noise=float(config.get('cavity_constraint_sigma',.25))
        self.link_noise=float(config.get('agreement_continuation_sigma',.05))
        self.cap=float(config.get('cavity_neighbor_cap',8.))
        self.rate=float(config.get('cavity_damping',.5))
        self.interval=int(config.get('cavity_update_every',100))
        if not (0<self.low<self.high and self.broad>0 and self.noise>0 and self.link_noise>0
                and self.cap>=1 and 0<self.rate<=1 and self.interval>=1):raise ValueError('Invalid agreement uncertainty settings')
        digest=hashlib.sha256()
        for k in ('source_id','i','j','ds_a','ds_b','sign','distance','parallel','links'):
            digest.update(np.asarray(d[k]).tobytes())
        digest.update(json.dumps([self.low,self.high,self.broad,self.noise,self.link_noise,self.cap,self.rate,
            self.interval,config.get('perpendicular_offset',.5),config.get('lasagna_winding_scale',1.)]).encode())
        if config.get('continuation_weight',10.)==0:digest.update(b'continuation-disabled')
        self.fingerprint=digest.hexdigest();self.ready=False;self.rounds=0;self.pending=None

    @torch.no_grad()
    def update(self,state,widths,d,iteration,extension=None):
        if self.ready and iteration%self.interval:return
        if extension is None:raise ValueError('Agreement uncertainty requires current endpoint geometry')
        if 'reference_owner' in d:raise ValueError('Use joint GT graph, not external fixed-reference events')
        from .fiber_implicit_winding import hypotheses
        from .fiber_intermediate_uv import SharedPairGeometry
        def offsets(i,j,edges,keys):
            geometry=SharedPairGeometry(extension,state,widths,edges,keys)
            a,la=hypotheses(state[i],self.values[i],self.high)
            b,lb=hypotheses(state[j],self.values[j],self.high)
            (_,ua,_),(_,ub,_)=geometry.points_both(a,b)
            return (la.exp()*ua).sum(1)-(lb.exp()*ub).sum(1)
        edges=torch.where(d['sign']!=0)[0];i=d['i'][edges];j=d['j'][edges]
        calibration=(~d['parallel'][edges])*(d['family'][j]-d['family'][i])*self.config.get('perpendicular_offset',.5)
        from .fiber_soft_winding import integer_separation_distribution
        targets,logmass=integer_separation_distribution(d['sign'][edges].to(state),
            d['distance'][edges].to(state)*self.config.get('lasagna_winding_scale',1.),calibration.to(state),
            self.config.get('distance_sigma',.5))
        mass=logmass.exp();target=(mass*targets).sum(1)
        delta=target+offsets(i,j,edges,('a','b'))
        variance=self.noise**2+(mass*(targets-target[:,None]).square()).sum(1)
        links=d['links']
        if len(links) and self.config.get('continuation_weight',10.)>0:
            a,b=links[:,:2].long().T
            delta=torch.cat((delta,offsets(a,b,torch.arange(len(links),device=state.device),('link_a','link_b'))))
            variance=torch.cat((variance,torch.full_like(a,self.link_noise**2,dtype=variance.dtype)))
            i=torch.cat((i,a));j=torch.cat((j,b))
        a,b,shift,noise,_=grouped_pairs(i,j,delta.double(),variance.double(),self.n)
        src,dst=torch.cat((a,b)),torch.cat((b,a))
        shift=torch.cat((shift,-shift));noise=torch.cat((noise,noise))
        previous=(self.mean,self.var,self.shift) if self.ready else None
        self.src,self.dst,self.shift=src,dst,shift
        center=state[:,0].double();reliability=state[src,3].sigmoid().double()
        if self.pending is not None:
            self.mean=torch.as_tensor(self.pending['agreement_mean'],device=state.device)
            self.var=torch.as_tensor(self.pending['agreement_var'],device=state.device)
            self.mean=self.mean+shift-torch.as_tensor(self.pending['agreement_shift'],device=state.device)
            self.rounds=int(self.pending['agreement_rounds']);self.pending=None
        elif previous is None:
            self.mean=center[src]+shift;self.var=torch.full_like(shift,self.broad**2)+noise
        else:self.mean=previous[0]+shift-previous[2];self.var=previous[1]
        total,keys,moments=source_moments(dst,self.sources[src],self.mean,self.var,reliability,self.n)
        # Remove ALL incoming evidence from the destination's source, not just
        # one reverse event. Split siblings cannot immediately echo confidence.
        remove_key=src*self.n+self.sources[dst]
        loc=torch.searchsorted(keys,remove_key)
        found=(loc<len(keys))&(keys[loc.clamp_max(max(0,len(keys)-1))]==remove_key) if len(keys) else torch.zeros_like(src,dtype=torch.bool)
        subtract=torch.zeros_like(total[src])
        if len(keys):subtract[found]=moments[loc[found]]
        cm,cv,_,_=combine_evidence(total[src]-subtract,center[src],self.broad**2,self.cap)
        new_mean=cm+shift;new_var=cv+noise
        rate=self.rate
        self.var=(1-rate)*self.var+rate*new_var+rate*(1-rate)*(self.mean-new_mean).square()
        self.mean=(1-rate)*self.mean+rate*new_mean
        total,_,_=source_moments(dst,self.sources[src],self.mean,self.var,reliability,self.n)
        self.node_mean,self.node_var,self.disagreement,self.degree=combine_evidence(total,center,self.broad**2,self.cap)
        self.values=self.node_var.sqrt().clamp(self.low,self.high).to(state)
        self.rounds+=1;self.ready=True

    def diagnostics(self,state=None):
        if not self.ready:return super().diagnostics(state)
        s=self.values
        result=dict(sigma_mode='agreement_cavity',sigma_mean=float(s.mean()),sigma_p05=float(s.quantile(.05)),
            sigma_p50=float(s.quantile(.5)),sigma_p95=float(s.quantile(.95)),cavity_rounds=self.rounds,
            cavity_degree_mean=float(self.degree.mean()),cavity_disagreement_mean=float(self.disagreement.mean()),
            cavity_sigma_upper_percent=float(100*(self.node_var.sqrt()>=self.high).double().mean()))
        if state is not None:
            from .fiber_implicit_winding import diagnostics
            result.update(diagnostics(state,self.config,self.values))
        return result

    def arrays(self):
        result=super().arrays()
        if self.ready:
            result.update({'agreement_'+k:getattr(self,k).cpu().numpy() for k in ('mean','var','shift','degree','disagreement')})
            result.update(agreement_rounds=np.array(self.rounds),agreement_fingerprint=np.array(self.fingerprint))
        return result

    def restore(self,saved):
        if 'agreement_fingerprint' not in saved:return
        if str(saved['agreement_fingerprint'])!=self.fingerprint:raise ValueError('Agreement graph/config changed')
        self.pending={k:np.array(saved[k],copy=True) for k in saved if k.startswith('agreement_')}
        self.values=torch.as_tensor(saved['winding_sigma'],device=self.values.device)
