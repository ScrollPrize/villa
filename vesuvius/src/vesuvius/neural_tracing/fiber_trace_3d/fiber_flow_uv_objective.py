"""Bounded exact parts of the fixed-winding straight-curve objective."""
import numpy as np
import torch

from .fiber_geometry_cache import width_boundaries
from .fiber_uv_layout import constraint_class_weights
from . import fiber_soft_winding as soft


class FixedFlowObjective:
    """Keep graph columns on the host; only one batch owns an autograd graph.

    Reductions retain the full-graph denominators, including the same-sheet
    count. This is deterministic full-objective optimization, not SGD sampling.
    """
    def __init__(self, curve, state, raw, data, initial_width, config):
        self.curve,self.state,self.raw=curve,state,raw
        self.data,self.initial_width,self.config=data,initial_width,config
        self.batch=int(config.get('batch_size',32768))
        if self.batch<1:raise ValueError('UV batch_size must be positive')
        self.weights=constraint_class_weights(data['parallel'],dict(config,constraint_class_balance=True))
        self.same=(data['fixed_target']==0)&~data['parallel']
        self.same_count=int(self.same.sum())
        if not self.same_count:raise ValueError('No fixed same-sheet perpendicular meetings')
        self.sample_count=np.bincount(data['sample_owner'],minlength=len(state))

    def tensor(self, value, dtype=None):
        return torch.as_tensor(value,device=self.state.device,dtype=dtype or self.state.dtype)

    def query(self, owner, arc, widths):
        owner=self.tensor(owner,torch.long);arc=self.tensor(arc)
        labels=self.state[owner,0,None]
        plan=(owner,arc,None,None,None)
        return tuple(value[:,0] for value in self.curve.hypothesis_queries(self.state,widths,plan,labels))

    def edge_values(self, sl, widths):
        d=self.data
        qa,_,va=self.query(d['i'][sl],d['ds_a'][sl],widths)
        qb,_,vb=self.query(d['j'][sl],d['ds_b'][sl],widths)
        bounds=width_boundaries(widths)
        xa=soft.uv.q_to_u(qa,widths,bounds);xb=soft.uv.q_to_u(qb,widths,bounds)
        return qa,qb,va,vb,xa,xb

    def parts(self):
        d,c=self.data,self.config;m=len(d['i']);n=len(self.state)
        for begin in range(0,m,self.batch):
            sl=slice(begin,min(begin+self.batch,m))
            widths=soft.decode_widths(self.raw,c)
            qa,qb,va,vb,xa,xb=self.edge_values(sl,widths)
            parallel=self.tensor(d['parallel'][sl],torch.bool)
            family=self.tensor(d['family'][d['i'][sl]],torch.long)
            use_u=(~parallel)|(family==1);use_v=(~parallel)|(family==0)
            weight=self.tensor(self.weights[sl]);target=self.tensor(d['fixed_target'][sl])
            same=self.tensor(self.same[sl],torch.bool)
            terms=dict(
                winding=(weight*use_u*(qb-qa-target).square()).sum()/m*c.get('distance_weight',1.),
                vertical=(weight*use_v*((vb-va)/c.get('v_pair_scale',1000.)).square()).sum()/m*c.get('metric_weight',1.),
                meeting=((xa[same]-xb[same]).square()+(va[same]-vb[same]).square()).sum()
                        /self.same_count/c.get('meeting_scale',128.)**2*c.get('meeting_weight',1.))
            yield sum(terms.values()),terms
        for begin in range(0,len(d['sample_owner']),self.batch):
            sl=slice(begin,begin+self.batch);owners=d['sample_owner'][sl]
            widths=soft.decode_widths(self.raw,c)
            _,phase,z=self.query(owners,d['sample_ds'][sl],widths)
            weight=1/self.tensor(self.sample_count[owners])
            terms=dict(phase_prior=(soft.circular(phase-self.tensor(d['sample_phase'][sl])).square()*weight).sum()
                       /n*c.get('prior_weight',.000625),
                       z_prior=(((z-self.tensor(d['sample_z'][sl]))/1000.).square()*weight).sum()
                       /n*c.get('v_prior_weight',.01))
            yield sum(terms.values()),terms
        widths=soft.decode_widths(self.raw,c)
        terms=soft.width_regularization(self.raw,widths,self.initial_width,c)
        yield sum(terms.values()),terms

    def __call__(self):
        terms={}
        for _,part in self.parts():
            for key,value in part.items():terms[key]=terms.get(key,0)+value
        return sum(terms.values()),terms

    @torch.no_grad()
    def errors(self):
        m=len(self.data['i']);same_u=np.empty(self.same_count);same_v=np.empty(self.same_count)
        same_q=np.empty(self.same_count);all_q=np.empty(m);offset=0
        widths=soft.decode_widths(self.raw,self.config)
        for begin in range(0,m,self.batch):
            sl=slice(begin,min(begin+self.batch,m));mask=self.same[sl]
            qa,qb,va,vb,xa,xb=[value.cpu().numpy() for value in self.edge_values(sl,widths)]
            size=int(mask.sum());take=slice(offset,offset+size)
            same_u[take]=(xb-xa)[mask];same_v[take]=(vb-va)[mask];same_q[take]=(qb-qa)[mask]
            all_q[sl]=qb-qa-self.data['fixed_target'][sl];offset+=size
        return same_u,same_v,same_q,all_q

    @torch.no_grad()
    def samples(self):
        d=self.data;count=len(d['sample_owner'])
        out=[np.empty(count) for _ in range(3)]
        widths=soft.decode_widths(self.raw,self.config);bounds=width_boundaries(widths)
        for begin in range(0,count,self.batch):
            sl=slice(begin,begin+self.batch)
            q,_,v=self.query(d['sample_owner'][sl],d['sample_ds'][sl],widths)
            x=soft.uv.q_to_u(q,widths,bounds)
            for target,value in zip(out,(q,x,v)):target[sl]=value.cpu().numpy()
        return out
