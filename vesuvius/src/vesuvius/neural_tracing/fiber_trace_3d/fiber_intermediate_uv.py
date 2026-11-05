"""Absolute per-label winding U / voxel V between shared and voxel geometry."""
import json
import numpy as np
import torch
from .fiber_geometry_cache import width_boundaries, profile_geometry
from . import fiber_soft_winding as soft
from .fiber_affine_geometry import remap_parameter, AffineOffsets
from .fiber_geometry_cache import cached_geometry, paired_gather


class IntermediateUV(torch.nn.Module):
    def __init__(self,config,device):
        super().__init__();self.config=config
        self.scale=float(config.get('intermediate_uv_u_scale',10000.))
        for key,default in [('intermediate_uv_u_scale',10000.),('intermediate_uv_lr',2.),
                            ('intermediate_uv_attach_weight',.001),('intermediate_uv_bend_weight',.0001)]:
            if not np.isfinite(config.get(key,default)) or config.get(key,default)<=0:
                raise ValueError(key+' must be positive and finite')
        self.values=torch.nn.Parameter(torch.empty(0,0,2,device=device))
        self.labels=None;self.active=None
        self.affine=None

    def prepare_affine(self,ext,d):
        if self.config.get('fiber_affine_geometry',False):
            self.affine=AffineOffsets(ext.owner,ext.v_offset_arc,d['start'],d['end'],
                                      ext.values.device,ext.values.dtype)

    def initialize(self,ext,state,widths):
        if self.labels is not None and torch.equal(ext.labels,self.labels) and torch.equal(ext.active,self.active):return
        with torch.no_grad():
            xy=ext.shared_positions(state,widths)
            bounds=width_boundaries(widths)
            initial=torch.stack((soft.uv.u_to_q(xy[...,0],widths,bounds)*self.scale,xy[...,1]),-1)
            if self.affine is not None:
                self.values=self.affine.remap(self.values,initial,self.labels,self.active,
                                             ext.labels,ext.active,ext.optimizer,
                                             reuse=self.config.get('affine_reuse_remap',False),
                                             batched=self.config.get('affine_batched_remap',True))
            else:
                self.values=remap_parameter(self.values,initial,ext.owner,self.labels,self.active,
                                           ext.labels,ext.active,ext.optimizer)
            self.labels=ext.labels.clone();self.active=ext.active.clone()

    @cached_geometry
    def scaled_coordinates(self):
        return self.affine.compose(self.values) if self.affine is not None else self.values

    @cached_geometry
    def coordinates(self):
        xy=self.scaled_coordinates()
        return torch.stack((xy[...,0]/self.scale,xy[...,1]),-1)

    @cached_geometry
    def physical(self,widths):
        qv=self.coordinates();bounds=width_boundaries(widths)
        return torch.stack((soft.uv.q_to_u(qv[...,0],widths,bounds),qv[...,1]),-1)

    def points(self,ext,state,widths,key,h,edges=None,prepared=None,plan=None,shared_winding=False):
        """Interpolate label deviations and arc deviations, zero outside window.

        Integer hypotheses read one persistent label. Continuous guidance samples
        between neighboring labels. Spatial interpolation uses winding coordinates;
        the shared physical baseline remains exact when deviations are zero.
        """
        ids,ds,l,r,t=ext.curve.plans[key] if plan is None else plan
        if edges is not None:ids,ds,l,r,t=(x[edges] for x in (ids,ds,l,r,t))
        if ext.curve.integer_geometry:
            q,p,v=ext.curve.hypothesis_queries(state,widths,(ids,ds,l,r,t),h[...,0])
            offsets=self.deviations(ext,state,widths) if prepared is None else prepared[2]
        elif prepared is None:
            base,travel=ext.curve.mapping_state(state,widths,key,edges)
            offsets=self.deviations(ext,state,widths)
        else:base,travel,offsets=prepared
        if not ext.curve.integer_geometry:
            mapped=h.clone();mapped[...,1]=base[:,None,1];mapped[...,2]=base[:,None,2]
            q,p,v=soft.points(mapped,ext.curve.family[ids,None],travel[:,None],widths,physical_u=True)
        if not len(ext.owner):return q,p,v
        if self.config.get('reuse_objective_geometry',False):
            # All four interpolation corners in one gather. Label validity is
            # identical at both spatial endpoints; compute it once, not four times.
            from .fiber_geometry_kernels import run_interpolation
            delta=run_interpolation(offsets,self.labels,self.active,ids,l,r,t.to(q),h[...,0],
                                    batched_backward=self.config.get('batched_interpolation_backward',False),
                                    fused=self.config.get('fused_geometry',False),
                                    audit_root=self.config.get('output_root') if self.config.get('geometry_kernel_audit') else None)
            return (q,p,v+delta[...,1]) if shared_winding else (q+delta[...,0],p+delta[...,0],v+delta[...,1])
        def at(nodes):
            nodes=nodes.clamp_min(0)
            k=h[...,0];lo=torch.floor(k).long();alpha=k-lo
            def read(label):
                col=label-self.labels[ids,:1]
                valid=(col>=0)&(col<self.labels.shape[1])&(l[:,None]>=0)
                col=col.clamp(0,self.labels.shape[1]-1)
                valid &= self.active[ids].gather(1,col)
                out=paired_gather(offsets,nodes,col)
                return torch.where(valid[...,None],out,0)
            return read(lo)*(1-alpha[...,None])+read(lo+1)*alpha[...,None]
        left=at(l);delta=left+(at(r)-left)*t.to(q)[:,None,None]
        return (q,p,v+delta[...,1]) if shared_winding else (q+delta[...,0],p+delta[...,0],v+delta[...,1])

    @cached_geometry
    def shared_prior(self,ext,state,widths):
        shared=ext.shared_positions(state,widths)
        bounds=width_boundaries(widths)
        return torch.stack((soft.uv.u_to_q(shared[...,0],widths,bounds),shared[...,1]),-1)

    @cached_geometry
    def deviations(self,ext,state,widths):
        prior=self.shared_prior(ext,state,widths)
        return self.coordinates()-prior

    def arrays(self):
        return dict(**(self.affine.arrays('intermediate_affine_',self.values) if self.affine is not None else {}),
                    intermediate_uv_scaled_qv=self.scaled_coordinates().detach().cpu().numpy(),
                    intermediate_uv_u_scale=np.array(self.scale),
                    intermediate_uv_config=np.array(json.dumps(self.config)))

    def restore(self,saved,ext):
        if 'intermediate_uv_scaled_qv' not in saved:raise ValueError('Missing intermediate UV layer')
        if float(saved['intermediate_uv_u_scale'])!=self.scale:raise ValueError('Intermediate U scale changed')
        value=torch.as_tensor(saved['intermediate_uv_scaled_qv'],device=ext.values.device)
        if value.shape!=ext.values.shape:raise ValueError('Intermediate layer topology differs')
        if self.affine is not None:
            if 'intermediate_affine_base' in saved:
                self.values=self.affine.restore(saved,'intermediate_affine_',ext.values.device)
            else:
                self.affine.initialize(value)
                self.values=torch.nn.Parameter(torch.zeros_like(value))
        else:
            if 'intermediate_affine_base' in saved:raise ValueError('Cannot drop affine intermediate parameterization')
            self.values=torch.nn.Parameter(value.clone())
        self.labels=ext.labels.clone();self.active=ext.active.clone()


class PairGeometry:
    """Read-only batch view; reuse normal pair reductions and sigma grouping."""
    def __init__(self,ext,state,widths,edges,prepared=None):
        self.ext,self.state,self.widths,self.edges=ext,state,widths,edges
        self.integer_geometry=ext.curve.integer_geometry
        if prepared is None:
            offsets=ext.intermediate.deviations(ext,state,widths)
            if ext.curve.integer_geometry:
                prepared={side:(state[ext.curve.plans[side][0][edges]],state.new_zeros(len(edges)),offsets) for side in ('a','b')}
            else:prepared={side:(*ext.curve.mapping_state(state,widths,side,edges),offsets) for side in ('a','b')}
        self.prepared=prepared
    def subset(self,index):
        prepared={key:(base[index],travel[index],offsets) for key,(base,travel,offsets) in self.prepared.items()}
        return PairGeometry(self.ext,self.state,self.widths,self.edges[index],prepared)
    def points(self,side,h):
        return self.ext.intermediate.points(self.ext,self.state,self.widths,side,h,self.edges,self.prepared[side],
            shared_winding=self.ext.config.get('pair_predicted_winding',False))

    def pack_groups(self,requests):
        """One ragged geometry evaluation; never pad pairwise probability support.

        Only the linear-size endpoint requests are flattened. Existing grouped
        all-to-all reductions retain their original shapes and normalization.
        """
        if not requests:return []
        indices=torch.cat([i.repeat_interleave(a.shape[1]) for i,a,b in requests])
        a=torch.cat([a.flatten(0,1) for i,a,b in requests])[:,None,:]
        b=torch.cat([b.flatten(0,1) for i,a,b in requests])[:,None,:]
        sides=self.subset(indices).points_both(a,b)
        output=[];start=0
        for i,a,b in requests:
            count=a.shape[0]*a.shape[1]
            values=tuple(tuple(x[start:start+count].reshape(a.shape[:2]) for x in side) for side in sides)
            output.append(PreparedPairGeometry(values,self.integer_geometry));start+=count
        return output

    @profile_geometry
    def points_both(self,a,b):
        """Batch both constraint sides through one mapping/interpolation graph."""
        count=len(a)
        if self.ext.config.get('packed_endpoint_gathers'):
            curve=self.ext.curve
            ids,l,r=curve.pair_plan_indices.index_select(0,self.edges).transpose(0,1).reshape(-1,3).unbind(-1)
            ds,t=curve.pair_plan_values.index_select(0,self.edges).transpose(0,1).reshape(-1,2).unbind(-1)
            plan=ids,ds,l,r,t
        else:
            plan=tuple(torch.cat((x[self.edges],y[self.edges]),0)
                       for x,y in zip(self.ext.curve.plans['a'],self.ext.curve.plans['b']))
        ba,ta,offsets=self.prepared['a'];bb,tb,_=self.prepared['b']
        values=self.ext.intermediate.points(self.ext,self.state,self.widths,'a',
            torch.cat((a,b),0),prepared=(torch.cat((ba,bb),0),torch.cat((ta,tb),0),offsets),plan=plan,
            shared_winding=self.ext.config.get('pair_predicted_winding',False))
        return tuple(v[:count] for v in values),tuple(v[count:] for v in values)


class PreparedPairGeometry:
    """Objective-local view of an already evaluated group, including soft query."""
    def __init__(self,values,integer_geometry=False):self.values=values;self.integer_geometry=integer_geometry
    def points_both(self,a,b):return self.values
    def points(self,side,h):
        values=self.values[0 if side=='a' else 1]
        if values[0].shape[1]==h.shape[1]:return values
        return tuple(x[:,-1:] if h.shape[1]==1 else x[:,:h.shape[1]] for x in values)


class SharedPairGeometry:
    """The same integer query path with no intermediate/local layer."""
    pack_groups=PairGeometry.pack_groups
    integer_geometry=True
    def __init__(self,ext,state,widths,edges,keys=('a','b')):
        self.ext,self.state,self.widths,self.edges=ext,state,widths,edges
        self.keys=keys
    def subset(self,index):return SharedPairGeometry(self.ext,self.state,self.widths,self.edges[index],self.keys)
    def points(self,side,h):
        plan=tuple(x[self.edges] for x in self.ext.curve.plans[self.keys[side=='b']])
        return self.ext.curve.hypothesis_queries(self.state,self.widths,plan,h[...,0])
    def points_both(self,a,b):
        plan=tuple(torch.cat((x[self.edges],y[self.edges])) for x,y in zip(self.ext.curve.plans[self.keys[0]],self.ext.curve.plans[self.keys[1]]))
        values=self.ext.curve.hypothesis_queries(self.state,self.widths,plan,torch.cat((a[...,0],b[...,0])))
        return tuple(x[:len(a)] for x in values),tuple(x[len(a):] for x in values)


def restore_model(d,saved,config=None):
    from .fiber_bucket_uv import BucketUV
    class Snapshot(dict):
        @property
        def files(self):return self.keys()
    cfg=json.loads(str(saved['intermediate_uv_config'])) if config is None else config
    cfg=dict(cfg,shared_curve_integer_geometry=bool(saved.get('shared_curve_integer_geometry',False)))
    ext=BucketUV(d,cfg,'cpu');ext.prepare_fields(d,{})
    ext.restore(Snapshot(saved));ext.cavity.values=torch.as_tensor(saved['winding_sigma'])
    return ext


def saved_points(d,saved,state,widths,ids,ds):
    ext=restore_model(d,saved)
    ext.curve.add_plan('saved_query',np.asarray(ids),np.asarray(ds))
    original=torch.as_tensor(saved['state']);w=torch.as_tensor(widths,dtype=original.dtype)
    requested=torch.as_tensor(state,dtype=original.dtype)[ids,None,:]
    with torch.no_grad():
        return tuple(x[:,0].numpy() for x in ext.intermediate.points(ext,original,w,'saved_query',requested))
