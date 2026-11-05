"""Shared, monotone cross-winding geometry at perpendicular constraint bases."""
import numpy as np
import torch
from .fiber_geometry_cache import width_boundaries, profile_geometry
from . import fiber_soft_winding as soft
from .fiber_affine_geometry import AffineOffsets, direction_cone, arc_length_energy
from .fiber_geometry_cache import cached_geometry


def interpolation_plan(owner, arc, query_owner, query_arc):
    """Linear interpolation AND endpoint extrapolation; no new free knots."""
    left=np.full(len(query_owner),-1,np.int64);right=left.copy()
    alpha=np.zeros(len(left),np.float64)
    for f in np.unique(query_owner):
        knots=np.flatnonzero(owner==f);rows=np.flatnonzero(query_owner==f)
        if not len(knots):raise ValueError(f'Fiber {f} has no perpendicular nodes; filter it before solving')
        if len(knots)==1:left[rows]=right[rows]=knots[0];continue
        a=arc[knots];idx=np.searchsorted(a,query_arc[rows],side='right')-1
        idx=np.clip(idx,0,len(a)-2);l=knots[idx];r=knots[idx+1]
        left[rows]=l;right[rows]=r;alpha[rows]=(query_arc[rows]-arc[l])/(arc[r]-arc[l])
    return left,right,alpha


def inverse_softplus(x):
    return x+torch.log(-torch.expm1(-x))


class SharedCurve(torch.nn.Module):
    def __init__(self,d,owner,arc,config,device):
        super().__init__();self.config=config;self.ready=False;self.plans={}
        self.integer_geometry=config.get('shared_curve_integer_geometry',True)
        self.straight=bool(config.get('straight_fixed_length_geometry',False))
        if self.straight and not self.integer_geometry:
            raise ValueError('Straight fixed-length geometry requires integer hypotheses')
        self.sigma_values=None
        self.meta=d
        for name,value in [('owner',owner),('arc',arc),('family',d['family']),
                           ('start',d['start']),('direction',np.sign(d['node_ds'][:,-1]))]:
            self.register_buffer(name,torch.as_tensor(value,device=device))
        n=len(owner)
        first=np.r_[True,np.diff(owner)!=0] if n else np.zeros(0,bool)
        first_index=np.maximum.accumulate(np.where(first,np.arange(n),0))
        self.register_buffer('first',torch.as_tensor(first,device=device))
        self.register_buffer('first_index',torch.as_tensor(first_index,device=device))
        # Absolute positions can be ~1e5 vx while adjacent knots are <.01 vx.
        # Keep this small geometry state double; pair/bucket tensors stay float32.
        parameter_count=len(d['family']) if self.straight else n
        self.u=torch.nn.Parameter(torch.zeros(parameter_count,device=device,dtype=torch.float64))
        self.v=torch.nn.Parameter(torch.zeros(parameter_count,device=device,dtype=torch.float64))
        self.affine=None
        if config.get('fiber_affine_geometry',False) and not self.straight:
            end=np.asarray(d.get('end',np.asarray(d['start'])+np.abs(d['node_ds'][:,-1])))
            self.affine=AffineOffsets(owner,arc,d['start'],end,device,torch.float64)
            self.affine.initialize(torch.zeros(n,1,2,device=device,dtype=torch.float64))
        triples=np.array([[j-2,j-1,j] for j in range(2,n) if owner[j]==owner[j-2]],dtype=np.int64).reshape(-1,3)
        self.register_buffer('triples',torch.as_tensor(triples,device=device))
        self.add_plan('_anchor',np.arange(len(d['family'])),np.zeros(len(d['family'])))
        for key,default in [('shared_curve_length_weight',10.),('shared_curve_alignment_weight',10.),('shared_curve_bend_weight',.1)]:
            value=float(config.get(key,default))
            if not np.isfinite(value) or value<=0:raise ValueError(key+' must be positive')

    def set_coordinates(self,coordinates):
        """Encode absolute phase/V through positive longitudinal increments."""
        if self.straight:raise ValueError('Straight model has anchors, not independent node coordinates')
        with torch.no_grad():
            values=coordinates.to(device=self.u.device,dtype=torch.float64)
            if self.affine is not None:
                self.affine.initialize(values[:,None,:])
                self.u.zero_();self.v.zero_();self.ready=True
                return
            if not len(values):self.ready=True;return
            signed=(values-values.roll(1,0))*self.direction[self.owner,None]
            longitudinal=signed.gather(1,self.family[self.owner,None].long()).squeeze(1)
            if bool(((longitudinal<0)&~self.first).any()):
                raise ValueError('Shared curve initialization reverses longitudinal direction')
            for axis,param in enumerate((self.u,self.v)):
                delta=signed[:,axis].clamp_min(torch.finfo(values.dtype).tiny)
                param.copy_(torch.where((self.family[self.owner]==axis)&~self.first,
                                       inverse_softplus(delta),values[:,axis]))
        self.ready=True

    def initialize(self,state,widths):
        if self.ready:return
        if self.straight:
            with torch.no_grad():
                self.u.copy_(state[:,1]);self.v.copy_(state[:,2])
            self.ready=True
            return
        if self.integer_geometry:
            from .fiber_implicit_winding import map_state
            state=map_state(state)  # Initialization only; never training through MAP.
        ds=(self.arc-self.start[self.owner])*self.direction[self.owner]
        _,phase,v=soft.points(state[self.owner].double(),self.family[self.owner],ds.double(),widths.double())
        self.set_coordinates(torch.stack((phase,v),-1))

    @cached_geometry
    def coordinates(self):
        if not self.ready:raise RuntimeError('Shared curve not initialized')
        if self.straight:raise ValueError('Straight coordinates require a winding hypothesis and widths')
        if self.affine is not None:
            return self.affine.compose(torch.stack((self.u,self.v),-1)[:,None,:])[:,0]
        result=[]
        for axis,param in enumerate((self.u,self.v)):
            increments=torch.where(self.first,torch.zeros_like(param),torch.nn.functional.softplus(param))
            total=increments.double().cumsum(0)
            longitudinal=param[self.first_index]+self.direction[self.owner].to(param)*(total-total[self.first_index]).to(param)
            result.append(torch.where(self.family[self.owner]==axis,longitudinal,param))
        return torch.stack(result,-1)

    def prepare_fields(self,d,samples):
        queries={'a':(d['i'],d['ds_a']),'b':(d['j'],d['ds_b'])}
        if 'owner' in samples:
            queries.update(unary=(samples['owner'],samples['ds']),v_unary=(samples['v_owner'],samples['v_ds']))
        if len(d['links']):
            queries.update(link_a=(d['links'][:,0].astype(int),d['links'][:,2]),
                           link_b=(d['links'][:,1].astype(int),d['links'][:,3]))
        for key,(ids,ds) in queries.items():
            self.add_plan(key,np.asarray(ids),np.asarray(ds))
        if 'reference_owner' in d:raise ValueError('Shared curve pilot has no fixed reference training')

    def add_plan(self,key,ids,ds):
        arc=self.meta['start'][ids]+ds*np.sign(self.meta['node_ds'][ids,-1])
        plan=interpolation_plan(self.owner.cpu().numpy(),self.arc.cpu().numpy(),ids,arc)
        self.plans[key]=tuple(torch.as_tensor(x,device=self.u.device) for x in (ids,ds,*plan))
        self.fixed_slices={};offset=0
        for name,p in self.plans.items():
            self.fixed_slices[name]=slice(offset,offset+len(p[0]));offset+=len(p[0])
        self.fixed_plan=tuple(torch.cat([p[i] for p in self.plans.values()]) for i in range(5))
        if key in ('a','b') and 'a' in self.plans and 'b' in self.plans:
            # Immutable endpoint records: two index domains/dtypes, rather than
            # ten separate field gathers for every sigma group.
            a,b=self.plans['a'],self.plans['b']
            self.pair_plan_indices=torch.stack([torch.stack([p[i] for i in (0,2,3)],-1) for p in (a,b)],1)
            self.pair_plan_values=torch.stack([torch.stack([p[i] for i in (1,4)],-1) for p in (a,b)],1)

    @profile_geometry
    def evaluate(self,state,widths,key,edges=None):
        if self.integer_geometry:
            from .fiber_implicit_winding import hypotheses
            h,lp=hypotheses(state,self.sigma_values if self.sigma_values is not None else self.config.get('winding_sigma',.5),
                            self.config.get('sigma_max',1.) if self.sigma_values is not None else None)
            plan=self.plans[key]
            if edges is not None:plan=tuple(x[edges] for x in plan)
            ids=plan[0]
            values=self.hypothesis_queries(state,widths,plan,h[ids,:,0])
            return tuple((v*lp[ids].exp()).sum(1).to(state) for v in values)
        if self.config.get('fixed_shared_queries',False):
            result=self.evaluate_fixed(state,widths)[self.fixed_slices[key]]
            if edges is not None:result=result.index_select(0,edges)
            return result.unbind(-1)
        ids,ds,l,r,t=self.plans[key]
        if edges is not None:ids,ds,l,r,t=(x[edges] for x in (ids,ds,l,r,t))
        endpoint_state=state.index_select(0,ids) if self.config.get('packed_endpoint_gathers') else state[ids]
        xy=self.coordinates()
        if self.config.get('packed_endpoint_gathers'):
            corners=xy.index_select(0,torch.stack((l,r),-1).clamp_min(0).flatten()).reshape(-1,2,2)
            a,b=corners.unbind(1)
        else:
            a=xy[l.clamp_min(0)];b=xy[r.clamp_min(0)]
        val=a+(b-a)*t.to(a)[:,None]
        if self.config.get('shared_curve_hypothesis_mapping'):
            ww=widths.double();bounds=width_boundaries(ww)
            n=endpoint_state[:,0].double()
            xa=soft.uv.q_to_u(n+a[:,0],ww,bounds);xb=soft.uv.q_to_u(n+b[:,0],ww,bounds)
            val=torch.stack((soft.uv.u_to_q(xa+(xb-xa)*t,ww,bounds)-n,val[:,1]),-1)
        pp,vv=val.unbind(-1)
        return endpoint_state[:,0]+pp.to(state),pp.to(state),vv.to(state)

    @cached_geometry
    @profile_geometry
    def evaluate_fixed(self,state,widths):
        """All static queries once per objective and dtype; no per-fiber work.

        Group membership is prepared with the plans, not decided on GPU each
        update. All queries use the same node interpolation.
        """
        if self.integer_geometry:
            from .fiber_implicit_winding import hypotheses
            h,lp=hypotheses(state,self.sigma_values if self.sigma_values is not None else self.config.get('winding_sigma',.5),
                            self.config.get('sigma_max',1.) if self.sigma_values is not None else None)
            ids=self.fixed_plan[0]
            values=self.hypothesis_queries(state,widths,self.fixed_plan,h[ids,:,0])
            return torch.stack([(v*lp[ids].exp()).sum(1).to(state) for v in values],-1)
        ids,ds,left,right,t=self.fixed_plan
        regular=torch.arange(len(ids),device=ids.device)
        xy=self.coordinates()
        result=state.new_zeros((len(ids),3))
        if len(regular):
            ii=ids;corner_index=torch.stack((left,right),-1).flatten();alpha=t
            st=state.index_select(0,ii)
            corners=xy.index_select(0,corner_index).reshape(-1,2,2)
            a,b=corners.unbind(1)
            val=a+(b-a)*alpha.to(a)[:,None]
            if self.config.get('shared_curve_hypothesis_mapping'):
                ww=widths.double();bounds=width_boundaries(ww);n=st[:,0].double()
                # Both endpoints in one conversion, instead of one call each.
                x=soft.uv.q_to_u(n[:,None]+corners[...,0],ww,bounds)
                val=torch.stack((soft.uv.u_to_q(x[:,0]+(x[:,1]-x[:,0])*alpha,ww,bounds)-n,val[:,1]),-1)
            phase=val[:,0].to(state);vertical=val[:,1].to(state)
            result=result.index_copy(0,regular,torch.stack((st[:,0]+phase,phase,vertical),-1))
        return result

    def endpoint_state(self,state,widths,key,edges=None):
        ids=self.plans[key][0] if edges is None else self.plans[key][0][edges]
        _,p,v=self.evaluate(state,widths,key,edges)
        return torch.stack((state[ids,0],p,v,state[ids,3]),-1)

    @cached_geometry
    def mapping_context(self,state,widths):
        """One shared double-precision anchor evaluation per objective.

        Keep conversions inside the cache: newly created state.double() tensors
        otherwise defeat identity-based memoization at every caller.
        """
        ww=widths.double();ss=state.double()
        bounds=width_boundaries(ww)
        anchor=self.evaluate(ss,ww,'_anchor')[1]
        return ss,ww,bounds,anchor

    @cached_geometry
    def mapping_state(self,state,widths,key,edges=None):
        """Anchor phase plus physical X travel, remapped for EACH hypothesis.

        V is already evaluated at the query. The source-start anchor is evaluated
        from the same curve, including interpolation/extrapolation.
        """
        if self.integer_geometry:
            raise ValueError('Integer shared geometry requires per-hypothesis queries, not one continuous mapping')
        ids=self.plans[key][0] if edges is None else self.plans[key][0][edges]
        if self.config.get('reuse_mapping_context',False):
            ss,ww,bounds,anchor=self.mapping_context(state,widths)
            endpoint=self.endpoint_state(ss,ww,key,edges)
            phase=anchor[ids]
        else:
            endpoint=self.endpoint_state(state.double(),widths.double(),key,edges)
            phase=self.evaluate(state.double(),widths.double(),'_anchor')[1][ids]
            ww=widths.double();bounds=width_boundaries(ww)
        q=state[ids,0].double()
        dx=soft.uv.q_to_u(q+endpoint[:,1].double(),ww,bounds)-soft.uv.q_to_u(q+phase,ww,bounds)
        base=torch.stack((state[ids,0],phase.to(state),endpoint[:,2].to(state),state[ids,3]),-1)
        return base,dx.to(state)

    def node_hypotheses(self,state,widths,labels,nodes=None):
        """One physical curve shape, shifted to each hypothesis's own widths."""
        if nodes is None:nodes=torch.arange(len(self.owner),device=self.owner.device)
        if self.straight:
            ids=self.owner[nodes];ds=(self.arc[nodes]-self.start[ids])*self.direction[ids]
            q,_,v=self.hypothesis_queries(state,widths,(ids,ds,None,None,None),labels)
            return q,v
        owner=self.owner[nodes];xy=self.coordinates();phase=xy[nodes,0]
        if self.integer_geometry:
            return (labels.to(phase)+phase[:,None]).to(state),xy[nodes,1,None].expand_as(labels).to(state)
        if self.config.get('reuse_mapping_context',False):
            _,ww,bounds,all_anchors=self.mapping_context(state,widths)
            anchor=all_anchors[owner]
        else:
            anchor=self.evaluate(state.double(),widths.double(),'_anchor')[1][owner]
            ww=widths.double();bounds=width_boundaries(ww)
        n=state[owner,0].double()
        dx=soft.uv.q_to_u(n+phase,ww,bounds)-soft.uv.q_to_u(n+anchor,ww,bounds)
        x=soft.uv.q_to_u(labels.double()+anchor[:,None],ww,bounds)+dx[:,None]
        q=soft.uv.u_to_q(x,ww,bounds)
        v=xy[nodes,1,None].expand_as(q)
        return q.to(state),v.to(state)

    def hypothesis_queries(self,state,widths,plan,labels):
        """Physical interpolation/extrapolation independently at each k+U.

        Labels may be fractional only for the explicitly continuous soft-loss
        surrogate. They never supply the geometry of another integer hypothesis.
        """
        ids,ds,l,r,t=plan
        ww=widths.double();bounds=width_boundaries(ww);k=labels.double()
        if self.straight:
            # One exact straight physical curve per integer anchor hypothesis.
            # The inverse uses the SAME learned widths, never a fitted inverse.
            anchor=soft.uv.q_to_u(k+self.u[ids,None],ww,bounds)
            x=anchor+(self.family[ids]==0)[:,None]*ds.double()[:,None]
            q=soft.uv.u_to_q(x,ww,bounds)
            v=(self.v[ids]+(self.family[ids]==1)*ds.double())[:,None].expand_as(k)
            return tuple(z.to(state) for z in (q,q-k,v))
        xy=self.coordinates();a=xy[l];b=xy[r]
        xa=soft.uv.q_to_u(k+a[:,None,0],ww,bounds)
        xb=soft.uv.q_to_u(k+b[:,None,0],ww,bounds)
        iq=soft.uv.u_to_q(xa+(xb-xa)*t[:,None],ww,bounds)
        iv=(a[:,1]+(b[:,1]-a[:,1])*t)[:,None].expand_as(k)
        return tuple(x.to(state) for x in (iq,iq-k,iv))

    def integer_terms(self,state,widths):
        """Expected geometry energy, never energy at a fractional mean label."""
        from .fiber_implicit_winding import hypotheses
        h,lp=hypotheses(state,self.sigma_values if self.sigma_values is not None else self.config.get('winding_sigma',.5),
                        self.config.get('sigma_max',1.) if self.sigma_values is not None else None)
        mass=lp.exp().double();xy=self.coordinates();ww=widths.double()
        q=h[self.owner,:,0].double()+xy[:,None,0]
        pos=torch.stack((soft.uv.q_to_u(q,ww,width_boundaries(ww)),xy[:,None,1].expand_as(q)),-1)
        r=torch.where(~self.first)[0];l=r-1;zero=pos.sum()*0
        if not len(r):return dict(shared_curve_length=zero,shared_curve_alignment=zero,shared_curve_bending=zero)
        gap=(self.arc[r]-self.arc[l]).to(pos);delta=pos[r]-pos[l]
        weights=mass[self.owner[r]];slope=delta/gap[:,None,None]
        transverse=slope.gather(-1,(1-self.family[self.owner[r]])[:,None,None].expand(-1,q.shape[1],1)).squeeze(-1)
        axis=(transverse.square()*weights*gap[:,None]).sum()/gap.sum()
        length=arc_length_energy(delta,gap,active=weights,weighting=self.config.get('length_segment_weighting','arc_length'))
        a,b,c=self.triples.unbind(1);bending=zero
        if len(c):
            dl=(self.arc[b]-self.arc[a]).to(pos);dr=(self.arc[c]-self.arc[b]).to(pos);span=.5*(dl+dr)
            curve=((pos[c]-pos[b])/dr[:,None,None]-(pos[b]-pos[a])/dl[:,None,None])/span[:,None,None]
            bending=((1000*curve).square().sum(-1)*mass[self.owner[b]]*span[:,None]).sum()/span.sum()
        extra={};weight=float(self.config.get('shared_direction_weight',0.))
        if weight:
            cone,_=direction_cone(pos,self.owner,self.arc,self.family,self.direction,self.config,active=mass)
            extra['shared_curve_direction']=weight*cone
        if not torch.is_grad_enabled():
            selected=lp.argmax(1)[self.owner[r]];rows=torch.arange(len(r),device=r.device)
            err=(torch.linalg.vector_norm(slope,dim=-1)-1).abs()[rows,selected]
            signed=slope.gather(-1,self.family[self.owner[r],None,None].expand(-1,q.shape[1],1)).squeeze(-1)*self.direction[self.owner[r],None]
            self.last=dict(shared_curve_length_relative_mean=float(err.mean()),shared_curve_length_relative_max=float(err.max()),
                shared_curve_reversals=int((signed[rows,selected]<0).sum()),
                shared_curve_transverse_slope_p95=float(transverse[rows,selected].abs().quantile(.95)))
        return dict(**extra,shared_curve_length=self.config.get('shared_curve_length_weight',10.)*length,
            shared_curve_alignment=self.config.get('shared_curve_alignment_weight',10.)*axis,
            shared_curve_bending=self.config.get('shared_curve_bend_weight',.1)*bending)

    def terms(self,state,widths):
        if self.straight:
            # References for serialization, not additional registered parameters.
            object.__setattr__(self,'_snapshot_state',state)
            object.__setattr__(self,'_snapshot_widths',widths)
            self.last=dict(shared_curve_length_relative_mean=0.,shared_curve_length_relative_max=0.,
                shared_curve_reversals=0,shared_curve_transverse_slope_p95=0.)
            # These are identities, not small weighted penalties.
            zero=(self.u.sum()+self.v.sum()+widths.sum())*0
            return dict(shared_curve_length=zero,shared_curve_alignment=zero,
                        shared_curve_bending=zero,shared_curve_direction=zero)
        if self.integer_geometry:return self.integer_terms(state,widths)
        xy=self.coordinates();q=state[self.owner,0]+xy[:,0]
        ww=widths.double();bounds=width_boundaries(ww)
        pos=torch.stack((soft.uv.q_to_u(q,ww,bounds),xy[:,1]),-1)
        r=torch.where(~self.first)[0];l=r-1
        zero=pos.sum()*0
        if not len(r):return {'shared_curve_length':zero,'shared_curve_alignment':zero,'shared_curve_bending':zero}
        gap=(self.arc[r]-self.arc[l]).to(pos);delta=pos[r]-pos[l]
        slope=delta/gap[:,None]
        transverse=slope.gather(1,(1-self.family[self.owner[r]])[:,None].long()).squeeze(1)
        axis=transverse.square()
        # Curvature of the shared physical curve (not offsets from a straight line).
        a,b,c=self.triples.unbind(1)
        if len(c):
            dl=(self.arc[b]-self.arc[a]).to(pos);dr=(self.arc[c]-self.arc[b]).to(pos)
            curve=((pos[c]-pos[b])/dr[:,None]-(pos[b]-pos[a])/dl[:,None])/(.5*(dl+dr))[:,None]
            bending=((1000*curve).square().sum(1)*(.5*(dl+dr))).sum()/(.5*(dl+dr)).sum()
        else:bending=zero
        if not torch.is_grad_enabled():
            err=(torch.linalg.vector_norm(slope,dim=1)-1).abs()
            forward=slope.gather(1,self.family[self.owner[r],None].long()).squeeze(1)*self.direction[self.owner[r]]
            self.last=dict(shared_curve_length_relative_mean=float(err.mean()),
                shared_curve_length_relative_max=float(err.max()),shared_curve_reversals=int((forward<0).sum()),
                shared_curve_transverse_slope_p95=float(transverse.abs().quantile(.95)))
            worst=err.argmax()
            self.last.update(shared_curve_length_worst_arc_gap=float(gap[worst]),
                shared_curve_length_worst_uv_distance=float(torch.linalg.vector_norm(delta[worst])),
                shared_curve_length_worst_node=int(r[worst]))
        extra={}
        weight=float(self.config.get('shared_direction_weight',0.))
        if weight:
            cone,stats=direction_cone(pos,self.owner,self.arc,self.family,self.direction,self.config)
            extra['shared_curve_direction']=weight*cone
            if not torch.is_grad_enabled():self.last.update({'direction_shared_'+k:v for k,v in stats.items()})
        return dict(**extra,shared_curve_length=self.config.get('shared_curve_length_weight',10.)*arc_length_energy(delta[:,None,:],gap,weighting=self.config.get('length_segment_weighting','arc_length')),
                    shared_curve_alignment=self.config.get('shared_curve_alignment_weight',10.)*(axis*gap).sum()/gap.sum(),
                    shared_curve_bending=self.config.get('shared_curve_bend_weight',.1)*bending)

    def arrays(self):
        if self.straight:
            from .fiber_implicit_winding import map_state
            with torch.no_grad():
                state=map_state(self._snapshot_state)
                q,v=self.node_hypotheses(state,self._snapshot_widths,state[self.owner,0,None])
                xy=torch.stack((q[:,0]-state[self.owner,0],v[:,0]),-1)
            return dict(straight_fixed_length_geometry=np.array(True),
                straight_anchor_u=self.u.detach().cpu().numpy(),straight_anchor_v=self.v.detach().cpu().numpy(),
                shared_curve_phase_v=xy.cpu().numpy(),shared_curve_integer_geometry=np.array(True),
                shared_curve_hypothesis_mapping=np.array(True),
                shared_curve_owner=self.owner.cpu().numpy(),shared_curve_arc=self.arc.cpu().numpy())
        extra=(self.affine.arrays('shared_affine_',torch.stack((self.u,self.v),-1)[:,None,:])
               if self.affine is not None else dict(shared_curve_raw_u=self.u.detach().cpu().numpy(),
                                                  shared_curve_raw_v=self.v.detach().cpu().numpy()))
        return dict(**extra,shared_curve_integer_geometry=np.array(self.integer_geometry),
                    shared_curve_phase_v=self.coordinates().detach().cpu().numpy(),
                    shared_curve_hypothesis_mapping=np.array(self.config.get('shared_curve_hypothesis_mapping',False)),
                    shared_curve_owner=self.owner.cpu().numpy(),shared_curve_arc=self.arc.cpu().numpy())

    def restore(self,saved):
        if self.straight != bool(saved.get('straight_fixed_length_geometry',False)):
            raise ValueError('Cannot restore across straight/free geometry models; initialize explicitly')
        if self.straight:
            for key,value in [('shared_curve_owner',self.owner),('shared_curve_arc',self.arc)]:
                if not np.array_equal(saved[key],value.cpu().numpy()):
                    raise ValueError('Shared curve topology changed')
            with torch.no_grad():
                self.u.copy_(torch.as_tensor(saved['straight_anchor_u'],device=self.u.device))
                self.v.copy_(torch.as_tensor(saved['straight_anchor_v'],device=self.v.device))
            self.ready=True
            if 'state' in saved and 'widths' in saved:
                object.__setattr__(self,'_snapshot_state',torch.as_tensor(saved['state'],device=self.u.device))
                object.__setattr__(self,'_snapshot_widths',torch.as_tensor(saved['widths'],device=self.u.device))
            return
        for key,value in [('shared_curve_owner',self.owner),('shared_curve_arc',self.arc)]:
            if not np.array_equal(saved[key],value.cpu().numpy()):raise ValueError('Shared curve topology changed')
        if 'shared_affine_base' in saved:
            if self.affine is None:raise ValueError('Cannot drop affine shared parameterization')
            local=self.affine.restore(saved,'shared_affine_',self.u.device)
            self.u=torch.nn.Parameter(local[:,0,0].detach().clone())
            self.v=torch.nn.Parameter(local[:,0,1].detach().clone())
            self.ready=True
        elif self.affine is not None:
            self.set_coordinates(torch.as_tensor(saved['shared_curve_phase_v']))
        elif 'shared_curve_raw_u' in saved:
            with torch.no_grad():
                self.u.copy_(torch.as_tensor(saved['shared_curve_raw_u'],device=self.u.device))
                self.v.copy_(torch.as_tensor(saved['shared_curve_raw_v'],device=self.v.device))
            self.ready=True
        else:
            # Earlier pilot snapshots stored only evaluated coordinates. Very
            # small positive increments can round to zero; that is not reversal.
            self.set_coordinates(torch.as_tensor(saved['shared_curve_phase_v']))


def saved_points(d,saved,state,widths,ids,ds,as_mapping=False,shared_only=False):
    """Read-only checkpoint evaluation; same interpolation as the trained curve."""
    if 'intermediate_uv_scaled_qv' in saved and not as_mapping and not shared_only:
        from .fiber_intermediate_uv import saved_points as intermediate_points
        return intermediate_points(d,saved,state,widths,ids,ds)
    if 'shared_curve_phase_v' not in saved:raise ValueError('Missing saved shared curve')
    if bool(saved.get('shared_curve_integer_geometry',False)):
        if as_mapping:raise ValueError('Integer shared geometry needs per-hypothesis queries')
        curve=restore_query_curve(d,saved)
        curve.add_plan('saved_query',np.asarray(ids),np.asarray(ds))
        requested=torch.as_tensor(state,dtype=torch.float64)
        from .fiber_implicit_winding import map_state
        requested=map_state(requested)
        with torch.no_grad():
            return tuple(x[:,0].numpy() for x in curve.hypothesis_queries(requested,torch.as_tensor(widths),
                curve.plans['saved_query'],requested[np.asarray(ids),0,None]))
    ids=np.asarray(ids);ds=np.asarray(ds)
    physical=bool(saved.get('shared_curve_hypothesis_mapping',False))
    if as_mapping and not physical:raise ValueError('Mapping requires corrected shared-curve snapshot')
    if physical:
        # Evaluate query and source anchor in the SAVED continuous placement,
        # then transplant that physical displacement to the requested labels.
        count=len(ids);ids=np.r_[ids,ids];ds=np.r_[ds,np.zeros_like(ds)]
    owner=saved['shared_curve_owner'];arc=saved['shared_curve_arc']
    query=d['start'][ids]+ds*np.sign(d['node_ds'][ids,-1])
    l,r,t=interpolation_plan(owner,arc,ids,query)
    requested=torch.as_tensor(state,dtype=torch.float64)
    st=torch.as_tensor(saved['state'] if physical else state,dtype=torch.float64)
    w=torch.as_tensor(widths,dtype=torch.float64)
    fam=torch.as_tensor(d['family'][ids]);dd=torch.as_tensor(ds,dtype=torch.float64)
    with torch.no_grad():
        if len(owner):
            xy=saved['shared_curve_phase_v'];val=xy[l.clip(0)]+(xy[r.clip(0)]-xy[l.clip(0)])*t[:,None]
            if physical:
                bounds=torch.cat((w.new_zeros(1),w.cumsum(0)));n=st[ids,0]
                a=torch.as_tensor(xy[l.clip(0),0]);b=torch.as_tensor(xy[r.clip(0),0])
                xa=soft.uv.q_to_u(n+a,w,bounds);xb=soft.uv.q_to_u(n+b,w,bounds)
                val[:,0]=(soft.uv.u_to_q(xa+(xb-xa)*torch.as_tensor(t),w,bounds)-n).numpy()
            p=torch.as_tensor(val[:,0]);v=torch.as_tensor(val[:,1])
        else:
            raise ValueError('Missing perpendicular curve nodes')
        if physical:
            bounds=torch.cat((w.new_zeros(1),w.cumsum(0)))
            ids=ids[:count];anchor=p[count:];n=st[ids,0]
            dx=soft.uv.q_to_u(n+p[:count],w,bounds)-soft.uv.q_to_u(n+anchor,w,bounds)
            base=torch.stack((requested[ids,0],anchor,v[:count],requested[ids,3]),-1)
            if as_mapping:return base.numpy(),dx.numpy()
            return tuple(x.numpy() for x in soft.points(base,fam[:count],dx,w,physical_u=True))
        return (st[ids,0]+p).numpy(),p.numpy(),v.numpy()


def restore_query_curve(d,saved):
    """Small read-only shared curve using the same query code as training."""
    curve=SharedCurve(d,saved['shared_curve_owner'],saved['shared_curve_arc'],
        dict(fiber_affine_geometry=True,shared_curve_integer_geometry=True,
             straight_fixed_length_geometry=bool(saved.get('straight_fixed_length_geometry',False))),'cpu')
    if curve.straight:curve.restore(saved)
    else:curve.set_coordinates(torch.as_tensor(saved['shared_curve_phase_v']))
    return curve
def saved_segment_metrics(d,s,excluded_sources=()):
    """MAP segment diagnostics, using the same physical curve as rendering."""
    from . import fiber_implicit_winding as density, fiber_soft_winding as soft
    from .fiber_geometry_cache import width_boundaries
    st=torch.as_tensor(s['state']);w=torch.as_tensor(s['widths']).double()
    mapped=density.map_state(st).numpy()
    owner=s['shared_curve_owner'];arc=s['shared_curve_arc']
    ds=(arc-d['start'][owner])*np.sign(d['node_ds'][owner,-1])
    q,_,v=saved_points(d,s,mapped,w.numpy(),owner,ds,shared_only=True)
    x=soft.uv.q_to_u(torch.tensor(q),w,width_boundaries(w)).numpy()
    pos=np.column_stack((x,v));right=np.flatnonzero(owner[1:]==owner[:-1])+1;left=right-1
    delta=pos[right]-pos[left];gap=arc[right]-arc[left];length=np.linalg.norm(delta,axis=1)
    family=d['family'][owner[right]]
    forward=delta[np.arange(len(delta)),family]*np.sign(d['node_ds'][owner[right],-1])
    angle=np.degrees(np.arccos(np.clip(forward/np.maximum(length,1e-12),-1,1)))
    def stats(a):
        return dict(mean=float(a.mean()),p95=float(np.percentile(a,95)),max=float(a.max()),count=len(a)) if len(a) else dict(mean=None,p95=None,max=None,count=0)
    included=~np.isin(d['source_id'][owner[right]],excluded_sources)
    return {label:dict(length_error_percent=stats(abs(length[mask]/gap[mask]-1)*100),
        angle_degrees=stats(angle[mask]),reversals=int((forward[mask]<0).sum()))
        for label,mask in [('H',(family==0)&included),('V',(family==1)&included)]}
