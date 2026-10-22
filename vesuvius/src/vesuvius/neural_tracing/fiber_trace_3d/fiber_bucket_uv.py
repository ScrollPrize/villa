"""Opt-in voxel UV hypotheses at endpoints, composed with cavity uncertainty."""
import math
import numpy as np
import torch
from .fiber_geometry_cache import width_boundaries
from . import fiber_implicit_winding as density
from . import fiber_soft_winding as soft
from .fiber_cavity_sigma import CavitySigma, FixedSigma
from .fiber_agreement_sigma import AgreementSigma
from .fiber_v_endpoint_offsets import topology
from .fiber_geometry_confidence import confidence_gate, meeting_confidence_gate
from .fiber_affine_geometry import AffineOffsets, direction_cone, direction_plan, arc_length_energy
from .fiber_geometry_cache import cached_geometry
from .fiber_winding_order import order_margin


def dynamic_same_sheet_pairs(qa, qb, sign, family_delta, leeway=.5):
    """Current continuous endpoint hypotheses, not an initialization diagonal.

    Half-open same-sheet interval prevents two adjacent winding branches from
    being selected at the boundary. No Lasagna distance enters this decision.
    """
    delta = qb[:, None, :] - qa[:, :, None]
    order = order_margin(delta, sign[:, None, None], family_delta[:, None, None],
                         torch.zeros_like(delta, dtype=torch.bool), leeway)
    return torch.where((delta >= -.5) & (delta < .5) & (order >= 0))


def fixed_carry(state,widths,d):
    """Initialize endpoint phase alignment ONCE, excluding integer labels."""
    mapped=state.detach().clone();mapped[:,0]=torch.floor(mapped[:,0]+.5)
    i,j=d['i'],d['j']
    qa,_,_=soft.points(mapped[i],d['family'][i],d['ds_a'],widths)
    qb,_,_=soft.points(mapped[j],d['family'][j],d['ds_b'],widths)
    return torch.floor((qb-mapped[j,0])-(qa-mapped[i,0])+.5).long()


def same_sheet_pairs(ka,kb,carry):
    """Fixed diagonal, independent of current U/N or local UV coordinates."""
    target=ka-carry[:,None]
    column=(target-kb[:,:1]).long()
    row,a=torch.where((column>=0)&(column<kb.shape[1]))
    return row,a,column[row,a]


def meeting_geometry_mass(mass,config):
    """Optional geometry-only meeting supervision; values/selection stay live."""
    return mass.detach() if config.get('bucket_uv_meeting_detach_mass',False) else mass


def meeting_cost(delta, huber_delta_vx=None):
    """Componentwise twice-Huber: identical to squared error near zero.

    The transition is in voxels; the historical 1000-vx normalization stays
    unchanged. Probability weighting is applied separately by the caller.
    """
    if huber_delta_vx is None:
        return (delta/1000).square().sum(-1)
    radius=float(huber_delta_vx)
    if not math.isfinite(radius) or radius<=0:
        raise ValueError('bucket_uv_huber_delta_vx must be positive and finite')
    magnitude=delta.abs()/1000
    quadratic=magnitude.clamp_max(radius/1000)
    return (quadratic.square()+2*(radius/1000)*(magnitude-quadratic)).sum(-1)


def bending(offsets, arc, triples):
    """Arc-normalized slope change; affine offsets have exactly zero curvature."""
    a,b,c=triples.unbind(1)
    left=arc[b]-arc[a];right=arc[c]-arc[b]
    slope_l=(offsets[b]-offsets[a])/left[:,None,None]
    slope_r=(offsets[c]-offsets[b])/right[:,None,None]
    return (slope_r-slope_l)/(0.5*(left+right))[:,None,None]


def contiguous_bending(offsets,arc,triples):
    """Same valid triples, but slice neighboring rows before the final selection.

    Boundary/duplicate-arc gaps are replaced only where not used by a valid
    triple. Invalid intermediate rows must never produce inf/NaN in backward.
    """
    if len(arc)<3:return offsets[:0]
    gaps=arc[1:]-arc[:-1]
    # Every within-fiber gap is strictly positive (topology merges equal arcs).
    safe=torch.where(gaps>0,gaps,torch.ones_like(gaps))
    slopes=(offsets[1:]-offsets[:-1])/safe[:,None,None]
    curve=(slopes[1:]-slopes[:-1])/(.5*(safe[:-1]+safe[1:]))[:,None,None]
    return curve.index_select(0,triples[:,0])


def bending_energy(curve, family, longitudinal_ratio=1.):
    """Independent label energies; H=0 longitudinal U, V=1 longitudinal V.

    A ratio of one preserves the historical reduction exactly. The outer
    bending weight is the transverse coefficient; the ratio scales only the
    longitudinal component, never probability or another bucket's curvature.
    """
    energy=(1000*curve).square()
    if longitudinal_ratio==1.:
        return energy.sum((1,2))
    longitudinal=torch.nn.functional.one_hot(family.long(),num_classes=2)
    return (energy*(1+(longitudinal_ratio-1)*longitudinal[:,None,:])).sum((1,2))


class BucketUV(torch.nn.Module):
    def __init__(self,d,config,device):
        super().__init__();self.config=config
        self.predicted_only=bool(config.get('predicted_only_geometry',False))
        if config.get('straight_fixed_length_geometry') and not self.predicted_only:
            raise ValueError('Straight fixed-length model requires predicted-only geometry')
        if self.predicted_only and not (config.get('shared_curve_geometry') and config.get('shared_curve_hypothesis_mapping')):
            raise ValueError('Predicted-only requires the shared hypothesis-mapped curve')
        if self.predicted_only and config.get('intermediate_uv_geometry'):
            raise ValueError('Predicted-only cannot include an intermediate layer')
        d,_=soft.constraint_family_filter(d,config.get('filter_constraint_families',False))
        if config.get('learned_winding_sigma'):
            raise ValueError('Bucket UV supports cavity or fixed uncertainty, not learned sigma')
        split=len(np.unique(d['source_id']))!=len(d['source_id'])
        if split and not (self.predicted_only and config.get('straight_fixed_length_geometry')):
            raise ValueError('Bucket UV pilot requires whole unsplit source fibers')
        if config.get('diagnostic_batch_size',len(d['i']))<len(d['i']):
            raise ValueError('Bucket UV pilot requires full-edge diagnostics')
        for key in ('bucket_uv_match_weight','bucket_uv_attach_weight','bucket_uv_bend_weight','bucket_uv_lr'):
            if not math.isfinite(config[key]) or config[key]<0 or (config[key]==0 and key!='bucket_uv_match_weight'):
                raise ValueError(key+' must be positive (meeting weight may be zero)')
        self.longitudinal_ratio=float(config.get('bucket_uv_bend_longitudinal_ratio',1.))
        for key in ('bucket_uv_bend_geometry','intermediate_uv_bend_geometry'):
            if config.get(key,'offset') not in ('offset','absolute'):
                raise ValueError(key+' must be offset or absolute')
        if not math.isfinite(self.longitudinal_ratio) or self.longitudinal_ratio<=0:
            raise ValueError('bucket_uv_bend_longitudinal_ratio must be positive and finite')
        sigma_class=(AgreementSigma if config.get('agreement_winding_sigma') else CavitySigma) if config.get('cavity_winding_sigma') else FixedSigma
        if config.get('agreement_winding_sigma') and not config.get('cavity_winding_sigma'):
            raise ValueError('Agreement uncertainty requires cavity_winding_sigma')
        self.cavity=sigma_class(d,config,device)
        self.perpendicular_nodes_only=bool(config.get('bucket_uv_perpendicular_nodes_only',True))
        event_mask=((~d['parallel'].astype(bool)) & (d['family'][d['i']]!=d['family'][d['j']])
                    if self.perpendicular_nodes_only else None)
        self.saved=topology(d,0.,event_mask=event_mask,piece_local=split)
        for k,v in self.saved.items():self.register_buffer(k,torch.as_tensor(v,device=device))
        padding=int(config.get('bucket_uv_padding',64))
        if padding<1:raise ValueError('Bucket UV padding must be positive')
        self.low=-padding;self.count=int(config['winding_count'])+2*padding
        self.values=(None if self.predicted_only else torch.nn.Parameter(torch.zeros(len(self.v_offset_arc),self.count,2,device=device)))
        source=np.arange(len(d['family'])) if split else d['source_id'];index={int(s):i for i,s in enumerate(source)}
        self.register_buffer('owner',torch.tensor([index[int(s)] for s in self.saved['v_offset_source']],device=device,dtype=torch.long))
        owners=self.owner.cpu().numpy()
        self.register_buffer('node_ds',torch.as_tensor(
            (self.saved['v_offset_arc']-d['start'][owners])*np.sign(d['node_ds'][owners,-1]),
            device=device,dtype=torch.float32))
        s=self.saved['v_offset_source'];triples=np.array([[i-1,i,i+1] for i in range(1,len(s)-1)
                                                       if s[i-1]==s[i]==s[i+1]],dtype=np.int64).reshape(-1,3)
        self.register_buffer('triples',torch.as_tensor(triples,device=device))
        gaps=self.v_offset_arc[1:]-self.v_offset_arc[:-1]
        self.register_buffer('bend_gaps',torch.where(gaps>0,gaps,torch.ones_like(gaps)))
        self.register_buffer('bend_spans',.5*(self.bend_gaps[:-1]+self.bend_gaps[1:]))
        self.register_buffer('bend_support',.5*(self.v_offset_arc[self.triples[:,2]]-self.v_offset_arc[self.triples[:,0]]))
        self.last={}
        self.carry=None
        self.positions_initialized=False
        self.windowed=bool(config.get('bucket_uv_active_window',False)) and not self.predicted_only
        self.full_count=self.count
        self.optimizer=None
        self.labels=None
        self.active=None
        if self.windowed:
            # Allocate actual compact storage at the first update, not a full ladder.
            self.values=torch.nn.Parameter(torch.empty(len(self.owner),0,2,device=device))
        self.curve=None
        self.affine=None
        if config.get('fiber_affine_geometry',False) and not self.predicted_only:
            if not self.windowed:raise ValueError('Affine pilot requires active bucket windows')
            self.affine=AffineOffsets(self.owner,self.v_offset_arc,d['start'],d['end'],device,self.values.dtype)
        for layer in ('shared','intermediate','bucket'):
            weight=float(config.get(layer+'_direction_weight',0.))
            if not math.isfinite(weight) or weight<0:raise ValueError('Direction weights must be finite and nonnegative')
        for layer in ('intermediate','bucket'):
            weight=float(config.get(layer+'_length_weight',0.))
            if not math.isfinite(weight) or weight<0:raise ValueError('Length weights must be finite and nonnegative')
        if config.get('shared_curve_geometry'):
            from .fiber_shared_curve import SharedCurve
            if not self.perpendicular_nodes_only:raise ValueError('Shared curve requires perpendicular-only nodes')
            self.curve=SharedCurve(d,owners,self.saved['v_offset_arc'],config,device)
        self.intermediate=None
        if config.get('intermediate_uv_geometry'):
            if self.curve is None or not self.windowed or not config.get('shared_curve_hypothesis_mapping'):
                raise ValueError('Intermediate UV requires corrected shared curve and active windows')
            from .fiber_intermediate_uv import IntermediateUV
            self.intermediate=IntermediateUV(config,device)
            self.intermediate.prepare_affine(self,d)
        if self.curve is not None:
            from .fiber_affine_geometry import direction_plan
            for k,value in enumerate(direction_plan(self.owner,self.v_offset_arc,self.curve.family,self.curve.direction)):
                self.register_buffer('direction_plan_'+str(k),value)

    @property
    def direction_plan(self):
        return tuple(getattr(self,'direction_plan_'+str(k)) for k in range(6))

    def fixed_bending(self,positions):
        if len(positions)<3:return positions[:0]
        slopes=(positions[1:]-positions[:-1])/self.bend_gaps[:,None,None]
        return ((slopes[1:]-slopes[:-1])/self.bend_spans[:,None,None]).index_select(0,self.triples[:,0])

    def intermediate_prior(self,state,widths,shared):
        if self.config.get('reuse_prior_conversion',False):
            prior=self.intermediate.shared_prior(self,state,widths)
            return torch.stack((prior[...,0]*self.intermediate.scale,prior[...,1]),-1)
        bounds=width_boundaries(widths)
        return torch.stack((soft.uv.u_to_q(shared[...,0],widths,bounds)*self.intermediate.scale,shared[...,1]),-1)

    def prepare_fields(self,d,samples):
        if self.curve is not None:
            filtered,_=soft.constraint_family_filter(d,self.config.get('filter_constraint_families',False))
            self.curve.prepare_fields(filtered,samples)

    def pair_states(self,state,widths,d,edges):
        if self.curve is None:
            return state[d['i'][edges]],state[d['j'][edges]],d['ds_a'][edges],d['ds_b'][edges]
        if self.config.get('shared_curve_hypothesis_mapping') and not self.curve.integer_geometry:
            sa,da=self.curve.mapping_state(state,widths,'a',edges)
            sb,db=self.curve.mapping_state(state,widths,'b',edges)
            return sa,sb,da,db
        sa=self.curve.endpoint_state(state,widths,'a',edges)
        sb=self.curve.endpoint_state(state,widths,'b',edges)
        z=state.new_zeros(len(edges))
        return sa,sb,z,z

    def field_points(self,state,widths,key):
        self.curve.sigma_values=self.cavity.values
        return self.curve.evaluate(state,widths,key)

    def phase_prior_error(self,state,widths,target):
        if not self.curve.integer_geometry:
            return soft.circular(self.field_points(state,widths,'unary')[1]-target).square()
        h,lp=density.hypotheses(state,self.cavity.values,self.config.get('sigma_max',1.))
        plan=self.curve.plans['unary'];ids=plan[0]
        phase=self.curve.hypothesis_queries(state,widths,plan,h[ids,:,0])[1]
        return (lp[ids].exp()*soft.circular(phase-target[:,None]).square()).sum(1)

    def pair_geometry(self,state,widths,edges):
        if self.curve is not None:self.curve.sigma_values=self.cavity.values
        if self.intermediate is None:
            if self.curve is not None and self.curve.integer_geometry:
                from .fiber_intermediate_uv import SharedPairGeometry
                return SharedPairGeometry(self,state,widths,edges)
            return None
        from .fiber_intermediate_uv import PairGeometry
        return PairGeometry(self,state,widths,edges)

    def optimizer_groups(self):
        if self.curve is None:return None
        groups=[dict(params=[self.values],lr=self.config['bucket_uv_lr']),
                dict(params=[self.curve.u],lr=self.config.get('shared_curve_u_lr',self.config['lr'])),
                dict(params=[self.curve.v],lr=self.config.get('shared_curve_v_lr',self.config['bucket_uv_lr']))]
        if self.intermediate is not None:
            groups.append(dict(params=[self.intermediate.values],lr=self.config.get('intermediate_uv_lr',2.)))
        if self.affine is not None or self.predicted_only and self.curve.affine is not None:
            local_factor=float(self.config.get('affine_local_lr_factor',1.))
            if not math.isfinite(local_factor) or local_factor<=0:raise ValueError('affine_local_lr_factor must be positive')
            for group in groups:group['lr']*=local_factor
        bucket_local_factor=float(self.config.get('bucket_local_lr_multiplier',1.))
        if not math.isfinite(bucket_local_factor) or bucket_local_factor<=0:
            raise ValueError('bucket_local_lr_multiplier must be positive')
        groups[0]['lr']*=bucket_local_factor
        if self.predicted_only:groups=groups[1:]
        for layer,affine,ulr,vlr in [
                ('shared',self.curve.affine,self.config.get('shared_curve_u_lr',self.config['lr']),self.config.get('shared_curve_v_lr',self.config['bucket_uv_lr'])),
                ('bucket',self.affine,self.config['bucket_uv_lr'],self.config['bucket_uv_lr']),
                ('intermediate',self.intermediate.affine if self.intermediate is not None else None,self.config.get('intermediate_uv_lr',2.),self.config.get('intermediate_uv_lr',2.))]:
            if affine is not None:
                factor=float(self.config.get('affine_coarse_lr_factor',1.))
                if not math.isfinite(factor) or factor<=0:raise ValueError('affine_coarse_lr_factor must be positive')
                groups.extend([dict(params=[affine.u],lr=ulr*factor),dict(params=[affine.v],lr=vlr*factor)])
        return groups

    def bind_optimizer(self,optimizer):
        self.optimizer=optimizer
        for affine in (self.affine,getattr(self.curve,'affine',None),getattr(self.intermediate,'affine',None)):
            if affine is not None:
                affine.fused=self.config.get('fused_affine_geometry',False)
                affine.batched_backward=self.config.get('batched_affine_backward',False)

    def refresh_window(self,state,widths):
        """Reindex absolute coordinates and Adam moments by persistent integer label."""
        with torch.no_grad():
            sigma=self.cavity.values.reshape(-1)
            multiple=float(self.config.get('bucket_uv_window_sigmas',4.))
            if not math.isfinite(multiple) or multiple<=0:
                raise ValueError('bucket_uv_window_sigmas must be positive')
            guard=int(self.config.get('bucket_uv_window_guard',0))
            if guard<0:raise ValueError('bucket_uv_window_guard must be nonnegative')
            radius=torch.ceil(multiple*sigma+.5).long().clamp_min(2)+guard
            maximum=int(radius.max())
            relative=torch.arange(-maximum,maximum+1,device=state.device)
            labels=torch.floor(state[:,0]+.5).long()[:,None]+relative
            active=relative.abs()[None,:]<=radius[:,None]
            if self.labels is not None and torch.equal(labels,self.labels) and torch.equal(active,self.active):
                return
            old=self.values;old_labels=self.labels;old_active=self.active
            self.labels=labels;self.active=active;self.count=labels.shape[1]
            if self.affine is not None:
                initial=self.shared_positions(state,widths)
                self.values=self.affine.remap(old,initial,old_labels,old_active,labels,active,self.optimizer,
                                             reuse=self.config.get('affine_reuse_remap',False),
                                             batched=self.config.get('affine_batched_remap',True))
                self.positions_initialized=True
                return
            if self.config.get('bucket_uv_reuse_storage') and old_labels is not None and old_labels.shape==labels.shape:
                changed=(labels!=old_labels).any(1)|(active!=old_active).any(1)
                nodes=torch.where(changed[self.owner])[0]
                owners=self.owner[nodes]
                column=labels[owners]-old_labels[owners,:1]
                valid=(column>=0)&(column<self.count)&active[owners]
                column=column.clamp(0,self.count-1)
                valid &= old_active[owners].gather(1,column)
                index=column[:,:,None].expand(-1,-1,2)
                old_rows=old.index_select(0,nodes)
                initial=self.shared_positions(state,widths,nodes)
                old.index_copy_(0,nodes,torch.where(valid[:,:,None],old_rows.gather(1,index),initial))
                if self.optimizer is not None:
                    for value in self.optimizer.state.get(old,{}).values():
                        if torch.is_tensor(value) and value.shape==old.shape:
                            rows=value.index_select(0,nodes).gather(1,index)
                            value.index_copy_(0,nodes,torch.where(valid[:,:,None],rows,0))
                return
            initial=self.shared_positions(state,widths)
            from .fiber_intermediate_uv import remap_parameter
            self.values=remap_parameter(old,initial,self.owner,old_labels,old_active,labels,active,self.optimizer)
            self.positions_initialized=True

    @cached_geometry
    def local_hypotheses(self,state):
        h,lp=density.hypotheses(state,self.cavity.values,self.config['sigma_max'])
        if not self.windowed:return h,lp
        index=self.labels-h[:,:,0].long()[:,:1]
        valid=(index>=0)&(index<h.shape[1])&self.active
        index=index.clamp(0,h.shape[1]-1)
        local=state[:,None,:].expand(-1,self.count,-1).clone()
        local[:,:,0]=self.labels
        return local,lp.gather(1,index).masked_fill(~valid,-torch.inf)

    @cached_geometry
    def shared_positions(self,state,widths,nodes=None):
        """Shared fiber prediction for every fixed label, used only as a prior."""
        owner=self.owner if nodes is None else self.owner[nodes]
        family=self.v_offset_family if nodes is None else self.v_offset_family[nodes]
        ds=self.node_ds if nodes is None else self.node_ds[nodes]
        if self.predicted_only:
            h=self.local_hypotheses(state)[0][owner]
        else:
            h=state[owner,None,:].expand(-1,self.count,-1).clone()
            h[:,:,0]=(self.labels[owner] if self.windowed else
                       torch.arange(self.low,self.low+self.count,device=state.device,dtype=state.dtype))
        if self.curve is None:
            q,_,v=density.hypothesis_points(h,family,ds,widths)
        else:
            if self.config.get('shared_curve_hypothesis_mapping'):
                q,v=self.curve.node_hypotheses(state,widths,h[:,:,0],nodes)
            else:
                xy=(self.curve.coordinates() if nodes is None else self.curve.coordinates()[nodes]).to(state)
                q=h[:,:,0]+xy[:,0,None];v=xy[:,1,None].expand_as(q)
        bounds=width_boundaries(widths)
        return torch.stack((soft.uv.q_to_u(q,widths,bounds),v),-1)

    def update(self,state,widths,d,iteration):
        if self.curve is not None:self.curve.initialize(state,widths)
        if isinstance(self.cavity,AgreementSigma):
            self.cavity.update(state,widths,d,iteration,extension=self)
        elif isinstance(self.cavity,CavitySigma):
            self.cavity.update(state,widths,d,iteration,pair_states=self.pair_states)
        else:
            self.cavity.update(state,widths,d,iteration)
        if self.carry is None and not self.config.get('bucket_uv_dynamic_same_sheet',False):
            with torch.no_grad():self.carry=fixed_carry(state,widths,d)
        if self.predicted_only:return
        if self.windowed:
            interval=int(self.config.get('bucket_uv_refresh_interval',1))
            if interval<1:raise ValueError('bucket_uv_refresh_interval must be positive')
            if self.labels is None or iteration%interval==0:
                from .fiber_geometry_cache import geometry_evaluation
                # Shared geometry is immutable here. refresh_window sets the new
                # labels BEFORE its first shared_positions query; only local
                # layers are remapped, and their coordinates are not cached here.
                # Discard this cache before the objective/any parameter update.
                with torch.no_grad(),geometry_evaluation(self.config.get('reuse_window_geometry',False),
                                                        self.config.get('profile_solver',False)):
                    self.refresh_window(state,widths)
                    if self.intermediate is not None:self.intermediate.initialize(self,state,widths)
            return
        if not self.positions_initialized:
            with torch.no_grad():self.values.copy_(self.shared_positions(state,widths))
            self.positions_initialized=True
    def pair_kwargs(self,*args):return self.cavity.pair_kwargs(*args)

    def lookup(self,nodes,k):
        index=k.long()-(self.labels[self.owner[nodes],:1] if self.windowed else self.low)
        if not self.config.get('bucket_uv_fast_gathers') and bool(((index<0)|(index>=self.count)).any()):
            raise ValueError('Bucket UV storage exhausted; enlarge padding, never clamp hypotheses')
        if self.config.get('bucket_uv_fast_gathers'):
            # Internal callers supply the same labels generated for these owners.
            return self.positions().flatten(0,1).index_select(0,(nodes[:,None]*self.count+index).flatten()).reshape(*index.shape,2)
        return self.positions()[nodes[:,None],index]

    @cached_geometry
    def positions(self):
        return self.affine.compose(self.values) if self.affine is not None else self.values

    def terms(self,state,widths,d,edges):
        if self.curve is not None:self.curve.sigma_values=self.cavity.values
        if self.predicted_only:return self.curve.terms(state,widths)
        h,lp=self.local_hypotheses(state)
        shared=self.shared_positions(state,widths)
        target=shared if self.intermediate is None else self.intermediate.physical(widths)
        positions=self.positions()
        offsets=positions-target
        if self.windowed:offsets=offsets*self.active[self.owner,:,None]
        weighted=self.config.get('bucket_uv_attachment_probability_weighted',True)
        if weighted or not self.config.get('bucket_uv_fast_gathers'):
            indices=(torch.arange(self.count,device=state.device)[None,:].expand(len(self.owner),-1)
                     if self.windowed else h[self.owner,:,0].long()-self.low)
            if not self.config.get('bucket_uv_fast_gathers'):
                self.lookup(torch.arange(len(self.owner),device=state.device),h[self.owner,:,0])
            node_offsets=offsets[torch.arange(len(self.owner),device=state.device)[:,None],indices]
            node_mass=lp[self.owner].exp()*state[self.owner,3:4].sigmoid()
        if weighted:
            attach=(node_mass*(node_offsets/1000).square().sum(-1)).sum()/max(1,len(self.owner))
        else:
            # Same per-bucket coefficient, including low-probability and currently
            # out-of-support labels. Do not divide by ladder size or gate by enabled.
            attach=(offsets/1000).square().sum()/max(1,len(self.owner))
        # Stable k labels are shared along each source; regularize all labels so
        # unoccupied hypotheses remain smooth if they become occupied later.
        bend_positions=positions if self.config.get('bucket_uv_bend_geometry','offset')=='absolute' else offsets
        if self.windowed:bend_positions=bend_positions*self.active[self.owner,:,None]
        bulk=self.config.get('batch_layer_regularizers',False)
        mid_offsets=None
        if bulk and self.intermediate is not None:
            prior=self.intermediate_prior(state,widths,shared)
            mid_offsets=(self.intermediate.scaled_coordinates()-prior)*self.active[self.owner,:,None]
            mid_positions=(target*self.active[self.owner,:,None]
                if self.config.get('intermediate_uv_bend_geometry','offset')=='absolute' else mid_offsets)
            if mid_positions.dtype==bend_positions.dtype:
                both=self.fixed_bending(torch.cat((bend_positions,mid_positions),1))
                curve,mid_curve=both.split(bend_positions.shape[1],dim=1)
            else:curve=self.fixed_bending(bend_positions);mid_curve=self.fixed_bending(mid_positions)
        else:
            curve=(self.fixed_bending(bend_positions) if bulk else
                (contiguous_bending if self.config.get('bucket_uv_contiguous') else bending)(
                    bend_positions,self.v_offset_arc,self.triples))
        support=self.bend_support if bulk else (self.v_offset_arc[self.triples[:,2]]-self.v_offset_arc[self.triples[:,0]])*.5
        bend=(bending_energy(curve,self.v_offset_family[self.triples[:,1]],
                             self.longitudinal_ratio)/(2*self.full_count)*support).sum()/support.sum() if curve.numel() else self.values.sum()*0
        intermediate_terms={}
        if self.intermediate is not None:
            if mid_offsets is None:
                prior=self.intermediate_prior(state,widths,shared)
                mid_offsets=(self.intermediate.scaled_coordinates()-prior)*self.active[self.owner,:,None]
            mid_attach=(mid_offsets/1000).square().sum()/max(1,len(self.owner))
            mid_positions=(target*self.active[self.owner,:,None]
                if self.config.get('intermediate_uv_bend_geometry','offset')=='absolute' else mid_offsets)
            if not bulk:
                mid_curve=(contiguous_bending if self.config.get('bucket_uv_contiguous') else bending)(
                    mid_positions,self.v_offset_arc,self.triples)
            mid_bend=(bending_energy(mid_curve,self.v_offset_family[self.triples[:,1]],
                self.config.get('intermediate_uv_bend_longitudinal_ratio',self.longitudinal_ratio))/(2*self.full_count)*support).sum()/support.sum() if mid_curve.numel() else mid_offsets.sum()*0
            intermediate_terms=dict(
                intermediate_uv_attachment=self.config.get('intermediate_uv_attach_weight',.001)*mid_attach,
                intermediate_uv_bending=self.config.get('intermediate_uv_bend_weight',.0001)*mid_bend)
        keep=(~d['parallel'][edges].bool())&(d['family'][d['i'][edges]]!=d['family'][d['j'][edges]])
        chosen=edges[keep];i=d['i'][chosen];j=d['j'][chosen]
        diagnostic_geometry=not torch.is_grad_enabled() or not self.config.get('reuse_constraint_work')
        if diagnostic_geometry:
            if self.curve is None:
                qa,_,va=density.hypothesis_points(h[i],d['family'][i],d['ds_a'][chosen],widths)
                qb,_,vb=density.hypothesis_points(h[j],d['family'][j],d['ds_b'][chosen],widths)
            else:
                if self.config.get('shared_curve_hypothesis_mapping'):
                    qa,va=self.curve.node_hypotheses(state,widths,h[i,:,0],self.v_offset_a[chosen])
                    qb,vb=self.curve.node_hypotheses(state,widths,h[j,:,0],self.v_offset_b[chosen])
                else:
                    xy=self.curve.coordinates().to(state)
                    qa=h[i,:,0]+xy[self.v_offset_a[chosen],0,None]
                    qb=h[j,:,0]+xy[self.v_offset_b[chosen],0,None]
                    va=xy[self.v_offset_a[chosen],1,None].expand_as(qa)
                    vb=xy[self.v_offset_b[chosen],1,None].expand_as(qb)
        direct=self.windowed and self.config.get('bucket_uv_direct_meetings',False)
        if not direct:
            a=self.lookup(self.v_offset_a[chosen],h[i,:,0]);b=self.lookup(self.v_offset_b[chosen],h[j,:,0])
        if diagnostic_geometry:
            bounds=width_boundaries(widths)
            xa=soft.uv.q_to_u(qa,widths,bounds);xb=soft.uv.q_to_u(qb,widths,bounds)
        if self.config.get('bucket_uv_dynamic_same_sheet',False):
            # Intermediate winding coordinates are the coordinates used by the
            # density/sign objective. Do not use final voxel positions to decide
            # their own attachment membership.
            if self.intermediate is not None and not self.config.get('pair_predicted_winding',False):
                q=self.intermediate.coordinates()[...,0]
                match_qa=q[self.v_offset_a[chosen]]
                match_qb=q[self.v_offset_b[chosen]]
            else:
                q=soft.uv.u_to_q(shared[...,0],widths,width_boundaries(widths))
                match_qa=q[self.v_offset_a[chosen]]
                match_qb=q[self.v_offset_b[chosen]]
            if not self.windowed:
                match_qa=match_qa.gather(1,h[i,:,0].long()-self.low)
                match_qb=match_qb.gather(1,h[j,:,0].long()-self.low)
            row,ka,kb=dynamic_same_sheet_pairs(match_qa,match_qb,d['sign'][chosen],
                d['family'][j]-d['family'][i],self.config.get('sign_order_leeway',.5))
        else:
            row,ka,kb=same_sheet_pairs(h[i,:,0],h[j,:,0],self.carry[chosen])
        if self.windowed:
            valid=self.active[i[row],ka]&self.active[j[row],kb]
            row,ka,kb=row[valid],ka[valid],kb[valid]
        if diagnostic_geometry:
            raw=torch.stack((xb[row,kb]-xa[row,ka],vb[row,kb]-va[row,ka]),-1)
        # Independent absolute coordinates: no shared-prediction addition here.
        if direct:
            # Select each endpoint/label once rather than gather full ladders
            # followed by another indexed selection. Repeated samples stay repeated.
            flat=positions.flatten(0,1)
            ai=self.v_offset_a[chosen[row]]*self.count+ka
            bi=self.v_offset_b[chosen[row]]*self.count+kb
            delta=flat.index_select(0,bi)-flat.index_select(0,ai)
            mass_by_fiber=lp.exp()*state[:,3:4].sigmoid()
            pa=mass_by_fiber.flatten().index_select(0,i[row]*self.count+ka)
            pb=mass_by_fiber.flatten().index_select(0,j[row]*self.count+kb)
        else:
            delta=b[row,kb]-a[row,ka]
            pa=lp[i[row],ka].exp()*state[i[row],3].sigmoid()
            pb=lp[j[row],kb].exp()*state[j[row],3].sigmoid()
        mass=pa*pb
        gate=meeting_confidence_gate(pa,pb,self.config)
        match=(d['class_weight'][chosen[row]]*meeting_geometry_mass(mass,self.config)*gate*meeting_cost(
            delta,self.config.get('bucket_uv_huber_delta_vx'))).sum()/len(edges)
        if not torch.is_grad_enabled():
            self.last={'bucket_uv_same_sheet_mass':float(mass.sum()),'bucket_uv_perp_events':len(chosen),
                       'bucket_uv_nodes':len(self.owner),'bucket_uv_smoothing_triples':len(self.triples)}
            self.last.update(bucket_uv_force_pairs=int((gate>0).sum()),
                bucket_uv_full_force_pairs=int((gate>=1).sum()),
                bucket_uv_gated_mass=float((mass*gate).sum()))
            if self.windowed:
                self.last.update(bucket_uv_window_slots=self.count,
                    bucket_uv_active_mean=float(self.active.sum(1).float().mean()),
                    bucket_uv_omitted_mass_mean=float((1-lp.exp().sum(1)).clamp_min(0).mean()),
                    bucket_uv_omitted_mass_max=float((1-lp.exp().sum(1)).clamp_min(0).max()))
            phase_error=(qb[row,kb]-qa[row,ka]).abs()
            # Endpoint displacement is distinct from the shared-path pair error.
            oa=offsets[self.v_offset_a[chosen[row]],ka]
            ob=offsets[self.v_offset_b[chosen[row]],kb]
            attachment_sum=oa.abs()+ob.abs()
            self.last['bucket_uv_phase_weighted_mean']=float((mass*phase_error).sum()/mass.sum().clamp_min(1e-30))
            self.last['bucket_uv_phase_incompatible_mass_fraction']=float((mass*(phase_error>.5)).sum()/mass.sum().clamp_min(1e-30))
            for cutoff in (.1,.5,.7):
                mask=(mass>cutoff) if self.config.get('bucket_uv_dynamic_same_sheet',False) else (pa>=cutoff)&(pb>=cutoff)
                self.last[f'bucket_uv_meetings_p{cutoff}']=int(mask.sum())
                if cutoff == .5 and self.config.get('convergence_monitor'):
                    self.last['bucket_uv_meeting_events']=sorted(chosen[row[mask]].cpu().tolist())
                phase=(qb[row,kb]-qa[row,ka])[mask].abs().float()
                self.last[f'bucket_uv_phase_p95_p{cutoff}']=float(phase.quantile(.95)) if len(phase) else None
                self.last[f'bucket_uv_phase_above_half_p{cutoff}']=int((phase>.5).sum())
                measurements=[('local',delta),('shared',raw),
                                    ('attachment',attachment_sum*.5),
                                    ('attachment_sum',attachment_sum),
                                    ('combined',delta.abs()+attachment_sum)]
                if self.intermediate is not None:
                    ma=target[self.v_offset_a[chosen[row]],ka];mb=target[self.v_offset_b[chosen[row]],kb]
                    sa=shared[self.v_offset_a[chosen[row]],ka];sb=shared[self.v_offset_b[chosen[row]],kb]
                    measurements += [('intermediate_meeting',mb-ma),
                                     ('shared_to_intermediate',(ma-sa).abs()+(mb-sb).abs()),
                                     ('shared_to_local',(ma+oa-sa).abs()+(mb+ob-sb).abs())]
                for name,values in measurements:
                    for axis in range(2):
                        val=values[mask,axis].abs().float();prefix=f'bucket_uv_{name}_{"uv"[axis]}_p{cutoff}'
                        if name=='attachment':
                            # Pool endpoints: max must not be the max of pair averages.
                            val=torch.cat((oa[mask,axis],ob[mask,axis])).abs().float()
                        if name=='shared_to_intermediate':
                            val=torch.cat(((ma-sa)[mask,axis],(mb-sb)[mask,axis])).abs().float()
                        if name=='shared_to_local':
                            val=torch.cat(((ma+oa-sa)[mask,axis],(mb+ob-sb)[mask,axis])).abs().float()
                        self.last.update({prefix+'_mean':float(val.mean()) if len(val) else None,
                                          prefix+'_p95':float(val.quantile(.95)) if len(val) else None,
                                          prefix+'_max':float(val.max()) if len(val) else None})
                # Convert positions separately through the integral width model;
                # dividing a voxel difference by one average width is not exact.
                fa=positions[self.v_offset_a[chosen[row]],ka]
                fb=positions[self.v_offset_b[chosen[row]],kb]
                def winding(x):
                    return soft.uv.u_to_q(x[...,0],widths,bounds)
                winding_diffs={'local':winding(fb)-winding(fa),
                               'shared':qb[row,kb]-qa[row,ka]}
                if self.intermediate is not None:
                    winding_diffs.update(
                        intermediate_meeting=winding(mb)-winding(ma),
                        shared_to_intermediate=torch.stack((winding(ma)-winding(sa),winding(mb)-winding(sb)),-1),
                        shared_to_local=torch.stack((winding(fa)-winding(sa),winding(fb)-winding(sb)),-1),
                        attachment=torch.stack((winding(fa)-winding(ma),winding(fb)-winding(mb)),-1))
                for name,values in winding_diffs.items():
                    val=values[mask].abs().flatten().float()
                    prefix=f'bucket_uv_{name}_u_windings_p{cutoff}'
                    self.last.update({prefix+'_mean':float(val.mean()) if len(val) else None,
                                      prefix+'_p95':float(val.quantile(.95)) if len(val) else None,
                                      prefix+'_max':float(val.max()) if len(val) else None,
                                      prefix+'_above_half':int((val>.5).sum()),
                                      prefix+'_count':len(val)})
            l=self.v_offset_left;r=self.v_offset_right
            gap=self.v_offset_arc[r]-self.v_offset_arc[l]
            rate=((offsets[r]-offsets[l])/gap[:,None,None]).abs()
            self.last['bucket_uv_max_deformation_rate']=float(rate.max()) if rate.numel() else 0.
            # Active-label geometry diagnostics, independent of confidence.
            # All labels on one fiber have the same active-window mask.
            active=(self.active[self.owner[l]] if self.windowed else
                    torch.ones(rate.shape[:2],dtype=torch.bool,device=rate.device))
            family=self.v_offset_family[l].long()
            for name,axis in [('longitudinal',family),('transverse',1-family)]:
                component=rate.gather(2,axis[:,None,None].expand(-1,rate.shape[1],1)).squeeze(-1)[active].float()
                prefix='bucket_uv_'+name+'_offset_slope'
                self.last[prefix+'_p95']=float(component.quantile(.95)) if component.numel() else 0.
                self.last[prefix+'_max']=float(component.max()) if component.numel() else 0.
            self.last['bucket_uv_max_offset_vx']=float(offsets.abs().max()) if offsets.numel() else 0.
            # Actual physical-curve curvature for the SAME >=.5 meeting-fiber
            # cohort, distinct from upstream-relative offset slope diagnostics.
            eligible=(pa>=.5)&(pb>=.5)
            shown=torch.zeros(len(state),dtype=torch.bool,device=state.device)
            shown[i[row][eligible]]=True;shown[j[row][eligible]]=True
            center=self.triples[:,1];owners=self.owner[center]
            cols=lp.argmax(1)[owners]
            selected=shown[owners]
            for name,curve_positions in [('shared',shared),('final',positions)]+(
                    [('intermediate',target)] if self.intermediate is not None else []):
                actual=bending(curve_positions.double(),self.v_offset_arc.double(),self.triples)
                val=torch.linalg.vector_norm(actual[torch.arange(len(center),device=state.device),cols],dim=-1)[selected]*1000
                prefix='actual_curve_'+name+'_curvature'
                self.last[prefix+'_p95']=float(val.quantile(.95)) if len(val) else None
                self.last[prefix+'_max']=float(val.max()) if len(val) else None
                self.last[prefix+'_rms']=float(val.square().mean().sqrt()) if len(val) else None
        direction_terms={}
        if bulk and torch.is_grad_enabled() and self.intermediate is not None and positions.dtype==target.dtype:
            cones,_=direction_cone(torch.cat((positions,target),1),self.owner,self.v_offset_arc,self.curve.family,
                self.curve.direction,self.config,self.active,plan=self.direction_plan,groups=2)
            direction_terms={name+'_uv_direction':float(self.config.get(name+'_direction_weight',0.))*cones[i]
                             for i,name in enumerate(('bucket','intermediate')) if self.config.get(name+'_direction_weight',0.)}
        for name,xy in [('bucket',positions)]+([('intermediate',target)] if self.intermediate is not None else []):
            if name+'_uv_direction' in direction_terms:continue
            weight=float(self.config.get(name+'_direction_weight',0.))
            if weight:
                cone,stats=direction_cone(xy,self.owner,self.v_offset_arc,self.curve.family,
                                         self.curve.direction,self.config,self.active,
                                         plan=self.direction_plan if bulk else None)
                direction_terms[name+'_uv_direction']=weight*cone
                if not torch.is_grad_enabled():self.last.update({'direction_'+name+'_'+k:v for k,v in stats.items()})
        length_terms={}
        if self.config.get('bucket_length_weight',0.) or self.config.get('intermediate_length_weight',0.):
            r,l,_,_,gap,owners=direction_plan(self.owner,self.v_offset_arc,self.curve.family,self.curve.direction)
            layers=[('bucket',positions)]+([('intermediate',target)] if self.intermediate is not None else [])
            # One batched length evaluation for all local layers, no fiber loops.
            delta=torch.cat([xy[r]-xy[l] for _,xy in layers],1)
            losses=arc_length_energy(delta,gap,self.active[owners],groups=len(layers),weighting=self.config.get('length_segment_weighting','arc_length'))
            if len(layers)==1:losses=losses[None]
            length_terms={name+'_uv_length':float(self.config.get(name+'_length_weight',0.))*losses[i]
                          for i,(name,_) in enumerate(layers)}
        return dict(**length_terms,**direction_terms,**(self.curve.terms(state,widths) if self.curve is not None else {}),**intermediate_terms,
                    bucket_uv_meeting=self.config['bucket_uv_match_weight']*match,
                    bucket_uv_attachment=self.config['bucket_uv_attach_weight']*attach,
                    bucket_uv_bending=self.config['bucket_uv_bend_weight']*bend)

    def diagnostics(self,state=None):return dict(self.cavity.diagnostics(state),**self.last,
        **(getattr(self.curve,'last',{}) if self.curve is not None else {}))

    def arrays(self):
        if self.predicted_only:
            return dict(self.cavity.arrays(),**self.saved,**self.curve.arrays(),
                        predicted_only_geometry=np.array(True))
        return dict(self.cavity.arrays(),**self.saved,bucket_uv_positions=self.positions().detach().cpu().numpy(),
                    **(self.affine.arrays('bucket_affine_',self.values) if self.affine is not None else {}),
                    **(self.curve.arrays() if self.curve is not None else {}),
                    **(self.intermediate.arrays() if self.intermediate is not None else {}),
                    bucket_uv_perpendicular_nodes_only=np.array(self.perpendicular_nodes_only),
                    bucket_uv_first_label=np.array(self.low),
                    **(dict(bucket_uv_carry=self.carry.cpu().numpy()) if self.carry is not None else {}),
                    **(dict(bucket_uv_labels=self.labels.cpu().numpy(),bucket_uv_active=self.active.cpu().numpy())
                       if self.windowed else {}))

    def restore(self,saved):
        if self.predicted_only:
            if 'bucket_uv_positions' in saved or 'intermediate_uv_scaled_qv' in saved:
                raise ValueError('Predicted-only must start fresh, not discard saved local layers')
            self.cavity.restore(saved)
            if 'shared_curve_phase_v' in saved:
                for key,value in self.saved.items():
                    if not np.array_equal(saved[key],value):raise ValueError('Shared endpoint topology changed')
                self.curve.restore(saved)
            return
        if bool(saved.get('predicted_only_geometry',False)):
            raise ValueError('Cannot silently add local layers to a predicted-only checkpoint')
        if 'bucket_uv_positions' in saved and ('intermediate_uv_scaled_qv' in saved)!=(self.intermediate is not None):
            raise ValueError('Cannot add/drop intermediate layer when restoring; start fresh')
        if 'shared_curve_phase_v' in saved:
            if self.curve is None:raise ValueError('Cannot drop saved shared curve')
            self.curve.restore(saved)
        elif self.curve is not None and 'bucket_uv_positions' in saved:
            raise ValueError('Shared curve pilot must start fresh, not from old bucket geometry')
        if ('bucket_uv_positions' in saved and
                bool(saved.get('bucket_uv_perpendicular_nodes_only',False))!=self.perpendicular_nodes_only):
            raise ValueError('Bucket node topology changed; start a fresh solve')
        self.cavity.restore(saved)
        if 'bucket_uv_offsets' in saved:
            raise ValueError('Legacy offset checkpoint is not an absolute-position checkpoint; start fresh')
        if 'bucket_uv_positions' in saved:
            if ('bucket_uv_labels' in saved)!=self.windowed:
                raise ValueError('Cannot mix compact-window and full-ladder UV checkpoints')
            if self.windowed:
                self.labels=torch.as_tensor(saved['bucket_uv_labels'],device=self.values.device)
                self.active=torch.as_tensor(saved['bucket_uv_active'],device=self.values.device)
                self.count=self.labels.shape[1]
                self.values=torch.nn.Parameter(torch.empty(len(self.owner),self.count,2,device=self.values.device))
            if not self.config.get('bucket_uv_dynamic_same_sheet',False):
                self.carry=torch.as_tensor(saved['bucket_uv_carry'],device=self.values.device)
            if int(saved['bucket_uv_first_label'])!=self.low:raise ValueError('Bucket labels changed')
            for k,v in self.saved.items():
                if not np.array_equal(saved[k],v):raise ValueError('Bucket endpoint topology changed')
            if self.affine is not None:
                if 'bucket_affine_base' in saved:
                    self.values=self.affine.restore(saved,'bucket_affine_',self.values.device)
                else:
                    value=torch.as_tensor(saved['bucket_uv_positions'],device=self.values.device)
                    self.affine.initialize(value)
                    self.values=torch.nn.Parameter(torch.zeros_like(value))
            else:
                if 'bucket_affine_base' in saved:raise ValueError('Cannot drop affine bucket parameterization')
                with torch.no_grad():self.values.copy_(torch.as_tensor(saved['bucket_uv_positions'],device=self.values.device))
            self.positions_initialized=True
            if self.intermediate is not None:self.intermediate.restore(saved,self)


def u_validity(ext,state,widths,source_ids,threshold=.2):
    """Diagnostic only: compare both local layers directly with predicted U."""
    bounds=width_boundaries(widths)
    shared=soft.uv.u_to_q(ext.shared_positions(state,widths)[...,0],widths,bounds)
    layers={'final':soft.uv.u_to_q(ext.positions()[...,0],widths,bounds)}
    if ext.intermediate is not None:layers['intermediate']=ext.intermediate.coordinates()[...,0]
    _,lp=ext.local_hypotheses(state)
    col=lp.argmax(1);prob=lp.exp().amax(1)*state[:,3].sigmoid()
    owner=ext.owner;nodes=torch.arange(len(owner),device=owner.device)
    active=ext.active[owner] if ext.windowed else torch.ones_like(shared,dtype=torch.bool)
    out={}
    for name,q in layers.items():
        diff=(q-shared).abs()
        selected=diff[nodes,col[owner]]
        cohorts={}
        for label,values in [('all_active',diff[active]),('confident_map',selected[prob[owner]>.5])]:
            cohorts[label]=dict(nodes=int(values.numel()),above_0p2=int((values>threshold).sum()),
                mean=float(values.mean()) if values.numel() else None,
                max=float(values.max()) if values.numel() else None)
        broken=[]
        for f in torch.unique(owner[(selected>threshold)&(prob[owner]>.5)]).tolist():
            ids=nodes[owner==f];worst=ids[selected[ids].argmax()]
            broken.append(dict(source_id=int(source_ids[f]),piece_index=f,probability=float(prob[f]),
                max_windings=float(selected[worst]),arc=float(ext.v_offset_arc[worst]),
                offending_nodes=int((selected[ids]>threshold).sum())))
        out[name]=dict(cohorts=cohorts,fibers=sorted(broken,key=lambda x:-x['max_windings']))
    return dict(threshold=threshold,definition='Absolute U difference from predicted; no penalty or projection',layers=out)


def report(settings):
    """Posthoc local-geometry audit using each solve's actual probabilities."""
    import json
    from pathlib import Path
    from lasagna.inference_provenance import atomic_write,sha256_file
    from .fiber_gt_constraint_score import arrays
    torch.set_num_threads(4)
    rows=[]
    for path in settings['solve_roots']:
        root=Path(path);cfg=json.loads((root/'config.json').read_text())
        cfg=dict(cfg)
        for key,value in dict(bucket_uv_match_weight=10.,bucket_uv_attach_weight=.1,
                 bucket_uv_bend_weight=10.,bucket_uv_lr=2.,bucket_uv_padding=64).items():cfg.setdefault(key,value)
        cfg.update(settings.get('parameters',{}))
        d=arrays(cfg['tensor_cache']);sol=arrays(root/'solution.npz');init=arrays(cfg['initial_solution'])
        cfg['shared_curve_integer_geometry']=bool(sol.get('shared_curve_integer_geometry',False))
        for key in ('source_id','start','end','family'):
            if not np.array_equal(d[key],sol[key]) or not np.array_equal(d[key],init[key]):
                raise ValueError('Local UV report identity mismatch')
        # Historical checkpoints explicitly retain their historical topology.
        cfg.setdefault('bucket_uv_perpendicular_nodes_only',bool(sol.get('bucket_uv_perpendicular_nodes_only',False)))
        data,_=soft.tensor_inputs(d,{},cfg,'cpu');ext=BucketUV(d,cfg,'cpu')
        if not cfg.get('bucket_uv_dynamic_same_sheet',False):
            ext.carry=fixed_carry(torch.as_tensor(init['state']),torch.as_tensor(init['widths']),data)
            initial_carry=ext.carry.clone()
        # No cavity iteration during audit; use exactly the saved probabilities.
        ext.cavity.values=torch.as_tensor(sol['winding_sigma'])
        if 'bucket_uv_offsets' in sol:
            raise ValueError('Legacy offset report requires its original implementation')
        if 'bucket_uv_positions' in sol:
            if ext.windowed:
                with np.load(root/'solution.npz') as checkpoint:ext.restore(checkpoint)
            for key,value in ext.saved.items():
                if not np.array_equal(sol[key],value):raise ValueError('Saved local topology mismatch')
            if int(sol['bucket_uv_first_label'])!=ext.low:raise ValueError('Local label bounds mismatch')
            if not cfg.get('bucket_uv_dynamic_same_sheet',False) and not torch.equal(initial_carry,torch.as_tensor(sol['bucket_uv_carry'])):
                raise ValueError('Saved matching is not the original fixed matching')
            if ext.affine is None:
                with torch.no_grad():ext.values.copy_(torch.as_tensor(sol['bucket_uv_positions']))
        else:
            with torch.no_grad():ext.values.copy_(ext.shared_positions(torch.as_tensor(sol['state']),torch.as_tensor(sol['widths'])))
        with torch.no_grad():
            terms=ext.terms(torch.as_tensor(sol['state']),torch.as_tensor(sol['widths']),data,torch.arange(len(data['i'])))
            validity=u_validity(ext,torch.as_tensor(sol['state']),torch.as_tensor(sol['widths']),d['source_id']) if ext.windowed else None
            full={}
            if settings.get('full_objective',False):
                samples=arrays(cfg['soft_samples'])
                ext.prepare_fields(d,samples)
                full_data,nodes=soft.tensor_inputs(d,samples,cfg,'cpu')
                _,initial=soft.uv.radial_initial_geometry(d,cfg)
                widths=torch.as_tensor(sol['widths'])
                w=widths-1e-4
                raw=w+torch.log(-torch.expm1(-w))
                total,all_terms=soft.diagnostic_objective(torch.as_tensor(sol['state']),raw,
                    torch.as_tensor(initial,dtype=torch.float32),full_data,nodes,cfg,
                    density.pair_cost,ext)
                full=dict(full_objective=float(total),full_terms={k:float(v) for k,v in all_terms.items()})
        rows.append(dict(root=str(root),solution_sha256=sha256_file(root/'solution.npz'),
                         **full,
                         u_validity=validity,
                         metrics=dict(ext.last,**(getattr(ext.curve,'last',{}) if ext.curve is not None else {})),
                         terms={k:float(v) for k,v in terms.items()}))
    out=Path(settings['output_root']);out.mkdir(exist_ok=True,parents=True)
    result=dict(rows=rows,definition='Saved sigma and per-run configured same-sheet selection; no optimization')
    atomic_write(out/'report.json',result,durable=False)
    print(json.dumps(result),flush=True)
    return result
