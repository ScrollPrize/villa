"""Fixed base + fiber translation + endpoint interpolation + local residual.

No cumulative coordinates, projection, or probability gating. U/V retain each
layer's native units; direction losses receive physical voxel coordinates.
"""
import math
import torch
from .fiber_geometry_cache import cached_geometry, row_gather, profile_geometry


def remap_rows(old, initial, owner, old_labels, old_active, labels, active):
    if old_labels is None:
        return initial, None, None
    col=labels-old_labels[:,:1]
    valid=(col>=0)&(col<old_labels.shape[1])&active
    col=col.clamp(0,old_labels.shape[1]-1)
    valid &= old_active.gather(1,col)
    index=col[owner,:,None].expand(-1,-1,initial.shape[-1])
    keep=valid[owner,:,None]
    return torch.where(keep,old.gather(1,index),initial),keep,index


def remap_parameter(old, initial, owner, old_labels, old_active, labels, active, optimizer):
    """Preserve parameter values and Adam moments by persistent integer label."""
    value,keep,index=remap_rows(old,initial,owner,old_labels,old_active,labels,active)
    new=torch.nn.Parameter(value)
    if optimizer is not None:
        for group in optimizer.param_groups:
            group['params']=[new if p is old else p for p in group['params']]
        prior=optimizer.state.pop(old,{})
        remapped={}
        for name,value in prior.items():
            if torch.is_tensor(value) and value.shape==old.shape:
                value=(torch.where(keep,value.gather(1,index),0) if keep is not None
                       else torch.zeros_like(initial))
            remapped[name]=value
        if remapped:optimizer.state[new]=remapped
    return new


def changed_row_plan(owner, old_labels, old_active, labels, active):
    """One batched index plan, shared by parameters and their Adam moments."""
    changed=(labels!=old_labels).any(1)|(active!=old_active).any(1)
    rows=torch.where(changed.index_select(0,owner))[0]
    fibers=owner.index_select(0,rows)
    col=labels.index_select(0,fibers)-old_labels.index_select(0,fibers)[:,:1]
    keep=(col>=0)&(col<old_labels.shape[1])&active.index_select(0,fibers)
    col=col.clamp(0,old_labels.shape[1]-1)
    keep &= old_active.index_select(0,fibers).gather(1,col)
    return rows,col,keep


def remap_inplace(value, plan, initial=None):
    rows,col,keep=plan
    index=col[...,None].expand(-1,-1,value.shape[-1])
    selected=value.index_select(0,rows).gather(1,index)
    replacement=0 if initial is None else initial.index_select(0,rows)
    value.index_copy_(0,rows,torch.where(keep[...,None],selected,replacement))


def remap_adam_inplace(parameter, plan, optimizer):
    remap_inplace(parameter,plan)
    if optimizer is not None:
        for value in optimizer.state.get(parameter,{}).values():
            if torch.is_tensor(value) and value.shape==parameter.shape:
                remap_inplace(value,plan)


def remap_adam_batched(parameters, plan, optimizer):
    """One packed gather/mask for all axes and Adam moments, over all rows.

    Python only enumerates the fixed set of optimizer tensors, never fibers,
    nodes or bucket labels. Scalar Adam step counters remain untouched.
    """
    values=[]
    for parameter in parameters:
        values.append(parameter)
        if optimizer is not None:
            values.extend(v for v in optimizer.state.get(parameter,{}).values()
                          if torch.is_tensor(v) and v.shape==parameter.shape)
    rows,col,keep=plan
    sizes=[v.shape[-1] for v in values]
    packed=torch.cat(values,dim=-1)
    selected=packed.index_select(0,rows).gather(
        1,col[...,None].expand(-1,-1,packed.shape[-1]))
    selected=torch.where(keep[...,None],selected,0)
    # Fixed small number of storage writes; the index arithmetic and remapping
    # kernel operate on the entire packed parameter/state block together.
    for value,part in zip(values,selected.split(sizes,dim=-1)):
        value.index_copy_(0,rows,part)


class AffineOffsets(torch.nn.Module):
    def __init__(self, owner, arc, start, end, device, dtype):
        super().__init__()
        owner=torch.as_tensor(owner,device=device,dtype=torch.long)
        arc=torch.as_tensor(arc,device=device,dtype=dtype)
        start=torch.as_tensor(start,device=device,dtype=dtype)
        end=torch.as_tensor(end,device=device,dtype=dtype)
        span=end-start
        self.register_buffer('owner',owner)
        # Single-point fibers have no identifiable tilt; both endpoints share weight.
        safe=torch.where(span!=0,span,torch.ones_like(span))
        t=torch.where(span[owner]!=0,(arc-start[owner])/safe[owner],.5)
        self.register_buffer('alpha',t)
        self.register_buffer('base',torch.empty(len(owner),0,2,device=device,dtype=dtype))
        self.u=torch.nn.Parameter(torch.empty(len(start),0,3,device=device,dtype=dtype))
        self.v=torch.nn.Parameter(torch.empty(len(start),0,3,device=device,dtype=dtype))

    @cached_geometry
    @profile_geometry
    def compose(self, local):
        if getattr(self,'batched_backward',False):
            from .fiber_geometry_kernels import BatchedAffine
            return BatchedAffine.apply(self.base,local,self.u,self.v,self.owner,self.alpha)
        if getattr(self,'fused',False):
            from .fiber_geometry_kernels import affine_coordinates,compiled_affine_coordinates
            # Keep diagnostic/no-grad and the singleton FP64 shared layer out
            # of the training graph's specialization cache.
            fn=(compiled_affine_coordinates() if local.is_cuda and torch.is_grad_enabled()
                and local.dtype==torch.float32 else affine_coordinates)
            return fn(self.base,local,self.u,self.v,self.owner,self.alpha)
        t=self.alpha[:,None]
        def axis(p):
            c=row_gather(p,self.owner)
            return c[...,0]+(1-t)*c[...,1]+t*c[...,2]
        return self.base+local+torch.stack((axis(self.u),axis(self.v)),-1)

    def initialize(self, initial):
        self.base=initial.detach().clone()
        shape=(len(self.u),initial.shape[1],3)
        # Shared geometry parameters already belong to an optimizer: retain identity.
        with torch.no_grad():
            if self.u.shape==shape:
                self.u.zero_();self.v.zero_()
            else:
                self.u=torch.nn.Parameter(initial.new_zeros(shape))
                self.v=torch.nn.Parameter(initial.new_zeros(shape))

    def remap(self, local, initial, old_labels, old_active, labels, active, optimizer, reuse=False, batched=True):
        with torch.no_grad():
            if reuse and old_labels is not None and old_labels.shape==labels.shape:
                # Keep parameter identity/storage and scalar Adam step intact.
                node_plan=changed_row_plan(self.owner,old_labels,old_active,labels,active)
                fibers=torch.arange(len(labels),device=labels.device)
                fiber_plan=changed_row_plan(fibers,old_labels,old_active,labels,active)
                remap_inplace(self.base,node_plan,initial)
                # Inactive bases retain the same current zero-offset prediction
                # as the original remapper; they never carry an optimized value.
                self.base.copy_(torch.where(active[self.owner,:,None],self.base,initial))
                if batched:
                    remap_adam_batched((self.u,self.v),fiber_plan,optimizer)
                    remap_adam_batched((local,),node_plan,optimizer)
                else:
                    remap_adam_inplace(self.u,fiber_plan,optimizer)
                    remap_adam_inplace(self.v,fiber_plan,optimizer)
                    remap_adam_inplace(local,node_plan,optimizer)
                return local
            self.base=remap_rows(self.base,initial,self.owner,old_labels,old_active,labels,active)[0]
            fibers=torch.arange(len(labels),device=labels.device)
            zeros=initial.new_zeros(len(labels),labels.shape[1],3)
            self.u=remap_parameter(self.u,zeros,fibers,old_labels,old_active,labels,active,optimizer)
            self.v=remap_parameter(self.v,zeros.clone(),fibers,old_labels,old_active,labels,active,optimizer)
            return remap_parameter(local,torch.zeros_like(initial),self.owner,
                                   old_labels,old_active,labels,active,optimizer)

    def arrays(self,prefix,local):
        return {prefix+k:v.detach().cpu().numpy() for k,v in
                [('base',self.base),('local',local),('global_end_u',self.u),('global_end_v',self.v)]}

    def restore(self,saved,prefix,device):
        self.base=torch.as_tensor(saved[prefix+'base'],device=device).clone()
        self.u=torch.nn.Parameter(torch.as_tensor(saved[prefix+'global_end_u'],device=device).clone())
        self.v=torch.nn.Parameter(torch.as_tensor(saved[prefix+'global_end_v'],device=device).clone())
        return torch.nn.Parameter(torch.as_tensor(saved[prefix+'local'],device=device).clone())


def direction_plan(owner,arc,family,direction):
    right=torch.where(owner[1:]==owner[:-1])[0]+1
    left=right-1
    return right,left,family[owner[right]].long(),direction[owner[right]],arc[right]-arc[left],owner[right]


def arc_length_energy(delta,gap,active=None,groups=1,weighting='arc_length'):
    """Squared relative length error; equal active-label weight.

    delta is [segments, labels, UV] in voxels, gap is reference arc length.
    Shared and local layers use the same physical length objective.
    uniform gives each segment equal weight, independent of reference length.
    arc_length preserves historical experiments.
    """
    if weighting not in ('uniform','arc_length'):
        raise ValueError('Unknown length segment weighting: '+str(weighting))
    if not len(delta):
        zero=delta.sum()*0
        return zero if groups==1 else zero.expand(groups)
    gap=gap.to(delta)[:,None]
    error=(torch.linalg.vector_norm(delta,dim=-1)/gap.clamp_min(1e-12)-1).square()
    mask=torch.ones_like(error) if active is None else active.to(error)
    if groups>1 and active is not None:mask=mask.repeat(1,groups)
    weights=mask if weighting=='uniform' else gap*mask
    loss=(error*weights).reshape(len(delta),groups,-1).sum((0,2))/weights.reshape(len(delta),groups,-1).sum((0,2)).clamp_min(1e-30)
    return loss[0] if groups==1 else loss


def direction_cone(positions, owner, arc, family, direction, config, active=None,plan=None,groups=1):
    """Reference-arc-weighted H/V direction energy, not a hard bound.

    normalized_vector uses 2*(1-cos(theta)), independent of segment length,
    plus the existing minimum forward-progress safeguard. Historical cosine
    and quadratic_vector modes remain available for exact experiment replay.
    Every active bucket has equal weight, including low-probability ones.
    Zero-length segments are violations, not silently discarded. Optional
    signed-progress hinge adds a non-stationary gradient at axial reversals.
    """
    angle=float(config.get('geometry_direction_deadzone_degrees',10.))
    if not math.isfinite(angle) or not 0<=angle<45:
        raise ValueError('geometry_direction_deadzone_degrees must be in [0,45)')
    progress_weight=float(config.get('geometry_forward_progress_weight',0.))
    minimum=float(config.get('geometry_minimum_forward_slope',.1))
    if not math.isfinite(progress_weight) or progress_weight<0:
        raise ValueError('geometry_forward_progress_weight must be finite and nonnegative')
    if not math.isfinite(minimum) or minimum<0:
        raise ValueError('geometry_minimum_forward_slope must be finite and nonnegative')
    if positions.ndim==2:positions=positions[:,None,:]
    right,left,f,d,gap,owners=direction_plan(owner,arc,family,direction) if plan is None else plan
    if not len(right):return positions.sum()*0 if groups==1 else (positions.sum()*0).expand(groups),{}
    delta=row_gather(positions,right)-row_gather(positions,left)
    longitudinal=delta.gather(-1,f[:,None,None].expand(-1,delta.shape[1],1)).squeeze(-1)
    signed=longitudinal*d[:,None]
    norm=torch.linalg.vector_norm(delta,dim=-1)
    cosine=signed/norm.clamp_min(1e-12)
    c0=math.cos(math.radians(angle));c45=math.sqrt(.5)
    mode=config.get('geometry_direction_loss','cosine')
    if mode in ('quadratic_vector','normalized_vector') and progress_weight:
        raise ValueError(mode+' includes its own progress hinge; set geometry_forward_progress_weight=0')
    forward=signed/gap.to(signed)[:,None].clamp_min(1e-12)
    if mode=='quadratic_vector':
        # Squared distance to a forward axial vector of the SAME length.
        # No angular dead zone or target unit speed. Nonzero restoring force
        # at 180deg; the small progress hinge also prevents zero-length escape.
        transverse=delta.gather(-1,(1-f)[:,None,None].expand(-1,delta.shape[1],1)).squeeze(-1)
        scale=gap.to(signed)[:,None].clamp_min(1e-12)
        penalty=(transverse/scale).square()+((norm-signed)/scale).square()+(minimum-forward).clamp_min(0).square()
    elif mode=='normalized_vector':
        # Normalize by CURRENT length: shrinkage cannot reduce the angular loss.
        # At collapse there is no direction: use a finite constant angular cost
        # and let the existing progress hinge supply the restoring gradient.
        # Clamp before division as well as masking to keep backward finite.
        unit_cosine=signed/norm.clamp_min(1e-6)
        angular=torch.where(norm>1e-6,2*(1-unit_cosine.clamp(-1,1)),
                            torch.full_like(norm,2.))
        penalty=angular+(minimum-forward).clamp_min(0).square()
    elif mode=='cosine':
        penalty=((c0-cosine).clamp_min(0)/(c0-c45)).square()
    else:
        raise ValueError('Unknown geometry_direction_loss: '+str(mode))
    # Unlike cosine, signed progress has a restoring gradient even at 180deg
    # or a collapsed segment. Apply to actual voxel coordinates in every layer.
    if progress_weight and mode=='cosine':
        penalty=penalty+progress_weight*(minimum-forward).clamp_min(0).square()
    mask=(torch.ones_like(penalty,dtype=torch.bool) if active is None else active[owners])
    if groups>1 and active is not None:mask=mask.repeat(1,groups)
    gap=gap.to(penalty)[:,None]
    weight=gap*mask
    loss=((penalty*weight).reshape(len(right),groups,-1).sum((0,2))/
          weight.reshape(len(right),groups,-1).sum((0,2)).clamp_min(1e-30))
    if groups==1:loss=loss[0]
    stats={}
    if not torch.is_grad_enabled():
        selected_mask=mask>0
        c=cosine[selected_mask];zero=norm[selected_mask]<=1e-12
        stats=dict(segments=int(c.numel()),above10=int((c<math.cos(math.radians(10))).sum()),
                   above45=int((c<c45).sum()),reversals=int((c<0).sum()),
                   degenerate=int(zero.sum()),angle_max=float(torch.rad2deg(torch.acos(c.clamp(-1,1))).max()) if c.numel() else 0.)
        if progress_weight or mode in ('quadratic_vector','normalized_vector'):
            selected=forward[selected_mask]
            stats.update(forward_slope_min=float(selected.min()) if selected.numel() else None,
                         below_minimum_progress=int((selected<minimum).sum()))
    return loss,stats
