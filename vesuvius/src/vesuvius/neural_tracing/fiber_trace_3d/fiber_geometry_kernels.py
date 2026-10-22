"""Small reusable fusion targets; no whole-solver compilation or CUDA graphs."""
from functools import lru_cache
from pathlib import Path
import torch


def affine_coordinates(base,local,u,v,owner,alpha):
    """Both axes, all fibers, all labels in one geometry graph."""
    coefficients=torch.stack((u,v),-1).index_select(0,owner)
    t=alpha[:,None,None]
    return base+local+(coefficients[...,0,:]+(1-t)*coefficients[...,1,:]+t*coefficients[...,2,:])


class BatchedAffine(torch.autograd.Function):
    """Both axes and three affine coefficients use a single scatter in backward."""
    @staticmethod
    def forward(ctx,base,local,u,v,owner,alpha):
        ctx.save_for_backward(owner,alpha);ctx.shape=u.shape
        return affine_coordinates(base,local,u,v,owner,alpha)

    @staticmethod
    def backward(ctx,grad):
        owner,alpha=ctx.saved_tensors
        t=alpha[:,None,None]
        coefficients=torch.stack((grad,grad*(1-t),grad*t),-2)
        out=grad.new_zeros((*ctx.shape,2))
        out.index_add_(0,owner,coefficients)
        return grad,grad,out[...,0],out[...,1],None,None


@lru_cache(maxsize=1)
def compiled_affine_coordinates():
    return torch.compile(affine_coordinates,fullgraph=True,dynamic=True,
                         options={'compile_threads':4,'use_fast_math':False})


def interpolate_offsets(offsets, labels, active, ids, left, right, t, k):
    """Four-corner interpolation of the existing winding-local deviations."""
    lo=torch.floor(k).long();alpha=k-lo
    label=torch.stack((lo,lo+1),-1)
    col=label-labels[ids,0,None,None]
    valid=(col>=0)&(col<labels.shape[1])&(left[:,None,None]>=0)
    col=col.clamp(0,labels.shape[1]-1)
    valid=valid&active.index_select(0,ids).gather(1,col.flatten(1)).reshape_as(col)
    nodes=torch.stack((left,right),-1).clamp_min(0)
    index=nodes[:,None,:,None]*offsets.shape[1]+col[:,:,None,:]
    corners=offsets.flatten(0,1).index_select(0,index.flatten()).reshape(*index.shape,2)
    corners=torch.where(valid[:,:,None,:,None],corners,0)
    a=corners[:,:,0,0]*(1-alpha[...,None])+corners[:,:,0,1]*alpha[...,None]
    b=corners[:,:,1,0]*(1-alpha[...,None])+corners[:,:,1,1]*alpha[...,None]
    return a+(b-a)*t[:,None,None]


@lru_cache(maxsize=1)
def compiled_interpolation():
    return torch.compile(interpolate_offsets,fullgraph=True,dynamic=True,
                         options={'compile_threads':4,'use_fast_math':False})


_audited=set()


def audit_interpolation(args,path):
    """Same-state forward/backward comparison on a nonzero fractional probe."""
    from lasagna.inference_provenance import atomic_write
    probe=list(args)
    for i in (0,6,7):probe[i]=probe[i].detach().clone().requires_grad_(True)
    with torch.no_grad():
        probe[0].add_((torch.arange(probe[0].numel(),device=probe[0].device).reshape_as(probe[0])%17)*.01)
        probe[7].add_(.125)
    params=[probe[i] for i in (0,6,7)]
    eager=interpolate_offsets(*probe);fused=compiled_interpolation()(*probe)
    weight=(torch.arange(eager.numel(),device=eager.device).reshape_as(eager)%7).to(eager)/7+.1
    ga=torch.autograd.grad((eager*weight).sum(),params)
    gb=torch.autograd.grad((fused*weight).sum(),params)
    pairs=[(eager,fused),*zip(ga,gb)]
    report=dict(shape=list(eager.shape),checks=[])
    for a,b in pairs:
        report['checks'].append(dict(max_abs=float((a-b).detach().abs().max()),
                                    reference_max=float(a.detach().abs().max())))
    atomic_write(path,report,durable=False)
    for a,b in pairs:torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)


def audit_pair_reduction(args,root):
    from .fiber_implicit_winding import pair_reductions,compiled_pair_cost
    from lasagna.inference_provenance import atomic_write
    key=('pair',root)
    if key in _audited:return
    probe=list(args);indices=(0,2,3,7,8,9)
    for i in indices:probe[i]=probe[i].detach().clone().requires_grad_(True)
    params=[probe[i] for i in indices]
    a=pair_reductions(*probe);b=compiled_pair_cost()(*probe)
    ga=torch.autograd.grad(sum(x.sum() for x in a),params)
    gb=torch.autograd.grad(sum(x.sum() for x in b),params)
    pairs=[*zip(a,b),*zip(ga,gb)]
    checks=[dict(max_abs=float((x-y).detach().abs().max()),reference_max=float(x.detach().abs().max()))
            for x,y in pairs]
    atomic_write(Path(root)/'pair_kernel_audit.json',dict(checks=checks,shape=list(args[0].shape)),durable=False)
    for x,y in pairs:torch.testing.assert_close(x,y,rtol=1e-4,atol=1e-5)
    _audited.add(key)


class BatchedInterpolation(torch.autograd.Function):
    """Same interpolation equations, one explicit batched offset scatter backward."""
    @staticmethod
    def forward(ctx,offsets,labels,active,ids,left,right,t,k):
        lo=torch.floor(k).long();alpha=k-lo
        col=torch.stack((lo,lo+1),-1)-labels[ids,0,None,None]
        valid=(col>=0)&(col<labels.shape[1])&(left[:,None,None]>=0)
        col=col.clamp(0,labels.shape[1]-1)
        valid &= active.index_select(0,ids).gather(1,col.flatten(1)).reshape_as(col)
        nodes=torch.stack((left,right),-1).clamp_min(0)
        index=nodes[:,None,:,None]*offsets.shape[1]+col[:,:,None,:]
        corners=offsets.flatten(0,1).index_select(0,index.flatten()).reshape(*index.shape,2)
        corners=torch.where(valid[:,:,None,:,None],corners,0)
        a=corners[:,:,0,0]*(1-alpha[...,None])+corners[:,:,0,1]*alpha[...,None]
        b=corners[:,:,1,0]*(1-alpha[...,None])+corners[:,:,1,1]*alpha[...,None]
        ctx.save_for_backward(index,valid,alpha,t,corners,b-a)
        ctx.shape=offsets.shape;ctx.dtype=offsets.dtype
        return a+(b-a)*t[:,None,None]

    @staticmethod
    def backward(ctx,grad):
        index,valid,alpha,t,corners,delta=ctx.saved_tensors
        # Keep the original nested interpolation: left receives g-g*t,
        # right receives g*t. Mask only after constructing corner derivatives.
        right=grad*t[:,None,None];left=grad-right
        spatial=torch.stack((left,right),2)
        gc=torch.stack((spatial*(1-alpha[:,:,None,None]),
                        spatial*alpha[:,:,None,None]),3)
        gc=torch.where(valid[:,:,None,:,None],gc,0)
        go=grad.new_zeros((ctx.shape[0]*ctx.shape[1],2),dtype=ctx.dtype)
        go.index_add_(0,index.flatten(),gc.reshape(-1,2).to(ctx.dtype))
        gt=(grad*delta).sum((1,2)) if ctx.needs_input_grad[6] else None
        gk=((spatial*(corners[:,:,:,1]-corners[:,:,:,0])).sum((2,3))
            if ctx.needs_input_grad[7] else None)
        return go.reshape(ctx.shape),None,None,None,None,None,gt,gk


def run_interpolation(*args,fused=False,audit_root=None,batched_backward=False):
    fn=(BatchedInterpolation.apply if batched_backward else
        compiled_interpolation() if fused and args[0].is_cuda else interpolate_offsets)
    if fused and args[0].is_cuda and torch.is_grad_enabled() and audit_root is not None and audit_root not in _audited:
        audit_interpolation(args,Path(audit_root)/'geometry_kernel_audit.json')
        _audited.add(audit_root)
    return fn(*args)
