"""CUDA sampling and deterministic pooling for the patch-normal exporter.

Geometry stays float64. Two atomic passes select minimum distance followed by
minimum source index, preserving the reference's stable overlap tie breaking.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _length(a, b, c):
    return libdevice.sqrt_rn((a*a + b*b) + c*c)


@triton.jit
def _geometry(Q, qi, u, v):
    az=tl.load(Q+qi*12+0); ay=tl.load(Q+qi*12+1); ax=tl.load(Q+qi*12+2)
    bz=tl.load(Q+qi*12+3); by=tl.load(Q+qi*12+4); bx=tl.load(Q+qi*12+5)
    cz=tl.load(Q+qi*12+6); cy=tl.load(Q+qi*12+7); cx=tl.load(Q+qi*12+8)
    dz=tl.load(Q+qi*12+9); dy=tl.load(Q+qi*12+10); dx=tl.load(Q+qi*12+11)
    pz=(az*(1-u)+cz*u)*(1-v)+(bz*(1-u)+dz*u)*v
    py=(ay*(1-u)+cy*u)*(1-v)+(by*(1-u)+dy*u)*v
    px=(ax*(1-u)+cx*u)*(1-v)+(bx*(1-u)+dx*u)*v
    uz=(cz-az)*(1-v)+(dz-bz)*v; vz=(bz-az)*(1-u)+(dz-cz)*u
    uy=(cy-ay)*(1-v)+(dy-by)*v; vy=(by-ay)*(1-u)+(dy-cy)*u
    ux=(cx-ax)*(1-v)+(dx-bx)*v; vx=(bx-ax)*(1-u)+(dx-cx)*u
    nz=uy*vx-ux*vy; ny=ux*vz-uz*vx; nx=uz*vy-uy*vz
    norm=_length(nz,ny,nx)
    valid=norm>((1.e-8*_length(uz,uy,ux))*_length(vz,vy,vx))
    return pz,py,px,nz,ny,nx,norm,valid


@triton.jit
def _quad_sizes(Q, NU, NV, BLOCKS, N, SPACING:tl.constexpr, B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B); ok=i<N
    # Padding loads repeat a real quad, then writes are masked.
    qi=tl.minimum(i,N-1)
    az=tl.load(Q+qi*12+0); ay=tl.load(Q+qi*12+1); ax=tl.load(Q+qi*12+2)
    bz=tl.load(Q+qi*12+3); by=tl.load(Q+qi*12+4); bx=tl.load(Q+qi*12+5)
    cz=tl.load(Q+qi*12+6); cy=tl.load(Q+qi*12+7); cx=tl.load(Q+qi*12+8)
    dz=tl.load(Q+qi*12+9); dy=tl.load(Q+qi*12+10); dx=tl.load(Q+qi*12+11)
    nu=tl.maximum(1,libdevice.ceil(tl.maximum(_length(cz-az,cy-ay,cx-ax),_length(dz-bz,dy-by,dx-bx))/SPACING)).to(tl.int64)
    nv=tl.maximum(1,libdevice.ceil(tl.maximum(_length(bz-az,by-ay,bx-ax),_length(dz-cz,dy-cy,dx-cx))/SPACING)).to(tl.int64)
    tl.store(NU+i,nu,ok);tl.store(NV+i,nv,ok);tl.store(BLOCKS+i,(nu*nv+B-1)//B,ok)


@triton.jit
def _pool_pass(Q, NU, NV, ENDS, STARTS, DIST, WIN,
               N, LZ, LY, LX,
               SZ:tl.constexpr, SY:tl.constexpr, SX:tl.constexpr,
               ZBEGIN:tl.constexpr, ZEND:tl.constexpr, PASS:tl.constexpr, B:tl.constexpr):
    block=tl.program_id(0)
    left=tl.full((),0,tl.int32);right=tl.full((),N,tl.int32)
    while left<right:
        mid=(left+right)//2
        end=tl.load(ENDS+mid)
        lower=end<=block
        left=tl.where(lower,mid+1,left);right=tl.where(lower,right,mid)
    qi=left
    before=tl.load(ENDS+qi-1,qi>0,other=0)
    nu=tl.load(NU+qi);nv=tl.load(NV+qi)
    local=(block-before)*B+tl.arange(0,B)
    u=(local//nv+.5).to(tl.float64)/nu;v=(local%nv+.5).to(tl.float64)/nv
    pz,py,px,nz,ny,nx,norm,valid=_geometry(Q,qi,u,v)
    z=libdevice.floor(pz).to(tl.int64);y=libdevice.floor(py).to(tl.int64);x=libdevice.floor(px).to(tl.int64)
    ok=(local<nu*nv)&valid&(pz>=ZBEGIN)&(pz<ZEND)&(z>=LZ)&(z<LZ+SZ)&(y>=LY)&(y<LY+SY)&(x>=LX)&(x<LX+SX)
    cell=((z-LZ)*SY+(y-LY))*SX+(x-LX)
    dz=pz-z-.5;dy=py-y-.5;dx=px-x-.5
    distance=((dz*dz+dy*dy)+dx*dx).to(tl.uint64,bitcast=True)
    if PASS==0:
        tl.atomic_min(DIST+cell,distance,ok,sem='relaxed')
    else:
        minimum=tl.load(DIST+cell,ok,other=0xffffffffffffffff)
        source=tl.load(STARTS+qi)+local
        tl.atomic_min(WIN+cell,source.to(tl.int64),ok&(distance==minimum),sem='relaxed')


@triton.jit
def _gather_geometry(Q, NU, NV, STARTS, WIN, CELLS, P, NORMAL,
                     NQ, N, B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);ok=i<N
    cell=tl.load(CELLS+i,ok,other=0)
    source=tl.load(WIN+cell,ok,other=0)
    left=tl.full((B,),0,tl.int32);right=tl.full((B,),NQ,tl.int32)
    while tl.sum((left<right).to(tl.int32),0)>0:
        mid=(left+right)//2
        start=tl.load(STARTS+tl.minimum(mid,NQ-1))
        lower=start<=source
        active=left<right
        left=tl.where(active&lower,mid+1,left);right=tl.where(active&~lower,mid,right)
    qi=tl.maximum(left-1,0)
    local=source-tl.load(STARTS+qi)
    nu=tl.load(NU+qi);nv=tl.load(NV+qi)
    u=(local//nv+.5).to(tl.float64)/nu;v=(local%nv+.5).to(tl.float64)/nv
    pz,py,px,nz,ny,nx,norm,valid=_geometry(Q,qi,u,v)
    tl.store(P+i*3,pz,ok);tl.store(P+i*3+1,py,ok);tl.store(P+i*3+2,px,ok)
    tl.store(NORMAL+i*3,nz/norm,ok);tl.store(NORMAL+i*3+1,ny/norm,ok);tl.store(NORMAL+i*3+2,nx/norm,ok)


def pool_quads_cuda(quads, low, high, z_roi, spacing=1.):
    """One representative per unit voxel, in lexicographic cell order."""
    q=torch.as_tensor(quads,dtype=torch.float64,device='cuda').contiguous()
    low=tuple(map(int,low));shape=tuple(int(b-a) for a,b in zip(low,high))
    if any(s<=0 for s in shape) or np.prod(shape)>512**3:
        raise ValueError('invalid CUDA pool bounds')
    if not len(q):
        return torch.empty((0,3),device='cuda',dtype=torch.float64),torch.empty((0,3),device='cuda',dtype=torch.float64)
    nu=torch.empty(len(q),device='cuda',dtype=torch.int64);nv=torch.empty_like(nu);blocks=torch.empty_like(nu)
    b=128
    _quad_sizes[(triton.cdiv(len(q),b),)](q,nu,nv,blocks,len(q),spacing,b,enable_fp_fusion=False)
    ends=torch.cumsum(blocks,0);starts=torch.cumsum(nu*nv,0)-nu*nv
    total_blocks=int(ends[-1].item())
    dist=torch.full((int(np.prod(shape)),),float('inf'),device='cuda',dtype=torch.float64)
    win=torch.full((len(dist),),torch.iinfo(torch.int64).max,device='cuda',dtype=torch.int64)
    args=(q,nu,nv,ends,starts,dist.view(torch.uint64),win,len(q),*low,*shape,*z_roi)
    for pass_id in (0,1):
        _pool_pass[(total_blocks,)](*args,pass_id,b,enable_fp_fusion=False)
    cells=torch.nonzero(win!=torch.iinfo(torch.int64).max).flatten()
    p=torch.empty((len(cells),3),device='cuda',dtype=torch.float64);n=torch.empty_like(p)
    if len(cells):
        _gather_geometry[(triton.cdiv(len(cells),b),)](q,nu,nv,starts,win,cells,p,n,len(q),len(cells),b,enable_fp_fusion=False)
    return p,n


def inward_cuda(points, transform, dr, chunk_size=65536):
    """Existing finite-difference checkpoint evaluation, retained on CUDA."""
    from sample_spiral import get_radial_covector_in_scroll_space
    result=torch.empty((len(points),3),device=points.device,dtype=torch.float32)
    with torch.no_grad():
        for start in range(0,len(points),chunk_size):
            cv=get_radial_covector_in_scroll_space(transform,points[start:start+chunk_size].float(),
                                                  epsilon=2.,dr_per_winding=dr)
            norm=torch.linalg.vector_norm(cv,dim=1,keepdim=True)
            result[start:start+chunk_size]=torch.where(norm>0,-cv/norm.clamp_min(1e-12),torch.zeros_like(cv))
    return result


@triton.jit
def _sign(NORMAL,INWARD,OUT,VALID,ALIGN,N,RENORMALIZE:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);ok=i<N
    z=tl.load(NORMAL+i*3,ok,other=0);y=tl.load(NORMAL+i*3+1,ok,other=0);x=tl.load(NORMAL+i*3+2,ok,other=0)
    length=_length(z,y,x);z=z/length;y=y/length;x=x/length
    iz=tl.load(INWARD+i*3,ok,other=0).to(tl.float64);iy=tl.load(INWARD+i*3+1,ok,other=0).to(tl.float64);ix=tl.load(INWARD+i*3+2,ok,other=0).to(tl.float64)
    dot=(z*iz+y*iy)+x*ix
    valid=(tl.abs(dot)>1e-6)&(tl.abs(dot)<float('inf'))
    factor=tl.where(dot<0,-1.,1.)
    z=tl.where(valid,z*factor,0).to(tl.float32);y=tl.where(valid,y*factor,0).to(tl.float32);x=tl.where(valid,x*factor,0).to(tl.float32)
    if RENORMALIZE:
        z=z.to(tl.float64);y=y.to(tl.float64);x=x.to(tl.float64)
        length=tl.maximum(_length(z,y,x),1e-20)
        z=z/length;y=y/length;x=x/length
    tl.store(OUT+i*3,z,ok);tl.store(OUT+i*3+1,y,ok);tl.store(OUT+i*3+2,x,ok)
    tl.store(VALID+i,valid,ok);tl.store(ALIGN+i,tl.where(valid,tl.minimum(tl.abs(dot),1.),0),ok)


def signed_cuda(p,n,transform,dr,grid=None,chunk_size=65536,renormalize=False):
    if grid is None:
        inward=inward_cuda(p,transform,dr,chunk_size);fallback=0
    else:
        inward,fallback=grid.directions(p,n)
    out=torch.empty_like(n,dtype=torch.float64 if renormalize else torch.float32)
    valid=torch.empty(len(p),device='cuda',dtype=torch.bool)
    alignment=torch.empty(len(p),device='cuda',dtype=torch.float32)
    if len(p):
        _sign[(triton.cdiv(len(p),128),)](n,inward,out,valid,alignment,len(p),renormalize,128,enable_fp_fusion=False)
    return out,valid,alignment,fallback


def coarse_rows_cuda(p,low,tile_edge,cell_size):
    lo=torch.as_tensor(low,device='cuda',dtype=torch.float64)
    rows=torch.nonzero(((p>=lo)&(p<lo+tile_edge)).all(1)).flatten()
    v=p[rows]/cell_size
    cell=torch.floor(v).long();local=cell-(lo/cell_size).long()
    edge=tile_edge//cell_size
    key=(local[:,0]*edge+local[:,1])*edge+local[:,2]
    delta=v-cell-.5
    dist=(delta[:,0].square()+delta[:,1].square())+delta[:,2].square()
    minimum=torch.full((edge**3,),float('inf'),device='cuda',dtype=torch.float64)
    minimum.scatter_reduce_(0,key,dist,reduce='amin',include_self=True)
    sentinel=torch.iinfo(torch.int64).max
    winner=torch.full((edge**3,),sentinel,device='cuda',dtype=torch.int64)
    winner.scatter_reduce_(0,key,torch.where(dist==minimum[key],rows,sentinel),reduce='amin',include_self=True)
    return winner[winner!=sentinel]


@triton.jit
def _box(P,NORMAL,VALID,LOOKUP,CHOSEN,OUT,OUTVALID,N,LZ,LY,LX,
         SZ:tl.constexpr,SY:tl.constexpr,SX:tl.constexpr,R:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);ok=i<N
    row=tl.load(CHOSEN+i,ok,other=0)
    z=libdevice.floor(tl.load(P+row*3,ok,other=0)).to(tl.int64)-LZ
    y=libdevice.floor(tl.load(P+row*3+1,ok,other=0)).to(tl.int64)-LY
    x=libdevice.floor(tl.load(P+row*3+2,ok,other=0)).to(tl.int64)-LX
    az=tl.full((B,),0,tl.float64);ay=tl.full((B,),0,tl.float64);ax=tl.full((B,),0,tl.float64)
    count=tl.full((B,),0,tl.int32)
    for dz in tl.static_range(-R,R+1):
      for dy in tl.static_range(-R,R+1):
       for dx in tl.static_range(-R,R+1):
        inside=ok&(z+dz>=0)&(z+dz<SZ)&(y+dy>=0)&(y+dy<SY)&(x+dx>=0)&(x+dx<SX)
        key=((z+dz)*SY+(y+dy))*SX+(x+dx)
        source=tl.load(LOOKUP+key,inside,other=-1)
        present=inside&(source>=0)&tl.load(VALID+tl.maximum(source,0),inside&(source>=0),other=False)
        az+=tl.load(NORMAL+source*3,present,other=0)
        ay+=tl.load(NORMAL+source*3+1,present,other=0)
        ax+=tl.load(NORMAL+source*3+2,present,other=0)
        count+=present.to(tl.int32)
    denom=tl.maximum(count,1).to(tl.float64);az=az/denom;ay=ay/denom;ax=ax/denom
    length=_length(az,ay,ax)
    valid=(count>0)&(length>1e-8)&tl.load(VALID+row,ok,other=False)
    tl.store(OUT+i*3,tl.where(valid,az/length,0),ok)
    tl.store(OUT+i*3+1,tl.where(valid,ay/length,0),ok)
    tl.store(OUT+i*3+2,tl.where(valid,ax/length,0),ok)
    tl.store(OUTVALID+i,valid,ok)


def filter_cuda(p,n,key,*,tile_edge,cell_size,width,transform,dr,grid=None,chunk_size=65536):
    low=np.asarray(key,dtype=np.int64)*tile_edge
    chosen=coarse_rows_cuda(p,low,tile_edge,int(cell_size))
    if not len(chosen):return None,{'fine_samples':len(p),'exact_fallback':0}
    signed,valid,_,fallback=signed_cuda(p,n,transform,dr,grid,chunk_size,renormalize=True)
    radius=width//2;origin=low-radius;edge=tile_edge+2*radius
    cells=torch.floor(p).long()-torch.as_tensor(origin,device='cuda')
    keys=(cells[:,0]*edge+cells[:,1])*edge+cells[:,2]
    lookup=torch.full((edge**3,),-1,device='cuda',dtype=torch.int32)
    lookup[keys]=torch.arange(len(p),device='cuda',dtype=torch.int32)
    mean=torch.empty((len(chosen),3),device='cuda',dtype=torch.float64)
    good=torch.empty(len(chosen),device='cuda',dtype=torch.bool)
    _box[(triton.cdiv(len(chosen),128),)](p,signed,valid,lookup,chosen,mean,good,len(chosen),
        *origin.tolist(),edge,edge,edge,radius,128,enable_fp_fusion=False)
    positions=p[chosen]
    out,outvalid,alignment,second=signed_cuda(positions,mean,transform,dr,grid,chunk_size)
    outvalid &= good
    out[~outvalid]=0;alignment[~outvalid]=0
    return dict(position_zyx=positions,normal_zyx=out,sign_valid=outvalid,
                inward_alignment=alignment,presence=outvalid.to(torch.uint8)*255),dict(fine_samples=len(p),exact_fallback=fallback+second)


@triton.jit(do_not_specialize=['N','LZ','LY','LX','SPACING','SZ','SY','SX'])
def _grid_lookup(P,NORMAL,FIELD,OUT,WEAK,N,LZ,LY,LX,SPACING,
                 SZ,SY,SX,MARGIN:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);ok=i<N
    z=(tl.load(P+i*3,ok,other=0).to(tl.float32)-LZ)/SPACING
    y=(tl.load(P+i*3+1,ok,other=0).to(tl.float32)-LY)/SPACING
    x=(tl.load(P+i*3+2,ok,other=0).to(tl.float32)-LX)/SPACING
    outside=(z<0)|(z>SZ-1)|(y<0)|(y>SY-1)|(x<0)|(x>SX-1)
    iz=tl.minimum(tl.maximum(tl.floor(z).to(tl.int32),0),SZ-2)
    iy=tl.minimum(tl.maximum(tl.floor(y).to(tl.int32),0),SY-2)
    ix=tl.minimum(tl.maximum(tl.floor(x).to(tl.int32),0),SX-2)
    fz=z-iz;fy=y-iy;fx=x-ix
    nz=tl.load(NORMAL+i*3,ok,other=0);ny=tl.load(NORMAL+i*3+1,ok,other=0);nx=tl.load(NORMAL+i*3+2,ok,other=0)
    az=tl.full((B,),0,tl.float32);ay=tl.full((B,),0,tl.float32);ax=tl.full((B,),0,tl.float32)
    mindot=tl.full((B,),float('inf'),tl.float64);maxdot=tl.full((B,),-float('inf'),tl.float64)
    for k in tl.static_range(8):
        dz=k//4;dy=k//2%2;dx=k%2
        at=((iz+dz)*SY+(iy+dy))*SX+(ix+dx)
        vz=tl.load(FIELD+at*3,ok,other=0);vy=tl.load(FIELD+at*3+1,ok,other=0);vx=tl.load(FIELD+at*3+2,ok,other=0)
        weight=tl.where(dz!=0,fz,1-fz)*tl.where(dy!=0,fy,1-fy)*tl.where(dx!=0,fx,1-fx)
        az+=weight*vz;ay+=weight*vy;ax+=weight*vx
        dot=(nz*vz+ny*vy)+nx*vx
        mindot=tl.minimum(mindot,dot);maxdot=tl.maximum(maxdot,dot)
    norm=tl.sqrt((az*az+ay*ay)+ax*ax)
    weak=outside|((mindot<MARGIN)&(maxdot>-MARGIN))|~(norm>1e-8)
    tl.store(OUT+i*3,az/tl.maximum(norm,1e-12),ok)
    tl.store(OUT+i*3+1,ay/tl.maximum(norm,1e-12),ok)
    tl.store(OUT+i*3+2,ax/tl.maximum(norm,1e-12),ok)
    tl.store(WEAK+i,weak,ok)


class InwardGrid:
    """Explicitly approximate checkpoint lookup; geometry is never interpolated here."""
    def __init__(self, low, high, spacing, transform, dr, chunk_size=65536):
        self.spacing=float(spacing);self.transform=transform;self.dr=dr;self.chunk_size=chunk_size
        low_values=np.floor(np.asarray(low)/spacing)*spacing
        self.low=torch.tensor(low_values,device='cuda',dtype=torch.float32)
        end=np.ceil(np.asarray(high)/spacing)*spacing
        shape=np.maximum(np.rint((end-low_values)/spacing).astype(int)+1,2).tolist()
        self.shape=shape
        # Flattened batches avoid allocating a second complete grid of positions.
        self.field=torch.empty((int(np.prod(shape)),3),device='cuda',dtype=torch.float32)
        with torch.no_grad():
            for start in range(0,len(self.field),chunk_size):
                at=torch.arange(start,min(start+chunk_size,len(self.field)),device='cuda')
                indices=torch.stack((at//(shape[1]*shape[2]),at//shape[2]%shape[1],at%shape[2]),1)
                p=self.low+indices.float()*spacing
                self.field[start:start+len(p)]=inward_cuda(p,transform,dr,chunk_size)
        self.volume=self.field.T.reshape(1,3,*shape)
        self.low_values=low_values.tolist()
        self.denominator=torch.tensor(np.asarray(shape)-1,device='cuda',dtype=torch.float32)*spacing

    def lookup(self, points):
        coords=(points.float()-self.low)/self.denominator*2-1
        values=F.grid_sample(self.volume,coords[:,[2,1,0]].reshape(1,1,1,-1,3),
                             mode='bilinear',padding_mode='border',align_corners=True)[0,:,0,0].T
        return F.normalize(values,dim=1)

    def directions(self, points, normals, margin=.75):
        inward=torch.empty((len(points),3),device='cuda',dtype=torch.float32)
        weak=torch.empty(len(points),device='cuda',dtype=torch.bool)
        _grid_lookup[(triton.cdiv(len(points),128),)](points,normals,self.field,inward,weak,len(points),
            *self.low_values,self.spacing,*self.shape,margin,128,enable_fp_fusion=False)
        rows=torch.nonzero(weak).flatten()
        if len(rows):
            inward[rows]=inward_cuda(points[rows],self.transform,self.dr,self.chunk_size)
        return inward,len(rows)
