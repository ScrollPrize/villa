"""Measured overlap witnesses on accepted fine-grid faces, independent of UV."""
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
from scipy.spatial import cKDTree

from winding_arclength import source_grid
from winding_topology import source_quad_mask


def closest_triangle(query, triangles):
    """Closest point and barycentric coordinates on a batch of 3D triangles."""
    a=triangles[:,:,0];ab=triangles[:,:,1]-a;ac=triangles[:,:,2]-a;q=query[:,None]-a
    aa=np.sum(ab*ab,axis=2);bb=np.sum(ab*ac,axis=2);cc=np.sum(ac*ac,axis=2)
    aq=np.sum(ab*q,axis=2);cq=np.sum(ac*q,axis=2);det=aa*cc-bb*bb
    s=(cc*aq-bb*cq)/np.maximum(det,1e-20);t=(aa*cq-bb*aq)/np.maximum(det,1e-20)
    weights=np.stack((1-s-t,s,t),axis=2)
    interior=np.all(weights>=0,axis=2)&(det>1e-15)
    best=np.sum((q-s[:,:,None]*ab-t[:,:,None]*ac)**2,axis=2);best[~interior]=np.inf
    for i,j in ((0,1),(1,2),(2,0)):
        base=triangles[:,:,i];delta=triangles[:,:,j]-base
        fraction=np.clip(np.sum((query[:,None]-base)*delta,axis=2)/np.maximum(np.sum(delta*delta,axis=2),1e-20),0,1)
        distance=np.sum((query[:,None]-base-fraction[:,:,None]*delta)**2,axis=2)
        better=distance<best;candidate=np.zeros_like(weights);candidate[:,:,i]=1-fraction;candidate[:,:,j]=fraction
        weights[better]=candidate[better];best=np.minimum(best,distance)
    return best,weights


class PatchSurface:
    """Triangle index containing only explicitly accepted source quads."""
    def __init__(self, grid, source):
        ij=np.argwhere(source_quad_mask(grid,source))
        self.domain=np.concatenate([ij[:,None]+offset for offset in
            (np.array([[0,0],[0,1],[1,0]]),np.array([[1,1],[1,0],[0,1]]))])
        self.points=grid[self.domain[:,:,0],self.domain[:,:,1]]
        normal=np.cross(self.points[:,1,:3]-self.points[:,0,:3],
                        self.points[:,2,:3]-self.points[:,0,:3])
        length=np.linalg.norm(normal,axis=1);good=length>1e-8
        self.points=self.points[good];self.domain=self.domain[good]
        self.normals=normal[good]/length[good,None]
        self.centers=self.points[:,:,:3].mean(axis=1)
        self.tree=cKDTree(self.centers)
        if len(self.points):
            xyz=self.points[:,:,:3]
            self.low=xyz.min(axis=(0,1));self.high=xyz.max(axis=(0,1))
            self.radius=float(np.linalg.norm(xyz-self.centers[:,None],axis=2).max())
        else:
            self.low=np.full(3,np.inf);self.high=-self.low;self.radius=0.

    def witnesses(self, other, *, distance=1., normal_cos=.95, samples=8, probes=256):
        """Return spatially spread source-domain pairs on real near-coincident faces.

        Broad phase uses a conservative triangle-radius bound. Neither current
        UV positions nor coarse control-mesh chords decide physical contact.
        Boundary vertices supplement centroids for narrow overlaps/touching edges.
        """
        if not len(self.points) or not len(other.points):
            return np.empty((0,2)),np.empty((0,2)),np.empty(0)
        query=np.concatenate((self.centers,self.points[:,:,:3].reshape(-1,3)))
        domain=np.concatenate((self.domain.mean(axis=1),self.domain.reshape(-1,2)))
        normals=np.concatenate((self.normals,np.repeat(self.normals,3,axis=0)))
        inside=np.all((query>=other.low-distance)&(query<=other.high+distance),axis=1)
        selected=np.flatnonzero(inside)
        if len(selected)>probes:
            selected=selected[np.linspace(0,len(selected)-1,probes,dtype=int)]
        query=query[selected];domain=domain[selected];normals=normals[selected]
        nearby=other.tree.query_ball_point(query,distance+other.radius)
        counts=np.array([len(x) for x in nearby],dtype=int)
        if not counts.sum():return np.empty((0,2)),np.empty((0,2)),np.empty(0)
        row=np.repeat(np.arange(len(query)),counts);face=np.concatenate(nearby).astype(int)
        compatible=np.abs(np.sum(normals[row]*other.normals[face],axis=1))>=normal_cos
        row=row[compatible];face=face[compatible]
        d2,w=closest_triangle(query[row],other.points[face,None,:,:3])
        d2=d2[:,0];w=w[:,0]
        order=np.lexsort((face,d2,row));row=row[order];face=face[order];d2=d2[order];w=w[order]
        take=np.r_[True,np.diff(row)!=0] if len(row) else np.empty(0,bool)
        take &= d2<=distance**2
        row=row[take];face=face[take];d2=d2[take];w=w[take]
        if not len(row):return np.empty((0,2)),np.empty((0,2)),np.empty(0)
        # Farthest-point selection gives rotation a measured baseline even in
        # a small overlap; dozens of near-identical witnesses do not do so.
        chosen=[int(np.argmin(d2))];separation=np.full(len(row),np.inf)
        while len(chosen)<min(samples,len(row)):
            separation=np.minimum(separation,np.sum((query[row]-query[row[chosen[-1]]])**2,axis=1))
            separation[chosen]=-1
            next_index=int(np.argmax(separation))
            if separation[next_index]<1e-12:break
            chosen.append(next_index)
        chosen=np.asarray(chosen)
        return (domain[row[chosen]],np.sum(other.domain[face[chosen]]*w[chosen,:,None],axis=1),
                np.sqrt(d2[chosen]))


def _surface_cache(payload, lookup):
    @lru_cache(maxsize=64)
    def surface(identity):
        source=lookup[identity]
        return PatchSurface(source_grid(payload,source),source)
    return surface


def _measure_pair(pair, surface, distance, normal_cos, samples):
    a,b=pair;first,second=surface(a),surface(b)
    qa,qb,d=first.witnesses(second,distance=distance,normal_cos=normal_cos,samples=samples)
    rb,ra,rd=second.witnesses(first,distance=distance,normal_cos=normal_cos,samples=samples)
    return a,b,np.concatenate((qa,ra)),np.concatenate((qb,rb)),np.r_[d,rd]


def _initialize_worker(filename, lookup, distance, normal_cos, samples):
    global _worker_surface, _worker_options
    payload=np.memmap(filename,dtype='<f8',mode='r').reshape(-1,5)
    _worker_surface=_surface_cache(payload,lookup)
    _worker_options=(distance,normal_cos,samples)


def _worker_pair(pair):
    return _measure_pair(pair,_worker_surface,*_worker_options)


def overlap_witnesses(payload, sources, evidence, *, distance=1., normal_cos=.95,
                      samples=8, progress=print, workers=1):
    """Visit every accepted patch pair, including redundant graph-cycle edges.

    File-backed payloads may be read by bounded independent worker processes.
    Results retain sorted pair order on every platform and worker count.
    """
    lookup={s['id']:s for s in sources if not s['fiber'] and not s.get('quarantined',False)}
    pairs=sorted({tuple(sorted((e['a'],e['b']))) for e in evidence
                  if e.get('kind')=='overlap' and e.get('status')=='consistent'
                  and e['a']!=e['b'] and e['a'] in lookup and e['b'] in lookup})
    if not np.isfinite(distance) or distance<=0 or not 0<=normal_cos<=1 or samples<1 or workers<1:
        raise ValueError('Invalid overlap sampling settings')
    executor=None
    try:
        if (workers>1 and isinstance(payload,np.memmap) and payload.offset==0
                and payload.nbytes==Path(payload.filename).stat().st_size
                and payload.dtype==np.dtype('<f8') and payload.flags.c_contiguous):
            executor=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                initializer=_initialize_worker,initargs=(str(payload.filename),lookup,distance,normal_cos,samples))
            results=executor.map(_worker_pair,pairs,chunksize=32,buffersize=workers*2)
        else:
            surface=_surface_cache(payload,lookup)
            results=(_measure_pair(pair,surface,distance,normal_cos,samples) for pair in pairs)
        for number,result in enumerate(results):
            yield result
            if number%5000==0:progress(f'Measured {number+1:,}/{len(pairs):,} accepted patch overlaps')
    finally:
        if executor is not None:executor.shutdown(wait=True,cancel_futures=True)
