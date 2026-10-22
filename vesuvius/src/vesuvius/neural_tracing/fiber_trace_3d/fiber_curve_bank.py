"""Original AFV curve interpolation and frozen native normal sampling."""
import json
from pathlib import Path
import numpy as np
from vesuvius.neural_tracing.tracers.shared.annotations.afv import read_afv_polylines


def unit(v):
    length=np.linalg.norm(v,axis=-1)
    valid=np.isfinite(v).all(-1)&(length>1e-8)
    return v/np.maximum(length[...,None],1e-8),valid


class CurveBank:
    """Vectorized exact polyline interpolation and shared native normal samples."""
    def __init__(self,afv,source_ids,normal_root,normal_solution):
        self.ids=np.unique(source_ids)
        curves=read_afv_polylines(afv,self.ids)
        if not np.array_equal([row['id'] for row,_ in curves],self.ids):raise ValueError('AFV source identity mismatch')
        lengths=[];arcs=[];points=[]
        for row,xyz in curves:
            arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
            keep=np.r_[True,np.diff(arc)>0]
            if keep.sum()<2:raise ValueError('Degenerate AFV curve')
            arcs.append(arc[keep]);points.append(xyz[keep]);lengths.append(arc[-1])
        self.lengths=np.asarray(lengths)
        self.base=np.r_[0.,np.cumsum(self.lengths[:-1]+1.)]
        self.keys=np.concatenate([a+b for a,b in zip(arcs,self.base)])
        self.xyz=np.concatenate(points)
        self.first=np.r_[0,np.cumsum([len(a) for a in arcs])[:-1]]
        self.last=self.first+np.asarray([len(a) for a in arcs])-1
        doc=json.loads((Path(normal_root)/'index.json').read_text())
        if doc['format']!='fiber-normal-curves' or doc['columns']!=['arc','nx','ny','nz','valid']:raise ValueError('Normal curve format')
        index=np.asarray(doc['index'],np.int64)
        at=np.searchsorted(index[:,0],self.ids)
        if not np.array_equal(index[at,0],self.ids):raise ValueError('Missing normal fibers')
        self.normal_first=index[at,1];self.normal_count=index[at,2]
        if (self.normal_count<2).any():raise ValueError('Degenerate normal curve')
        self.normals=np.memmap(Path(normal_root)/'normals.bin',dtype='<f8',mode='r',shape=(doc['statistics']['samples'],5))
        self.spacing=doc['statistics']['sample_spacing_base']
        loc=np.searchsorted(normal_solution['ids'],self.ids)
        if not np.array_equal(normal_solution['ids'][loc],self.ids):raise ValueError('Normal sign identity')
        if normal_solution['broken'][loc].any():raise ValueError('Selected normal-stage broken fiber')
        self.sign=normal_solution['normal_sign'][loc]

    def sample(self,source,arc):
        source=np.asarray(source);arc=np.asarray(arc)
        valid=np.isfinite(arc)&(arc>=0)&(arc<=self.lengths[source]+1e-6)
        if not valid.all():raise ValueError('Curve sampling outside source span')
        arc=np.minimum(arc,self.lengths[source])  # endpoint roundoff only
        q=self.base[source]+arc
        i=np.searchsorted(self.keys,q,side='right')-1
        i=np.maximum(self.first[source],np.minimum(i,self.last[source]-1))
        t=(q-self.keys[i])/(self.keys[i+1]-self.keys[i])
        xyz=self.xyz[i]*(1-t[:,None])+self.xyz[i+1]*t[:,None]
        j=self.normal_first[source]+np.minimum((arc/self.spacing).astype(np.int64),self.normal_count[source]-2)
        a=self.normals[j];b=self.normals[j+1]
        f=(arc-a[:,0])/(b[:,0]-a[:,0]);raw=a[:,1:4]*(1-f[:,None])+b[:,1:4]*f[:,None]
        n,nok=unit(raw*self.sign[source,None])
        # Invalid sample gaps are not filled, even if nearby normals are valid.
        nok&=(a[:,4]>0)&(b[:,4]>0)&(np.sum(a[:,1:4]*b[:,1:4],axis=1)>0)
        return xyz,n,nok
