"""H-anchored tifxyz surfaces from fixed UV assignments and original AFV curves."""
import json
import time
from pathlib import Path

import numpy as np

from . import fiber_uv_layout as uv
from .fiber_normal_convention import inward_solution
from vesuvius.neural_tracing.fiber_follow.data.afv import read_afv_polylines
from vesuvius.tifxyz.types import Tifxyz
from vesuvius.tifxyz.writer import write_tifxyz


def unit(v):
    length=np.linalg.norm(v,axis=-1)
    valid=np.isfinite(v).all(-1)&(length>1e-8)
    return v/np.maximum(length[...,None],1e-8),valid


def blend_surface(h0,h1,n0,n1,left,right,t,s):
    """left/right are (V(t), V(0), V(1), inward V normal)."""
    n,valid=unit((1-t[:,None])*n0+t[:,None]*n1)
    valid&=np.sum(n0*n1,axis=1)>0
    def residual(values):
        point,a,b,normal=values
        return np.sum((point-((1-t[:,None])*a+t[:,None]*b))*normal,axis=1)
    delta=(1-s)*residual(left)+s*residual(right)
    # Exact endpoint value even if intermediate arithmetic produces roundoff.
    delta=np.where((t==0)|(t==1),0.,delta)
    xyz=(1-t[:,None])*h0+t[:,None]*h1+delta[:,None]*n
    return xyz,valid&np.isfinite(xyz).all(1)


def line_bins(ids,lo,hi,grid):
    """Exact interval/grid intersections, O(total intersections), not all pairs."""
    first=np.searchsorted(grid,lo[ids],side='left')
    end=np.searchsorted(grid,hi[ids],side='right')
    counts=end-first
    owners=np.repeat(ids,counts)
    if not len(owners):return [np.empty(0,np.int64) for _ in grid]
    bins=np.repeat(first,counts)+np.arange(len(owners))-np.repeat(np.cumsum(counts)-counts,counts)
    order=np.argsort(bins,kind='stable');edges=np.searchsorted(bins[order],np.arange(len(grid)+1))
    return [owners[order[edges[i]:edges[i+1]]] for i in range(len(grid))]


def brackets(candidates,positions,query,max_distance,lo=None,hi=None,target_lo=None,target_hi=None):
    """Nearest bracketing lines, optionally requiring full target interval coverage.

    Ties are deterministic by piece ID. The span filter is applied BEFORE
    selection: a closer short V cannot obscure a farther usable V.
    """
    result=np.full((2,len(query)),-1,np.int32)
    if not len(candidates):return result
    order=np.lexsort((candidates,positions[candidates]));ids=candidates[order];p=positions[ids]
    for side in range(2):
        indices=np.searchsorted(p,query,side='right' if side==0 else 'left')-(side==0)
        pending=np.arange(len(query))
        while len(pending):
            inside=(indices[pending]>=0)&(indices[pending]<len(ids))
            pending=pending[inside]
            if not len(pending):break
            chosen=ids[indices[pending]]
            pending=pending[abs(positions[chosen]-query[pending])<=max_distance]
            if not len(pending):break
            chosen=ids[indices[pending]]
            good=np.ones(len(pending),bool) if lo is None else ((lo[chosen]<=target_lo[pending])&(hi[chosen]>=target_hi[pending]))
            result[side,pending[good]]=chosen[good]
            pending=pending[~good];indices[pending]+=(-1 if side==0 else 1)
    return result


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


def construct(d,state,widths,cfg,bank,settings):
    p,_,_,pick,u,v,_=uv.hard_geometry(d,state,widths,cfg)
    confidence=p[np.arange(len(p)),pick]
    accepted=confidence>settings.get('joint_threshold',.5)
    family=d['family'];hids=np.flatnonzero(accepted&(family==0));vids=np.flatnonzero(accepted&(family==1))
    if not len(hids) or not len(vids):raise ValueError('Both H and V supports required')
    ds=d['node_ds'][:,2]-d['node_ds'][:,0]
    direction=np.sign(ds)
    if not np.isin(direction[accepted],[-1,1]).all():raise ValueError('Degenerate piece direction')
    # Recover signed source arc exactly, not the float32 cached endpoint length.
    length=d['end']-d['start'];anchor=np.where(family==0,u,v)
    end=anchor+direction*length
    lo=np.minimum(anchor,end);hi=np.maximum(anchor,end)
    orth=np.where(family==0,v,u)
    step=float(settings.get('grid_step',100.));reach=float(settings.get('max_search',1000.))
    hreach=float(settings.get('max_search_h',reach));vreach=float(settings.get('max_search_v',reach))
    if not np.isfinite(step+hreach+vreach) or min(step,hreach,vreach)<=0:raise ValueError('Invalid grid/search distance')
    xlim=(min(lo[hids].min(),u[vids].min()),max(hi[hids].max(),u[vids].max()))
    ylim=(min(v[hids].min(),lo[vids].min()),max(v[hids].max(),hi[vids].max()))
    xgrid=np.arange(np.floor(xlim[0]/step),np.ceil(xlim[1]/step)+1)*step
    ygrid=np.arange(np.floor(ylim[0]/step),np.ceil(ylim[1]/step)+1)*step
    shape=(len(ygrid),len(xgrid))
    print(f'Surface grid {shape} ({np.prod(shape):,} vertices); supports H={len(hids):,} V={len(vids):,}',flush=True)
    hbins=line_bins(hids,lo,hi,xgrid);vbins=line_bins(vids,lo,hi,ygrid)
    lower=np.full(shape,-1,np.int32);upper=lower.copy()
    for col,candidates in enumerate(hbins):
        lower[:,col],upper[:,col]=brackets(candidates,orth,ygrid,hreach)
    del hbins
    xyz=np.full((*shape,3),-1.,np.float32)
    source=np.searchsorted(bank.ids,d['source_id'])
    def sample(ids,q):
        arc=d['start'][ids]+direction[ids]*(q-anchor[ids])
        if np.any(arc<d['start'][ids]-1e-6) or np.any(arc>d['end'][ids]+1e-6):raise ValueError('Piece extrapolation')
        return bank.sample(source[ids],np.maximum(0,arc))
    counts=dict(h_bracket=0,v_bracket=0,invalid_normals=0,exact_h=0)
    for row,y in enumerate(ygrid):
        col=np.flatnonzero((lower[row]>=0)&(upper[row]>=0));counts['h_bracket']+=len(col)
        if not len(col):continue
        a=lower[row,col];b=upper[row,col];y0=orth[a];y1=orth[b]
        exact=y0==y1
        if exact.any():
            point,_,valid=sample(a[exact],xgrid[col[exact]])
            xyz[row,col[exact][valid]]=point[valid];counts['exact_h']+=int(valid.sum())
        col=col[~exact];a=a[~exact];b=b[~exact];y0=y0[~exact];y1=y1[~exact]
        if not len(col):continue
        left,right=brackets(vbins[row],orth,xgrid[col],vreach,lo,hi,y0,y1)
        good=(left>=0)&(right>=0);counts['v_bracket']+=int(good.sum())
        col=col[good];a=a[good];b=b[good];y0=y0[good];y1=y1[good];left=left[good];right=right[good]
        if not len(col):continue
        t=(y-y0)/(y1-y0);span=orth[right]-orth[left]
        s=np.divide(xgrid[col]-orth[left],span,out=np.zeros(len(col)),where=span!=0)
        h0,n0,valid0=sample(a,xgrid[col]);h1,n1,valid1=sample(b,xgrid[col]);valid=valid0&valid1
        values=[]
        for ids in (left,right):
            point,n,ok=sample(ids,np.full(len(ids),y));pa,_,oka=sample(ids,y0);pb,_,okb=sample(ids,y1)
            valid&=ok&oka&okb;values.append((point,pa,pb,n))
        points,ok=blend_surface(h0,h1,n0,n1,*values,t,s);valid&=ok&(points[:,2]>0)
        counts['invalid_normals']+=int((~valid).sum());xyz[row,col[valid]]=points[valid]
        if row%100==0:print(f'Surface row {row}/{len(ygrid)} valid={int((xyz[:row+1,:,2]>0).sum()):,}',flush=True)
    valid=xyz[:,:,2]>0;xyz[~valid]=-1
    quads=valid[:-1,:-1]&valid[:-1,1:]&valid[1:,:-1]&valid[1:,1:]
    lengths=[]
    for axis in (0,1):
        pair=(valid[:-1]&valid[1:]) if axis==0 else (valid[:,:-1]&valid[:,1:])
        lengths.append(np.linalg.norm(np.diff(xyz,axis=axis),axis=2)[pair])
    edges=np.concatenate(lengths)
    # Opposing triangle normals identify twisted/degenerate individual quads.
    tl=xyz[:-1,:-1];tr=xyz[:-1,1:];bl=xyz[1:,:-1];br=xyz[1:,1:]
    n1=np.cross(tr-tl,bl-tl);n2=np.cross(bl-br,tr-br)
    twisted=int(((np.sum(n1*n2,axis=2)<=0)&quads).sum())
    stats=dict(selected_pieces=int(accepted.sum()),selected_h=len(hids),selected_v=len(vids),
        grid_shape=list(shape),uv_origin=[float(xgrid[0]),float(ygrid[0])],grid_step=step,
        valid_vertices=int(valid.sum()),valid_vertex_percent=float(100*valid.mean()),valid_quads=int(quads.sum()),
        twisted_or_degenerate_quads=twisted,edge_length_quantiles=np.quantile(edges,[0,.5,.95,.99,1]).tolist() if len(edges) else [],
        edges_over_5_grid_steps=int((edges>5*step).sum()),counts=counts)
    return xyz,valid,stats


def validate_sources(cfg,normal_doc):
    signature=json.loads(Path(cfg['tensor_cache']).with_suffix('.json').read_text())['signature']
    hashes={key:uv.sha256_file(cfg[key]) for key in ('database','afv','stage1')}
    for key,value in hashes.items():
        if value!=signature[key]:raise ValueError('Surface source hash mismatch: '+key)
    native=normal_doc['provenance']['config']
    if Path(native['orient_database']).resolve()!=Path(cfg['database']).resolve() or Path(native['afv']).resolve()!=Path(cfg['afv']).resolve():
        raise ValueError('Native normal source mismatch')
    return hashes


def run(settings):
    root=Path(settings['output_root']);root.mkdir(parents=True,exist_ok=True);reports=[]
    for item in settings['solutions']:
        started=time.monotonic();run_root=Path(item['run_root']);out=root/item['label'].lower()
        if out.exists():raise FileExistsError(out)
        cfg=json.loads((run_root/'summary.json').read_text())['config']
        doc=json.loads((Path(settings['normal_curves'])/'index.json').read_text())
        hashes=validate_sources(cfg,doc)
        normal_cfg=doc['provenance']['config']
        if Path(normal_cfg['orient_database']).resolve()!=Path(cfg['database']).resolve():raise ValueError('Normal database mismatch')
        with np.load(cfg['tensor_cache']) as f:d={k:f[k] for k in f.files}
        with np.load(run_root/'solution.npz') as f:
            state=f['state'].copy();widths=f['widths'].copy()
            for key in ('source_id','start','end','family'):
                if not np.array_equal(d[key],f[key]):raise ValueError('Solution/tensor piece mismatch')
        with np.load(cfg['stage1']) as f:normal=inward_solution({k:f[k] for k in f.files})
        p,_,_,pick,_,_,_=uv.hard_geometry(d,state,widths,cfg)
        accepted=p[np.arange(len(p)),pick]>settings.get('joint_threshold',.5)
        del p
        print(f'{item["label"]}: loading {accepted.sum():,} selected piece curves',flush=True)
        bank=CurveBank(cfg['afv'],d['source_id'][accepted],settings['normal_curves'],normal)
        xyz,valid,stats=construct(d,state,widths,cfg,bank,settings)
        if not valid.any():raise ValueError('No supported surface vertices')
        meta=dict(uv_origin=stats['uv_origin'],coordinate_space='PHercParis4/20260411134726',
            provenance=dict(solution=str(run_root/'solution.npz'),solution_sha256=uv.sha256_file(run_root/'solution.npz'),
                afv_sha256=uv.sha256_file(cfg['afv']),tensor_sha256=uv.sha256_file(cfg['tensor_cache']),
                normal_sign_sha256=uv.sha256_file(cfg['stage1']),normal_curves=str(Path(settings['normal_curves']).resolve()),
                normal_index_sha256=uv.sha256_file(Path(settings['normal_curves'])/'index.json'),settings=settings),
            method='H ruled surface plus scalar normal V chord residual; joint>0.5; bounded brackets; holes')
        surface=Tifxyz(xyz[:,:,0],xyz[:,:,1],xyz[:,:,2],uuid='Paris4-uv20-'+item['label'].lower(),
            _scale=(1/stats['grid_step'],)*2,_mask=valid,extra=meta)
        write_tifxyz(out,surface,compression='lzw')
        uv.atomic_write(out/'source_validation.json',hashes,durable=False)
        stats.update(label=item['label'],output=str(out),seconds=time.monotonic()-started)
        uv.atomic_write(out/'surface_statistics.json',stats,durable=False);reports.append(stats)
        print(json.dumps(stats),flush=True)
        del surface,xyz,valid,bank,d,state
    uv.atomic_write(root/'results.json',reports,durable=False)
    return reports


def audit(settings):
    """Audit completed artifact bands and provenance; never render a surface."""
    import tifffile
    results=[]
    for item in settings['solutions']:
        cfg=json.loads((Path(item['run_root'])/'summary.json').read_text())['config']
        doc=json.loads((Path(settings['normal_curves'])/'index.json').read_text())
        hashes=validate_sources(cfg,doc)
        path=Path(settings['output_root'])/item['label'].lower()
        meta=json.loads((path/'meta.json').read_text());stats=json.loads((path/'surface_statistics.json').read_text())
        assert meta['scale']==[1/settings['grid_step']]*2
        bands=[tifffile.imread(path/(a+'.tif')) for a in ('x','y','z')]
        mask=tifffile.imread(path/'mask.tif')
        assert all(b.dtype==np.float32 and b.shape==mask.shape for b in bands)
        valid=mask==255
        assert np.array_equal(valid,bands[2]>0)
        assert all(np.isfinite(b).all() and (b[~valid]==-1).all() for b in bands)
        assert int(valid.sum())==stats['valid_vertices']
        result=dict(label=item['label'],source_hashes=hashes,vertices=int(valid.sum()),
                    shape=list(mask.shape),tifxyz_valid=True)
        uv.atomic_write(path/'validation.json',result,durable=False);results.append(result)
        del bands,mask,valid
    return results
