"""Exact Z-clipped induced subgraphs of reference-selected original fibers.

Reference labels only select this benchmark; they never become training losses.
Original AFV/source arcs and the full-corpus radial prior are retained.
"""
import json
from pathlib import Path
import time

import numpy as np
from lasagna.inference_provenance import atomic_write, sha256_file
from . import fiber_uv_layout as uv


def clipped_intervals(xyz, z_range):
    """Return original-arc intervals inside the closed Z slab, including crossings."""
    xyz=np.asarray(xyz,dtype=float)
    lo,hi=map(float,z_range)
    if not np.isfinite(xyz).all() or not np.isfinite([lo,hi]).all() or lo>=hi:
        raise ValueError('Invalid curve or Z slab')
    length=np.linalg.norm(np.diff(xyz,axis=0),axis=1)
    arc=np.r_[0.,np.cumsum(length)]; result=[]
    for n,(a,b) in enumerate(zip(xyz[:-1,2],xyz[1:,2])):
        if length[n]==0:continue
        if a==b:
            if not lo<=a<=hi:continue
            t0,t1=0.,1.
        else:
            bounds=sorted(((lo-a)/(b-a),(hi-a)/(b-a)))
            t0=max(0.,bounds[0]);t1=min(1.,bounds[1])
            if t1<=t0:continue
        start=arc[n]+length[n]*t0;end=arc[n]+length[n]*t1
        if result and abs(result[-1][1]-start)<1e-8:result[-1][1]=end
        else:result.append([start,end])
    return np.asarray(result,dtype=float).reshape(-1,2)


def remap_endpoints(old_owner, old_ds, d, parents, starts, ends, directions):
    """Remap only actual retained endpoints; never clamp across an excision."""
    old_owner=np.asarray(old_owner);old_ds=np.asarray(old_ds)
    arc=d['start'][old_owner]+old_ds/directions[old_owner]
    owner=np.full(len(arc),-1,np.int64);ds=np.zeros(len(arc),np.float32)
    order=np.argsort(old_owner,kind='stable');sorted_owner=old_owner[order]
    for new,old in enumerate(parents):
        idx=order[np.searchsorted(sorted_owner,old):np.searchsorted(sorted_owner,old,side='right')]
        idx=idx[(arc[idx]>=starts[new])&(arc[idx]<=ends[new])]
        owner[idx]=new;ds[idx]=(arc[idx]-starts[new])*directions[old]
    return owner,ds


def retained_links(d, parents, starts, ends, directions):
    links=d['links']
    if not len(links):return np.empty((0,4))
    # These are exact shared cuts, not generic events. Recovering a cut using
    # start+(end-start) can move it a float ulp outside the retained interval.
    old_a,old_b=links[:,:2].astype(int).T
    if not np.array_equal(d['end'][old_a],d['start'][old_b]):raise ValueError('Parent continuation is not a shared cut')
    tail={int(old):new for new,old in enumerate(parents) if ends[new]==d['end'][old]}
    head={int(old):new for new,old in enumerate(parents) if starts[new]==d['start'][old]}
    result=[]
    for a,b in zip(old_a,old_b):
        if a in tail and b in head:
            na,nb=tail[a],head[b]
            result.append((na,nb,float(np.float32((ends[na]-starts[na])*directions[a])),0.))
    return np.asarray(result,dtype=float).reshape(-1,4)


def reanchored_centers(d, parents, starts, phases, directions, centers, widths):
    """Translate the full-data radial mean to new anchors without stretching it."""
    from .fiber_reference_comparison import inverse_u
    q=centers[parents]+d['phase'][parents]
    bucket=np.floor(q).astype(int).clip(0,len(widths)-1)
    boundaries=np.r_[0.,np.cumsum(widths)]
    u=boundaries[bucket]+(q-bucket)*widths[bucket]
    u+=(d['family'][parents]==0)*(starts-d['start'][parents])*directions[parents]
    return (inverse_u(u,widths)-phases).astype(np.float32)


def audit_subset(d, parent, projection, z_range, winding_range):
    """Independent post-extraction assertions, also usable on existing artifacts."""
    count=len(d['source_id']);parents=d['parent_piece']
    for key,values in d.items():
        if np.issubdtype(np.asarray(values).dtype,np.number) and not np.isfinite(values).all():
            raise ValueError('Nonfinite subset array: '+key)
    for key in ('source_id','source_index','family'):
        if not np.array_equal(d[key],parent[key][parents]):raise ValueError('Subset identity mismatch: '+key)
    if np.any(d['start']<parent['start'][parents]) or np.any(d['end']>parent['end'][parents]) or np.any(d['lengths']<=0):
        raise ValueError('Invalid clipped intervals')
    if np.any(np.diff(d['source_id'])<0):raise ValueError('Unsorted subset sources')
    if not np.allclose(d['lengths'],d['end']-d['start'],rtol=0,atol=1e-8):raise ValueError('Wrong clipped lengths')
    directions=np.sign(d['node_ds'][:,2]-d['node_ds'][:,0])
    for key,ds in (('i','ds_a'),('j','ds_b')):
        owner=d[key]
        if np.any(owner<0) or np.any(owner>=count):raise ValueError('Bad event owner')
        offset=d[ds]/directions[owner]
        if np.any(offset<0) or np.any(offset>d['lengths'][owner]+1e-3):raise ValueError('Event outside clipped piece')
    expected=np.bincount(np.r_[d['i'],d['j']],minlength=count)
    if not np.array_equal(expected,d['degree']):raise ValueError('Wrong induced degree')
    if len(d['links']):
        a,b=d['links'][:,:2].astype(int).T
        if np.any(a<0) or np.any(b<0) or np.any(a>=count) or np.any(b>=count):raise ValueError('Bad link owner')
        if not np.array_equal(d['source_id'][a],d['source_id'][b]) or not np.array_equal(d['end'][a],d['start'][b]):
            raise ValueError('Link bridges removed geometry')
        if not np.allclose(d['links'][:,2],directions[a]*d['lengths'][a],atol=1e-3) or not np.allclose(d['links'][:,3],0,atol=1e-6):
            raise ValueError('Link does not join cut endpoints')
    for key in ('source_id','start','end'):
        if not np.array_equal(projection[key],d[key]):raise ValueError('Projection identity mismatch')
    owner=projection['owner'];arc=projection['arc'];offset=projection['offset']
    if len(offset)!=count+1 or offset[0]!=0 or offset[-1]!=len(owner) or np.any(np.diff(offset)<2):raise ValueError('Bad projection offsets')
    if not np.array_equal(owner,np.repeat(np.arange(count),np.diff(offset))):raise ValueError('Bad projection owners')
    if np.any(arc<d['start'][owner]) or np.any(arc>d['end'][owner]):raise ValueError('Bad projected arcs')
    z=projection['xyz'][:,2];chosen=projection['chosen_winding']
    if winding_range is not None and (not np.isfinite(chosen).all() or np.any(chosen<winding_range[0]) or np.any(chosen>winding_range[1])):raise ValueError('Bad reference selection')
    if np.any(z<z_range[0]-.01) or np.any(z>z_range[1]+.01):raise ValueError('Geometry outside Z slab')
    np.testing.assert_allclose(np.add.reduceat(projection['weight'],offset[:-1]),d['lengths'],rtol=1e-6,atol=1e-6)
    return dict(pieces=count,events=len(d['i']),links=len(d['links']),samples=len(owner),validated=True)


def validate_cache(config):
    manifest=json.loads(Path(config['reference_subset']).read_text())
    if Path(config['tensor_cache']).resolve()!=Path(manifest['tensor_cache']).resolve():
        raise ValueError('Subset tensor path mismatch')
    if int(config.get('winding_count',100))!=manifest['winding_count']:
        raise ValueError('Subset winding count mismatch')
    for key in ('afv','umbilicus','database','features','stage1','hv_solution','h_direction_solution','v_direction_solution'):
        if sha256_file(config[key])!=manifest['signature']['inputs'][key]:
            raise ValueError('Subset source provenance mismatch: '+key)
    if config.get('h_convention','ccw')!=manifest['h_convention'] or not uv.anchored(config):
        raise ValueError('Subset geometry convention mismatch')
    for name,digest in manifest['outputs'].items():
        if sha256_file(Path(config['reference_subset']).parent/name)!=digest:
            raise ValueError('Subset output changed: '+name)


def run(settings):
    from . import fiber_reference_comparison as ref
    from .fiber_normal_sign_visualization import load_umbilicus_frame
    from vesuvius.neural_tracing.tracers.shared.annotations.afv import read_afv_polylines
    started=time.monotonic()
    source_path=Path(settings['source_config'])
    config=json.loads(source_path.read_text());config=config.get('fiber_uv_layout',config)
    if not uv.anchored(config):raise ValueError('Subset extraction requires aligned shared-anchor geometry')
    root=Path(settings['output_root']);manifest_path=root/'manifest.json'
    reference_selected=settings.get('winding_range') is not None
    input_keys=('afv','umbilicus','database','features','stage1','hv_solution','h_direction_solution','v_direction_solution','tensor_cache')
    inputs={key:sha256_file(config[key]) for key in input_keys+(('reference_projection',) if reference_selected else ())}
    parent_signature=json.loads(Path(config['tensor_cache']).with_suffix('.json').read_text())['signature']
    for key in ('afv','umbilicus','database','features','stage1','hv_solution','h_direction_solution','v_direction_solution'):
        if parent_signature.get(key)!=inputs[key]:raise ValueError('Parent tensor provenance mismatch: '+key)
    if parent_signature.get('h_convention')!=config.get('h_convention','ccw') or parent_signature.get('layout_mode')!='shared_anchor':
        raise ValueError('Parent tensor geometry convention mismatch')
    signature=dict(version=1,inputs=inputs,source_config=sha256_file(source_path),
                   z_range=list(settings['z_range']),winding_range=list(settings['winding_range']) if reference_selected else None,
                   sample_spacing=float(settings.get('sample_spacing',64.)))
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['signature']!=signature:raise ValueError('Refusing stale subset overwrite')
        validate_cache(dict(config,reference_subset=str(manifest_path),tensor_cache=manifest['tensor_cache']))
        print(json.dumps(manifest),flush=True);return manifest
    if root.exists() and any(root.iterdir()):raise FileExistsError('Incomplete/nonempty subset output: '+str(root))
    if signature['sample_spacing']<=0:raise ValueError('Invalid sample spacing')
    with np.load(config['tensor_cache']) as src:d={k:src[k] for k in src.files}
    chosen=(ref.load_metric_projection(config['reference_projection'],d,config)['chosen_winding']
            if reference_selected else np.full(len(d['family']),np.nan))
    wlo,whi=map(float,settings['winding_range']) if reference_selected else (-np.inf,np.inf)
    if wlo>whi:raise ValueError('Invalid winding range')
    selected=np.flatnonzero(np.isfinite(chosen)&(chosen>=wlo)&(chosen<=whi)) if reference_selected else np.arange(len(chosen))
    selected_ids=np.unique(d['source_id'][selected])
    if not len(selected_ids):raise ValueError('Empty reference selection')
    with uv._ro(config['afv']) as db:meta={k:json.loads(v) for k,v in db.execute('SELECT key,value FROM metadata')}
    axis=load_umbilicus_frame(config['umbilicus'],meta['root'])
    directions=(d['node_ds'][:,2]-d['node_ds'][:,0])/d['lengths']
    if not np.allclose(abs(directions),1,atol=1e-5):raise ValueError('Invalid aligned direction')
    directions=np.sign(directions)
    centers,widths=uv.radial_initial_geometry(d,config)
    rows=[];curves=[];curve_arcs=[];nodes=[];parents=[]
    geometry=read_afv_polylines(config['afv'],selected_ids)
    for row,xyz in geometry:
        sid=row['id'];first=np.searchsorted(d['source_id'],sid);last=np.searchsorted(d['source_id'],sid,side='right')
        intervals=clipped_intervals(xyz,settings['z_range'])
        if not len(intervals):continue
        arc,travel,phase,radius=uv.aligned_geometry(xyz,d['family'][first],axis,directions[first],config.get('h_convention','ccw'))
        for old in range(first,last):
            if reference_selected and not wlo<=chosen[old]<=whi:continue
            for lo,hi in intervals:
                a=max(lo,d['start'][old]);b=min(hi,d['end'][old])
                if b<=a:continue
                midpoint=(a+b)/2;anchors=np.array([a,midpoint,b]);base=np.interp(a,arc,travel)
                parents.append(old);rows.append((a,b,np.interp(a,arc,phase)%1,np.interp(a,arc,xyz[:,2]),np.interp(midpoint,arc,radius)))
                nodes.append((np.interp(anchors,arc,travel)-base,np.interp(anchors,arc,phase),np.interp(anchors,arc,xyz[:,2])))
                s=np.r_[a,arc[(arc>a)&(arc<b)],b]
                curve_arcs.append(s);curves.append(np.column_stack([np.interp(s,arc,xyz[:,q]) for q in range(3)]))
    if not rows:raise ValueError('Empty clipped subset')
    parents=np.asarray(parents,np.int64);rows=np.asarray(rows);nodes=np.asarray(nodes)
    out={key:d[key][parents].copy() for key in ('source_index','source_id','family','initial_k')}
    out.update(ids=d['ids'],earlier_labels=d['earlier_labels'],start=rows[:,0],end=rows[:,1],phase=rows[:,2],z=rows[:,3],radius=rows[:,4],
               lengths=rows[:,1]-rows[:,0],node_ds=nodes[:,0].astype(np.float32),node_phase=nodes[:,1].astype(np.float32),node_z=nodes[:,2].astype(np.float32),
               width_initial=d['width_initial'],radial_initial_widths=widths,parent_piece=parents)
    out['radial_initial_centers']=reanchored_centers(d,parents,out['start'],out['phase'],directions,centers,widths)
    i,da=remap_endpoints(d['i'],d['ds_a'],d,parents,out['start'],out['end'],directions)
    j,db=remap_endpoints(d['j'],d['ds_b'],d,parents,out['start'],out['end'],directions)
    keep=(i>=0)&(j>=0)
    if not keep.any():raise ValueError('Subset has no internal constraints')
    out.update(i=i[keep],j=j[keep],ds_a=da[keep],ds_b=db[keep])
    for key in ('phase_a','phase_b','phase_delta','distance','sign','parallel','event_ids'):out[key]=d[key][keep]
    out['degree']=np.bincount(np.r_[out['i'],out['j']],minlength=len(parents)).astype(np.float32)
    out['links']=retained_links(d,parents,out['start'],out['end'],directions)
    sample=ref.samples(out,config,signature['sample_spacing'])
    projection_out=dict(sample,chosen_winding=chosen[parents],**{key:out[key] for key in ('source_id','start','end')})
    audit_subset(out,d,projection_out,settings['z_range'],settings.get('winding_range'))
    zlo,zhi=settings['z_range']
    if np.any(sample['xyz'][:,2]<zlo-.01) or np.any(sample['xyz'][:,2]>zhi+.01):raise ValueError('Clipped geometry escaped Z range')
    root.mkdir(parents=True,exist_ok=True)
    np.savez(root/'tensors.npz',**out)
    np.savez(root/'projection.npz',**projection_out)
    projection_signature=(json.loads(Path(config['reference_projection']).with_name('projection_signature.json').read_text())
                          if reference_selected else dict(afv=inputs['afv'],reference_selected=False))
    projection_signature['reference_subset']=signature
    atomic_write(root/'projection_signature.json',projection_signature,durable=False)
    np.savez(root/'clipped_curves.npz',xyz=np.concatenate(curves),arc=np.concatenate(curve_arcs),
             offsets=np.r_[0,np.cumsum([len(c) for c in curves])],source_id=out['source_id'],start=out['start'],end=out['end'],parent_piece=parents)
    stats=dict(signature=signature,tensor_cache=str((root/'tensors.npz').resolve()),winding_count=int(config.get('winding_count',100)),
               h_convention=config.get('h_convention','ccw'),selected_sources_before_z=len(selected_ids),sources=len(np.unique(out['source_id'])),pieces=len(parents),
               events=int(keep.sum()),parent_events=len(keep),continuations=len(out['links']),parent_continuations=len(d['links']),
               length_base=float(out['lengths'].sum()),z_range=[float(sample['xyz'][:,2].min()),float(sample['xyz'][:,2].max())],
               winding_range=[float(chosen[parents].min()),float(chosen[parents].max())] if reference_selected else None,seconds=time.monotonic()-started)
    atomic_write(root/'tensors.json',dict(stats,signature=signature),durable=False)
    stats['outputs']={name:sha256_file(root/name) for name in ('tensors.npz','tensors.json','projection.npz','projection_signature.json','clipped_curves.npz')}
    atomic_write(manifest_path,stats,durable=False)
    print(json.dumps(stats),flush=True);return stats
