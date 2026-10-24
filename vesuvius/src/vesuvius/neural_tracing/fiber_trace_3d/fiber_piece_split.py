"""Deterministic, measured-geometry piece splitting; preserve source/arc identity."""
import numpy as np
from . import fiber_uv_layout as uv


def split_graph(d, config, protected_sources=()):
    target=float(config.get('split_target_length_vx',0))
    if target==0:return d
    minimum=int(config.get('split_min_perpendicular',4))
    if target<=0 or minimum<1:raise ValueError('Invalid piece split settings')
    if len(d['links']):raise ValueError('Split input must be whole fibers, not already split')
    centers,widths=uv.radial_initial_geometry(d,config)
    d=dict(d,radial_initial_centers=centers,radial_initial_widths=widths)
    from vesuvius.neural_tracing.tracers.shared.annotations.afv import read_afv_polylines
    import json
    from .fiber_normal_sign_visualization import load_umbilicus_frame
    with uv._ro(config['afv']) as db:
        meta={k:json.loads(v) for k,v in db.execute('SELECT key,value FROM metadata')}
    axis=load_umbilicus_frame(config['umbilicus'],meta['root'])
    curves={r['id']:xyz for r,xyz in read_afv_polylines(config['afv'],np.unique(d['source_id']))}
    return split_graph_geometry(d,target,minimum,curves,axis,config.get('h_convention','ccw'),protected_sources)


def split_graph_geometry(d,target,minimum,curves,axis,convention='ccw',protected_sources=()):
    protected=set(map(int,protected_sources));parents=[];starts=[];ends=[];nodes=[];phases=[];zs=[];radii=[];links=[]
    mapped=[np.empty(len(d['i']),np.int64),np.empty(len(d['j']),np.int64)]
    offsets=[np.empty(len(d['i'])),np.empty(len(d['j']))]
    for f,source in enumerate(d['source_id']):
        direction=np.sign(d['node_ds'][f,-1])
        if direction not in (-1,1):raise ValueError('Missing resolved arc direction')
        a,b=float(d['start'][f]),float(d['end'][f])
        sides=[np.flatnonzero(d[k]==f) for k in ('i','j')]
        arcs=[a+direction*d[k][rows] for k,rows in zip(('ds_a','ds_b'),sides)]
        perp=np.concatenate([x[~d['parallel'][rows]] for x,rows in zip(arcs,sides)])
        bounds=(np.array([a,b]) if int(source) in protected else
                a+uv.split_bounds(b-a,perp-a,target,minimum))
        xyz=curves[int(source)];arc,_,angle,radius=uv.fiber_geometry(xyz,int(d['family'][f]),axis)
        angle*=1 if convention=='ccw' else -1
        first=len(parents)
        for lo,hi in zip(bounds[:-1],bounds[1:]):
            idx=len(parents);samples=np.array([lo,(lo+hi)/2,hi])
            parents.append(f);starts.append(lo);ends.append(hi)
            nodes.append((direction*(samples-lo),np.interp(samples,arc,angle),np.interp(samples,arc,xyz[:,2])))
            phases.append(np.interp(lo,arc,angle)%1);zs.append(np.interp(lo,arc,xyz[:,2]));radii.append(np.interp((lo+hi)/2,arc,radius))
            if idx>first:links.append((idx-1,idx,direction*(lo-starts[idx-1]),0.))
        for side,(rows,x) in enumerate(zip(sides,arcs)):
            local=np.searchsorted(bounds[1:-1],x,side='right')
            mapped[side][rows]=first+local;offsets[side][rows]=direction*(x-bounds[local])
    parent=np.asarray(parents);out=dict(d)
    for key in ('source_id','source_index','family','initial_k','radial_initial_centers'):
        if key in d:out[key]=d[key][parent]
    nodes=np.asarray(nodes)
    out.update(start=np.asarray(starts),end=np.asarray(ends),lengths=np.asarray(ends)-starts,
        phase=np.asarray(phases),z=np.asarray(zs),radius=np.asarray(radii),
        node_ds=nodes[:,0],node_phase=nodes[:,1],node_z=nodes[:,2],
        i=mapped[0],j=mapped[1],ds_a=offsets[0],ds_b=offsets[1],
        links=np.asarray(links,dtype=float).reshape(-1,4),split_parent=parent)
    out['degree']=np.bincount(np.r_[out['i'],out['j']],minlength=len(parent)).astype(np.float32)
    out['split_origin_phase']=d['phase'][parent]
    out['split_origin_z']=d['z'][parent]
    out['split_anchor_ds']=(out['start']-d['start'][parent])*np.sign(d['node_ds'][parent,-1])
    return out


def initial_state(d,centers,widths):
    """Split the original radial initial curve, not its phase modulo one.

    This is initialization only; pieces subsequently optimize independently.
    Avoid injecting an artificial one-turn jump at a split crossing the seam.
    """
    state=np.column_stack((centers,d['phase'],d['z'],np.full(len(centers),4.)))
    if 'split_anchor_ds' not in d:return state
    import torch
    from .fiber_soft_winding import points
    state[:,1]=d['split_origin_phase'];state[:,2]=d['split_origin_z']
    with torch.no_grad():
        _,phase,v=points(torch.as_tensor(state),torch.as_tensor(d['family']),
            torch.as_tensor(d['split_anchor_ds']),torch.as_tensor(widths,dtype=torch.float64))
    state[:,1]=phase.numpy();state[:,2]=v.numpy()
    return state


def rejection_costs(d,config):
    n=len(d['family']);cost=np.full(n,float(config.get('rejection_cost',20.)))
    for kind,mask in [('perpendicular',~d['parallel']),('parallel',d['parallel'])]:
        weight=float(config.get('rejection_'+kind+'_cost',0.))
        if not np.isfinite(weight) or weight<0:raise ValueError('Invalid rejection coefficient')
        cost+=weight*np.bincount(np.r_[d['i'][mask],d['j'][mask]],minlength=n)
    weight=float(config.get('rejection_continuation_cost',0.))
    if not np.isfinite(weight) or weight<0:raise ValueError('Invalid continuation rejection coefficient')
    cost+=weight*np.bincount(d['links'][:,:2].astype(int).ravel(),minlength=n)
    return cost
