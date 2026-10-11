"""Probabilistic winding / physical U,V layout of fixed-family fiber pieces.

Opt-in experimental stage. Input constraint records and earlier solutions remain
immutable. All winding hypotheses participate; constraint points have no state.
"""
import argparse
import json
import math
from pathlib import Path
import sqlite3
import time

import numpy as np
from lasagna.inference_provenance import atomic_write, sha256_file
from .fiber_normal_sign import launch_tensor_solve
from .fiber_normal_convention import inward_solution


def split_bounds(length, perpendicular_arcs, minimum, min_events=3):
    """Greedy earliest eligible midpoint, keeping an eligible remainder.

    Repeated event IDs are counted once by the caller. Coincident arcs are real
    distinct observations; never split through them. Original short fibers stay.
    """
    arcs=np.sort(np.asarray(perpendicular_arcs,dtype=float))
    if minimum<=0 or min_events<1:raise ValueError('Invalid split limits')
    if np.any(arcs<0) or np.any(arcs>length+1e-5):raise ValueError('Arc out of bounds')
    cuts=[0.];start=0
    for n in range(min_events,len(arcs)-min_events+1):
        if n-start<min_events or arcs[n-1]==arcs[n]:continue
        cut=(arcs[n-1]+arcs[n])/2
        if cut-cuts[-1]>=minimum and length-cut>=minimum:
            cuts.append(cut);start=n
    return np.asarray(cuts+[length])


def fiber_geometry(points, family, umbilicus):
    """Arc maps independent of original point order, no curve repair/densification."""
    xyz=np.asarray(points);delta=np.diff(xyz,axis=0)
    ds=np.linalg.norm(delta,axis=1);arc=np.r_[0.,np.cumsum(ds)]
    order=np.argsort(umbilicus[:,2]);axis=umbilicus[order]
    cx=np.interp(xyz[:,2],axis[:,2],axis[:,0]);cy=np.interp(xyz[:,2],axis[:,2],axis[:,1])
    radius=np.hypot(xyz[:,0]-cx,xyz[:,1]-cy)
    angle=np.unwrap(np.arctan2(xyz[:,1]-cy,xyz[:,0]-cx))/(2*np.pi)
    # Integrate signed physical arc, not angular magnitude. This can reverse in
    # a fold; it is not a claim that all fibers are monotone in global angle/z.
    mid=(xyz[1:]+xyz[:-1])/2
    rx=mid[:,0]-np.interp(mid[:,2],axis[:,2],axis[:,0])
    ry=mid[:,1]-np.interp(mid[:,2],axis[:,2],axis[:,1])
    projection=(-ry*delta[:,0]+rx*delta[:,1]) if family==0 else delta[:,2]
    travel=np.r_[0.,np.cumsum(ds*np.sign(projection))]
    return arc,travel,angle,radius


def signed_order(connector, inward_a, inward_b, cosine_min=.1):
    length=np.linalg.norm(connector,axis=1)
    unit=connector/np.maximum(length[:,None],1e-12)
    a=-np.sum(unit*inward_a,axis=1);b=-np.sum(unit*inward_b,axis=1)
    known=(a*b>0)&(np.minimum(abs(a),abs(b))>=cosine_min)&(length>0)
    return np.where(known,np.sign(a),0).astype(np.float32)


def _ro(path):
    return sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)


def anchored(config):
    return config.get('layout_mode', 'legacy') == 'shared_anchor'


def aligned_geometry(points,family,umbilicus,direction,h_convention):
    if direction not in (-1,1) or h_convention not in ('ccw','cw'):
        raise ValueError('Invalid aligned geometry convention')
    arc,_,angle,radius=fiber_geometry(points,family,umbilicus)
    return arc,arc*direction,angle*(1 if h_convention=='ccw' else -1),radius


def direction_inputs(config, signature, ids, labels):
    """Validate immutable previous stages, including V's inherited H solution."""
    outputs=[]
    for key in ('h_direction_solution','v_direction_solution'):
        path=Path(config[key])
        meta=json.loads((path.parent/'summary.json').read_text())['input']['signature']
        for source in ('database','afv','features','stage1','hv_solution','umbilicus'):
            if meta[source]!=signature[source]:raise ValueError('Direction provenance mismatch: '+source)
        with np.load(path) as src:outputs.append({k:src[k] for k in src.files})
    h,v=outputs
    for solution in outputs:
        if not np.array_equal(ids,solution['ids']):raise ValueError('Direction IDs mismatch')
    if np.any((h['labels']!=labels)&~((labels==0)&(h['labels']==2))):
        raise ValueError('H direction changed frozen labels')
    if np.any((v['labels']!=h['labels'])&~((h['labels']==1)&(v['labels']==2))):
        raise ValueError('V direction changed frozen labels')
    for name in ('ccw','cw'):
        if not np.array_equal(v['inherited_h_direction_sign_'+name],h['direction_sign_'+name]):
            raise ValueError('Inherited H signs mismatch')
    vmeta=json.loads((Path(config['v_direction_solution']).parent/'summary.json').read_text())['input']['signature']
    if vmeta['inherited_h_solution']!=signature['h_direction_solution']:raise ValueError('Inherited H hash mismatch')
    convention=config.get('h_convention','ccw')
    if convention not in ('ccw','cw'):raise ValueError('Invalid H convention')
    result=np.where(v['labels']==0,h['direction_sign_'+convention],v['direction_sign_up'])
    if not np.isin(result[v['labels']!=2],[-1,1]).all():raise ValueError('Missing retained direction')
    return v['labels'].copy(),result


def prepare(config):
    from .fiber_hv_stage import verify_feature_source
    from .fiber_normal_sign_visualization import load_umbilicus_frame
    from vesuvius.neural_tracing.fiber_follow.data.afv import read_afv_polylines
    started=time.monotonic();cache=Path(config['tensor_cache']);stamp=cache.with_suffix('.json')
    signature={key:sha256_file(config[key]) for key in ('database','features','stage1','hv_solution','afv','umbilicus')}
    signature.update(version=2,minimum_length_mm=config.get('minimum_length_mm',10.),
                     sign_cosine_min=config.get('sign_cosine_min',.1),pilot_fibers=config.get('pilot_fibers',0))
    if anchored(config):
        signature.update(version=3,layout_mode='shared_anchor',h_convention=config.get('h_convention','ccw'))
        signature.update({key:sha256_file(config[key]) for key in ('h_direction_solution','v_direction_solution')})
    if cache.exists() and stamp.exists() and json.loads(stamp.read_text())['signature']==signature:return
    feature=json.loads(Path(config['features']).read_text());verify_feature_source(config['database'],feature)
    hv_summary=json.loads((Path(config['hv_solution']).parent/'summary.json').read_text())
    if (hv_summary['input']['signature']['database']!=signature['database'] or
            hv_summary['input']['signature']['stage1']!=signature['stage1']):raise ValueError('H/V input mismatch')
    with np.load(config['stage1']) as src:normal=inward_solution({k:src[k] for k in src.files})
    with np.load(config['hv_solution']) as h:ids=h['ids'].copy();labels=h['labels'].copy()
    if not np.array_equal(ids,normal['ids']):raise ValueError('Normal/HV IDs mismatch')
    if not np.all(np.diff(ids)>0) or not np.isin(labels,[0,1,2]).all():raise ValueError('Invalid IDs/labels')
    if np.any(normal['broken']&(labels!=2)):raise ValueError('Stage1 exclusions not preserved')
    if anchored(config):labels,directions=direction_inputs(config,signature,ids,labels)
    if signature['afv']!=feature['afv_sha256'] or signature['umbilicus']!=feature['umbilicus_sha256']:
        raise ValueError('Geometry/axial input mismatch')
    with _ro(config['afv']) as db:meta={k:json.loads(v) for k,v in db.execute('SELECT key,value FROM metadata')}
    axis=load_umbilicus_frame(config['umbilicus'],meta['root'])
    kept=labels!=2
    pilot=int(config.get('pilot_fibers',0))
    if pilot:
        # Deterministic central-z induced subgraph, used ONLY for timing pilot.
        with _ro(config['afv']) as db:rows=np.asarray(db.execute('SELECT id,(min_z+max_z)/2 FROM fibers ORDER BY id').fetchall())
        if not np.array_equal(rows[:,0],ids):raise ValueError('Pilot AFV IDs mismatch')
        center=np.median(rows[kept,1]);select=np.flatnonzero(kept)
        select=select[np.argsort(abs(rows[select,1]-center),kind='stable')[:pilot]]
        kept[:]=False;kept[select]=True
    print(f'UV prepare: {kept.sum():,} retained fibers; reading events',flush=True)
    columns='id,fiber_a,fiber_b,arc_a,arc_b,ax,ay,az,bx,by,bz,parallel_score,perpendicular_score,winding_distance,parallel_winding_distance,normal_a,normal_b'
    batches=[];original=0
    with _ro(config['database']) as db:
        cursor=db.execute('SELECT '+columns+' FROM constraints ORDER BY id')
        while rows:=cursor.fetchmany(65536):
            original+=len(rows)
            numeric=np.asarray([r[:15] for r in rows],dtype=float)
            ends=np.searchsorted(ids,numeric[:,1:3].astype(np.int64))
            if np.any(ends>=len(ids)) or not np.array_equal(ids[ends],numeric[:,1:3]):raise ValueError('Unknown constraint fiber')
            keep=kept[ends].all(1)
            chosen=np.flatnonzero(keep);n=numeric[keep]
            def normals(column):
                records=[json.loads(rows[i][column]) for i in chosen]
                return np.asarray([r['xyz'] if r['valid'] else [np.nan]*3 for r in records],float).reshape(-1,3)
            na=normals(15);nb=normals(16)
            na*=normal['normal_sign'][ends[keep,0],None];nb*=normal['normal_sign'][ends[keep,1],None]
            sign=signed_order(n[:,8:11]-n[:,5:8],na,nb,config.get('sign_cosine_min',.1))
            distance=np.where(n[:,11]>n[:,12],n[:,14],n[:,13])
            if not np.isfinite(distance).all() or np.any(distance<0):raise ValueError('Invalid distance measurement')
            batches.append(np.column_stack([n[:,:11],n[:,11]>n[:,12],distance,sign]))
    events=np.concatenate(batches);del batches
    if not len(events):raise ValueError('Empty induced graph')
    # One sorted adjacency index for both endpoint columns, no per-fiber E scan.
    ecount=len(events);end_ids=events[:,1:3].astype(np.int64).T.reshape(-1)
    endpoint_arcs=events[:,3:5].T.reshape(-1);order=np.argsort(end_ids,kind='stable')
    sorted_ids=end_ids[order];positions=np.searchsorted(sorted_ids,ids)
    last=np.searchsorted(sorted_ids,ids,side='right')
    owner=np.empty(2*ecount,np.int64);offset=np.empty(2*ecount,np.float32)
    phase=np.empty(2*ecount,np.float32);zpos=np.empty(2*ecount,np.float32)
    pieces=[];links=[];nodes=[]
    minimum=signature['minimum_length_mm']/meta['root']['voxel_size_mm']
    geometry=read_afv_polylines(config['afv'],ids[kept]);print('UV prepare: events loaded; deriving pieces',flush=True)
    for fi,(row,xyz) in enumerate(geometry):
        idx=np.searchsorted(ids,row['id']);family=labels[idx]
        if anchored(config):
            arc,travel,angle,radius=aligned_geometry(xyz,family,axis,directions[idx],config.get('h_convention','ccw'))
        else:arc,travel,angle,radius=fiber_geometry(xyz,family,axis)
        ep=order[positions[idx]:last[idx]];ea=endpoint_arcs[ep]
        if not np.isfinite(ea).all() or np.any(ea<0) or np.any(ea>arc[-1]+1e-5):
            raise ValueError(f'Constraint arc outside original fiber {row["id"]}')
        perp=ea[events[ep%ecount,11]==0]
        bounds=split_bounds(arc[-1],perp,minimum);first=len(pieces)
        middle=(bounds[:-1]+bounds[1:])/2
        anchors=bounds[:-1] if anchored(config) else middle
        tr=np.interp(anchors,arc,travel);ph=np.interp(anchors,arc,angle)
        for p in range(len(middle)):
            a,b=bounds[p:p+2];m=middle[p]
            pieces.append((idx,row['id'],a,b,family,tr[p],ph[p],np.interp(anchors[p],arc,xyz[:,2]),np.interp(m,arc,radius)))
            # Uniform piece regularizer evaluates both ends and midpoint.
            nodes.append((np.interp([a,m,b],arc,travel)-tr[p],np.interp([a,m,b],arc,angle),np.interp([a,m,b],arc,xyz[:,2])))
            if p:
                cut=a;tc=np.interp(cut,arc,travel)
                links.append((first+p-1,first+p,tc-tr[p-1],tc-tr[p]))
        local=np.searchsorted(bounds[1:-1],ea,side='right')
        owner[ep]=first+local;offset[ep]=np.interp(ea,arc,travel)-tr[local]
        phase[ep]=np.interp(ea,arc,angle);zpos[ep]=np.interp(ea,arc,xyz[:,2])
        if fi and fi%100000==0:print(f'UV prepare: {fi:,}/{len(geometry):,} fibers',flush=True)
    p=np.asarray(pieces);nodes=np.asarray(nodes);i,j=owner.reshape(2,-1)
    ds_a,ds_b=offset.reshape(2,-1);phi_a,phi_b=phase.reshape(2,-1)
    phase_delta=(phi_b-phi_a+.5)%1-.5
    degree=np.bincount(owner,minlength=len(p)).astype(np.float32)
    lengths=p[:,3]-p[:,2];r=p[:,8]
    # Geometric initialization only; no winding labels/radii are hard evidence.
    lo,hi=np.quantile(r,[.001,.999]);initial_k=np.clip(5+85*(r-lo)/max(hi-lo,1),0,99)
    radii=np.interp(np.arange(100),np.linspace(5,90,100),np.linspace(lo,hi,100))
    width_initial=np.maximum(2*np.pi*radii,100.)
    arrays=dict(ids=ids,earlier_labels=labels,source_index=p[:,0].astype(np.int64),source_id=p[:,1].astype(np.int64),
        start=p[:,2],end=p[:,3],family=p[:,4].astype(np.int64),phase=p[:,6]%1,z=p[:,7],radius=r,
        lengths=lengths,degree=degree,initial_k=initial_k.astype(np.float32),width_initial=width_initial.astype(np.float32),
        i=i,j=j,ds_a=ds_a,ds_b=ds_b,phase_a=phi_a,phase_b=phi_b,phase_delta=phase_delta,
        distance=events[:,12].astype(np.float32),sign=events[:,13].astype(np.float32),
        parallel=events[:,11].astype(bool),event_ids=events[:,0].astype(np.int64),
        links=np.asarray(links,dtype=np.float64).reshape(-1,4),node_ds=nodes[:,0].astype(np.float32),
        node_phase=nodes[:,1].astype(np.float32),node_z=nodes[:,2].astype(np.float32))
    cache.parent.mkdir(parents=True,exist_ok=True);temporary=cache.with_suffix('.partial.npz')
    np.savez(temporary,**arrays);temporary.replace(cache)
    stats=dict(signature=signature,original_fibers=len(ids),earlier_rejected=int((labels==2).sum()),
        active_fibers=int(kept.sum()),pieces=len(p),continuations=len(links),original_events=original,events=ecount,
        known_sign_percent=float(100*np.mean(events[:,13]!=0)),parallel_events=int(events[:,11].sum()),
        minimum_piece_base=minimum,additional_pieces=int(len(p)-len(geometry)),
        split_fibers=int(np.sum(np.bincount(p[:,0].astype(int),minlength=len(ids))>1)),seconds=time.monotonic()-started,
        width_initial_range=[float(width_initial.min()),float(width_initial.max())],
        distance_quantiles=np.quantile(events[:,12],[0,.5,.95,.99,1]).tolist())
    atomic_write(stamp,stats,durable=False);print('UV prepared '+json.dumps(stats),flush=True)


def width_boundaries(raw, minimum=100.):
    import torch
    widths=minimum+torch.nn.functional.softplus(raw)
    return widths,torch.cat((widths.new_zeros(1),widths.cumsum(0)))


def u_to_q(u, widths, boundaries):
    import torch
    bucket=torch.searchsorted(boundaries.detach(),u.detach().contiguous(),right=True)-1
    bucket=bucket.clamp(0,len(widths)-1)
    return bucket+(u-boundaries[bucket])/widths[bucket]


def q_to_u(q, widths, boundaries):
    """Continuous cumulative-width map; extrapolate end widths, never wrap."""
    bucket=q.detach().floor().long().clamp(0,len(widths)-1)
    return boundaries[bucket]+(q-bucket)*widths[bucket]


def anchor_offsets(state,k):
    du=state[:,k+3:2*k+3];dv=state[:,2*k+3:3*k+3]
    return du-du.mean(1,keepdim=True),dv-dv.mean(1,keepdim=True)


def offset_regularization(state,k,config):
    import torch
    du,dv=anchor_offsets(state,k)
    return (config.get('offset_soft_weight',.001)*(du.square().mean()+dv.square().mean())+
            config.get('offset_limit_weight',10000.)*torch.relu(du.abs()-.25).square().mean())


def coordinates(state, phase, z, family, ds, widths, boundaries):
    """Only piece state has parameters. All base points are derived here."""
    k=len(widths)
    if state.shape[1]==3*k+3:
        import torch
        du,dv=anchor_offsets(state,k)
        shared_phase=phase+state[:,k+1]
        q=torch.arange(k,device=state.device)[None,:]+shared_phase[:,None]
        # Offsets in local turn units; mapping remains continuous at every seam.
        u=q_to_u(q+du,widths,boundaries)+(family==0)[:,None]*ds[:,None]
        v=z[:,None]+1000*(state[:,k+2,None]+dv)+(family==1)[:,None]*ds[:,None]
        return u,v,u_to_q(u,widths,boundaries)
    frac=phase[:,None]+.25*state[:,k+1:2*k+1].tanh()
    u=boundaries[:-1]+widths*frac+(family==0)[:,None]*ds[:,None]
    v=z[:,None]+1000*state[:,2*k+1:]+(family==1)[:,None]*ds[:,None]
    return u,v,u_to_q(u,widths,boundaries)


def make_distance_table(maximum, sigma, step=.0625, tau=.25):
    """Discrete half-grid Gaussian, truncated+renormalized at zero, log mixture.

    Tabulation (bilinear) is an explicit numerical approximation, tested against
    direct logsumexp. Neither sign cost nor hypothesis set is approximated.
    """
    from scipy.special import logsumexp
    mu_grid=np.arange(0,np.ceil(maximum/step)+2)*step
    x_grid=np.arange(0,104/step+1)*step
    support=np.arange(0,max(104.,maximum+8*sigma)+.5,.5)
    table=np.empty((len(mu_grid),len(x_grid)),np.float32)
    for n,mu in enumerate(mu_grid):
        weights=-.5*((support-mu)/sigma)**2;weights-=logsumexp(weights)
        table[n]=-logsumexp(weights[:,None]-.5*((support[:,None]-x_grid)/tau)**2,axis=0)
    return table


def table_cost(delta, mu, table, step=.0625):
    """Bilinear table lookup; explicit linear extrapolation beyond final x bin."""
    import torch
    m=(mu/step).clamp(0,table.shape[0]-1.00001);mi=m.long();mf=m-mi
    x=delta.abs()/step;xi=x.long().clamp(0,table.shape[1]-2);xf=x-xi
    low=table[mi[:,None,None],xi]*(1-xf)+table[mi[:,None,None],xi+1]*xf
    high=table[mi[:,None,None]+1,xi]*(1-xf)+table[mi[:,None,None]+1,xi+1]*xf
    return low*(1-mf[:,None,None])+high*mf[:,None,None]


def rejection_weights(degree,mean_degree,mode='fixed'):
    import torch
    if mode=='fixed':return torch.ones_like(degree)
    if mode=='degree':return .5+.5*degree/mean_degree.clamp_min(1.)
    raise ValueError('rejection_weighting must be fixed or degree')


def validate_joint_rejection(config):
    if float(config.get('rejection_warmup',0))!=0:
        raise ValueError('Joint optimization no longer permits rejection_warmup; broken logits train from update0')
    if config.get('rejection_weighting','fixed') not in ('fixed','degree'):
        raise ValueError('rejection_weighting must be fixed or degree')
    if config.get('reference_initialization') and config.get('warm_start'):
        raise ValueError('Reference initialization and checkpoint warm start are mutually exclusive')
    mode=config.get('update_mode','minibatch')
    if mode not in ('minibatch','global_accumulated'):
        raise ValueError('Unknown update_mode')
    if mode=='global_accumulated':
        if config.get('traversal','full_pass')!='full_pass':
            raise ValueError('Global accumulation requires full_pass traversal')
        if config.get('entropy_ramp',True):
            raise ValueError('Global accumulation requires entropy_ramp=false for a fixed pass objective')


def calibrated_sign_charge(degree_a,degree_b,mean_degree,event_count,piece_count,rejection_cost,multiplier,
                           rejection_weighting='degree'):
    """Cheaper-endpoint dominance under the actual event/node normalization."""
    import torch
    ca=rejection_weights(degree_a,mean_degree,rejection_weighting)
    cb=rejection_weights(degree_b,mean_degree,rejection_weighting)
    return multiplier*(event_count/piece_count)*rejection_cost*torch.minimum(ca,cb)


def fixed_sign_terms(pa,pb,ordered,charge,selected_ordered=None):
    """A diffuse posterior must not hide a wrong selected retained assignment.

    Exact indicators have no coordinate derivative; the separate magnitude
    hinge supplies that gradient. No stale audited edge is forced wrong.
    """
    import torch
    mass=pa[:,:,None]*pb[:,None,:]
    expected=(mass*(ordered<0)).sum((1,2))
    if selected_ordered is None:
        row=torch.arange(len(pa),device=pa.device)
        selected_ordered=ordered[row,pa.argmax(1),pb.argmax(1)]
    selected=pa.sum(1)*pb.sum(1)*(selected_ordered<0)
    return charge*torch.maximum(expected,selected)


def vertical_layout_prior(v,z,config):
    """Weak unary layout calibration, not a pairwise displacement measurement.

    Returns per-hypothesis costs; callers apply retention probabilities and
    uniform piece/point averaging. No slope or monotonicity condition.
    """
    scale=float(config.get('v_prior_scale',1000.))
    weight=float(config.get('v_prior_weight',.01))
    if not math.isfinite(scale) or scale<=0 or not math.isfinite(weight) or weight<0:
        raise ValueError('Invalid vertical layout prior scale/weight')
    return weight*((v-z)/scale).square()


def pair_geometry(qa,qb,va,vb,phase_delta,family_delta,sign,config):
    """Shared pair geometry; used by likelihood and expected-distance objectives."""
    import torch
    delta=qb[:,None,:]-qa[:,:,None]-phase_delta[:,None,None]+.5*family_delta[:,None,None]
    # Finite optimization surrogate for truncated support: never reflect a known
    # wrong sign into a good distance. Boundary cost + uncapped corrective hinge.
    ordered=sign[:,None,None]*delta
    wrong=torch.relu(-ordered)*4
    # Angular anchoring is an individual prior, not another pairwise penalty.
    # Keep observed phase_delta above: it corrects the winding separation.
    vertical=((vb[:,None,:]-va[:,:,None])/config.get('metric_scale',128.)).square().clamp(max=100.)
    metric=config.get('metric_weight',.01)*vertical
    return delta,ordered,wrong,metric


def pair_cost_components(qa,qb,va,vb,phase_delta,family_delta,sign,mu,table,config):
    """Shared K*K geometric costs, independent of assignment probabilities."""
    import torch
    delta,ordered,wrong,metric=pair_geometry(qa,qb,va,vb,phase_delta,family_delta,sign,config)
    distance=torch.where((sign!=0)[:,None,None],ordered.clamp_min(0),delta.abs())
    soft=table_cost(distance,mu,table)
    soft_cost=config.get('distance_weight',1.)*soft+metric
    cost=config.get('sign_weight',10.)*wrong+soft_cost
    return cost,ordered,soft_cost,wrong


def prefix_sign_terms(pa,pb,qa,qb,phase_delta,family_delta,sign):
    """Exact expected sign hinge without a K*K tensor (piecewise gradients)."""
    import torch
    origin=qb[:,:1]
    x=sign[:,None]*(qa-origin+phase_delta[:,None]-.5*family_delta[:,None])
    y=sign[:,None]*(qb-origin)
    y,order=y.sort(dim=1);b=pb.gather(1,order)
    mass=torch.cat((b.new_zeros((len(b),1)),b.cumsum(1)),1)
    moment=torch.cat((b.new_zeros((len(b),1)),(b*y).cumsum(1)),1)
    count=torch.searchsorted(y.contiguous(),x.contiguous(),right=False)
    return 4*(pa*(x*mass.gather(1,count)-moment.gather(1,count))).sum(1)


def local_winding_window(p,q,v,radius):
    """Conditional local distribution, preserving unconditional retained mass."""
    import torch
    k=p.shape[1]
    ids=p.argmax(1)[:,None]+torch.arange(-radius,radius+1,device=p.device)[None,:]
    valid=(ids>=0)&(ids<k);ids=ids.clamp(0,k-1)
    local=p.gather(1,ids)*valid
    total=p.sum(1,keepdim=True);selected=local.sum(1,keepdim=True)
    local=local/selected.clamp_min(1e-30)*total
    return local,q.gather(1,ids),v.gather(1,ids),selected/total.clamp_min(1e-30)


def pair_objective(pa,pb,qa,qb,va,vb,phase_delta,family_delta,sign,mu,table,config,
                   charge=None,event_weights=None,selected_ordered=None):
    """Legacy posterior objective, optionally with selected-sign safeguard."""
    import torch
    mode=config.get('distance_objective','likelihood')
    radius=config.get('local_window_radius')
    if radius is not None:
        if not isinstance(radius,int) or radius<0:raise ValueError('Invalid local_window_radius')
        if mode!='squared' or charge is not None:
            raise ValueError('Local windows require squared objective without fixed sign charge')
        if radius<pa.shape[1]-1:
            sign_terms=prefix_sign_terms(pa,pb,qa,qb,phase_delta,family_delta,sign)
            a,x,u,_=local_winding_window(pa,qa,va,radius)
            b,y,v,_=local_winding_window(pb,qb,vb,radius)
            local_config=dict(config);local_config.pop('local_window_radius');local_config['sign_weight']=0.
            # Evaluate shared squared/V formula; exact global hinge is separate.
            soft=pair_objective(a,b,x,y,u,v,phase_delta,family_delta,sign,mu,table,
                                local_config,event_weights=event_weights)
            if event_weights is not None:sign_terms=sign_terms*event_weights
            return soft+config.get('sign_weight',10.)*sign_terms.mean()
    if mode not in ('likelihood','squared','huber','huber_likelihood'):
        raise ValueError('Invalid distance_objective')
    if mode!='likelihood':
        if not anchored(config) or config.get('pair_mode','likelihood')!='likelihood':
            raise ValueError('Distance alternatives require shared-anchor likelihood layout')
        sigma=float(config.get('distance_sigma',.5))
        if not math.isfinite(sigma) or sigma<=0:raise ValueError('distance_sigma must be finite and positive')
        delta,ordered,wrong,metric=pair_geometry(qa,qb,va,vb,phase_delta,family_delta,sign,config)
        distance=torch.where((sign!=0)[:,None,None],ordered,delta.abs())
        residual=(distance-mu[:,None,None])/sigma
        penalty=(.5*residual.square() if mode=='squared' else
                 torch.where(residual.abs()<=1,.5*residual.square(),residual.abs()-.5))
        mass=pa[:,:,None]*pb[:,None,:]
        terms=(mass*(config.get('distance_weight',1.)*penalty+
                     config.get('sign_weight',10.)*wrong)).sum((1,2))
        if mode=='huber_likelihood':
            soft=table_cost(torch.where((sign!=0)[:,None,None],ordered.clamp_min(0),delta.abs()),mu,table)
            terms=terms+compatibility_terms(pa,pb,config.get('distance_weight',1.)*soft+metric,
                                            config.get('temperature',1.))
        else:terms=terms+(mass*metric).sum((1,2))
        if charge is not None:terms=terms+fixed_sign_terms(pa,pb,ordered,charge,selected_ordered)
        if event_weights is not None:terms=terms*event_weights
        return terms.mean()
    cost,ordered,soft_cost,wrong=pair_cost_components(qa,qb,va,vb,phase_delta,family_delta,
                                                   sign,mu,table,config)
    mass=pa[:,:,None]*pb[:,None,:]
    if anchored(config) and config.get('pair_mode','likelihood')=='likelihood':
        # Likelihood only soft evidence; expected sign cost cannot hide behind
        # one compatible pair. Retention gating is outside conditional mixture.
        terms=compatibility_terms(pa,pb,soft_cost,config.get('temperature',1.))+(
            mass*config.get('sign_weight',10.)*wrong).sum((1,2))
    else:terms=(mass*cost).sum((1,2))
    if charge is not None:terms=terms+fixed_sign_terms(pa,pb,ordered,charge,selected_ordered)
    if event_weights is not None:terms=terms*event_weights
    return terms.mean()


def compatibility_terms(pa,pb,cost,temperature=1.):
    import torch
    if temperature<=0:raise ValueError('Temperature must be positive')
    ma=pa.sum(1);mb=pb.sum(1)
    a=pa/ma[:,None].clamp_min(1e-30);b=pb/mb[:,None].clamp_min(1e-30)
    logmass=a.clamp_min(1e-30).log()[:,:,None]+b.clamp_min(1e-30).log()[:,None,:]
    return ma*mb*(-temperature*torch.logsumexp((logmass-cost/temperature).flatten(1),1))


def pair_evaluator(config):
    """Select once per solve; explicit compilation never silently falls back."""
    backend=config.get('pair_backend','eager')
    if backend=='eager':return pair_objective
    if backend!='compiled':raise ValueError('pair_backend must be eager or compiled')
    import torch
    return torch.compile(pair_objective,fullgraph=True,dynamic=False)


def compatibility_loss(pa,pb,cost,temperature=1.):
    return compatibility_terms(pa,pb,cost,temperature).mean()


def proposal_weights(in_active,event_count,active_count,fraction):
    if not 0<=fraction<1:raise ValueError('Active fraction must preserve uniform support')
    if active_count==0:return in_active.float()*0+1
    return 1/((1-fraction)+fraction*(event_count/active_count)*in_active)


def draw_events(event_count,batch,active,mask,fraction,device):
    import torch
    edges=torch.randint(event_count,(batch,),device=device)
    if len(active) and fraction:
        choose=torch.rand(batch,device=device)<fraction
        candidates=active[torch.randint(len(active),(batch,),device=device)]
        edges=torch.where(choose,candidates,edges)
    return edges,proposal_weights(mask[edges],event_count,len(active),fraction)


def selected_ordering(sa,sb,phase_a,phase_b,z_a,z_b,family_a,family_b,ds_a,ds_b,widths,phase_delta,sign):
    """Fresh double-precision selected coordinates; no stale violation forcing."""
    import torch
    w=widths.double();bounds=torch.cat((w.new_zeros(1),w.cumsum(0)))
    qa=coordinates(sa.double(),phase_a.double(),z_a.double(),family_a,ds_a.double(),w,bounds)[2]
    qb=coordinates(sb.double(),phase_b.double(),z_b.double(),family_b,ds_b.double(),w,bounds)[2]
    a=qa.gather(1,sa[:,:len(w)].argmax(1,keepdim=True)).squeeze(1)
    b=qb.gather(1,sb[:,:len(w)].argmax(1,keepdim=True)).squeeze(1)
    return sign.double()*(b-a-phase_delta.double()+.5*(family_b-family_a))


def continuation_likelihood(pa,pb,ua,ub,va,vb,scale,temperature=1.):
    cost=((ua[:,:,None]-ub[:,None,:])/scale).square()+((va[:,:,None]-vb[:,None,:])/scale).square()
    return compatibility_loss(pa,pb,cost,temperature)


def continuation_cost(pa,pb,ua,ub,va,vb,scale):
    """Exact quadratic pair expectation via moments, including rejection gating."""
    def moments(p,x):return p.sum(1),(p*x).sum(1),(p*x*x).sum(1)
    result=ua.new_zeros(len(pa))
    for a,b in ((ua/scale,ub/scale),(va/scale,vb/scale)):
        origin=a.gather(1,pa.argmax(1,keepdim=True)).detach()
        a=a-origin;b=b-origin
        ma,aa,aa2=moments(pa,a);mb,bb,bb2=moments(pb,b)
        result+=aa2*mb+bb2*ma-2*aa*bb
    return result.clamp_min(0).mean()


def _tensor_data(d,device):
    import torch
    keys=['family','phase','z','degree','node_ds','node_phase','node_z','i','j','ds_a','ds_b',
          'phase_a','phase_b','phase_delta','distance','sign','parallel','links']
    return {k:torch.as_tensor(d[k],device=device,dtype=(torch.long if k in ('family','i','j') else torch.float32)) for k in keys}


def hard_geometry(d,state,widths,config):
    """Shared canonical readout for full audits and active-set selection."""
    from scipy.special import softmax
    if config.get('broken_threshold') is not None:state=state.astype(np.float64,copy=False)
    k=len(widths);p=softmax(state[:,:k+1],axis=1);labels=p.argmax(1);keep=labels<k
    if config.get('broken_threshold') is not None:
        threshold=config['broken_threshold']
        if not 0<threshold<1:raise ValueError('Invalid broken threshold')
        keep=p[:,k]<=threshold
        labels=np.where(keep,state[:,:k].argmax(1),k)
    pick=np.minimum(labels,k-1)
    if config.get('broken_threshold') is not None:pick=state[:,:k].argmax(1)
    frac=d['phase']+.25*np.tanh(state[np.arange(len(state)),k+1+pick])
    b=np.r_[0,np.cumsum(widths,dtype=np.float64)]
    u=b[pick]+widths[pick]*frac;v=d['z']+1000*state[np.arange(len(state)),2*k+1+pick]
    if anchored(config):
        du=state[:,k+3:2*k+3];du=du-du.mean(1,keepdims=True)
        dv=state[:,2*k+3:];dv=dv-dv.mean(1,keepdims=True)
        phase=d['phase']+state[:,k+1]
        q=pick+phase+du[np.arange(len(state)),pick]
        bucket=np.floor(q).astype(int).clip(0,k-1)
        u=b[bucket]+widths[bucket]*(q-bucket)
        v=d['z']+1000*(state[:,k+2]+dv[np.arange(len(state)),pick])
    return p,labels,keep,pick,u,v,b


def geometry_points(d,u,v,b,widths,ids,ds):
    val=u[ids]+(d['family'][ids]==0)*ds
    bucket=np.clip(np.searchsorted(b,val,side='right')-1,0,len(widths)-1)
    return bucket+(val-b[bucket])/widths[bucket],v[ids]+(d['family'][ids]==1)*ds


def edge_readout(d,geometry,widths,e):
    _,_,keep,_,u,v,b=geometry;i=d['i'][e];j=d['j'][e]
    qa,_=geometry_points(d,u,v,b,widths,i,d['ds_a'][e])
    qb,_=geometry_points(d,u,v,b,widths,j,d['ds_b'][e])
    delta=qb-qa-d['phase_delta'][e]+.5*(d['family'][j]-d['family'][i])
    return qa,qb,delta,keep[i]&keep[j]


def violation_indices(d,state,widths,config):
    geometry=hard_geometry(d,state,widths,config);found=[]
    for start in range(0,len(d['i']),131072):
        e=np.arange(start,min(start+131072,len(d['i'])))
        _,_,delta,active=edge_readout(d,geometry,widths,e)
        wrong=active&(d['sign'][e]!=0)&(delta*d['sign'][e]<-config.get('sign_tolerance',1e-4))
        found.append(e[wrong])
    return np.concatenate(found)


def audit(d, state, widths, config, indices=None):
    """Vectorized CPU hard readout, full graph at final; no GPU needed."""
    geometry=hard_geometry(d,state,widths,config)
    p,labels,keep,pick,u,v,b=geometry;k=len(widths)
    if config.get('broken_threshold') is not None:state=state.astype(np.float64,copy=False)
    extra={}
    if config.get('local_window_radius') is not None:
        radius=config['local_window_radius'];center=p[:,:k].argmax(1)
        mass=p[:,:k].sum(1);selected=np.zeros(len(p),dtype=np.float64)
        rows=np.arange(len(p))
        for offset in range(-radius,radius+1):
            index=center+offset;valid=(index>=0)&(index<k)
            selected[valid]+=p[rows[valid],index[valid]]
        coverage=selected/np.maximum(mass,1e-30)
        extra.update(window_retained_mass_mean=float(mass.mean()),
            window_captured_conditional_mean_percent=float(100*coverage.mean()),
            window_captured_conditional_q10_q50_q90_percent=(100*np.quantile(coverage,[.1,.5,.9])).tolist())
    if anchored(config):
        du=state[:,k+3:2*k+3];du=du-du.mean(1,keepdims=True)
        dv=state[:,2*k+3:];dv=dv-dv.mean(1,keepdims=True)
        phase=d['phase']+state[:,k+1]
        q=pick+phase+du[np.arange(len(state)),pick]
        separation=1+np.diff(du,axis=1)
        extra.update(dict(offset_over_quarter_percent=float(100*np.mean(abs(du)>.25)),
            chosen_retained_offset_over_quarter_percent=float(100*np.mean(
                abs(du[np.arange(len(state)),pick][keep])>.25)) if keep.any() else None,
            probability_mass_offset_over_quarter_percent=float(100*np.sum(p[:,:k]*(abs(du)>.25))/max(float(p[:,:k].sum()),1e-30)),
            chosen_retained_abs_offset_quantiles=np.quantile(abs(du[np.arange(len(state)),pick][keep]),[.5,.95,.99,1]).tolist() if keep.any() else None,
            abs_offset_turn_quantiles=np.quantile(abs(du),[.5,.95,.99,1]).tolist(),
            min_bucket_separation_turns=float(separation.min()),
            nonpositive_bucket_separation_percent=float(100*np.mean(separation<=0)),
            shared_phase_range=[float(phase.min()),float(phase.max())],
            retained_anchor_q_quantiles=np.quantile(q[keep],[0,.5,1]).tolist() if keep.any() else None,
            occupied_anchor_windings=np.unique(np.floor(q[keep]).astype(int)).tolist(),
            bucket_indices_are_relative_to_unwrapped_anchor=True))
    sel=np.arange(len(d['i'])) if indices is None else indices
    total=known=bad=retained=0;errors=[];bytype={};minq=np.inf;maxq=-np.inf
    for a in range(0,len(sel),131072):
        e=sel[a:a+131072];i=d['i'][e];j=d['j'][e]
        qa,qb,delta,active=edge_readout(d,geometry,widths,e)
        minq=min(minq,qa.min(),qb.min());maxq=max(maxq,qa.max(),qb.max())
        ks=(d['sign'][e]!=0)&active
        wrong=(delta*d['sign'][e]<-config.get('sign_tolerance',1e-4))&ks
        total+=len(e);retained+=int(active.sum());known+=int(ks.sum());bad+=int(wrong.sum())
        errors.append(abs(abs(delta[active])-d['distance'][e][active]))
        for name,mask in [('parallel',d['parallel'][e]),('perpendicular',~d['parallel'][e])]:
            r=bytype.setdefault(name,dict(events=0,retained=0,known=0,violations=0))
            r['events']+=int(mask.sum());r['retained']+=int((mask&active).sum());r['known']+=int((mask&ks).sum());r['violations']+=int((mask&wrong).sum())
    error=np.concatenate(errors);links=d['links'];continuity=[];retained_links=0
    if len(links):
        i=links[:,0].astype(int);j=links[:,1].astype(int);active=keep[i]&keep[j]
        ua=u[i]+(d['family'][i]==0)*links[:,2];ub=u[j]+(d['family'][j]==0)*links[:,3]
        va=v[i]+(d['family'][i]==1)*links[:,2];vb=v[j]+(d['family'][j]==1)*links[:,3]
        continuity=np.hypot(ua-ub,va-vb)[active]
        retained_links=int(active.sum())
    def quantiles(x):return np.quantile(x,[.5,.95,.99]).tolist() if len(x) else None
    for r in bytype.values():r['sign_violation_percent']=100*r['violations']/r['known'] if r['known'] else None
    return dict(pieces=len(state),rejected_pieces=int((~keep).sum()),rejected_percent=float(100*(~keep).mean()),
        retained_arc_percent=float(100*d['lengths'][keep].sum()/d['lengths'].sum()),events_evaluated=total,
        retained_events=retained,retained_events_percent=100*retained/total,known_sign_events=known,sign_violations=bad,
        sign_feasible=bool(bad==0 and known>0),broken_threshold=config.get('broken_threshold'),
        sign_tolerance=config.get('sign_tolerance',1e-4),
        sign_violation_percent=100*bad/known if known else None,distance_error_quantiles=quantiles(error),
        distance_mean_error=float(error.mean()) if len(error) else None,continuation_error_base_quantiles=quantiles(continuity),
        continuation_links=len(links),retained_continuation_links=retained_links,
        ambiguous_percent=float(100*(p.max(1)<.9).mean()),zero_probability_percent=float(100*(p==0).mean()),
        mean_rejection_probability=float(p[:,-1].mean()),occupied_windings=np.unique(labels[keep]).tolist(),
        boundary_hypothesis_percent=float(100*((labels[keep]==0)|(labels[keep]==k-1)).mean()) if keep.any() else None,
        endpoint_q_range=[float(minq),float(maxq)],by_type=bytype,**extra)


def expand_winding_state(state,widths,count):
    """Expand shared-anchor hypotheses without changing old physical geometry."""
    from scipy.special import logsumexp
    old=len(widths)
    if count<old or state.shape[1]!=3*old+3:raise ValueError('Invalid shared-anchor expansion')
    if count==old:return state,widths
    out=np.zeros((len(state),3*count+3),dtype=state.dtype)
    logits=state[:,:old+1].astype(np.float64)
    logp=logits-logsumexp(logits,axis=1,keepdims=True)
    out[:,:old]=logp[:,:old]+np.log(old/count)
    out[:,old:count]=(logsumexp(logp[:,:old],axis=1)-np.log(count))[:,None]
    out[:,count]=logp[:,old]
    out[:,count+1:count+3]=state[:,old+1:old+3]
    for a,b in ((old+3,count+3),(2*old+3,2*count+3)):
        offsets=state[:,a:a+old]
        out[:,b:b+old]=offsets-offsets.mean(1,keepdims=True)
    return out,np.r_[widths,np.full(count-old,widths[-1],dtype=widths.dtype)]


def load_warm_start(config,d,shape):
    path=Path(config['warm_start'])
    if sha256_file(path)!=config['warm_start_sha256']:raise ValueError('Changed warm start')
    summary=json.loads((path.parent/'summary.json').read_text())
    expected=json.loads(Path(config['tensor_cache']).with_suffix('.json').read_text())['signature']
    if summary['input']['signature']!=expected:raise ValueError('Warm-start evidence mismatch')
    with np.load(path) as src:
        for key in ('source_id','start','end','family'):
            if not np.array_equal(src[key],d[key]):raise ValueError('Warm-start piece mismatch: '+key)
        state=src['state'].copy();widths=src['widths'].copy()
    old=len(widths);expected_columns=3*old+(3 if anchored(config) else 1)
    if state.shape!=(shape[0],expected_columns) or not np.isfinite(state).all() or not np.isfinite(widths).all() or np.any(widths<=100):
        raise ValueError('Invalid warm-start state/widths')
    count=int(config.get('winding_count',100))
    if old!=count:
        if not config.get('expand_windings') or not anchored(config):raise ValueError('Explicit shared-anchor expansion required')
        state,widths=expand_winding_state(state,widths,count)
    if state.shape!=shape:raise ValueError('Warm-start target shape mismatch')
    return state,widths


def inverse_widths(widths):
    import torch
    x=widths-100
    if bool((x<=0).any()):raise ValueError('Widths must exceed100')
    return x+torch.log(-torch.expm1(-x))


def reference_initial_centers(d,widths,reference,radial_centers):
    """Place each piece midpoint at its source's reference winding, no GT loss."""
    reference=np.asarray(reference);centers=np.asarray(radial_centers).copy()
    valid=np.isfinite(reference)
    if np.any((reference[valid]<0)|(reference[valid]>=len(widths))):
        raise ValueError('Reference winding outside configured hypotheses')
    ix=reference[valid].astype(int)
    phase=d['phase'][valid]+(d['family'][valid]==0)*d['node_ds'][valid,1]/widths[ix]
    centers[valid]=reference[valid]-np.floor(phase+.5)
    if np.any((centers[valid]<0)|(centers[valid]>=len(widths))):
        raise ValueError('Reference anchor-corrected winding outside configured hypotheses')
    return centers


def fresh_initial_state(d,config):
    """Geometric initialization only; never consumes a previous U/V solution.

    Physical evidence caches predate configurable K. Their radius observations
    remain reusable; regenerate the radial prior across the requested range.
    Preserve the historical 100-candidate initialization byte-for-byte.
    """
    from scipy.special import logsumexp
    k=int(config.get('winding_count',100))
    sigma=float(config.get('initial_sigma',3.))
    if k<2 or not np.isfinite(sigma) or sigma<=0:
        raise ValueError('Require at least two windings and positive finite initial_sigma')
    if k==100:
        centers=np.asarray(d['initial_k'])
        widths=np.asarray(d['width_initial']).copy()
    else:
        r=np.asarray(d['radius']);lo,hi=np.quantile(r,[.001,.999]);scale=(k-1)/99
        centers=np.clip(scale*(5+85*(r-lo)/max(hi-lo,1)),0,k-1).astype(np.float32)
        radii=np.interp(np.arange(k),np.linspace(5*scale,90*scale,k),np.linspace(lo,hi,k))
        widths=np.maximum(2*np.pi*radii,100.).astype(np.float32)
    if config.get('reference_initialization'):
        if not anchored(config):raise ValueError('Reference initialization requires shared_anchor layout')
        from .fiber_reference_comparison import load_metric_projection
        projection=load_metric_projection(config['reference_initialization'],d,config)
        centers=reference_initial_centers(d,widths,projection['chosen_winding'],centers)
    state=np.zeros((len(d['family']),3*k+(3 if anchored(config) else 1)),np.float32)
    state[:,:k]=np.maximum(-14.,-.5*((np.arange(k)[None,:]-centers[:,None])/sigma)**2)
    state[:,k]=logsumexp(state[:,:k],axis=1)+np.log(.01/.99)
    return state,widths


def pass_partition(count, updates, index):
    """Balanced disjoint partition; every item appears exactly once per pass."""
    return slice(count*index//updates, count*(index+1)//updates)


class FullPassGradientAccumulator:
    """Bounded storage for repeated sparse rows; all nodes occur in each pass."""
    def __init__(self,weight,width):
        import torch
        self.weight=torch.zeros_like(weight,requires_grad=False)
        self.width=torch.zeros_like(width,requires_grad=False)
        self.rows=torch.arange(len(weight),device=weight.device)[None,:]

    def reset(self):
        self.weight.zero_();self.width.zero_()

    def add(self,weight_grad,width_grad):
        import torch
        with torch.no_grad():
            grad=weight_grad.detach().coalesce();rows=grad.indices()[0]
            # Coalescing guarantees unique rows; no duplicate-index writes.
            self.weight.index_copy_(0,rows,self.weight.index_select(0,rows)+grad.values())
            self.width.add_(width_grad.detach())

    def install(self,weight,width):
        import torch
        weight.grad=torch.sparse_coo_tensor(self.rows,self.weight,size=weight.shape,
            device=weight.device,dtype=weight.dtype,is_coalesced=True,check_invariants=True)
        width.grad=self.width


def solve(config):
    if config.get('experiment_matrix'):
        from .fiber_normal_sign import solve_tensor_matrix
        return solve_tensor_matrix(config,'vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_layout')
    if config.get('benchmark_pair_fusion'):
        from .fiber_uv_pair_benchmark import benchmark
        return benchmark(config)
    if config.get('assignment_update')=='cost_target':
        from .fiber_uv_categorical import solve_categorical
        return solve_categorical(config)
    validate_joint_rejection(config)
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Production U/V solve requires CUDA')
    torch.set_num_threads(4);torch.manual_seed(config.get('seed',0))
    torch.use_deterministic_algorithms(True)
    with np.load(config['tensor_cache']) as archive:d={key:archive[key] for key in archive.files}
    device='cuda:0';data=_tensor_data(d,device)
    n=len(d['family']);k=int(config.get('winding_count',100));root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    if (root/'summary.json').exists():raise FileExistsError('Completed U/V output exists')
    rng=np.random.default_rng(config.get('seed',0))
    diagnostic=np.sort(rng.choice(len(d['i']),min(100000,len(d['i'])),replace=False))
    # Shared table lookup makes broad Gaussian hypotheses cheap without K*K*R tensors.
    table=torch.from_numpy(make_distance_table(float(d['distance'].max()),config.get('distance_sigma',.5))).to(device)
    state,start_width=fresh_initial_state(d,config)
    initial_width=torch.as_tensor(start_width,device=device)
    if config.get('warm_start'):state,start_width=load_warm_start(config,d,state.shape)
    embedding=torch.nn.Embedding.from_pretrained(torch.from_numpy(state).to(device),freeze=False,sparse=True)
    raw=torch.nn.Parameter(inverse_widths(torch.as_tensor(start_width,device=device)))
    opt=torch.optim.SparseAdam(embedding.parameters(),lr=config.get('lr',.03))
    widthopt=torch.optim.Adam([raw],lr=config.get('width_lr',2.))
    steps=int(config.get('steps',1000));batch=int(config.get('batch_size',1024));every=int(config.get('log_every',100))
    full_pass=config.get('traversal','full_pass')=='full_pass'
    if config.get('traversal','full_pass') not in ('full_pass','sampled'):
        raise ValueError('Unknown joint traversal')
    if full_pass:
        passes=int(config.get('passes',10))
        if passes<1 or batch<1 or not len(d['i']):raise ValueError('Invalid full-pass budget')
        updates=(len(d['i'])+batch-1)//batch
        if updates>n:raise ValueError('Increase batch_size: full-pass node partitions must be nonempty')
        steps=passes*updates;every=updates
    benchmark_updates=config.get('benchmark_updates')
    if benchmark_updates is not None:
        if config.get('update_mode','minibatch')!='minibatch':
            raise ValueError('Short update benchmark requires minibatch updates')
        steps=int(benchmark_updates);every=steps
        if steps<2:raise ValueError('Benchmark requires at least two updates')
    evaluate_pair=pair_evaluator(config)
    update_timings=[]
    segment_timings=[]
    update_profile=None;profile_rows=None
    def timing_marker():
        event=torch.cuda.Event(enable_timing=True);event.record();return event
    accumulated=config.get('update_mode','minibatch')=='global_accumulated'
    gradients=FullPassGradientAccumulator(embedding.weight,raw) if accumulated else None
    history=[];started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    baseline=audit(d,state,start_width,config)
    reference_projection=None
    if config.get('reference_projection'):
        from .fiber_reference_comparison import load_metric_projection,reference_winding_metrics
        reference_projection=load_metric_projection(config['reference_projection'],d,config)
    pass_losses=torch.zeros(3,device=device)
    charged=config.get('sign_charge_multiplier',0.)>0
    fraction=float(config.get('active_fraction',0.))
    if full_pass and fraction:raise ValueError('Full passes forbid active oversampling; set active_fraction=0')
    if not 0<=fraction<1:raise ValueError('Active fraction must preserve uniform support')
    active=torch.empty(0,device=device,dtype=torch.long)
    active_mask=torch.zeros(len(d['i']),device=device,dtype=torch.bool)
    if charged:
        phase64=torch.as_tensor(d['phase'],device=device,dtype=torch.float64)
        z64=torch.as_tensor(d['z'],device=device,dtype=torch.float64)
        precise={key:torch.as_tensor(d[key],device=device,dtype=torch.float64)
                 for key in ('ds_a','ds_b','phase_delta')}
    active_every=int(config.get('active_refresh',every))
    if active_every<1:raise ValueError('Invalid active refresh')
    def coords(ids,ds,widths,bounds):
        s=embedding(ids);p=s[:,:k+1].softmax(-1)
        u,v,q=coordinates(s,data['phase'][ids],data['z'][ids],data['family'][ids],ds,widths,bounds)
        return p,u,v,q
    for step in range(steps+1):
        widths,bounds=width_boundaries(raw)
        refreshed=False
        if fraction and (step%active_every==0 or step==steps):
            snapshot=embedding.weight.detach().cpu().numpy();w=widths.detach().cpu().numpy()
            bad_indices=violation_indices(d,snapshot,w,config)
            active=torch.as_tensor(bad_indices,device=device,dtype=torch.long)
            active_mask.zero_();active_mask[active]=True;refreshed=True
            print(f'UV active step={step} strict_violations={len(active)}',flush=True)
        feasible_check=refreshed and not len(active) and step>=config.get('minimum_steps',100) and config.get('stop_when_feasible',False)
        if step%every==0 or step==steps or feasible_check:
            torch.cuda.synchronize();snapshot=embedding.weight.detach().cpu().numpy()
            w=widths.detach().cpu().numpy();metrics=audit(d,snapshot,w,config,None if accumulated or step==steps or feasible_check or charged else diagnostic)
            record=dict(step=step,seconds=time.monotonic()-started,event_samples=step*batch,
                equivalent_event_epochs=step*batch/len(d['i']),importance_sampled=bool(fraction),
                active_violations=len(active) if fraction else None,width_range=[float(w.min()),float(w.max())],**metrics)
            if full_pass:
                processed=(step//updates)*len(d['i'])+(step%updates)*len(d['i'])//updates
                record.update(completed_passes=step//updates,event_samples=processed,
                              equivalent_event_epochs=processed/len(d['i']),traversal='full_pass')
            record.update(update_mode=config.get('update_mode','minibatch'),
                          optimizer_updates=step//updates if accumulated else step)
            if reference_projection is not None:
                record['reference']=reference_winding_metrics(d,snapshot,w,config,reference_projection)
            if step:record.update(loss=float(loss.detach()),pair_loss=float(pair.detach()),regularization=float(reg.detach()))
            if step and full_pass:
                means=(pass_losses/(step if benchmark_updates is not None else updates)).cpu().tolist()
                record.update(loss=means[0],pair_loss=means[1],regularization=means[2],
                              loss_scope=('full objective at pre-update state; diagnostics are post-update' if accumulated else
                                          'mean online objective across pass updates, not frozen-state global objective'))
                if benchmark_updates is not None:record['loss_scope']='mean online objective across short benchmark updates'
                pass_losses.zero_()
            elif step:
                means=(pass_losses/(step%every or every)).cpu().tolist()
                record.update(loss=means[0],pair_loss=means[1],regularization=means[2],
                              loss_scope='mean online objective since previous diagnostic; sampled minibatch updates')
                pass_losses.zero_()
            history.append(record);atomic_write(root/'history.json',history,durable=False)
            print('UV '+json.dumps(record),flush=True)
            if step==steps or (feasible_check and metrics['sign_feasible']):break
        if benchmark_updates is not None:
            if config.get('benchmark_torch_profile') and step==steps-1:
                torch.cuda.synchronize()
                update_profile=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                     torch.profiler.ProfilerActivity.CUDA])
                update_profile.__enter__()
            tick=torch.cuda.Event(enable_timing=True);tock=torch.cuda.Event(enable_timing=True);tick.record()
        if full_pass:
            position=step%updates
            if position==0:
                if accumulated:gradients.reset()
                event_order=torch.randperm(len(d['i']),device=device)
                node_order=torch.randperm(n,device=device)
            edges=event_order[pass_partition(len(d['i']),updates,position)]
            event_weights=torch.ones(len(edges),device=device)
        else:
            edges,event_weights=draw_events(len(d['i']),batch,active,active_mask,fraction,device)
        i=data['i'][edges];j=data['j'][edges]
        pa,ua,va,qa=coords(i,data['ds_a'][edges],widths,bounds)
        pb,ub,vb,qb=coords(j,data['ds_b'][edges],widths,bounds)
        charge=selected=None
        if charged:
            charge=calibrated_sign_charge(data['degree'][i],data['degree'][j],data['degree'].mean(),
                len(d['i']),n,config.get('rejection_cost',10.),config['sign_charge_multiplier'],
                config.get('rejection_weighting','fixed'))
            selected=selected_ordering(embedding(i),embedding(j),phase64[i],phase64[j],z64[i],z64[j],
                data['family'][i],data['family'][j],precise['ds_a'][edges],precise['ds_b'][edges],widths,
                precise['phase_delta'][edges],data['sign'][edges])
        if benchmark_updates is not None:pair_start=timing_marker()
        pair=evaluate_pair(pa[:,:k],pb[:,:k],qa,qb,va,vb,data['phase_delta'][edges],data['family'][j]-data['family'][i],
            data['sign'][edges],data['distance'][edges],table,config,charge,event_weights,selected)
        if benchmark_updates is not None:pair_end=timing_marker()
        for pp,qq,phase_key in ((pa,qa,'phase_a'),(pb,qb,'phase_b')):
            angle=qq-data[phase_key][edges,None];angle=angle-angle.detach().round()
            pair=pair+config.get('prior_weight',.01)*.5*(event_weights*(pp[:,:k]*angle.square()).sum(-1)).mean()
        # Uniform node sample prevents degree-biased rejection/prior normalization.
        ni=(node_order[pass_partition(n,updates,position)] if full_pass else
            torch.randint(n,(batch,),device=device))
        ns=embedding(ni);npb=ns[:,:k+1].softmax(-1)
        degree=data['degree'];cost=rejection_weights(degree[ni],degree.mean(),config.get('rejection_weighting','fixed'))
        reg=config.get('rejection_cost',10.)*(npb[:,k]*cost).mean()
        if anchored(config):reg=reg+offset_regularization(ns,k,config)
        ramp=min(1.,max(0.,(step/steps-.5)/.3)) if config.get('entropy_ramp',True) else 1.
        reg=reg+config.get('entropy_weight',.05)*ramp*(-npb*npb.clamp_min(1e-20).log()).sum(-1).mean()
        for point in range(3):
            nu,nv,nq=coordinates(ns,data['phase'][ni],data['z'][ni],data['family'][ni],data['node_ds'][ni,point],widths,bounds)
            angle=nq-data['node_phase'][ni,point,None];angle=angle-angle.detach().round()
            prior=angle.square()
            prior=prior+torch.relu(-nq).square()+torch.relu(nq-k).square()
            unary=config.get('prior_weight',.01)*prior+vertical_layout_prior(nv,data['node_z'][ni,point,None],config)
            reg=reg+(npb[:,:k]*unary).sum(-1).mean()/3
        if full_pass:
            # Balanced partitions can differ by one item; preserve full-objective weights.
            pair=pair*(len(edges)*updates/len(d['i']))
            reg=reg*(len(ni)*updates/n)
        reg=reg+config.get('width_prior',.1)*((widths/initial_width-1).square().mean())
        reg=reg+config.get('width_smooth',.01)*torch.diff(widths/initial_width).square().mean()
        if len(data['links']):
            links=(data['links'] if full_pass else data['links'][torch.randint(len(data['links']),(min(batch,len(data['links'])),),device=device)])
            ca,cu,cv,_=coords(links[:,0].long(),links[:,2],widths,bounds)
            cb,du,dv,_=coords(links[:,1].long(),links[:,3],widths,bounds)
            continuation=continuation_likelihood if anchored(config) else continuation_cost
            link_normalization=len(data['links'])/len(d['i']) if anchored(config) else 1.
            reg=reg+config.get('continuation_weight',10.)*link_normalization*continuation(
                ca[:,:k],cb[:,:k],cu,du,cv,dv,config.get('continuation_scale',128.) if anchored(config) else initial_width.mean())
        loss=pair+reg
        pass_losses+=torch.stack((loss.detach(),pair.detach(),reg.detach()))
        opt.zero_grad(set_to_none=True);widthopt.zero_grad(set_to_none=True)
        if benchmark_updates is not None:backward_start=timing_marker()
        (loss/updates if accumulated else loss).backward()
        if benchmark_updates is not None:backward_end=timing_marker()
        # Broken is an ordinary jointly optimized category from the first update.
        if accumulated:
            gradients.add(embedding.weight.grad,raw.grad)
            if position==updates-1:
                gradients.install(embedding.weight,raw)
                opt.step();widthopt.step()
        else:
            embedding.weight.grad=embedding.weight.grad.coalesce()
            opt.step();widthopt.step()
        if benchmark_updates is not None:
            tock.record();update_timings.append((tick,tock))
            segment_timings.append((tick,pair_start,pair_end,backward_start,backward_end,tock))
            if update_profile is not None:
                torch.cuda.synchronize();update_profile.__exit__(None,None,None)
                top=sorted(update_profile.key_averages(),key=lambda e:e.self_device_time_total,reverse=True)[:30]
                profile_rows=[dict(name=e.key,self_cuda_us=e.self_device_time_total,
                                   self_cpu_us=e.self_cpu_time_total,count=e.count) for e in top]
                update_profile.export_chrome_trace(str(root/'update_trace.json'))
                update_profile=None
    np.savez(root/'solution.npz',state=snapshot,widths=w,source_id=d['source_id'],start=d['start'],end=d['end'],family=d['family'])
    summary=dict(config=config,input=json.loads(Path(config['tensor_cache']).with_suffix('.json').read_text()),
        baseline=baseline,final=history[-1],gpu=torch.cuda.get_device_name(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        state_layout=(f'logits[{k+1}],shared_phase,shared_v,centered_u_turn_offsets[{k}],centered_v_offsets[{k}]'
                      if anchored(config) else f'logits[{k+1}],fraction_adjustment[{k}],v_adjustment[{k}]'),
        initialization=dict(kind=('checkpoint' if config.get('warm_start') else
                                 'reference_winding' if config.get('reference_initialization') else 'fresh_radial'),
            winding_count=k,initial_sigma=config.get('initial_sigma',3.),
            geometry_trainable=True,widths_trainable=True),
        interpretation='Model consistency only; joint Adam, '+('exhaustive full passes' if full_pass else 'sampled batches'))
    if benchmark_updates is not None:
        torch.cuda.synchronize()
        ms=[a.elapsed_time(b) for a,b in update_timings]
        warmup=int(config.get('benchmark_warmup',5))
        end=-1 if profile_rows is not None else None
        steady=ms[warmup:end]
        if not steady:raise ValueError('Benchmark warmup consumed all updates')
        summary['update_benchmark']=dict(milliseconds=ms,warmup=warmup,mean_ms=float(np.mean(steady)),
            p50_ms=float(np.quantile(steady,.5)),p95_ms=float(np.quantile(steady,.95)),
            min_ms=float(min(steady)),max_ms=float(max(steady)),updates=len(ms),
            note='Production minibatch update excluding initial width-boundary calculation and logging/audit; first updates include compilation')
        names=('sampling_coordinates','pair_forward','other_losses_forward','all_backward','coalescing_optimizer')
        summary['update_benchmark']['segments']={name:dict(
            mean_ms=float(np.mean(values)),p50_ms=float(np.quantile(values,.5)),
            p95_ms=float(np.quantile(values,.95))) for index,name in enumerate(names)
            for values in [[row[index].elapsed_time(row[index+1]) for row in segment_timings[warmup:end]]]}
        summary['update_benchmark']['profile']=profile_rows
        summary['update_benchmark']['profiled_last_update_excluded']=profile_rows is not None
    atomic_write(root/'summary.json',summary,durable=False)


def run(config):
    if config.get('assignment_update')!='cost_target':
        validate_joint_rejection(config)
        config=dict(config,rejection_warmup=0,rejection_weighting=config.get('rejection_weighting','fixed'))
    prepare(config)
    if config.get('warm_start'):
        digest=sha256_file(config['warm_start'])
        if config.get('warm_start_sha256',digest)!=digest:raise ValueError('Changed warm start')
        config=dict(config,warm_start_sha256=digest)
    return launch_tensor_solve(config,__name__,Path(__file__))


def report(config):
    """Audit persisted partitions/states against original observations, then report."""
    roots=[Path(p) for p in config['runs']];reports=[]
    for root in roots:
        summary=json.loads((root/'summary.json').read_text());cfg=summary['config']
        with np.load(cfg['tensor_cache']) as source:d={k:source[k] for k in source.files}
        with np.load(root/'solution.npz') as source:s={k:source[k] for k in source.files}
        for key in ('source_id','start','end','family'):
            if not np.array_equal(d[key],s[key]):raise ValueError('Saved piece identity mismatch: '+key)
        if not np.isfinite(s['state']).all() or not np.isfinite(s['widths']).all() or np.any(s['widths']<=0):
            raise ValueError('Invalid solution state')
        k=int(cfg.get('winding_count',100))
        if s['state'].shape!=(len(d['family']),3*k+(3 if anchored(cfg) else 1)) or s['widths'].shape!=(k,):raise ValueError('State shape')
        signature=summary['input']['signature']
        for key in ('database','features','stage1','hv_solution','afv','umbilicus'):
            if sha256_file(cfg[key])!=signature[key]:raise ValueError('Changed input: '+key)
        if anchored(cfg):
            for key in ('h_direction_solution','v_direction_solution'):
                if sha256_file(cfg[key])!=signature[key]:raise ValueError('Changed input: '+key)
            with np.load(cfg['hv_solution']) as prior:
                frozen,_=direction_inputs(cfg,signature,d['ids'],prior['labels'])
            if not np.array_equal(frozen,d['earlier_labels']):raise ValueError('Frozen labels changed')
        if np.any(d['earlier_labels'][d['source_index']]==2):raise ValueError('Earlier rejection reintroduced')
        with _ro(cfg['database']) as db:
            lengths=dict(db.execute('SELECT id,length_base FROM fibers'))
            events=np.asarray(db.execute('SELECT id,fiber_a,fiber_b,arc_a,arc_b,parallel_score,perpendicular_score FROM constraints ORDER BY id').fetchall())
        included=np.unique(d['source_id']);mask=np.isin(events[:,1],included)&np.isin(events[:,2],included)
        events=events[mask]
        if not np.array_equal(events[:,0],d['event_ids']):raise ValueError('Observation IDs lost/duplicated')
        for endpoint,key in enumerate(('i','j')):
            owner=d[key];ea=events[:,3+endpoint]
            if not np.array_equal(d['source_id'][owner],events[:,1+endpoint]):raise ValueError('Endpoint fiber ownership')
            if np.any(ea<d['start'][owner]-1e-5) or np.any(ea>d['end'][owner]+1e-5):raise ValueError('Endpoint arc ownership')
        perp=~d['parallel'];counts=np.bincount(np.r_[d['i'][perp],d['j'][perp]],minlength=len(d['family']))
        starts=np.r_[0,np.flatnonzero(np.diff(d['source_id']))+1];ends=np.r_[starts[1:],len(d['family'])]
        expected_links=[];split_fibers=0
        for a,b in zip(starts,ends):
            fid=int(d['source_id'][a])
            if abs(d['start'][a])>1e-6 or abs(d['end'][b-1]-lengths[fid])>1e-4:raise ValueError('Missing source coverage')
            if not np.array_equal(d['end'][a:b-1],d['start'][a+1:b]):raise ValueError('Gap/overlap in partition')
            if b-a>1:
                split_fibers+=1
                if np.any(d['lengths'][a:b]<summary['input']['minimum_piece_base']-1e-5) or np.any(counts[a:b]<3):
                    raise ValueError('Invalid short/underconstrained split')
                expected_links.extend((i,i+1) for i in range(a,b-1))
        if not np.array_equal(np.asarray(expected_links).reshape(-1,2),d['links'][:,:2]):raise ValueError('Invalid continuation graph')
        metrics=audit(d,s['state'],s['widths'],cfg)
        if cfg.get('assignment_update')=='cost_target':
            warm,ww=load_warm_start(cfg,d,s['state'].shape)
            if not np.array_equal(warm[:,k+1:],s['state'][:,k+1:]) or not np.array_equal(ww,s['widths']):
                raise ValueError('Categorical run changed frozen geometry/widths')
        for key in ('rejected_pieces','retained_events','sign_violations'):
            if metrics[key]!=summary['final'][key]:raise ValueError('Recomputed readout mismatch: '+key)
        from scipy.special import softmax
        probs=softmax(s['state'][:,:k+1],axis=1)
        if not np.allclose(probs.sum(1),1,atol=1e-6):raise ValueError('Unnormalized probabilities')
        row=dict(run=str(root),solution_sha256=sha256_file(root/'solution.npz'),tensor_cache_sha256=sha256_file(cfg['tensor_cache']),
            assignment_update=cfg.get('assignment_update','sampled_adam'),
            geometry_frozen=cfg.get('assignment_update')=='cost_target',
            partition_audit=dict(fibers=len(included),pieces=len(d['family']),split_fibers=split_fibers,events=len(events),
                                 continuations=len(expected_links),no_observations_removed_by_splitting=True),
            baseline=summary['baseline'],final=metrics,seconds=summary['final']['seconds'],
            equivalent_epochs=summary['final']['equivalent_event_epochs'],peak_gpu_gib=summary['peak_allocated_bytes']/1024**3)
        if cfg.get('sign_charge_multiplier',0)>0:
            initial,initial_width=(load_warm_start(cfg,d,s['state'].shape)
                                   if cfg.get('warm_start') else fresh_initial_state(d,cfg))
            before=violation_indices(d,initial,initial_width,cfg)
            after=violation_indices(d,s['state'],s['widths'],cfg)
            geometry=hard_geometry(d,s['state'],s['widths'],cfg)
            p,_,keep,*_=geometry
            stats=dict(multiplier=cfg['sign_charge_multiplier'],broken_threshold=cfg['broken_threshold'],
                       sign_tolerance=cfg.get('sign_tolerance',1e-4),
                       original_violations=len(before),remaining_violations=len(after),
                       newly_violating_events=len(np.setdiff1d(after,before)))
            i=d['i'][before];j=d['j'][before]
            disabled=~(keep[i]&keep[j]);stats['original_violations_disabled']=int(disabled.sum())
            stats['original_violations_corrected_while_retained']=int((~disabled).sum()-len(np.intersect1d(before,after)))
            if len(after):
                _,_,delta,_=edge_readout(d,geometry,s['widths'],after)
                stats['remaining_violation_magnitude_quantiles']=np.quantile(-delta*d['sign'][after],[0,.5,.95,1]).tolist()
                stats['remaining_max_endpoint_broken_probability_quantiles']=np.quantile(
                    np.maximum(p[d['i'][after],-1],p[d['j'][after],-1]),[0,.5,.95,1]).tolist()
                ids=np.unique(np.r_[d['i'][after],d['j'][after]])
                probs=p[ids,-1]
                stats['remaining_unique_endpoint_pieces']=len(ids)
                stats['remaining_unique_source_fibers']=len(np.unique(d['source_id'][ids]))
                stats['remaining_endpoint_broken_probability_quantile_levels']=[0,.05,.25,.5,.75,.95,1]
                stats['remaining_endpoint_broken_probability_quantiles']=np.quantile(probs,[0,.05,.25,.5,.75,.95,1]).tolist()
                stats['remaining_endpoint_broken_probability_mean']=float(probs.mean())
                bins=np.array([0,.01,.05,.1,.25,.4,.5,1.000001])
                stats['remaining_endpoint_broken_probability_bin_edges']=bins.tolist()
                stats['remaining_endpoint_broken_probability_bin_counts']=np.histogram(probs,bins)[0].tolist()
                pair_max=np.maximum(p[d['i'][after],-1],p[d['j'][after],-1])
                pair_min=np.minimum(p[d['i'][after],-1],p[d['j'][after],-1])
                stats['remaining_max_endpoint_broken_probability_mean']=float(pair_max.mean())
                stats['remaining_min_endpoint_broken_probability_quantiles']=np.quantile(pair_min,[0,.5,.95,1]).tolist()
                stats['remaining_pairs_both_below_probability']={str(t):int((pair_max<t).sum()) for t in (.01,.05,.1,.25,.4)}
                stats['all_retained_piece_broken_probability_quantiles']=np.quantile(p[keep,-1],[0,.5,.95,1]).tolist()
                stats['all_retained_piece_broken_probability_mean']=float(p[keep,-1].mean())
            row['fixed_charge_audit']=stats
        reports.append(row);atomic_write(root/'audit.json',row,durable=False)
    destination=Path(config['report']);lines=['# Global U/V layout: first experiments','',
        'All earlier rejections fixed; baked H/V. Full-graph readout, no ground-truth layout labels.',
        'Ordering errors use retained known-sign events; each run records its tolerance and broken readout below. Finite penalty is not a hard feasibility guarantee.',
        'Rejection and arc retention use input pieces/arc after earlier stages. Distance errors are winding units.','',
        '| Run | Rejected pieces | Retained arc | Retained events | Sign errors | Mean distance error | Ambiguous (<90%) | Seconds |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    details=[]
    for r in reports:
        for label,m in [('initial',r['baseline']),('final',r['final'])]:
            lines.append(f"|{Path(r['run']).name} {label}|{m['rejected_percent']:.4f}%|{m['retained_arc_percent']:.4f}%|"
                f"{m['retained_events_percent']:.4f}%|{m['sign_violation_percent']:.4f}%|{m['distance_mean_error']:.4f}|"
                f"{m['ambiguous_percent']:.4f}%|{r['seconds']:.2f}|" )
        details+=['',f"{Path(r['run']).name}: {r['partition_audit']}; sampled epochs={r['equivalent_epochs']:.3f}, peakGPU={r['peak_gpu_gib']:.3f}GiB.",
            f"Readout: broken threshold={r['final']['broken_threshold']} (None=joint argmax), sign tolerance={r['final']['sign_tolerance']}; feasible={r['final']['sign_feasible']}.",
            f"Final per-type: {r['final']['by_type']}",f"Continuation links retained: {r['final']['retained_continuation_links']}/{r['final']['continuation_links']}; median/p95/p99(basevx): {r['final']['continuation_error_base_quantiles']}",
            f"Numerically zero probabilities: {r['final']['zero_probability_percent']:.6f}%; boundary hypotheses: {r['final']['boundary_hypothesis_percent']}%."]
        if 'fixed_charge_audit' in r:details.append(f"Fixed-charge audit: {r['fixed_charge_audit']}")
        if 'offset_over_quarter_percent' in r['final']:
            m=r['final']
            details += [f"Quarter-turn exceedance: all buckets {m['offset_over_quarter_percent']:.4f}%; chosen retained {m['chosen_retained_offset_over_quarter_percent']:.4f}%; probability mass {m['probability_mass_offset_over_quarter_percent']:.4f}%.",
                f"Chosen retained |offset| median/p95/p99/max: {m['chosen_retained_abs_offset_quantiles']}; minimum bucket separation {m['min_bucket_separation_turns']:.6f} turns.",
                f"Shared phase range {m['shared_phase_range']}; actual retained anchor-q min/median/max {m['retained_anchor_q_quantiles']}. Bucket IDs are relative to the unwrapped anchor."]
    lines+=details
    destination.write_text('\n'.join(lines)+'\n');atomic_write(destination.with_suffix('.json'),reports,durable=False)
    print('\n'.join(lines),flush=True);return dict(report=str(destination),runs=len(reports))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--unmanaged',action='store_true');parser.add_argument('--devices')
    args=parser.parse_args()
    if not args.unmanaged:raise RuntimeError('Use the approved managed runner')
    solve(json.loads(args.config.read_text()))
