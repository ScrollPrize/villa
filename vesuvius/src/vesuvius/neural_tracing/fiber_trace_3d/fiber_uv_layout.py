"""Probabilistic winding / physical U,V layout of fixed-family fiber pieces.

Opt-in experimental stage. Input constraint records and earlier solutions remain
immutable. All winding hypotheses participate; constraint points have no state.
"""
import argparse
import json
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


def prepare(config):
    from .fiber_hv_stage import verify_feature_source
    from .fiber_normal_sign_visualization import load_umbilicus_frame
    from vesuvius.neural_tracing.fiber_follow.data.afv import read_afv_polylines
    started=time.monotonic();cache=Path(config['tensor_cache']);stamp=cache.with_suffix('.json')
    signature={key:sha256_file(config[key]) for key in ('database','features','stage1','hv_solution','afv','umbilicus')}
    signature.update(version=2,minimum_length_mm=config.get('minimum_length_mm',10.),
                     sign_cosine_min=config.get('sign_cosine_min',.1),pilot_fibers=config.get('pilot_fibers',0))
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
        arc,travel,angle,radius=fiber_geometry(xyz,family,axis)
        ep=order[positions[idx]:last[idx]];ea=endpoint_arcs[ep]
        if not np.isfinite(ea).all() or np.any(ea<0) or np.any(ea>arc[-1]+1e-5):
            raise ValueError(f'Constraint arc outside original fiber {row["id"]}')
        perp=ea[events[ep%ecount,11]==0]
        bounds=split_bounds(arc[-1],perp,minimum);first=len(pieces)
        middle=(bounds[:-1]+bounds[1:])/2
        tr=np.interp(middle,arc,travel);ph=np.interp(middle,arc,angle)
        for p in range(len(middle)):
            a,b=bounds[p:p+2];m=middle[p]
            pieces.append((idx,row['id'],a,b,family,tr[p],ph[p],np.interp(m,arc,xyz[:,2]),np.interp(m,arc,radius)))
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
    # V metric target uses actual native axial displacement; cylinder is a weak prior.
    target_v=events[:,10]-events[:,7]
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
        target_v=target_v.astype(np.float32),distance=events[:,12].astype(np.float32),sign=events[:,13].astype(np.float32),
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


def coordinates(state, phase, z, family, ds, widths, boundaries):
    """Only piece state has parameters. All base points are derived here."""
    k=len(widths);frac=phase[:,None]+.25*state[:,k+1:2*k+1].tanh()
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


def pair_objective(pa,pb,qa,qb,va,vb,phase_delta,family_delta,sign,mu,target_v,table,config):
    """Exact joint posterior expectation over all K*K assignments in this batch."""
    import torch
    delta=qb[:,None,:]-qa[:,:,None]-phase_delta[:,None,None]+.5*family_delta[:,None,None]
    # Finite optimization surrogate for truncated support: never reflect a known
    # wrong sign into a good distance. Boundary cost + uncapped corrective hinge.
    ordered=sign[:,None,None]*delta
    wrong=torch.relu(-ordered)*4
    distance=torch.where((sign!=0)[:,None,None],ordered.clamp_min(0),delta.abs())
    soft=table_cost(distance,mu,table)
    # Local tangential displacement is accounted by observed angle difference.
    phase_error=delta-.5*family_delta[:,None,None]
    phase_error=phase_error-phase_error.detach().round()
    vertical=((vb[:,None,:]-va[:,:,None]-target_v[:,None,None])/config.get('metric_scale',128.)).square().clamp(max=100.)
    cost=config.get('sign_weight',10.)*wrong+config.get('distance_weight',1.)*soft
    cost=cost+config.get('metric_weight',.01)*(phase_error.square()+vertical)
    mass=pa[:,:,None]*pb[:,None,:]
    return (mass*cost).sum((1,2)).mean()


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
          'phase_a','phase_b','phase_delta','target_v','distance','sign','parallel','links']
    return {k:torch.as_tensor(d[k],device=device,dtype=(torch.long if k in ('family','i','j') else torch.float32)) for k in keys}


def audit(d, state, widths, config, indices=None):
    """Vectorized CPU hard readout, full graph at final; no GPU needed."""
    from scipy.special import softmax
    k=len(widths);p=softmax(state[:,:k+1],axis=1);labels=p.argmax(1);keep=labels<k
    pick=np.minimum(labels,k-1);frac=d['phase']+.25*np.tanh(state[np.arange(len(state)),k+1+pick])
    b=np.r_[0,np.cumsum(widths,dtype=np.float64)]
    u=b[pick]+widths[pick]*frac;v=d['z']+1000*state[np.arange(len(state)),2*k+1+pick]
    def point(ids,ds):
        val=u[ids]+(d['family'][ids]==0)*ds
        bucket=np.clip(np.searchsorted(b,val,side='right')-1,0,k-1)
        return bucket+(val-b[bucket])/widths[bucket],v[ids]+(d['family'][ids]==1)*ds
    sel=np.arange(len(d['i'])) if indices is None else indices
    total=known=bad=retained=0;errors=[];bytype={};minq=np.inf;maxq=-np.inf
    for a in range(0,len(sel),131072):
        e=sel[a:a+131072];i=d['i'][e];j=d['j'][e]
        qa,_=point(i,d['ds_a'][e]);qb,_=point(j,d['ds_b'][e]);minq=min(minq,qa.min(),qb.min());maxq=max(maxq,qa.max(),qb.max())
        delta=qb-qa-d['phase_delta'][e]+.5*(d['family'][j]-d['family'][i])
        active=keep[i]&keep[j];ks=(d['sign'][e]!=0)&active
        wrong=(delta*d['sign'][e]<-1e-4)&ks
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
        sign_violation_percent=100*bad/known if known else None,distance_error_quantiles=quantiles(error),
        distance_mean_error=float(error.mean()) if len(error) else None,continuation_error_base_quantiles=quantiles(continuity),
        continuation_links=len(links),retained_continuation_links=retained_links,
        ambiguous_percent=float(100*(p.max(1)<.9).mean()),zero_probability_percent=float(100*(p==0).mean()),
        mean_rejection_probability=float(p[:,-1].mean()),occupied_windings=np.unique(labels[keep]).tolist(),
        boundary_hypothesis_percent=float(100*((labels[keep]==0)|(labels[keep]==k-1)).mean()) if keep.any() else None,
        endpoint_q_range=[float(minq),float(maxq)],by_type=bytype)


def solve(config):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Production U/V solve requires CUDA')
    torch.set_num_threads(4);torch.manual_seed(config.get('seed',0))
    torch.use_deterministic_algorithms(True)
    with np.load(config['tensor_cache']) as archive:d={key:archive[key] for key in archive.files}
    device='cuda:0';data=_tensor_data(d,device)
    n=len(d['family']);k=100;root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    if (root/'summary.json').exists():raise FileExistsError('Completed U/V output exists')
    rng=np.random.default_rng(config.get('seed',0))
    diagnostic=np.sort(rng.choice(len(d['i']),min(100000,len(d['i'])),replace=False))
    # Shared table lookup makes broad Gaussian hypotheses cheap without K*K*R tensors.
    table=torch.from_numpy(make_distance_table(float(d['distance'].max()),config.get('distance_sigma',.5))).to(device)
    state=np.zeros((n,3*k+1),np.float32)
    state[:,:k]=np.maximum(-14.,-.5*((np.arange(k)[None,:]-d['initial_k'][:,None])/config.get('initial_sigma',3.))**2)
    from scipy.special import logsumexp
    # Give rejection a real trainable gradient; -20 would be below Adam epsilon
    # after averaging a minibatch and would effectively disable this alternative.
    state[:,k]=logsumexp(state[:,:k],axis=1)+np.log(.01/.99)
    embedding=torch.nn.Embedding.from_pretrained(torch.from_numpy(state).to(device),freeze=False,sparse=True)
    initial_width=torch.as_tensor(d['width_initial'],device=device)
    raw=torch.nn.Parameter((initial_width-100).clamp_min(1.))
    opt=torch.optim.SparseAdam(embedding.parameters(),lr=config.get('lr',.03))
    widthopt=torch.optim.Adam([raw],lr=config.get('width_lr',2.))
    steps=int(config.get('steps',1000));batch=int(config.get('batch_size',1024));every=int(config.get('log_every',100))
    history=[];started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    baseline=audit(d,state,d['width_initial'],config)
    def coords(ids,ds,widths,bounds):
        s=embedding(ids);p=s[:,:k+1].softmax(-1)
        u,v,q=coordinates(s,data['phase'][ids],data['z'][ids],data['family'][ids],ds,widths,bounds)
        return p,u,v,q
    for step in range(steps+1):
        widths,bounds=width_boundaries(raw)
        if step%every==0 or step==steps:
            torch.cuda.synchronize();snapshot=embedding.weight.detach().cpu().numpy()
            w=widths.detach().cpu().numpy();metrics=audit(d,snapshot,w,config,None if step==steps else diagnostic)
            record=dict(step=step,seconds=time.monotonic()-started,event_samples=step*batch,
                equivalent_event_epochs=step*batch/len(d['i']),width_range=[float(w.min()),float(w.max())],**metrics)
            if step:record.update(loss=float(loss.detach()),pair_loss=float(pair.detach()),regularization=float(reg.detach()))
            history.append(record);atomic_write(root/'history.json',history,durable=False)
            print('UV '+json.dumps(record),flush=True)
            if step==steps:break
        edges=torch.randint(len(d['i']),(batch,),device=device)
        i=data['i'][edges];j=data['j'][edges]
        pa,ua,va,qa=coords(i,data['ds_a'][edges],widths,bounds)
        pb,ub,vb,qb=coords(j,data['ds_b'][edges],widths,bounds)
        pair=pair_objective(pa[:,:k],pb[:,:k],qa,qb,va,vb,data['phase_delta'][edges],data['family'][j]-data['family'][i],
            data['sign'][edges],data['distance'][edges],data['target_v'][edges],table,config)
        for pp,qq,phase_key in ((pa,qa,'phase_a'),(pb,qb,'phase_b')):
            angle=qq-data[phase_key][edges,None];angle=angle-angle.detach().round()
            pair=pair+config.get('prior_weight',.01)*.5*(pp[:,:k]*angle.square()).sum(-1).mean()
        # Uniform node sample prevents degree-biased rejection/prior normalization.
        ni=torch.randint(n,(batch,),device=device);ns=embedding(ni);npb=ns[:,:k+1].softmax(-1)
        degree=data['degree'];cost=.5+.5*degree[ni]/degree.mean().clamp_min(1.)
        reg=config.get('rejection_cost',10.)*(npb[:,k]*cost).mean()
        ramp=min(1.,max(0.,(step/steps-.5)/.3))
        reg=reg+config.get('entropy_weight',.05)*ramp*(-npb*npb.clamp_min(1e-20).log()).sum(-1).mean()
        for point in range(3):
            nu,nv,nq=coordinates(ns,data['phase'][ni],data['z'][ni],data['family'][ni],data['node_ds'][ni,point],widths,bounds)
            angle=nq-data['node_phase'][ni,point,None];angle=angle-angle.detach().round()
            prior=angle.square()+((nv-data['node_z'][ni,point,None])/1000).square()
            prior=prior+torch.relu(-nq).square()+torch.relu(nq-100).square()
            reg=reg+config.get('prior_weight',.01)*(npb[:,:k]*prior).sum(-1).mean()/3
        reg=reg+config.get('width_prior',.1)*((widths/initial_width-1).square().mean())
        reg=reg+config.get('width_smooth',.01)*torch.diff(widths/initial_width).square().mean()
        if len(data['links']):
            links=data['links'][torch.randint(len(data['links']),(min(batch,len(data['links'])),),device=device)]
            ca,cu,cv,_=coords(links[:,0].long(),links[:,2],widths,bounds)
            cb,du,dv,_=coords(links[:,1].long(),links[:,3],widths,bounds)
            reg=reg+config.get('continuation_weight',10.)*continuation_cost(ca[:,:k],cb[:,:k],cu,du,cv,dv,initial_width.mean())
        loss=pair+reg
        opt.zero_grad(set_to_none=True);widthopt.zero_grad(set_to_none=True);loss.backward()
        # Initial assignment settling before opening the rejection alternative.
        grad=embedding.weight.grad.coalesce()
        if step<steps*config.get('rejection_warmup',.5):grad.values()[:,k]=0
        embedding.weight.grad=grad
        opt.step();widthopt.step()
    np.savez(root/'solution.npz',state=snapshot,widths=w,source_id=d['source_id'],start=d['start'],end=d['end'],family=d['family'])
    summary=dict(config=config,input=json.loads(Path(config['tensor_cache']).with_suffix('.json').read_text()),
        baseline=baseline,final=history[-1],gpu=torch.cuda.get_device_name(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        state_layout='logits[101],fraction_adjustment[100],v_adjustment[100]',
        interpretation='Model consistency only, finite sign penalty; fixed arc maps, sampled Adam; no anatomical GT')
    atomic_write(root/'summary.json',summary,durable=False)


def run(config):
    prepare(config)
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
        if s['state'].shape!=(len(d['family']),301) or s['widths'].shape!=(100,):raise ValueError('State shape')
        signature=summary['input']['signature']
        for key in ('database','features','stage1','hv_solution','afv','umbilicus'):
            if sha256_file(cfg[key])!=signature[key]:raise ValueError('Changed input: '+key)
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
        for key in ('rejected_pieces','retained_events','sign_violations'):
            if metrics[key]!=summary['final'][key]:raise ValueError('Recomputed readout mismatch: '+key)
        from scipy.special import softmax
        probs=softmax(s['state'][:,:101],axis=1)
        if not np.allclose(probs.sum(1),1,atol=1e-6):raise ValueError('Unnormalized probabilities')
        row=dict(run=str(root),solution_sha256=sha256_file(root/'solution.npz'),tensor_cache_sha256=sha256_file(cfg['tensor_cache']),
            partition_audit=dict(fibers=len(included),pieces=len(d['family']),split_fibers=split_fibers,events=len(events),
                                 continuations=len(expected_links),no_observations_removed_by_splitting=True),
            baseline=summary['baseline'],final=metrics,seconds=summary['final']['seconds'],
            equivalent_epochs=summary['final']['equivalent_event_epochs'],peak_gpu_gib=summary['peak_allocated_bytes']/1024**3)
        reports.append(row);atomic_write(root/'audit.json',row,durable=False)
    destination=Path(config['report']);lines=['# Global U/V layout: first experiments','',
        'All earlier rejections fixed; baked H/V. Full-graph readout, no ground-truth layout labels.',
        'Ordering errors use retained known-sign events (tolerance1e-4winding); finite penalty, not hard feasibility.',
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
            f"Final per-type: {r['final']['by_type']}",f"Continuation links retained: {r['final']['retained_continuation_links']}/{r['final']['continuation_links']}; median/p95/p99(basevx): {r['final']['continuation_error_base_quantiles']}",
            f"Numerically zero probabilities: {r['final']['zero_probability_percent']:.6f}%; boundary hypotheses: {r['final']['boundary_hypothesis_percent']}%."]
    lines+=details
    destination.write_text('\n'.join(lines)+'\n');atomic_write(destination.with_suffix('.json'),reports,durable=False)
    print('\n'.join(lines),flush=True);return dict(report=str(destination),runs=len(reports))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--unmanaged',action='store_true');parser.add_argument('--devices')
    args=parser.parse_args()
    if not args.unmanaged:raise RuntimeError('Use the approved managed runner')
    solve(json.loads(args.config.read_text()))
