"""Fine UV deformation at distinct constraint endpoints, in base voxels."""
import math
import numpy as np

# Arc refinement can report the same point with double-roundoff differences.
# Preserve distinct nodes and original arcs; only their coupling distance changes.
COINCIDENT_ARC_EPS=1e-8


def validate(config):
    if not config.get('event_uv_offsets',False):return
    if (not config.get('bucket_shared_phase') or not config.get('bucket_position_split') or
            config.get('layout_mode')!='shared_anchor' or config.get('warm_start') or
            config.get('update_mode','minibatch')!='minibatch' or
            config.get('assignment_update')=='cost_target' or config.get('benchmark_pair_fusion')):
        raise ValueError('Event offsets require fresh joint shared-phase minibatch solve')
    for key,default in [('event_offset_longitudinal_weight',100.),
                        ('event_offset_transverse_weight',100.),
                        ('event_offset_anchor_weight',.01),('event_offset_lr',.001)]:
        value=float(config.get(key,default))
        if not math.isfinite(value) or value<=0:raise ValueError('Positive finite '+key+' required')


def adjacency(source,arc,family,tie_distance=0.):
    """Endpoint order is A[0:E],B[0:E]; do not merge coincident observations."""
    source=np.asarray(source);arc=np.asarray(arc,dtype=np.float64);family=np.asarray(family)
    if source.shape!=arc.shape or family.shape!=arc.shape or not np.isfinite(arc).all():
        raise ValueError('Invalid endpoint metadata')
    order=np.lexsort((np.arange(len(source)),arc,source))
    left,right=order[:-1],order[1:]
    same=source[left]==source[right];left,right=left[same],right[same]
    if np.any(family[left]!=family[right]):raise ValueError('Family changes within source fiber')
    gap=arc[right]-arc[left]
    if not math.isfinite(tie_distance) or tie_distance<0:raise ValueError('Invalid tie distance')
    new_group=np.ones(len(order),dtype=bool)
    if len(order)>1:
        new_group[1:]=(source[order[1:]]!=source[order[:-1]]) | (arc[order[1:]]-arc[order[:-1]]>=tie_distance)
    parameter_index=np.empty(len(order),dtype=np.int64)
    parameter_index[order]=np.cumsum(new_group)-1
    if tie_distance==0:parameter_index=np.arange(len(order),dtype=np.int64)
    return dict(source=source,arc=arc,family=family,left=left,right=right,gap=gap,
                parameter_index=parameter_index,parameter_count=int(new_group.sum()),tie_distance=tie_distance)


def prepare(d,config):
    from .fiber_uv_layout import _ro
    ids=np.asarray(d['event_ids'],dtype=np.int64)
    records={}
    with _ro(config['database']) as db:
        for a in range(0,len(ids),500):
            batch=ids[a:a+500].tolist()
            query='SELECT id,fiber_a,fiber_b,arc_a,arc_b FROM constraints WHERE id IN ('+','.join('?' for _ in batch)+')'
            records.update({int(r[0]):r[1:] for r in db.execute(query,batch)})
    if len(records)!=len(ids):raise ValueError('Missing/duplicate constraint endpoint IDs')
    rows=np.asarray([records[int(i)] for i in ids])
    owners=np.r_[d['i'],d['j']]
    source=d['source_id'][owners]
    if not np.array_equal(rows[:,:2].T.reshape(-1),source):
        raise ValueError('Endpoint source order does not match cached constraints')
    arcs=rows[:,2:].T.reshape(-1)
    if np.any(arcs<d['start'][owners]-1e-5) or np.any(arcs>d['end'][owners]+1e-5):
        raise ValueError('Endpoint arc outside owning piece')
    event_nodes=len(owners)
    links=np.asarray(d.get('links',[])).reshape(-1,4)
    if len(links):
        a,b=links[:,:2].astype(np.int64).T
        if not np.array_equal(d['source_id'][a],d['source_id'][b]):
            raise ValueError('Continuation does not join one original fiber')
        if not np.allclose(d['end'][a],d['start'][b],rtol=0,atol=1e-5):
            raise ValueError('Continuation cut arc mismatch')
        owners=np.r_[owners,a,b]
        arcs=np.r_[arcs,d['end'][a],d['start'][b]]
        source=d['source_id'][owners]
    result=adjacency(source,arcs,d['family'][owners],float(config.get('event_offset_tie_distance_base',4.)))
    result.update(piece=owners,kind=np.r_[np.zeros(event_nodes,dtype=np.int64),np.ones(2*len(links),dtype=np.int64)],
        constraint_id=np.r_[ids,ids,np.arange(len(links)),np.arange(len(links))],
        side=np.r_[np.zeros(len(ids),dtype=np.int64),np.ones(len(ids),dtype=np.int64),
                   np.zeros(len(links),dtype=np.int64),np.ones(len(links),dtype=np.int64)])
    return result


def tensor_adjacency(meta,device):
    import torch
    return {key:torch.as_tensor(meta[key],device=device,
            dtype=torch.long if key in ('left','right','family','parameter_index') else torch.float32)
            for key in ('left','right','family','gap','parameter_index')}


def regularization(parameters,topology,config):
    """Integral squared slope / covered arc; absolute anchor in 1000vx units."""
    expanded=parameters[topology['parameter_index']]
    physical=1000*expanded
    left,right,gap=topology['left'],topology['right'],topology['gap']
    delta=physical[right]-physical[left]
    family=topology['family'][left]
    longitudinal=delta.gather(1,family[:,None]).squeeze(1)
    transverse=delta.gather(1,(1-family)[:,None]).squeeze(1)
    # Numerically coincident observations use a finite 1vx coupling distance.
    denominator=gap.where(gap>COINCIDENT_ARC_EPS,gap.new_ones(()))
    total=gap.sum().clamp_min(1.)
    along=(longitudinal.square()/denominator).sum()/total
    across=(transverse.square()/denominator).sum()/total
    anchor=expanded.square().mean()
    loss=(config.get('event_offset_longitudinal_weight',100.)*along+
          config.get('event_offset_transverse_weight',100.)*across+
          config.get('event_offset_anchor_weight',.01)*anchor)
    return loss,along,across,anchor


def diagnostics(physical,meta):
    delta=physical[meta['right']]-physical[meta['left']]
    gap=meta['gap'];family=meta['family'][meta['left']]
    along=delta[np.arange(len(delta)),family]
    across=delta[np.arange(len(delta)),1-family]
    denominator=np.where(gap>COINCIDENT_ARC_EPS,gap,1.)
    def quantiles(x):return np.quantile(np.abs(x),[.5,.95,1]).tolist() if len(x) else None
    tied=meta['parameter_index'][meta['left']]==meta['parameter_index'][meta['right']]
    idx=meta['parameter_index'];count=meta['parameter_count']
    lo=np.full(count,np.inf);hi=np.full(count,-np.inf)
    np.minimum.at(lo,idx,meta['arc']);np.maximum.at(hi,idx,meta['arc'])
    return dict(nodes=len(physical),parameter_nodes=meta['parameter_count'],
        group_size_q50_q95_max=quantiles(np.bincount(idx,minlength=count)),
        group_span_base_q50_q95_max=quantiles(hi-lo),
        tie_distance_base=meta['tie_distance'],tied_edges=int(tied.sum()),
        tied_delta_max=float(np.max(np.abs(delta[tied]))) if tied.any() else 0.,
        rms_base=float(np.sqrt(np.mean(physical**2))),
        abs_base_q50_q95_max=quantiles(physical),
        longitudinal_slope_q50_q95_max=quantiles(along/denominator),
        transverse_slope_q50_q95_max=quantiles(across/denominator),
        coincident_edges=int((gap<=COINCIDENT_ARC_EPS).sum()),
        roundoff_coincident_edges=int(((gap>0)&(gap<=COINCIDENT_ARC_EPS)).sum()),
        coincident_arc_tolerance_base=COINCIDENT_ARC_EPS,
        coincident_delta_q50_q95_max=quantiles(delta[gap<=COINCIDENT_ARC_EPS]),
        longitudinal_energy=float(np.sum(along**2/denominator)/max(gap.sum(),1)),
        transverse_energy=float(np.sum(across**2/denominator)/max(gap.sum(),1)),
        anchor_energy=float(np.mean((physical/1000)**2)),
        note='UV offsets in base voxels; coarse GT/sign metrics exclude offsets')
