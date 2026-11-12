"""Vectorized exact fixed-label geometry priors from shared packed paths."""
import numpy as np


def geometry_priors(paths, ids, family, spacing=1000., batch_fibers=4096, batch_samples=65536):
    """Match the original per-polyline UV priors without source-store rereads.

    Radius medians use bounded fiber batches and segmented sorting. All sample
    positions are queried together through the existing packed interpolator.
    No reference labels or geometric approximations enter this preparation.
    """
    if spacing<=0 or batch_fibers<1 or batch_samples<1:raise ValueError('Invalid geometry sampling settings')
    ids=np.asarray(ids);n=len(ids);indices=np.searchsorted(paths.ids,ids)
    if np.any(indices>=len(paths.ids)) or not np.array_equal(paths.ids[indices],ids):
        raise ValueError('Unknown packed geometry fiber')
    begin=paths.offset[indices];end=paths.offset[indices+1]
    first=np.asarray(paths.xyz[begin]);last=np.asarray(paths.xyz[end-1])
    rel=paths.seam.relative(first)
    phase=((np.arctan2(rel[:,1],rel[:,0])-paths.seam.angle)%(2*np.pi))/(2*np.pi)
    tail=paths.seam.relative(last)
    tail_phase=((np.arctan2(tail[:,1],tail[:,0])-paths.seam.angle)%(2*np.pi))/(2*np.pi)
    directions=np.sign(np.where(np.asarray(family)==0,paths.turns[end-1]+tail_phase-phase,last[:,2]-first[:,2]))
    if np.any(directions==0):raise ValueError('Unidentifiable fiber direction')
    radii=np.empty(n);minimum=np.empty(n,np.int64);maximum=np.empty(n,np.int64)
    for start in range(0,n,batch_fibers):
        stop=min(n,start+batch_fibers);sizes=end[start:stop]-begin[start:stop]
        offsets=np.r_[0,np.cumsum(sizes)];owner=np.repeat(np.arange(stop-start),sizes)
        point=begin[start:stop][owner]+np.arange(offsets[-1])-offsets[owner]
        xyz=paths.seam.relative(paths.xyz[point]);radius=np.linalg.norm(xyz,axis=1)
        ordered=radius[np.lexsort((radius,owner))]
        lower=offsets[:-1]+(sizes-1)//2;upper=offsets[:-1]+sizes//2
        radii[start:stop]=(ordered[lower]+ordered[upper])*.5
        turns=np.asarray(paths.turns[point])
        minimum[start:stop]=np.minimum.reduceat(turns,offsets[:-1])
        maximum[start:stop]=np.maximum.reduceat(turns,offsets[:-1])
    lengths=np.asarray(paths.keys['arc'][end-1])
    count=np.maximum(2,np.ceil(lengths/spacing).astype(np.int64)+1)
    offsets=np.r_[0,np.cumsum(count)];owner=np.repeat(np.arange(n),count)
    local=np.arange(offsets[-1])-offsets[owner]
    arc=lengths[owner]*local/(count[owner]-1)
    # np.linspace fixes endpoints explicitly; preserve that for arc validation.
    arc[local==count[owner]-1]=lengths[owner[local==count[owner]-1]]
    sample_phase=np.empty(len(arc));sample_z=np.empty(len(arc))
    for start in range(0,len(arc),batch_samples):
        end=min(start+batch_samples,len(arc))
        position,_=paths.at_many(ids[owner[start:end]],arc[start:end])
        relative=paths.seam.relative(position)
        sample_phase[start:end]=((np.arctan2(relative[:,1],relative[:,0])-paths.seam.angle)%(2*np.pi))/(2*np.pi)
        sample_z[start:end]=position[:,2]
    return dict(directions=directions,phase=phase,z=first[:,2],radii=radii,
                sample_owner=owner,sample_arc=arc,sample_phase=sample_phase,sample_z=sample_z,
                minimum_turn=minimum,maximum_turn=maximum)


def fixed_targets(paths, data, labels, batch_size=65536):
    """Evaluate endpoint transport in bounded batches, never all-edge UUID arrays."""
    if batch_size<1:raise ValueError('Invalid endpoint batch size')
    target=np.empty(len(data['i']),dtype=np.result_type(labels,np.int64))
    for begin in range(0,len(target),batch_size):
        sl=slice(begin,begin+batch_size);i,j=data['i'][sl],data['j'][sl]
        a,ta=paths.at_many(data['source_id'][i],data['ds_a'][sl])
        b,tb=paths.at_many(data['source_id'][j],data['ds_b'][sl])
        target[sl]=labels[j]-labels[i]+tb-ta-paths.seam.segments(a,b)
    return target
