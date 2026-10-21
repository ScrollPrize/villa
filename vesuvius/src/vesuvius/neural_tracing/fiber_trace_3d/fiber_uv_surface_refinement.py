"""Optional fixed-assignment UV geometry refinement; parent solve is immutable."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import fiber_uv_layout as uv
from . import fiber_uv_event_offsets as eo
from .fiber_normal_sign import launch_tensor_solve,sha256_file
from lasagna.inference_provenance import atomic_write


def select(d,state,widths,config,threshold=.5):
    if not .5<=threshold<=1:raise ValueError('Bucket probability threshold must be in [0.5,1]')
    geometry=uv.hard_geometry(d,state,widths,config,best_nonbroken=True)
    p,_,keep,pick,*_=geometry
    confidence=p[np.arange(len(p)),pick]
    confident=keep&(confidence>=threshold)
    _,_,delta,_=uv.edge_readout(d,geometry,widths,np.arange(len(d['i'])),config,state)
    if config.get('arc_phase_constraints',False):delta=np.rint(delta)
    threshold=0.
    if config.get('integer_sheet_constraints',False):
        from .fiber_uv_integer_sheets import margin
        threshold=margin(d['sign'],d['family'][d['j']]-d['family'][d['i']],d['parallel'])
    bad=(confident[d['i']]&confident[d['j']]&(d['sign']!=0)&
         (delta*d['sign']-threshold < -config.get('sign_tolerance',1e-4)))
    removed=np.unique(np.r_[d['source_id'][d['i'][bad]],d['source_id'][d['j'][bad]]])
    retained=confident&~np.isin(d['source_id'],removed)
    edges=retained[d['i']]&retained[d['j']]
    fd=d['family'][d['j']]-d['family'][d['i']]
    sheet_delta=delta if config.get('integer_sheet_constraints',False) else delta-.5*fd
    same=edges&~d['parallel']&(fd!=0)&(abs(sheet_delta)<.5)
    links=d['links'];li=links[:,:2].astype(np.int64)
    link_mask=retained[li].all(1)
    return dict(retained=retained,confident=confident,confidence=confidence,pick=pick,
        removed_sources=removed,bad_edges=bad,edges=edges,same_sheet=same,
        sheet_delta=sheet_delta,link_mask=link_mask,geometry=geometry)


def initialize(d,s,parent,config):
    selection=select(d,s['state'],s['widths'],parent,config.get('minimum_bucket_probability',.5))
    piece=np.flatnonzero(selection['retained']);edges=np.flatnonzero(selection['edges'])
    links=np.flatnonzero(selection['link_mask'])
    if not len(piece) or not selection['same_sheet'].any():raise ValueError('No retained same-sheet H/V matches')
    sub=dict(d)
    for key in ('i','j','event_ids'):sub[key]=d[key][edges]
    sub['links']=d['links'][links]
    meta=eo.prepare(sub,parent)
    meta['constraint_id'][meta['kind']==1]=np.r_[links,links]
    old_e=len(d['i']);old_l=len(d['links'])
    original_nodes=np.r_[edges,edges+old_e,2*old_e+links,2*old_e+old_l+links]
    seed=np.zeros((len(meta['arc']),2),np.float64)
    if parent.get('event_uv_offsets') and 'event_uv_offsets_base' not in s:
        raise ValueError('Parent is missing learned endpoint offsets')
    if 'event_uv_offsets_base' in s:
        if not np.array_equal(s['event_ids'],d['event_ids']):raise ValueError('Parent event identity mismatch')
        for field in ('source','arc','piece','kind','constraint_id','side'):
            if not np.array_equal(s['event_endpoint_'+field][original_nodes],meta[field]):
                raise ValueError('Parent endpoint identity mismatch: '+field)
        seed=s['event_uv_offsets_base'][original_nodes].astype(np.float64)
    idx=meta['parameter_index'];groups=meta['parameter_count']
    grouped=np.zeros((groups,2));np.add.at(grouped,idx,seed)
    grouped/=np.bincount(idx,minlength=groups)[:,None]
    if not np.allclose(grouped[idx],seed,rtol=0,atol=1e-5):raise ValueError('Parent hard-tied offsets disagree')
    local=np.full(len(d['family']),-1,np.int64);local[piece]=np.arange(len(piece))
    meta['piece']=local[meta['piece']]
    _,_,_,_,u,v,_=selection['geometry'];anchors=np.column_stack([u[piece],v[piece]])
    ds=np.r_[d['ds_a'][edges],d['ds_b'][edges],d['links'][links,2],d['links'][links,3]]
    along=np.column_stack([meta['family']==0,meta['family']==1])*ds[:,None]
    base=anchors[meta['piece']]+along
    return dict(selection=selection,piece=piece,edges=edges,links=links,meta=meta,
        grouped=grouped,anchors=anchors,base=base,ds=ds,seed=seed)


def build_system(model,config):
    """Sparse affine residuals in physical base voxels; no all-bucket tensors."""
    from scipy import sparse
    meta=model['meta'];n=len(model['piece']);g=meta['parameter_count'];size=n+g
    e=len(model['edges']);l=len(model['links']);idx=meta['parameter_index']
    same=model['selection']['same_sheet'][model['edges']]
    weights=dict(match=config.get('match_weight',1000.),approximate=config.get('approximate_weight',.01),
        continuation=config.get('continuation_weight',1000.),piece_anchor=config.get('piece_anchor_weight',.01),
        longitudinal=config.get('event_offset_longitudinal_weight',100.),
        transverse=config.get('event_offset_transverse_weight',100.),offset_anchor=config.get('event_offset_anchor_weight',.01))
    if any(not np.isfinite(v) or v<=0 for v in weights.values()):raise ValueError('Refinement weights must be positive finite')
    scale=float(config.get('match_scale_base',128.))
    if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid match scale')
    groups=[];targets=[];components=[]
    for axis in range(2):
        blocks=[];rhs=[];labels=[]
        def add(name,cols,coeff,target):
            target=np.asarray(target,dtype=np.float64);count=len(target)
            if not count:return
            rows=np.repeat(np.arange(count),cols.shape[1])
            matrix=sparse.coo_matrix((np.broadcast_to(coeff,cols.shape).ravel(),(rows,cols.ravel())),shape=(count,size)).tocsr()
            matrix.sum_duplicates();matrix.eliminate_zeros()
            blocks.append(matrix);rhs.append(target);labels.extend([name]*count)
        def pair(name,a,b,target,weight):
            count=len(a)
            if not count:return
            factor=np.sqrt(weight/count)/scale
            cols=np.column_stack([meta['piece'][a],meta['piece'][b],n+idx[a],n+idx[b]])
            add(name,cols,factor*np.array([-1,1,-1,1]),factor*target)
        a=np.arange(e);b=a+e
        delta=model['base'][b,axis]-model['base'][a,axis]
        pair('same_sheet',a[same],b[same],-delta[same],weights['match'])
        # Other constraints preserve the main solve's approximate relative layout.
        target=model['seed'][b,axis]-model['seed'][a,axis]
        pair('approximate',a[~same],b[~same],target[~same],weights['approximate'])
        ca=2*e+np.arange(l);cb=ca+l
        pair('continuation',ca,cb,model['base'][ca,axis]-model['base'][cb,axis],weights['continuation'])
        left,right,gap=meta['left'],meta['right'],meta['gap']
        active=idx[left]!=idx[right];left=left[active];right=right[active]
        factor=np.sqrt(np.where(meta['family'][left]==axis,weights['longitudinal'],weights['transverse'])/
            (max(meta['gap'].sum(),1.)*np.maximum(gap[active],eo.COINCIDENT_ARC_EPS)))
        add('offset_slope',np.column_stack([n+idx[left],n+idx[right]]),factor[:,None]*np.array([-1,1]),np.zeros(len(left)))
        add('offset_anchor',(n+idx)[:,None],np.sqrt(weights['offset_anchor']/(2*len(idx)))/1000,np.zeros(len(idx)))
        add('piece_anchor',np.arange(n)[:,None],np.sqrt(weights['piece_anchor']/(2*n))/1000,np.zeros(n))
        groups.append(sparse.vstack(blocks,format='csr'));targets.append(np.concatenate(rhs));components.append(np.asarray(labels))
    matrix=sparse.block_diag(groups,format='csr');target=np.concatenate(targets)
    initial=np.vstack([np.zeros((n,2)),model['grouped']]).T.ravel()
    return matrix,target,initial,np.concatenate(components)


def pcg(matrix,target,initial,device='cpu',rtol=1e-8,max_iterations=10000,report=None):
    """Jacobi-preconditioned SPD least squares, sparse float64 PyTorch matvecs."""
    import torch
    if rtol<=0 or max_iterations<1:raise ValueError('Invalid CG limits')
    h=(matrix.T@matrix).tocsr();h.sum_duplicates();h.eliminate_zeros()
    diagonal=h.diagonal()
    if np.any(diagonal<=0):raise ValueError('Unanchored refinement parameter')
    def tensor(x):return torch.as_tensor(x,device=device,dtype=torch.float64)
    op=torch.sparse_csr_tensor(torch.as_tensor(h.indptr,device=device),torch.as_tensor(h.indices,device=device),tensor(h.data),size=h.shape)
    def mv(x):return torch.sparse.mm(op,x[:,None])[:,0]
    b=tensor(matrix.T@target);x=tensor(initial).clone();inv=1/tensor(diagonal)
    r=b-mv(x);z=inv*r;p=z.clone();rz=r.dot(z)
    norm=max(float(torch.linalg.vector_norm(b)),1e-30);initial_relative=float(torch.linalg.vector_norm(r))/norm
    if initial_relative<=rtol:return x.cpu().numpy(),dict(iterations=0,converged=True,relative_residual=initial_relative)
    relative=initial_relative
    for step in range(1,max_iterations+1):
        hp=mv(p);alpha=rz/p.dot(hp).clamp_min(1e-300)
        x=x+alpha*p;r=r-alpha*hp
        z=inv*r;next_rz=r.dot(z);p=z+(next_rz/rz.clamp_min(1e-300))*p;rz=next_rz
        if step%25==0 or step==max_iterations:
            relative=float(torch.linalg.vector_norm(b-mv(x)))/norm
            if report and (step%500==0 or relative<=rtol):report(step,relative)
            if not np.isfinite(relative):raise ValueError('Nonfinite CG residual')
            if relative<=rtol:break
    return x.cpu().numpy(),dict(iterations=step,converged=relative<=rtol,relative_residual=relative,
        initial_relative_residual=initial_relative,normal_nnz=int(h.nnz))


def mismatch(points,model,reference_points=None):
    e=len(model['edges']);delta=points[e:2*e]-points[:e]
    same=model['selection']['same_sheet'][model['edges']]
    def stats(x):
        if not len(x):return dict(events=0)
        def values(y):return dict(median=float(np.median(abs(y))),rms=float(np.sqrt(np.mean(y*y))),
            p95=float(np.quantile(abs(y),.95)),maximum=float(np.max(abs(y))))
        return dict(events=len(x),u=values(x[:,0]),v=values(x[:,1]),uv=values(np.linalg.norm(x,axis=1)))
    out=dict(same_sheet_perpendicular=stats(delta[same]),other_relative_layout=stats(delta[~same]))
    if reference_points is not None:
        baseline=reference_points[e:2*e]-reference_points[:e]
        out['other_change_from_parent']=stats((delta-baseline)[~same])
    return out


def evaluate_points(solution,pieces,arcs,signed_ds):
    """Refined curve evaluator: local retained-piece indices, original true arcs."""
    pieces=np.asarray(pieces,dtype=np.int64);arcs=np.asarray(arcs);signed_ds=np.asarray(signed_ds)
    if pieces.shape!=arcs.shape or arcs.shape!=signed_ds.shape:raise ValueError('Query shape mismatch')
    if np.any(pieces<0) or np.any(pieces>=len(solution['source_id'])):raise ValueError('Unknown retained piece')
    out=solution['piece_uv_base'][pieces].copy()
    out+=np.column_stack([solution['family'][pieces]==0,solution['family'][pieces]==1])*signed_ds[:,None]
    source=solution['endpoint_source'];arc=solution['endpoint_arc'];offset=solution['endpoint_uv_offset_base']
    query=solution['source_id'][pieces]
    if not len(source):return out
    order=np.lexsort((arc,source));source=source[order];arc=arc[order];offset=offset[order]
    ids,first,counts=np.unique(source,return_index=True,return_counts=True)
    last=first+counts-1;lo=arc[first];hi=arc[last]
    base=np.r_[0.,np.cumsum((hi-lo+1.)[:-1])]
    keys=arc-np.repeat(lo,counts)+np.repeat(base,counts)
    loc=np.searchsorted(ids,query);valid=loc<len(ids)
    loc=loc.clip(0,len(ids)-1);valid&=ids[loc]==query
    q=(np.clip(arcs[valid],lo[loc[valid]],hi[loc[valid]])-lo[loc[valid]]+base[loc[valid]])
    for axis in range(2):out[valid,axis]+=np.interp(q,keys,offset[:,axis])
    return out


def solve(config):
    import torch
    started=time.monotonic();root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    if (root/'summary.json').exists():raise FileExistsError('Completed refinement exists')
    if not torch.cuda.is_available():raise RuntimeError('Production refinement requires CUDA')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    parent_path=Path(config['parent_solution']);parent=json.loads((parent_path.parent/'config.json').read_text())
    if not parent.get('bucket_shared_phase') or not parent.get('bucket_position_split'):raise ValueError('Unsupported parent formulation')
    paths=[parent_path,parent_path.parent/'config.json',Path(parent['tensor_cache'])]
    hashes={str(p):sha256_file(p) for p in paths}
    with np.load(parent['tensor_cache']) as f:d={key:f[key] for key in f.files}
    with np.load(parent_path) as f:s={key:f[key] for key in f.files}
    for key in ('source_id','start','end','family'):
        if not np.array_equal(s[key],d[key]):raise ValueError('Parent piece identity mismatch: '+key)
    model=initialize(d,s,parent,config);matrix,target,initial,components=build_system(model,config)
    meta=model['meta'];n=len(model['piece']);g=meta['parameter_count'];sel=model['selection']
    def points(x):
        x=x.reshape(2,n+g).T
        return model['base']+x[:n][meta['piece']]+x[n:][meta['parameter_index']]
    def objective(x):
        error=np.square(matrix@x-target)
        return {name:float(error[components==name].sum()) for name in np.unique(components)}
    before=mismatch(points(initial),model)
    print('REFINE FILTER '+json.dumps(dict(pieces=n,sources=len(np.unique(d['source_id'][model['piece']])),
        removed_sources=len(sel['removed_sources']),sign_violations=int(sel['bad_edges'].sum()),
        same_sheet=int(sel['same_sheet'].sum()),before=before)),flush=True)
    x,solver=pcg(matrix,target,initial,'cuda',config.get('cg_rtol',1e-8),config.get('cg_max_iterations',100000),
        lambda step,residual:print('REFINE CG '+json.dumps(dict(step=step,relative_residual=residual)),flush=True))
    after=mismatch(points(x),model,points(initial));parameters=x.reshape(2,n+g).T;piece=model['piece']
    output=dict(format=np.array('fiber-uv-refinement-v1'),source_id=d['source_id'][piece],original_piece_index=piece,
        start=d['start'][piece],end=d['end'][piece],family=d['family'][piece],
        bucket=sel['pick'][piece],bucket_probability=sel['confidence'][piece],
        coarse_shared_phase=s['state'][piece,len(s['widths'])+1],anchor_phase=d['phase'][piece],widths=s['widths'],
        piece_uv_base=model['anchors']+parameters[:n],piece_delta_uv_base=parameters[:n],
        endpoint_source=meta['source'],endpoint_arc=meta['arc'],endpoint_piece=meta['piece'],endpoint_ds=model['ds'],
        endpoint_kind=meta['kind'],endpoint_constraint_id=meta['constraint_id'],endpoint_side=meta['side'],
        endpoint_parameter_index=meta['parameter_index'],endpoint_uv_offset_base=parameters[n:][meta['parameter_index']],
        original_event_index=model['edges'],event_ids=d['event_ids'][model['edges']],
        same_sheet_perpendicular=sel['same_sheet'][model['edges']],removed_source_ids=sel['removed_sources'],
        event_coarse_sheet_delta=sel['sheet_delta'][model['edges']],
        removed_sign_violation_event_ids=d['event_ids'][sel['bad_edges']],
        parent_solution=np.array(str(parent_path)),original_retained_mask=sel['retained'])
    reconstructed=evaluate_points(output,meta['piece'],meta['arc'],model['ds'])
    np.testing.assert_allclose(reconstructed,points(x),rtol=0,atol=1e-7)
    for path,digest in hashes.items():
        if sha256_file(path)!=digest:raise ValueError('Parent changed during refinement: '+path)
    np.savez(root/'solution.npz',**output)
    summary=dict(format='fiber-uv-refinement-v1',config=config,parent_hashes=hashes,solver=solver,
        input_pieces=len(d['family']),confident_pieces=int(sel['confident'].sum()),retained_pieces=n,
        retained_sources=len(np.unique(output['source_id'])),removed_source_ids=sel['removed_sources'].tolist(),
        sign_violation_edges=int(sel['bad_edges'].sum()),retained_events=len(model['edges']),
        same_sheet_perpendicular_events=int(sel['same_sheet'].sum()),retained_continuations=len(model['links']),
        before=before,after=after,objective_before=objective(initial),objective_after=objective(x),
        piece_displacement_rms_base=float(np.sqrt(np.mean(parameters[:n]**2))),
        offsets=eo.diagnostics(output['endpoint_uv_offset_base'],meta),
        solve_wall_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        fixed_assignments=True,disablement_possible=False)
    atomic_write(root/'summary.json',summary,durable=False)
    print('REFINE COMPLETE '+json.dumps(summary),flush=True)
    return summary


def run(config):
    return launch_tensor_solve(config,__name__,Path(__file__))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--unmanaged',action='store_true');parser.add_argument('--devices')
    args=parser.parse_args()
    if not args.unmanaged:raise RuntimeError('Use the approved managed evaluation runner')
    solve(json.loads(args.config.read_text()))
