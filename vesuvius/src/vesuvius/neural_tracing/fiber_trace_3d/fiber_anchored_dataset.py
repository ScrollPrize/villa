"""Broad surface selection, fixed manual-reference density solve, curated export."""
import json
from pathlib import Path
import time
import numpy as np
from lasagna.inference_provenance import atomic_write, sha256_file
from . import fiber_uv_layout as uv
from .fiber_gt_constraint_score import arrays


def curve_samples(xyz, spacing=64.):
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
    s=np.unique(np.r_[arc,np.linspace(0,arc[-1],max(2,int(np.ceil(arc[-1]/spacing))+1))])
    return s,np.column_stack([np.interp(s,arc,xyz[:,a]) for a in range(3)])


def induced_graph(d, sources, config):
    keep=np.isin(d['source_id'],sources); parents=np.flatnonzero(keep)
    remap=np.full(len(keep),-1,np.int64);remap[parents]=np.arange(len(parents))
    edge=keep[d['i']]&keep[d['j']]
    piece_keys=('source_index','source_id','family','initial_k','start','end','phase','z','radius',
                'lengths','node_ds','node_phase','node_z')
    out={k:d[k][parents].copy() for k in piece_keys}
    for k in ('ids','earlier_labels','width_initial'):out[k]=d[k].copy()
    for k in ('ds_a','ds_b','phase_a','phase_b','phase_delta','distance','sign','parallel','event_ids'):
        out[k]=d[k][edge].copy()
    out.update(i=remap[d['i'][edge]],j=remap[d['j'][edge]],parent_piece=parents)
    links=d['links'];lk=keep[links[:,0].astype(int)]&keep[links[:,1].astype(int)]
    out['links']=links[lk].copy();out['links'][:,:2]=remap[out['links'][:,:2].astype(int)]
    out['degree']=np.bincount(np.r_[out['i'],out['j']],minlength=len(parents)).astype(np.float32)
    centers,widths=uv.radial_initial_geometry(d,config)
    out.update(radial_initial_centers=centers[parents],radial_initial_widths=widths)
    return out


def one_hop(d, seeds):
    seed=np.isin(d['source_id'],seeds)
    edge=seed[d['i']]|seed[d['j']]
    return np.unique(np.r_[seeds,d['source_id'][d['i'][edge]],d['source_id'][d['j'][edge]]])


def perpendicular_supported_graph(d, config):
    """Induce the graph on fibers with actual compatible perpendicular events.

    No connectivity/length cut and no invented endpoint geometry. Removing these
    fibers cannot orphan a perpendicular event: both of its owners are retained.
    """
    perp=~np.asarray(d['parallel'],bool)
    perp &= d['family'][d['i']]!=d['family'][d['j']]
    owners=np.unique(np.r_[d['i'][perp],d['j'][perp]])
    keep=np.zeros(len(d['family']),bool);keep[owners]=True
    if not keep.any():raise ValueError('No fibers with perpendicular nodes remain')
    removed_edges=~(keep[d['i']]&keep[d['j']])
    stats=dict(input_fibers=len(keep),retained_fibers=int(keep.sum()),
        removed_source_ids=d['source_id'][~keep].tolist(),
        removed_constraints=int(removed_edges.sum()),
        removed_perpendicular_constraints=int((removed_edges&perp).sum()))
    return (d if keep.all() else induced_graph(d,d['source_id'][keep],config)),stats


def anchored_mask(d, reference_owner, active=None):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    a=np.r_[d['i'],d['links'][:,0]].astype(int);b=np.r_[d['j'],d['links'][:,1]].astype(int)
    active=np.ones(len(d['family']),bool) if active is None else np.asarray(active,bool)
    valid=active[a]&active[b];a=a[valid];b=b[valid]
    graph=coo_matrix((np.ones(len(a)),(a,b)),shape=(len(d['family']),)*2)
    _,labels=connected_components(graph,directed=False)
    seeds=np.asarray(reference_owner,dtype=int);seeds=seeds[active[seeds]]
    return active&np.isin(labels,labels[seeds])


def fixed_reference_endpoints(d, events, gd, gs, gcfg, shift=0):
    """Evaluate immutable whole GT curves using their own fitted geometry."""
    import torch
    from .fiber_uv_arc_phase import event_signed_arcs
    _, _, keep, pick = uv.hard_assignment(gs['state'],len(gs['widths']),gcfg,best_nonbroken=True)
    if not keep.all(): raise ValueError('All manual reference fibers must be active')
    _,_,_,_,gu,gv,gb=uv.hard_geometry(gd,gs['state'],gs['widths'],gcfg)
    gi=np.array([e['gt_piece'] for e in events]); arcs=np.array([e['gt_arc'] for e in events])
    q,v=uv.geometry_points(gd,gu,gv,gb,gs['widths'],gi,event_signed_arcs(gd,gi,arcs))
    widths=torch.as_tensor(gs['widths'],dtype=torch.float64)
    bounds=torch.cat((widths.new_zeros(1),widths.cumsum(0)))
    physical_u=uv.q_to_u(torch.as_tensor(q,dtype=torch.float64),widths,bounds).numpy()
    q=q+shift; n=pick[gi]+shift
    owner=np.array([e['test_piece'] for e in events]); signed=np.array([e['signed_distance'] for e in events])
    return dict(owner=owner,source=gd['source_id'][gi],arc=arcs,u=physical_u,
        state=np.column_stack((n,q-n,v,np.full(len(n),100.))),family=gd['family'][gi],
        ds=event_signed_arcs(d,owner,np.array([e['test_arc'] for e in events])),sign=np.sign(signed),
        distance=abs(signed),parallel=np.array([e['parallel'] for e in events]))


def target_spans(arc, xyz, q, z_range, winding_range):
    """Intersect each linear sampled segment with the target Z/winding box."""
    result=[]
    for n in range(len(arc)-1):
        lo,hi=0.,1.
        for values,bounds in ((xyz[:,2],z_range),(q,(winding_range[0]-.5,winding_range[1]+.5))):
            a,b=values[n:n+2];delta=b-a
            if delta==0:
                if not bounds[0]<=a<=bounds[1]:hi=-1.;break
            else:
                x,y=sorted(((bounds[0]-a)/delta,(bounds[1]-a)/delta))
                lo=max(lo,x);hi=min(hi,y)
        if hi<=lo:continue
        start=arc[n]+lo*(arc[n+1]-arc[n]);end=arc[n]+hi*(arc[n+1]-arc[n])
        if result and abs(start-result[-1][1])<1e-6:result[-1][1]=float(end)
        else:result.append([float(start),float(end)])
    return result


def run(settings):
    from .fiber_full_slab import reference_scale
    reference_scale(settings)
    from . import fiber_reference_comparison as ref, fiber_soft_winding as soft
    from .fiber_uv_arc_phase import event_arc_points,event_signed_arcs
    from .fiber_gt_constraint_score import run as cross_run
    from .fiber_workflows import run as dispatch
    from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import fiber_volume,fibers_digest,read_polylines,write_fibers
    import torch
    started=time.monotonic();root=Path(settings['output_root']);root.mkdir(parents=True,exist_ok=True)
    base=json.loads(Path(settings['base_config']).read_text())['fiber_implicit_winding']
    parent_cfg=json.loads(Path(settings['parent_config']).read_text())['fiber_uv_layout']
    parent=arrays(parent_cfg['tensor_cache'])
    gtroot=Path(settings['gt_root']);gcfg=json.loads((gtroot/'uv_whole/config.json').read_text())
    gd=arrays(gcfg['tensor_cache']);gs=arrays(gtroot/'uv_whole/solution.npz')
    _,_,gkeep,gpick=uv.hard_assignment(gs['state'],len(gs['widths']),gcfg,best_nonbroken=True)
    if not gkeep.all():raise ValueError('GT must be intact and enabled')
    _,_,_,_,gu,gv,gb=uv.hard_geometry(gd,gs['state'],gs['widths'],gcfg)
    def gt_points(ids,arcs):
        return uv.geometry_points(gd,gu,gv,gb,gs['widths'],ids,event_signed_arcs(gd,ids,arcs))
    paths=[Path(settings[k]) for k in ('base_config','parent_config','extract_template','orient_template')]
    paths += [Path(parent_cfg[k]) for k in ('tensor_cache','database','stage1','hv_solution','h_direction_solution','v_direction_solution')]
    paths += [gtroot/'uv_whole/config.json',gtroot/'uv_whole/solution.npz',Path(gcfg['tensor_cache'])]
    paths += [Path(settings['reference'])/(c+'.tif') for c in 'xyz']+[Path(settings['reference'])/'meta.json']
    paths += [Path(__file__),Path(soft.__file__),Path(ref.__file__)]
    from . import fiber_implicit_winding as density, fiber_gt_constraint_score as cross_module
    paths += [Path(density.__file__),Path(cross_module.__file__)]
    signature=dict(settings=settings,inputs={str(p):sha256_file(p) for p in paths},
                   fibers=fibers_digest(parent_cfg['fibers']),gt_fibers=fibers_digest(gcfg['fibers']))
    stamp=root/'inputs.json'
    if stamp.exists() and json.loads(stamp.read_text())!=signature:raise ValueError('Changed dataset inputs; use a new output directory')
    atomic_write(stamp,signature,durable=False)
    if (root/'summary.json').exists():return json.loads((root/'summary.json').read_text())
    selection_path=root/'selection.json'
    surface=None
    def get_surface():
        nonlocal surface
        if surface is None:surface=ref.Surface(settings['reference'],settings['reference_xyz_scale'],
            settings.get('projection_stride',2),settings.get('threads',128),settings.get('projection_candidates',8))
        return surface
    if not selection_path.exists():
        surf=get_surface();wr=settings['winding_range'];halo=settings.get('halo_windings',2)
        cells=np.flatnonzero((surf.labels>=wr[0]-halo)&(surf.labels<=wr[1]+halo))
        from scipy.spatial import cKDTree
        tree=cKDTree(surf.centers[cells])
        r,c=surf.r[cells],surf.c[cells]
        radius=max(float(np.linalg.norm(surf.xyz[r+dr,c+dc]-surf.centers[cells],axis=1).max()) for dr,dc in ((0,0),(0,1),(1,0),(1,1)))
        zlo,zhi=np.array(settings['z_range'])+[-settings.get('z_halo',1000),settings.get('z_halo',1000)]
        threshold=settings.get('surface_distance',200)+settings.get('sample_spacing',64)/2
        print('selection: reading full source curves',flush=True)
        curves=read_polylines(parent_cfg['fibers']);seeds=[];batch=[];owners=[];seen=0
        def flush():
            if not batch:return
            xyz=np.concatenate(batch);ids=np.concatenate(owners)
            near,_=tree.query(xyz,k=1,distance_upper_bound=threshold+radius,workers=settings.get('threads',128))
            possible=np.isfinite(near)
            _,dist=surf.project(xyz[possible],cells,tree=tree)
            seeds.extend(np.unique(ids[possible][dist<=threshold]).tolist());batch.clear();owners.clear()
        for row,xyz in curves:
            if xyz[:,2].max()<zlo or xyz[:,2].min()>zhi:continue
            _,points=curve_samples(xyz,settings.get('sample_spacing',64))
            points=points[(points[:,2]>=zlo)&(points[:,2]<=zhi)]
            if not len(points):continue
            batch.append(points);owners.append(np.full(len(points),row['id']));seen+=len(points)
            if sum(map(len,batch))>=65536:flush();print(f'selection: {seen} samples, {len(set(seeds))} sources',flush=True)
        flush();seeds=np.unique(np.asarray(seeds,dtype=str))
        eligible=np.intersect1d(seeds,parent['source_id']);excluded=np.setdiff1d(seeds,parent['source_id'])
        sources=one_hop(parent,eligible)
        selection=dict(seed_sources=seeds.tolist(),eligible_seed_sources=eligible.tolist(),
            earlier_stage_excluded_sources=excluded.tolist(),candidate_sources=sources.tolist(),
            store_sources=len(curves),sample_count=seen,distance_with_sampling_padding=threshold)
        atomic_write(selection_path,selection,durable=False)
        del curves
    selection=json.loads(selection_path.read_text());print(json.dumps({k:len(v) if isinstance(v,list) else v for k,v in selection.items()}),flush=True)
    d=induced_graph(parent,selection['candidate_sources'],parent_cfg)
    d,filter_stats=soft.constraint_family_filter(d,True)
    d['degree']=np.bincount(np.r_[d['i'],d['j']],minlength=len(d['family'])).astype(np.float32)
    np.savez(root/'tensors.npz',**d)
    config=dict(base,tensor_cache=str(root/'tensors.npz'),layout_mode='shared_anchor',
        output_root=str(root/'solve'),initial_solution=str(root/'init/solution.npz'),
        fixed_reference_events=str(root/'fixed_reference_events.npz'),
        devices=settings.get('devices',base.get('devices','0')),bucket_shared_phase=True)
    config.pop('reference_subset',None)
    config['gpu_manager']=dict(base['gpu_manager'],label='broad-reference-anchored-density',exclusive=False)
    atomic_write(root/'candidate_config.json',{'fiber_uv_layout':config},durable=False)
    crossroot=root/'cross'
    if not (crossroot/'cross_events.json').exists():
        cross_run(dict(configs=[str(root/'candidate_config.json')],gt_root=str(gtroot),
            output_root=str(crossroot),extract_template=settings['extract_template'],
            orient_template=settings['orient_template'],threads=settings.get('threads',128),events_only=True,
            native_binary=settings.get('native_binary','vc_fiber_constraints')))
    events=json.loads((crossroot/'cross_events.json').read_text())['events']
    events=[e for e in events if e['class_matches_families']]
    if not events:raise ValueError('No family-compatible GT constraints')
    # Calibrate the immutable GT gauge to the surface before the candidate solve.
    gaugepath=root/'reference_gauge.json'
    if not gaugepath.exists():
        sample=ref.samples(gd,gcfg,settings.get('sample_spacing',64));rc,distance=get_surface().project(sample['xyz'])
        refq=ref.winding_at(rc[:,1],get_surface().x,get_surface().q)
        gq,_=gt_points(sample['owner'],sample['arc'])
        good=np.isfinite(refq)&(distance<=settings.get('surface_distance',200))
        if not good.any():raise ValueError('No GT-to-reference gauge support')
        shift=int(np.floor(ref.grouped_weighted_median(refq[good]-gq[good],sample['weight'][good],np.array([0,int(good.sum())]))[0]+.5))
        atomic_write(gaugepath,dict(gt_to_surface_offset=shift,samples=int(good.sum()),
            median_absolute_error=float(np.median(abs(refq[good]-gq[good]-shift)))),durable=False)
    gauge=json.loads(gaugepath.read_text());shift=gauge['gt_to_surface_offset']
    references=fixed_reference_endpoints(d,events,gd,gs,gcfg,shift)
    owner=references['owner']
    np.savez(root/'fixed_reference_events.npz',**references)
    init=root/'init';init.mkdir(exist_ok=True)
    centers,widths=uv.radial_initial_geometry(d,config)
    state=np.column_stack((centers,d['phase'],d['z'],np.full(len(centers),4.))).astype(np.float32)
    np.savez(init/'solution.npz',state=state,widths=widths,**{k:d[k] for k in ('source_id','start','end','family')})
    np.savez(init/'samples.npz',**soft.prepare_samples(config,d))
    atomic_write(init/'config.json',dict(tensor_sha256=sha256_file(root/'tensors.npz'),soft_samples=str(init/'samples.npz')),durable=False)
    atomic_write(root/'solve_config.json',{'fiber_implicit_winding':config},durable=False)
    if not (root/'solve/summary.json').exists():dispatch({'fiber_implicit_winding':config})
    sol=arrays(root/'solve/solution.npz');st=sol['state'];widths=sol['widths']
    from .fiber_implicit_winding import hypotheses
    with torch.no_grad():
        _,lp=hypotheses(torch.as_tensor(st),config['winding_sigma'])
        confidence=(lp.exp().amax(1)*torch.as_tensor(st[:,3]).sigmoid()).numpy()
    connected=anchored_mask(d,owner,st[:,3]>=0);records=[];groups={'selected':set(),'uncertain':set(selection['earlier_stage_excluded_sources']),'context':set()}
    all_ids=np.unique(np.r_[d['source_id'],np.asarray(selection['earlier_stage_excluded_sources'],dtype=str)])
    curves=read_polylines(config['fibers'],all_ids)
    for row,xyz in curves:
        sid=row['id'];pieces=np.flatnonzero(d['source_id']==sid);arc,points=curve_samples(xyz,settings.get('sample_spacing',64))
        items=[];has_target=False;uncertain=not len(pieces)
        for p in pieces:
            s=np.unique(np.r_[d['start'][p],arc[(arc>d['start'][p])&(arc<d['end'][p])],d['end'][p]])
            xyzp=np.column_stack([np.interp(s,arc,points[:,a]) for a in range(3)])
            states=np.repeat(st[p:p+1],len(s),axis=0);states[:,0]=np.floor(states[:,0]+.5)
            with torch.no_grad():q=soft.points(torch.as_tensor(states),torch.full((len(s),),int(d['family'][p])),
                torch.as_tensor(event_signed_arcs(d,np.full(len(s),p),s),dtype=torch.float32),torch.as_tensor(widths))[0].numpy()
            spans=target_spans(s,xyzp,q,settings['z_range'],settings['winding_range'])
            trusted=bool(connected[p] and confidence[p]>=settings.get('confidence',.5))
            has_target|=bool(spans and trusted);uncertain|=not trusted
            items.append(dict(piece=int(p),start=float(d['start'][p]),end=float(d['end'][p]),
                joint_probability=float(confidence[p]),enabled=bool(st[p,3]>=0),anchored=bool(connected[p]),
                q_range=[float(q.min()),float(q.max())],target_spans=spans))
        category='selected' if has_target else 'uncertain' if uncertain else 'context'
        groups[category].add(sid)
        records.append(dict(source_id=sid,category=category,pieces=items,has_uncertain_pieces=uncertain))
    for name,ids in groups.items():
        if not ids:continue
        path=root/(name+'.vcobj')
        if not path.exists():write_fibers(path,[(r,x) for r,x in curves if r['id'] in ids],volume=fiber_volume(config['fibers']))
        saved=read_polylines(path)
        original={r['id']:xyz for r,xyz in curves if r['id'] in ids}
        if set(r['id'] for r,_ in saved)!=ids:raise ValueError('Export source IDs differ')
        for row,xyz in saved:
            if not np.array_equal(xyz,original[row['id']]):raise ValueError('Export truncated/changed a source curve')
    atomic_write(root/'fibers.json',records,durable=False)
    old_sources=np.unique(arrays(base['tensor_cache'])['source_id']).tolist()
    comparison=dict(old_sources=len(old_sources),new_selected=len(groups['selected']),
        recovered_sources=sorted(groups['selected']-set(old_sources)),
        old_sources_not_selected=sorted(set(old_sources)-groups['selected']),
        old_sources_by_new_category={name:int(len(set(old_sources)&ids)) for name,ids in groups.items()},
        old_sources_missing_from_candidates=sorted(set(old_sources)-set(all_ids)))
    atomic_write(root/'old_set_comparison.json',comparison,durable=False)
    mapped=st[owner].copy();mapped[:,0]=np.floor(mapped[:,0]+.5)
    with torch.no_grad():
        tq,_,tv=soft.points(torch.as_tensor(mapped),torch.as_tensor(d['family'][owner]),
            torch.as_tensor(references['ds'],dtype=torch.float32),torch.as_tensor(widths))
    reference_q=references['state'][:,0]+references['state'][:,1]
    required=reference_q-tq.numpy()+config.get('perpendicular_offset',.5)*(~references['parallel'])*(references['family']-d['family'][owner])-references['sign']*references['distance']
    enabled=st[owner,3]>=0;parallel=references['parallel']
    def residual_stats(mask):
        error=abs(required[mask])
        return dict(events=int(mask.sum()),l1=float(error.mean()) if len(error) else None,
            correct_percent=float(100*np.mean(error<.5)) if len(error) else None,
            max=float(error.max()) if len(error) else None)
    residuals=dict(enabled=residual_stats(enabled),parallel=residual_stats(enabled&parallel),
        perpendicular=residual_stats(enabled&~parallel),disabled=residual_stats(~enabled),
        enabled_percent=float(100*enabled.mean()),
        interpretation='Supervised GT residuals at fixed absolute gauge, no postfit correction; not independent evaluation')
    atomic_write(root/'reference_metrics.json',residuals,durable=False)
    result=dict(selection_counts={k:len(v) if isinstance(v,list) else v for k,v in selection.items()},
        pieces=len(d['family']),constraints=len(d['i']),reference_constraints=len(events),gauge=gauge,
        constraint_filter=filter_stats,reference_metrics=residuals,
        recovered_sources=len(comparison['recovered_sources']),
        old_sources_not_selected=len(comparison['old_sources_not_selected']),
        outputs={k:dict(sources=len(v),path=str(root/(k+'.vcobj'))) for k,v in groups.items()},
        solve=json.loads((root/'solve/summary.json').read_text())['final'],seconds=time.monotonic()-started,
        interpretation='Model-assisted selection with manual GT supervision; not independent GT')
    atomic_write(root/'summary.json',result,durable=False);print(json.dumps(result),flush=True);return result
