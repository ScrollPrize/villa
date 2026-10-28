"""Prepare a packed full-volume corpus for the existing independent SGD model.

No fitted winding labels enter initialization. AFC decoding and sampled priors
are shared with the existing adapters; paths remain memory mapped.
"""
import json
from pathlib import Path
import time
import numpy as np

from .fiber_min_flow_experiment import arrays,write,digest
from .fiber_min_flow_full import PackedPaths,event_batches,parallel_stage,ro


def measured_geometry(d, paths, ids, family, lengths):
    """Complete whole-fiber solver tensors from measured paths, without fitted labels.

    Input event arcs are unsigned native arclength; output arcs follow the
    measured H angular / V axial traversal. Shared by small and full adapters.
    """
    ids=np.asarray(ids);family=np.asarray(family,np.int64);n=len(ids)
    owner=np.repeat(np.arange(n),3);arc=(lengths[:,None]*np.array([0,.5,1])).ravel()
    xyz,turns=paths.at_many(ids[owner],arc);rel=paths.seam.relative(xyz)
    phase=(turns+(np.arctan2(rel[:,1],rel[:,0])-paths.seam.angle)%(2*np.pi)/(2*np.pi)+paths.seam.angle/(2*np.pi)).reshape(n,3)
    phase-=np.floor(phase[:,0]+.5)[:,None]
    z=xyz[:,2].reshape(n,3)
    direction=np.sign(np.where(family==0,phase[:,-1]-phase[:,0],z[:,-1]-z[:,0]))
    if np.any(direction==0):raise ValueError('Ambiguous geometric traversal direction; no silent fallback')
    radius=np.linalg.norm(rel,axis=1).reshape(n,3)[:,1]
    d.update(source_id=ids,source_index=np.arange(n),ids=ids,earlier_labels=family,
        family=family,start=np.zeros(n),end=lengths,lengths=lengths,radius=radius,
        phase=phase[:,0]%1,z=z[:,0],node_ds=arc.reshape(n,3)*direction[:,None],
        node_phase=phase,node_z=z,links=np.empty((0,4)))
    lo,hi=np.quantile(radius,[.001,.999])
    d['initial_k']=np.clip(5+85*(radius-lo)/max(hi-lo,1),0,99).astype(np.float32)
    d['width_initial']=np.maximum(2*np.pi*np.interp(np.arange(100),np.linspace(5,90,100),np.linspace(lo,hi,100)),100.).astype(np.float32)
    for side,key in [('a','i'),('b','j')]:d['ds_'+side]*=direction[d[key]]
    return d


def event_job(job):
    from . import fiber_min_flow_full as full
    config,paths,db=full._worker
    first,last=job;root=Path(config['output_root'])/'events';root.mkdir(exist_ok=True)
    out=root/f'{first:010d}.npz';marker=out.with_suffix('.json')
    if out.exists() and marker.exists():return json.loads(marker.read_text())
    started=time.monotonic();chunks=[];read=0
    for count,d in event_batches(config,paths,db,first,last):
        read+=count
        for side,key in [('a','i'),('b','j')]:
            xyz,turns=paths.at_many(paths.ids[d[key]],d['ds_'+side])
            rel=paths.seam.relative(xyz)
            d['phase_'+side]=turns+(np.arctan2(rel[:,1],rel[:,0])-paths.seam.angle)%(2*np.pi)/(2*np.pi)+paths.seam.angle/(2*np.pi)
        d['phase_delta']=(d['phase_b']-d['phase_a']+.5)%1-.5
        chunks.append({k:d[k] for k in ('i','j','ds_a','ds_b','distance','sign','parallel','event_ids')})
        chunks[-1].update({k:d[k] for k in ('phase_a','phase_b','phase_delta')})
    data={k:np.concatenate([d[k] for d in chunks]) for k in chunks[0]}
    np.savez(out,**data)
    result=dict(first=first,last=last,read=read,events=len(data['i']),seconds=time.monotonic()-started)
    write(marker,result);return result


def prepare(settings,config):
    from . import fiber_soft_winding as soft,fiber_uv_layout as uv
    from .fiber_anchored_dataset import perpendicular_supported_graph
    root=Path(settings['output_root']);root.mkdir(parents=True,exist_ok=True)
    source=Path(settings['source_root']);parent=json.loads((source/'request.json').read_text())['config']
    (root/'geometry').symlink_to(source/'geometry',target_is_directory=True) if not (root/'geometry').exists() else None
    paths=PackedPaths(source/'geometry');ids=paths.ids;n=len(ids)
    if not (root/'tensors.npz').exists():
        native=dict(parent,output_root=str(root),workers=settings.get('workers',32))
        with ro(parent['database']) as db:lo,hi=db.execute('SELECT min(id),max(id) FROM constraints').fetchone()
        batch=parent.get('constraint_batch',500000)
        jobs=[(a,min(a+batch,hi+1)) for a in range(lo,hi+1,batch)]
        parallel_stage(native,'events',jobs,event_job)
        write(root/'progress.json',dict(stage='assemble_tensors'))
        parts=sorted((root/'events').glob('*.npz'))
        keys=('i','j','ds_a','ds_b','distance','sign','parallel','event_ids','phase_a','phase_b','phase_delta')
        # One column at a time: do not retain every decompressed part twice.
        def column(p,k):
            with np.load(p) as src:return src[k]
        d={k:np.concatenate([column(p,k) for p in parts]) for k in keys}
        lengths=np.asarray(paths.keys['arc'][paths.offset[1:]-1])
        d=measured_geometry(d,paths,ids,paths.catalog['family'],lengths)
        # Keep only the same graph's largest component, then the existing
        # model-required perpendicular support filter (no fallback geometry).
        selected=arrays(settings['component_solution'])['source_id']
        from .fiber_anchored_dataset import induced_graph
        d=induced_graph(d,selected,config)
        d,stats=perpendicular_supported_graph(d,config)
        d['degree']=np.bincount(np.r_[d['i'],d['j']],minlength=len(d['family'])).astype(np.float32)
        centers,widths=uv.radial_initial_geometry(d,config)
        d.update(radial_initial_centers=centers,radial_initial_widths=widths)
        np.savez(root/'tensors.npz',**d)
        write(root/'selection.json',dict(fibers=len(d['family']),events=len(d['i']),
            perpendicular_support=stats,direction='signed measured endpoint angular progression for H; Z progression for V',
            initialization='radial only; no fitted labels or reference surface'))
    d=arrays(root/'tensors.npz');init=root/'init';init.mkdir(exist_ok=True)
    if not (init/'samples.npz').exists():
        started=time.monotonic()
        write(root/'progress.json',dict(stage='measured_prior_samples',fibers=len(d['family'])))
        indices=np.searchsorted(paths.ids,d['source_id'])
        geometry=((dict(id=int(paths.ids[i])),paths.xyz[paths.offset[i]:paths.offset[i+1]]) for i in indices)
        sample_config=dict(config,prior_sample_workers=settings.get('prior_sample_workers',64),
            prior_sample_batch=settings.get('prior_sample_batch',512),prior_sample_progress=True)
        samples=soft.prepare_samples_geometry(sample_config,d,geometry,paths.catalog['axis'])
        computed=time.monotonic()
        temporary=init/'samples.partial.npz'
        np.savez(temporary,**samples)
        temporary.replace(init/'samples.npz')
        write(root/'prior_sample_timing.json',dict(compute_seconds=computed-started,
            save_seconds=time.monotonic()-computed,fibers=len(d['family']),
            phase_samples=len(samples['owner']),z_samples=len(samples['v_owner']),
            workers=sample_config['prior_sample_workers']))
    centers,widths=uv.radial_initial_geometry(d,config)
    state=np.column_stack((centers,d['phase'],d['z'],np.full(len(centers),4.))).astype(np.float32)
    np.savez(init/'solution.npz',state=state,widths=widths,**{k:d[k] for k in ('source_id','start','end','family')})
    write(init/'config.json',dict(tensor_sha256=digest(root/'tensors.npz'),soft_samples=str(init/'samples.npz')))
    return d


def run(settings):
    from . import fiber_implicit_winding as density
    from .fiber_full_slab import clean_config
    root=Path(settings['output_root']);root.mkdir(parents=True,exist_ok=True)
    base=json.loads(Path(settings['base_config']).read_text())
    config=clean_config(base.get('fiber_implicit_winding',base))
    config.update(tensor_cache=str(root/'tensors.npz'),initial_solution=str(root/'init/solution.npz'),
        output_root=str(root/'solve'),devices=settings.get('devices','1'),
        estimate_fixed_length_vram=False,gpu_manager=settings['gpu_manager'])
    config.pop('gt_benchmark',None)
    config['startup_trace_seconds']=settings.get('startup_trace_seconds',0)
    config['lbfgs_prior_batch_size']=settings.get('prior_batch_size',config.get('lbfgs_edge_batch_size',32768))
    config['lbfgs_objective_progress_every']=settings.get('objective_progress_every',256)
    if 'log_every' in settings:config['log_every']=settings['log_every']
    parent=json.loads((Path(settings['source_root'])/'request.json').read_text())['config']
    config.update(afv=parent['afv'],umbilicus=parent['umbilicus'])
    write(root/'request.json',settings)
    d=prepare(settings,config)
    if settings.get('prepare_only',False):
        write(root/'progress.json',dict(stage='prepared',fibers=len(d['family']),events=len(d['i'])))
        return
    config['diagnostic_batch_size']=len(d['i'])
    write(root/'solve_config.json',{'fiber_implicit_winding':config})
    write(root/'progress.json',dict(stage='managed_solve',fibers=len(d['family']),events=len(d['i']),steps=config['steps']))
    del d
    result=density.run(config)
    write(root/'progress.json',dict(stage='complete'))
    return result
