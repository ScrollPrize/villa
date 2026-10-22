"""GT-free full-cross-section slab baseline and posthoc spiral metrics."""
import json
from pathlib import Path
import time
import numpy as np
from lasagna.inference_provenance import atomic_write, sha256_file
from . import fiber_uv_layout as uv, fiber_soft_winding as soft
from . import fiber_uv_reference_subset as subset
from .fiber_gt_constraint_score import arrays


def reference_scale(settings):
    """Require an explicit physical scale before expensive work or output writes."""
    scale=float(settings['reference_xyz_scale'])
    if not np.isfinite(scale) or scale<=0:
        raise ValueError('reference_xyz_scale must be positive and finite')
    return scale


def clean_config(base):
    return {k:v for k,v in base.items() if not (
        k.startswith(('reference_', 'fixed_reference', 'executed_')) or
        k.endswith('_sha256') or k in ('initial_solution','soft_samples','code_provenance',
                                     'initialization_geometry_cache'))}


def reference_metrics(d,sol,config,sample,reference,*,xyz_scale,continuous=False):
    from . import fiber_reference_comparison as ref
    metric_cfg=dict(config,reference_widths=dict(meta=str(Path(reference)/'meta.json'),xyz_scale=xyz_scale))
    state=sol['state'];widths=sol['widths'];b=np.r_[0.,np.cumsum(widths,dtype=float)]
    q=(state[:,0] if continuous else np.floor(state[:,0]+.5))+state[:,1]
    k=np.floor(q).astype(int).clip(0,len(widths)-1)
    anchor=b[k]+(q-k)*widths[k]
    geometry=(None,None,state[:,3]>=0,None,anchor,state[:,2],b)
    metrics=ref.reference_winding_metrics(d,state,widths,metric_cfg,sample,geometry=geometry)
    offset=metrics['global_offset']
    if offset is not None:
        occupied=np.unique(np.floor(q[state[:,3]>=0]).astype(int))
        occupied=occupied[(occupied>=0)&(occupied<len(widths))]
        reference_widths=ref.reference_interval_widths(metric_cfg,occupied+offset)
        valid=np.isfinite(reference_widths)&(reference_widths>0)
        ratios=widths[occupied[valid]]/reference_widths[valid]
        metrics['occupied_anchor_widths']=dict(compared_windings=int(valid.sum()),
            mean_normalized_factor=float(ratios.mean()) if len(ratios) else None,
            mean_absolute_relative_error_percent=float(100*np.abs(ratios-1).mean()) if len(ratios) else None)
    return metrics


def score_checkpoints(root,parent,config,reference,*,xyz_scale,continuous=False):
    """Shared posthoc scoring; never feed projection data into optimization."""
    root=Path(root);parent=Path(parent)
    if not (parent/'results.json').exists():
        raise RuntimeError('Parent projection not yet published; retry scoring after completion')
    d=arrays(config['tensor_cache']);sample=arrays(parent/'reference_projection.npz')
    for key in ('source_id','start','end'):
        if not np.array_equal(d[key],sample[key]):raise ValueError('Projection identity mismatch')
    rows=[]
    for record in json.loads((root/'history.json').read_text()):
        sol=arrays(root/'checkpoints'/f'{record["iteration"]:06d}.npz')
        row=dict(record,reference=reference_metrics(d,sol,config,sample,reference,xyz_scale=xyz_scale,continuous=continuous))
        rows.append(row)
        print(json.dumps(dict(iteration=row['iteration'],loss=row['loss'],enabled_percent=row['enabled_percent'],
                              reference=row['reference'])),flush=True)
        atomic_write(root/'reference_history.json',rows,durable=False)
    return rows


def soft_trial_config(original,settings):
    config=clean_config(original)
    for key in list(config):
        if key.startswith(('cavity_','sigma_','sign_probability')) or key in (
                'winding_sigma','soft_pair_weight','probability_pair_weight','same_sheet_match_multiplier',
                'sign_violation_weight','learned_winding_sigma','pair_weight_schedule'):
            config.pop(key)
    config.update(output_root=settings['output_root'],initial_solution=original['initial_solution'],
        soft_samples=original['soft_samples'],tensor_sha256=original['tensor_sha256'],
        samples_sha256=original['samples_sha256'],save_diagnostic_checkpoints=True,
        devices=settings['devices'],width_lr=settings['width_lr'])
    config['gpu_manager']=dict(original['gpu_manager'],exclusive=False,label='soft-slab')
    soft.validate(config)
    return config


def run_soft_trial(settings):
    """Fresh, same-input soft-only control; only width LR varies across trials."""
    reference_scale(settings)
    parent=Path(settings['parent_root']);root=Path(settings['output_root'])
    original=json.loads((parent/'solve/config.json').read_text())
    config=soft_trial_config(original,settings)
    signature=dict(settings=settings,parent_config=sha256_file(parent/'solve/config.json'),
        inputs={key:sha256_file(config[key]) for key in ('tensor_cache','initial_solution','soft_samples')},
        solver=sha256_file(soft.__file__),orchestration=sha256_file(__file__),
        projection=sha256_file(parent/'reference_projection.npz'),
        reference_meta=sha256_file(Path(settings['reference'])/'meta.json'))
    root.mkdir(parents=True,exist_ok=True);stamp=root/'trial_inputs.json'
    if stamp.exists() and json.loads(stamp.read_text())!=signature:raise ValueError('Soft trial inputs changed')
    atomic_write(stamp,signature,durable=False)
    if not (root/'summary.json').exists():
        uv.launch_tensor_solve(config,soft.__name__,Path(soft.__file__))
    saved=json.loads((root/'config.json').read_text())
    if any(saved.get(k)!=v for k,v in config.items()):raise ValueError('Soft trial config mismatch')
    if json.loads((root/'summary.json').read_text())['final']['iteration']!=config['steps']:
        raise ValueError('Incomplete soft trial')
    rows=score_checkpoints(root,parent,config,settings['reference'],xyz_scale=settings['reference_xyz_scale'],continuous=True)
    result=dict(width_lr=config['width_lr'],rows=rows,model='continuous_soft_winding_v1',
                solution_sha256=sha256_file(root/'solution.npz'),gt_supervision=False)
    atomic_write(root/'reference_results.json',result,durable=False)
    return result


def run_progress(settings):
    """Repeat prepared seeded solve with checkpoints; reference remains posthoc."""
    reference_scale(settings)
    from . import fiber_implicit_winding as density
    root=Path(settings['output_root']);parent=Path(settings['parent_root'])
    original=json.loads((parent/'solve/config.json').read_text())
    config=clean_config(original)
    config.update(output_root=str(root),initial_solution=original['initial_solution'],
                  save_diagnostic_checkpoints=True)
    root.mkdir(parents=True,exist_ok=True)
    signature=dict(settings=settings,tensors=sha256_file(config['tensor_cache']),
        init=sha256_file(config['initial_solution']),parent_config=sha256_file(parent/'solve/config.json'),
        reference={name:sha256_file(Path(settings['reference'])/name) for name in ('x.tif','y.tif','z.tif','meta.json')},
        solver=sha256_file(soft.__file__),density=sha256_file(density.__file__))
    stamp=root/'repeat_inputs.json'
    if stamp.exists() and json.loads(stamp.read_text())!=signature:raise ValueError('Progress repeat inputs changed')
    atomic_write(stamp,signature,durable=False)
    if not (root/'summary.json').exists():density.run(config)
    saved=json.loads((root/'config.json').read_text())
    if any(saved.get(k)!=v for k,v in config.items()):raise ValueError('Repeat solve config mismatch')
    if json.loads((root/'summary.json').read_text())['final']['iteration']!=config['steps']:
        raise ValueError('Incomplete repeat solve')
    if not (parent/'results.json').exists():
        raise RuntimeError('Parent projection not yet published; rerun this command to score saved checkpoints without repeating optimization')
    rows=score_checkpoints(root,parent,config,settings['reference'],xyz_scale=settings['reference_xyz_scale'])
    previous=arrays(parent/'solve/solution.npz');current=arrays(root/'solution.npz')
    comparison={k:float(np.max(np.abs(previous[k]-current[k]))) for k in ('state','widths')}
    result=dict(rows=rows,repeat_max_abs_difference=comparison,
                projection_sha256=sha256_file(parent/'reference_projection.npz'),
                report_code_sha256=sha256_file(__file__),
                cavity_solver_sha256=saved.get('cavity_solver_sha256'))
    atomic_write(root/'reference_results.json',result,durable=False)
    return result


def run(settings):
    reference_scale(settings)
    from . import fiber_implicit_winding as density
    from . import fiber_reference_comparison as ref
    from . import fiber_cavity_sigma as cavity
    started=time.monotonic()
    root=Path(settings['output_root']);root.mkdir(parents=True,exist_ok=True)
    base_path=Path(settings['base_config'])
    config=clean_config(json.loads(base_path.read_text())['fiber_implicit_winding'])
    signature=dict(settings=settings,base=sha256_file(base_path),
        code={Path(m.__file__).name:sha256_file(m.__file__) for m in (subset,uv,soft,density,cavity,ref)},
        orchestration=sha256_file(__file__),
        reference={name:sha256_file(Path(settings['reference'])/name) for name in ('x.tif','y.tif','z.tif','meta.json')},
        inputs={k:sha256_file(config[k]) for k in ('afv','database','features','stage1','hv_solution',
                     'h_direction_solution','v_direction_solution','umbilicus')})
    stamp=root/'inputs.json'
    if stamp.exists() and json.loads(stamp.read_text())!=signature:raise ValueError('Slab inputs changed')
    atomic_write(stamp,signature,durable=False)
    if (root/'results.json').exists():return json.loads((root/'results.json').read_text())
    parent=dict(config,tensor_cache=str(root/'whole_tensors.npz'),keep_whole_fibers=True)
    print('slab: preparing full-corpus whole-fiber graph',flush=True)
    uv.prepare(parent)
    atomic_write(root/'parent_config.json',{'fiber_uv_layout':parent},durable=False)
    manifest=subset.run(dict(source_config=str(root/'parent_config.json'),output_root=str(root/'slab'),
                            z_range=settings['z_range'],sample_spacing=settings.get('sample_spacing',200)))
    d=arrays(root/'slab/tensors.npz')
    d,filter_stats=soft.constraint_family_filter(d,True)
    if not len(d['family']) or not len(d['i']):raise ValueError('Empty slab constraint graph')
    d['degree']=np.bincount(np.r_[d['i'],d['j']],minlength=len(d['family'])).astype(np.float32)
    np.savez(root/'tensors.npz',**d)
    config.update(tensor_cache=str(root/'tensors.npz'),output_root=str(root/'solve'),
                  devices=settings.get('devices','0'),initial_solution=str(root/'init/solution.npz'),
                  cavity_group_source_pieces=True,diagnostic_batch_size=config['batch_size'])
    config['gpu_manager']=dict(config['gpu_manager'],exclusive=False,label='full-cross-section-slab')
    init=root/'init';init.mkdir(exist_ok=True)
    centers,widths=uv.radial_initial_geometry(d,config)
    state=np.column_stack((centers,d['phase'],d['z'],np.full(len(centers),4.))).astype(np.float32)
    np.savez(init/'solution.npz',state=state,widths=widths,**{k:d[k] for k in ('source_id','start','end','family')})
    print('slab: preparing angular samples',flush=True)
    np.savez(init/'samples.npz',**soft.prepare_samples(config,d))
    atomic_write(init/'config.json',dict(tensor_sha256=sha256_file(root/'tensors.npz'),
                                       soft_samples=str(init/'samples.npz')),durable=False)
    atomic_write(root/'solve_config.json',{'fiber_implicit_winding':config},durable=False)
    print(json.dumps(dict(sources=manifest['sources'],pieces=len(d['family']),events=len(d['i']),
        filter=filter_stats,steps=config['steps'],batch_size=config['batch_size'],gt_supervision=False)),flush=True)
    if not (root/'solve/summary.json').exists():density.run(config)
    saved=json.loads((root/'solve/config.json').read_text())
    if any(saved.get(k)!=v for k,v in config.items()):raise ValueError('Solve config mismatch')
    summary=json.loads((root/'solve/summary.json').read_text())
    if summary['final']['iteration']!=config['steps']:raise ValueError('Incomplete solve')
    print('slab: posthoc spiral projection (no reference used by optimization)',flush=True)
    sol=arrays(root/'solve/solution.npz')
    sample=arrays(root/'slab/projection.npz')
    surface=ref.Surface(settings['reference'],settings['reference_xyz_scale'],settings.get('projection_stride',2),
                        settings.get('workers',128),settings.get('projection_candidates',4))
    rc,distance=surface.project(sample['xyz'])
    q=ref.winding_at(rc[:,1],surface.x,surface.q)
    max_distance=settings.get('projection_max_distance',200.)
    chosen=ref.assign_sources(d,sample,q,distance,max_distance,surface.limits)
    sample['chosen_winding']=chosen
    np.savez(root/'reference_projection.npz',**sample,nearest_distance=distance,nearest_q=q)
    metrics=reference_metrics(d,sol,saved,sample,settings['reference'],xyz_scale=settings['reference_xyz_scale'])
    support=distance<=max_distance
    result=dict(final=summary['final'],reference=metrics,sources=manifest['sources'],pieces=len(d['family']),
        events=len(d['i']),batch_size=config['batch_size'],updates=config['steps'],
        sampled_pass_equivalents=config['steps']*min(config['batch_size'],len(d['i']))/len(d['i']),
        projection_supported_length_percent=float(100*sample['weight'][support].sum()/sample['weight'].sum()),
        z_range=settings['z_range'],gt_supervision=False,reference_selection=False,
        solution_sha256=sha256_file(root/'solve/solution.npz'),seconds=time.monotonic()-started)
    atomic_write(root/'results.json',result,durable=False)
    print(json.dumps(result),flush=True)
    return result
