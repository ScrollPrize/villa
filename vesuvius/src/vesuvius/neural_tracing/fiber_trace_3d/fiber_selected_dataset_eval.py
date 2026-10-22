"""GT-free baseline solve on selected whole sources, followed by frozen GT scoring."""
import json
from pathlib import Path
import time
import numpy as np
from lasagna.inference_provenance import atomic_write,sha256_file
from . import fiber_uv_layout as uv, fiber_soft_winding as soft
from .fiber_anchored_dataset import induced_graph
from .fiber_gt_constraint_score import arrays,piece_at,event_residuals,enabled_l1,gt_solve_width_metrics


def remap_events(events,d):
    sources=set(map(int,d['source_id']));result=[]
    for event in events:
        if event['test_source'] not in sources:continue
        piece=piece_at(d,event['test_source'],event['test_arc'])
        if piece is None:raise ValueError('Selected whole source lost GT event arc')
        result.append(dict(event,test_piece=piece))
    if not result:raise ValueError('Selected set has no GT evaluation events')
    return result


def initialization_config(template, config, output, initial=None):
    """Reuse caller-supplied stage hyperparameters, never its data/state/provenance."""
    result={k:v for k,v in template.items() if not (
        k.startswith(('reference_', 'fixed_reference', 'executed_')) or
        k.endswith('_sha256') or k in ('code_provenance','initial_solution','soft_samples'))}
    for key in ('tensor_cache','afv','umbilicus','devices','gpu_manager','layout_mode','filter_constraint_families'):
        result[key]=config[key]
    result.update(output_root=str(output))
    if initial is not None:result['initial_solution']=str(initial)
    return result


def aligned_widths(radial, donor, offset):
    """Test interval k uses donor interval k+offset; preserve unmapped tails."""
    radial=np.asarray(radial);donor=np.asarray(donor)
    if donor.ndim!=1 or not np.isfinite(donor).all() or np.any(donor<=0):
        raise ValueError('Invalid donor widths')
    ids=np.arange(len(radial))+int(offset)
    valid=(ids>=0)&(ids<len(donor))
    result=radial.copy();result[valid]=donor[ids[valid]]
    return result,int(valid.sum())


def checkpoint_scores(settings):
    """Posthoc scoring of published checkpoints; no GT enters optimization."""
    from zipfile import BadZipFile
    root=Path(settings['root']);cfg=json.loads((root/'solve/config.json').read_text())
    d=arrays(cfg['tensor_cache']);gtroot=Path(settings['gt_root'])
    gcfg=json.loads((gtroot/'uv_whole/config.json').read_text())
    gd=arrays(gcfg['tensor_cache']);gs=arrays(gtroot/'uv_whole/solution.npz')
    events=remap_events(json.loads((Path(settings['dataset'])/'cross/cross_events.json').read_text())['events'],d)
    _,_,gkeep,gpick=uv.hard_assignment(gs['state'],len(gs['widths']),gcfg,best_nonbroken=True)
    if not gkeep.all():raise ValueError('Trusted GT disabled')
    for e in events:
        if int(gd['source_id'][e['gt_piece']])!=e['gt_source']:raise ValueError('GT event identity mismatch')
    parallel=np.array([e['parallel'] for e in events]);owner=np.array([e['test_piece'] for e in events])
    rows=[]
    for record in json.loads((root/'solve/history.json').read_text()):
        path=root/'solve/checkpoints'/f'{record["iteration"]:06d}.npz'
        # History is published before its checkpoint. A live report may skip
        # this newest row and pick it up on the next call.
        if not path.exists():
            if (root/'solve/summary.json').exists():raise FileNotFoundError(path)
            continue
        try:sol=arrays(path)
        except (OSError,ValueError,EOFError,BadZipFile):
            if (root/'solve/summary.json').exists():raise
            continue
        mapped=sol['state'].copy();mapped[:,0]=np.floor(mapped[:,0]+.5)
        residual=event_residuals(events,d,mapped,None,len(sol['widths']),gd,gs,gpick,
                                 dict(cfg,soft_winding_model=True),gcfg,sol['widths'],saved_geometry=sol)
        keep=mapped[owner,3]>=0;score=enabled_l1(residual,keep,parallel)
        row=dict(record,gt=score);rows.append(row)
    atomic_write(root/'checkpoint_metrics.json',rows,durable=False)
    print(json.dumps(rows),flush=True)
    return rows


def run(settings):
    from .fiber_workflows import run as dispatch
    from .fiber_implicit_match_report import run as match_report
    from .fiber_uv_arc_phase import event_arc_points
    from vesuvius.neural_tracing.tracers.shared.annotations.afv import read_afv_polylines
    started=time.monotonic();root=Path(settings['output_root']);root.mkdir(parents=True,exist_ok=True)
    dataset=Path(settings['dataset']);base_path=Path(settings['base_config'])
    config=json.loads(base_path.read_text())['fiber_implicit_winding']
    config.update(settings.get('parameters',{}))
    parent=arrays(dataset/'tensors.npz')
    curves=read_afv_polylines(dataset/'selected.afv');sources=np.array([r['id'] for r,_ in curves])
    if len(sources)!=len(np.unique(sources)):raise ValueError('Duplicate selected source')
    for (row,xyz),(original,points) in zip(curves,read_afv_polylines(config['afv'],sources)):
        if row['id']!=original['id'] or not np.array_equal(xyz,points):raise ValueError('Changed selected geometry')
    paths=[base_path,dataset/'selected.afv',dataset/'tensors.npz',dataset/'cross/cross_events.json',Path(__file__),Path(soft.__file__),Path(config['afv']),Path(config['umbilicus'])]
    gtroot=Path(settings['gt_root']);gcfg_path=gtroot/'uv_whole/config.json'
    gcfg=json.loads(gcfg_path.read_text())
    paths += [gcfg_path,gtroot/'uv_whole/solution.npz',Path(gcfg['tensor_cache'])]
    from . import fiber_implicit_winding as density, fiber_gt_constraint_score as score_module, fiber_anchored_dataset as selection
    paths += [Path(m.__file__) for m in (density,score_module,selection,uv)]
    if config.get('optimizer_kind')=='scaled_lbfgs':
        from . import fiber_scaled_lbfgs
        paths.append(Path(fiber_scaled_lbfgs.__file__))
    if config.get('cavity_winding_sigma'):
        from . import fiber_cavity_sigma
        paths.append(Path(fiber_cavity_sigma.__file__))
    paths += [Path(stage['config']) for stage in settings.get('initialization_stages',[])]
    width_source=settings.get('width_initialization',{})
    if width_source and settings.get('initialization_stages'):
        raise ValueError('Width diagnostic requires fresh fiber state')
    if width_source.get('solution'):paths.append(Path(width_source['solution']))
    signature=dict(settings=settings,inputs={str(p):sha256_file(p) for p in paths})
    stamp=root/'inputs.json'
    if stamp.exists() and json.loads(stamp.read_text())!=signature:raise ValueError('Changed evaluation inputs/code')
    atomic_write(stamp,signature,durable=False)
    if (root/'results.json').exists():
        completed=json.loads((root/'results.json').read_text())
        if completed['solution_sha256']!=sha256_file(root/'solve/solution.npz'):raise ValueError('Completed solution changed')
        return completed
    d=induced_graph(parent,sources,config)
    if not np.array_equal(np.unique(d['source_id']),sources):raise ValueError('Missing selected source')
    d,filter_stats=soft.constraint_family_filter(d,True)
    if config.get('shared_curve_geometry'):
        from .fiber_anchored_dataset import perpendicular_supported_graph
        d,geometry_filter=perpendicular_supported_graph(d,config)
        atomic_write(root/'perpendicular_support_filter.json',geometry_filter,durable=False)
        print(json.dumps(dict(perpendicular_support_filter=geometry_filter)),flush=True)
    from .fiber_piece_split import split_graph
    d=split_graph(d,config)
    d['degree']=np.bincount(np.r_[d['i'],d['j']],minlength=len(d['family'])).astype(np.float32)
    np.savez(root/'tensors.npz',**d)
    for key in list(config):
        if key.startswith('reference_') or key.startswith('fixed_reference'):
            config.pop(key)
    config.update(tensor_cache=str(root/'tensors.npz'),output_root=str(root/'solve'),
        initial_solution=str(root/'init/solution.npz'),devices=settings.get('devices',config.get('devices','0')),
        layout_mode='shared_anchor',filter_constraint_families=True)
    config['gpu_manager']=dict(config['gpu_manager'],label='selected-dataset-gt-free')
    # Only radial geometry, no candidate supervised solution or fitted widths.
    centers,widths=uv.radial_initial_geometry(d,config)
    from .fiber_piece_split import initial_state
    initial=initial_state(d,centers,widths).astype(np.float32)
    width_info=dict(mode=width_source.get('mode','radial'),gt_assisted=False)
    if width_info['mode']=='solution':
        donor=arrays(width_source['solution'])
        for key in ('source_id','start','end','family'):
            if not np.array_equal(donor[key],d[key]):raise ValueError('Width donor identity mismatch')
        offset=int(np.floor(np.median(donor['state'][:,0]-initial[:,0])+.5))
        widths,count=aligned_widths(widths,donor['widths'],offset)
        width_info.update(offset=offset,mapped_intervals=count,sha256=sha256_file(width_source['solution']))
    elif width_info['mode']=='gt':
        events=remap_events(json.loads((dataset/'cross/cross_events.json').read_text())['events'],d)
        gd=arrays(gcfg['tensor_cache']);gs=arrays(gtroot/'uv_whole/solution.npz')
        _,_,keep,pick=uv.hard_assignment(gs['state'],len(gs['widths']),gcfg,best_nonbroken=True)
        if not keep.all():raise ValueError('GT must retain every manual reference')
        required=event_residuals(events,d,initial,None,len(widths),gd,gs,pick,
            dict(config,soft_winding_model=True),gcfg,widths)
        inferred_offset=enabled_l1(required,np.ones(len(required),dtype=bool),
            np.array([e['parallel'] for e in events]))['gt_global_offset']
        offset=width_source.get('global_offset',inferred_offset)
        if not np.isfinite(offset) or int(offset)!=offset:raise ValueError('Width offset must be integral')
        offset=int(offset)
        widths,count=aligned_widths(widths,gs['widths'],offset)
        gtq=event_arc_points(gd,gs['state'],pick,gs['widths'],
            np.array([e['gt_piece'] for e in events]),np.array([e['gt_arc'] for e in events]))
        width_info.update(offset=offset,initial_inferred_offset=inferred_offset,mapped_intervals=count,gt_assisted=True,
            sha256=sha256_file(gtroot/'uv_whole/solution.npz'),
            gt_supported_intervals=np.unique(np.floor(gtq).astype(int)).tolist(),
            alignment_residual_p10_p50_p90=np.quantile(required,[.1,.5,.9]).tolist())
    elif width_info['mode']!='radial':raise ValueError('Unknown width initialization')
    atomic_write(root/'width_initialization.json',width_info,durable=False)
    init=root/'init';init.mkdir(exist_ok=True)
    np.savez(init/'solution.npz',state=initial,widths=widths,**{k:d[k] for k in ('source_id','start','end','family')})
    np.savez(init/'samples.npz',**soft.prepare_samples(config,d))
    atomic_write(init/'config.json',dict(tensor_sha256=sha256_file(root/'tensors.npz'),soft_samples=str(init/'samples.npz')),durable=False)
    stages=[];previous=None
    for index,stage in enumerate(settings.get('initialization_stages',[])):
        template=json.loads(Path(stage['config']).read_text())
        stage_root=root/f'initialization_{index}'
        stage_config=initialization_config(template,config,stage_root,previous)
        if stage['model'] not in ('fiber_soft_winding','fiber_implicit_winding'):
            raise ValueError('Unknown initialization model')
        atomic_write(root/f'initialization_{index}_config.json',{stage['model']:stage_config},durable=False)
        if not (stage_root/'summary.json').exists():
            if stage['model']=='fiber_soft_winding':soft.run_prepared(stage_config)
            else:dispatch({stage['model']:stage_config})
        stage_saved=json.loads((stage_root/'config.json').read_text())
        if any(stage_saved.get(k)!=v for k,v in stage_config.items()):raise ValueError('Initialization configuration mismatch')
        stage_summary=json.loads((stage_root/'summary.json').read_text())
        if stage_summary['final']['iteration']!=stage_config['steps']:raise ValueError('Incomplete initialization')
        previous=stage_root/'solution.npz'
        stage_sol=arrays(previous)
        if any(not np.array_equal(stage_sol[k],d[k]) for k in ('source_id','start','end','family')):
            raise ValueError('Initialization identity mismatch')
        stages.append(dict(model=stage['model'],final=stage_summary['final'],solution_sha256=sha256_file(previous)))
    if previous is not None:config['initial_solution']=str(previous)
    initialization=(' -> '.join(stage['model'] for stage in stages)+' -> fiber_implicit_winding; fresh optimizer per stage' if stages else
                    'fresh radial; scaled joint L-BFGS' if config.get('optimizer_kind')=='scaled_lbfgs' else
                    'fresh radial; reset Adam')
    atomic_write(root/'solve_config.json',{'fiber_implicit_winding':config},durable=False)
    print(json.dumps(dict(sources=len(sources),pieces=len(d['family']),internal_events=len(d['i']),
        gt_supervision=width_info['gt_assisted'],initialization=initialization,width_initialization=width_info)),flush=True)
    if not (root/'solve/summary.json').exists():dispatch({'fiber_implicit_winding':config})
    saved=json.loads((root/'solve/config.json').read_text())
    if config.get('estimate_fixed_length_vram',False):
        from .fiber_scaled_lbfgs import estimated_fixed_length_vram
        config['gpu_manager']=dict(config['gpu_manager'],vram_bytes_per_gpu=
            estimated_fixed_length_vram(len(d['i']),config.get('lbfgs_edge_batch_size',len(d['i'])),config.get('winding_sigma',.5)))
    for key,value in config.items():
        if saved.get(key)!=value:raise ValueError('Saved solve config mismatch: '+key)
    summary=json.loads((root/'solve/summary.json').read_text())
    if config.get('optimizer_kind')=='scaled_lbfgs':
        if not summary.get('completed'):raise ValueError('Incomplete L-BFGS solve')
    elif summary['final']['iteration']!=config['steps']:raise ValueError('Incomplete solve')
    if any(k.startswith('fixed_reference') or k.startswith('reference_') for k in saved):
        raise ValueError('Evaluation solve must not consume references')
    return score_completed(settings,initialization=initialization,stages=stages,started=started)


def score_completed(settings,initialization=None,stages=None,started=None):
    """Established posthoc scoring, independently retryable without a solve."""
    from .fiber_implicit_match_report import run as match_report
    from .fiber_uv_arc_phase import event_arc_points
    started=time.monotonic() if started is None else started
    root=Path(settings['output_root']);dataset=Path(settings['dataset'])
    config=json.loads((root/'solve/config.json').read_text())
    d=arrays(config['tensor_cache']);sources=np.unique(d['source_id']);init=root/'init'
    gtroot=Path(settings['gt_root']);gcfg=json.loads((gtroot/'uv_whole/config.json').read_text())
    width_info=json.loads((root/'width_initialization.json').read_text())
    if initialization is None:
        initialization='fresh radial; scaled joint L-BFGS' if config.get('optimizer_kind')=='scaled_lbfgs' else 'see saved initialization provenance'
    stages=[] if stages is None else stages
    sol=arrays(root/'solve/solution.npz');state=sol['state'];widths=sol['widths']
    if float(config.get('width_lr',2))==0:
        if not np.allclose(widths,arrays(init/'solution.npz')['widths'],rtol=0,atol=.002):
            raise ValueError('Frozen widths changed')
    for key in ('source_id','start','end','family'):
        if not np.array_equal(sol[key],d[key]):raise ValueError('Changed solved identity')
    events=remap_events(json.loads((dataset/'cross/cross_events.json').read_text())['events'],d)
    gd=arrays(gcfg['tensor_cache']);gs=arrays(gtroot/'uv_whole/solution.npz')
    for event in events:
        if int(d['source_id'][event['test_piece']])!=event['test_source'] or int(gd['source_id'][event['gt_piece']])!=event['gt_source']:
            raise ValueError('Cross-event source identity mismatch')
    _,_,gkeep,gpick=uv.hard_assignment(gs['state'],len(gs['widths']),gcfg,best_nonbroken=True)
    if not gkeep.all():raise ValueError('GT must retain every manual reference')
    mapped=state.copy();mapped[:,0]=np.floor(mapped[:,0]+.5)
    required=event_residuals(events,d,mapped,None,len(widths),gd,gs,gpick,
        dict(config,soft_winding_model=True),gcfg,widths,saved_geometry=sol)
    enabled=state[np.array([e['test_piece'] for e in events]),3]>=0
    parallel=np.array([e['parallel'] for e in events])
    score=enabled_l1(required,enabled,parallel)
    gtq=event_arc_points(gd,gs['state'],gpick,gs['widths'],
        np.array([e['gt_piece'] for e in events]),np.array([e['gt_arc'] for e in events]))
    score['gt_solve_widths']=gt_solve_width_metrics(widths,gs['widths'],np.floor(gtq).astype(int),score['gt_global_offset'])
    error=abs(required-score['gt_global_offset'])
    for key,mask in [('enabled',enabled),('parallel',enabled&parallel),('perpendicular',enabled&~parallel),('disabled',~enabled)]:
        score[key+'_error_quantiles']=dict(zip(('median','p95','max'),map(float,np.quantile(error[mask],[.5,.95,1])))) if mask.any() else None
    score['coverage_by_type']={name:dict(total=int(mask.sum()),enabled=int((enabled&mask).sum()),
        percent=float(100*enabled[mask].mean()) if mask.any() else None) for name,mask in [('parallel',parallel),('perpendicular',~parallel)]}
    atomic_write(root/'gt_events.json',[dict(e,enabled=bool(enabled[n]),residual=float(required[n]-score['gt_global_offset'])) for n,e in enumerate(events)],durable=False)
    matches=match_report(dict(solve_roots=[str(root/'solve')],output_root=str(root/'matches'),cutoffs=[.5,.7]))
    result=dict(sources=len(sources),pieces=len(d['family']),internal_events=len(d['i']),
        internal_parallel=int(d['parallel'].sum()),internal_perpendicular=int((~d['parallel']).sum()),
        continuation_links=len(d['links']),gt_supervision=width_info['gt_assisted'],width_initialization=width_info,initialization=initialization,initialization_stages=stages,
        final=json.loads((root/'solve/summary.json').read_text())['final'],score=score,matches=matches,
        wall_seconds=time.monotonic()-started,solution_sha256=sha256_file(root/'solve/solution.npz'),
        caveat=('GT-assisted width diagnostic; not GT-free.' if width_info['gt_assisted'] else
            'Optimization is GT-free; dataset membership was curated using GT. Not a wholly independent held-out set.'))
    atomic_write(root/'results.json',result,durable=False);print(json.dumps(result),flush=True);return result
