"""Store-to-store winding and fixed-assignment UV workflow adapters.

Public inputs and authoritative outputs are immutable referenced vcobj stores;
no reference/GT labels are read. UV preparation returns native owned arrays
directly to the solver in the managed process, without tensor staging files.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from .fiber_graph_store import write_results


def _input_lineage(config, winding_run=None):
    """Resolve exact stage ancestors; explicit selectors may not override pins."""
    from .fiber_wgraph_stages import _selected_run,_run_inputs,_uuid
    database=config['database']
    def parent(run,stage):
        rows=[row for row in _run_inputs(database,run) if row['role']=='input_0']
        if len(rows)!=1 or rows[0]['input_run_uuid'] is None:
            raise ValueError(f'{stage} ancestor is not pinned by selected input run')
        return _selected_run(database,stage,_uuid(rows[0]['input_run_uuid']))
    if winding_run is None:
        sign=_selected_run(database,'normal_sign',config.get('normal_sign_run_uuid'))
    else:
        sign=parent(winding_run,'normal_sign')
        explicit=config.get('normal_sign_run_uuid')
        if explicit is not None and _uuid(explicit)!=sign:
            raise ValueError('Selected normal-sign run differs from winding input ancestry')
    classified=parent(sign,'classification')
    explicit=config.get('classification_run_uuid')
    if explicit is not None and _uuid(explicit)!=classified:
        raise ValueError('Selected classification run differs from normal-sign input ancestry')
    return dict(classification=classified,normal_sign=sign)


def _prepare(config, *, build_graph=True, in_memory=False):
    from .fiber_wgraph_stages import _fiber_columns, _graph, _base_database
    from . import fiber_min_flow_full as flow
    from .fiber_graph_store import store_digest
    from .fiber_min_flow_experiment import arrays, write
    started=time.monotonic();timings={}
    database=config['database']
    selected=_input_lineage(config)
    config=dict(config,classification_run_uuid=selected['classification'],normal_sign_run_uuid=selected['normal_sign'])
    root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    timings['lineage_and_setup']=time.monotonic()-started
    phase=time.monotonic()
    base=_base_database(config)
    graph=_graph(config,base)
    timings['graph_cache_open']=time.monotonic()-phase;phase=time.monotonic()
    classifications=_fiber_columns(database,selected['classification'],graph.ids,required=('family','enabled'),exact=True)
    timings['classification_columns']=time.monotonic()-phase;phase=time.monotonic()
    signs=_fiber_columns(database,selected['normal_sign'],graph.ids,required=('normal_sign',),exact=True)
    timings['sign_columns']=time.monotonic()-phase;phase=time.monotonic()
    binding=dict(config=config,source=store_digest(database),input_runs=selected)
    manifest=root/'request.json'
    if manifest.exists() and json.loads(manifest.read_text())!=binding:
        raise ValueError('Geometry work directory belongs to different inputs/settings')
    write(manifest,binding)
    geometry_binding=dict(source=store_digest(base),input_runs=selected)
    geometry=root/'geometry'
    if config.get('geometry_cache'):
        cache=Path(config['geometry_cache']).resolve()
        source=cache/'source.json'
        if not source.exists() or json.loads(source.read_text())!=geometry_binding:
            raise ValueError('Geometry cache source snapshot or selected input runs differ')
        if geometry.exists() and geometry.resolve()!=cache:
            raise ValueError('Work geometry path differs from configured geometry_cache')
        if not geometry.exists():geometry.symlink_to(cache,target_is_directory=True)
    labels=classifications['family'];enabled=classifications['enabled']
    if not np.isin(enabled,[0,1]).all() or not np.isin(labels,[0,1,2]).all():
        raise ValueError('Invalid classification readout')
    labels=np.where(enabled,labels,2)
    if not np.any(labels!=2):raise ValueError('No enabled H/V fibers remain')
    normal_sign=np.asarray(signs['normal_sign'],np.int8)
    if not np.isin(normal_sign,[-1,1]).all():raise ValueError('Invalid normal sign')
    comparison=root/'classification.npz';normal=root/'normal_sign.npz'
    if not in_memory:
        np.savez(comparison,ids=graph.ids,labels=labels)
        np.savez(normal,ids=graph.ids,normal_sign=normal_sign,normal_convention=np.array('inward'))
    settings=dict(config,database=str(base),fibers=database,comparison=str(comparison),normal_solution=str(normal),
                  graph_cache=str(config.get('graph_cache',str(base)+'.graph-cache')))
    geometry.mkdir(exist_ok=True)
    timings['bind_and_stage_columns']=time.monotonic()-phase;phase=time.monotonic()
    print(json.dumps(dict(stage='geometry_readout_prepared',seconds=time.monotonic()-started,timings=timings)),flush=True)
    if in_memory:
        catalog=flow.prepare(settings,build_graph=build_graph,reuse_catalog=bool(config.get('geometry_cache')),
            comparison=dict(ids=graph.ids,labels=labels),
            normal=dict(ids=graph.ids,normal_sign=normal_sign,normal_convention=np.array('inward')))
    elif config.get('geometry_cache'):
        catalog=flow.prepare(settings,build_graph=build_graph,reuse_catalog=True)
    else:
        catalog=flow.prepare(settings) if build_graph else flow.prepare(settings,build_graph=False)
    timings['catalog_and_geometry_and_constraints']=time.monotonic()-phase;phase=time.monotonic()
    source=geometry/'source.json'
    if source.exists() and json.loads(source.read_text())!=geometry_binding:
        raise ValueError('Geometry cache source snapshot changed')
    if not source.exists():write(source,geometry_binding)
    if not build_graph:
        write(root/'preparation-timings.json',dict(seconds=time.monotonic()-started,stages=timings))
        return catalog,selected,settings
    parts=[arrays(p) for p in sorted((root/'graph_parts').glob('*.npz')) if '.partial.' not in p.name]
    if not parts:raise ValueError('No usable winding constraints')
    from .fiber_min_flow import aggregate_l1_edges
    g=aggregate_l1_edges(**{k:np.concatenate([p[k] for p in parts]) for k in ('a','b','target','weight')})
    np.savez(root/'graph.npz',**g)
    timings['concatenate_aggregate_save']=time.monotonic()-phase
    write(root/'preparation-timings.json',dict(seconds=time.monotonic()-started,stages=timings))
    return catalog,selected,settings


def solve(config):
    """Managed CUDA-child entry; exact historic linear IRLS implementation."""
    if config.get('stage_mode')=='uv_direct':
        return _solve_uv_direct(config)
    import torch
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from .fiber_linear_irls import solve_l1,round_components
    from .fiber_min_flow_experiment import arrays,write
    root=Path(config['output_root']);g=arrays(root/'graph.npz')
    catalog=arrays(root/'geometry'/'catalog.npz');n=len(catalog['source_id'])
    _,component=connected_components(coo_matrix((np.ones(len(g['a'])),(g['a'],g['b'])),shape=(n,n)),directed=False)
    torch.set_num_threads(config.get('torch_threads',4))
    started=time.monotonic()
    x,residual=solve_l1(n,g['a'],g['b'],g['target'],g['weight'],np.zeros(n),
        config.get('prior_weight',1e-6),dev='cuda',irls=config.get('irls',15),
        cg_iters=config.get('cg',500),eps=config.get('eps',.05),log=lambda x:print(x,flush=True))
    if not np.isfinite(x).all():raise RuntimeError('IRLS returned nonfinite winding potentials')
    labels=round_components(x,component)
    np.savez(root/'solution.npz',source_id=catalog['source_id'],family=catalog['family'],
             winding=labels,continuous=x,component=component)
    summary=dict(fibers=n,arcs=len(g['a']),seconds=time.monotonic()-started,
        weighted_l1=float(np.dot(g['weight'],residual)),mean_residual=float(residual.mean()),
        max_residual=float(residual.max()),expanded_arcs_within_half_turn_percent=float(100*np.mean(residual<=.5)))
    write(root/'summary.json',summary)
    return summary


def run_stage(stage,config):
    from .fiber_normal_sign import launch_tensor_solve
    from .fiber_min_flow_experiment import arrays
    destination=Path(config['result_store'])
    if destination.exists():raise FileExistsError(destination)
    root=Path(config.get('output_root',str(destination)+'.work'))
    settings=dict(config,output_root=str(root),devices=config.get('devices','1'))
    manager=dict(settings.get('gpu_manager',{}));manager.setdefault('exclusive',False);settings['gpu_manager']=manager
    if stage=='uv':return _run_uv(settings)
    if stage!='winding':raise ValueError(stage)
    if settings.get('optimizer','irls')!='irls':raise ValueError('Winding currently supports the tested linear IRLS backend')
    catalog,selected,_=_prepare(settings)
    summary=launch_tensor_solve(settings,__name__,__file__)
    saved=arrays(root/'solution.npz')
    rows=[dict(fiber_uuid=str(fid),enabled=1,family=int(catalog['family'][i]),
               selected_winding=int(saved['winding'][i]),winding_mean=float(saved['continuous'][i]),
               component=int(saved['component'][i])) for i,fid in enumerate(catalog['source_id'])]
    # IRLS returns point estimates, not a calibrated uncertainty distribution.
    metrics=[dict(iteration=int(settings.get('irls',15)),scope='all',name=name,value=float(summary[name]),
                  units='turns' if name.endswith('residual') else 'weighted_turns')
             for name in ('weighted_l1','mean_residual','max_residual')]
    run=write_results(destination,[settings['database']],'winding','linear_irls',dict(fibers=rows,metrics=metrics),
        settings={k:settings.get(k,v) for k,v in dict(irls=15,cg=500,eps=.05,prior_weight=1e-6).items()},
        input_runs=[selected['normal_sign']])
    return dict(result_store=str(destination),run_uuid=run,solver_summary=summary)


def _run_uv(config):
    from .fiber_normal_sign import launch_tensor_solve
    from .fiber_wgraph_stages import _selected_run
    winding_run=_selected_run(config['database'],'winding',config.get('input_run_uuid'))
    _input_lineage(config,winding_run)
    result=launch_tensor_solve(dict(config,stage_mode='uv_direct'),__name__,__file__)
    return dict(result_store=config['result_store'],run_uuid=result['run_uuid'],solver_summary=result)


def _solve_uv_direct(config):
    """Native parallel preparation and GPU optimization share one process."""
    from .fiber_wgraph_stages import _selected_run,_fiber_columns
    from .fiber_min_flow_full import PackedPaths
    from .fiber_uv_native_prepare import prepare_events
    from .fiber_flow_uv import prepare_data,solve as solve_uv
    from .fiber_min_flow_experiment import write
    from vc_object_store import CollectionReader
    root=Path(config['output_root'])
    started=time.monotonic()
    with CollectionReader(config['database']) as source:
        input_source=source.source
    winding_run=_selected_run(config['database'],'winding',config.get('input_run_uuid'))
    lineage=_input_lineage(config,winding_run)
    config=dict(config,classification_run_uuid=lineage['classification'],normal_sign_run_uuid=lineage['normal_sign'])
    catalog,_,native=_prepare(config,build_graph=False,in_memory=True)
    ids=catalog['source_id']
    winding=_fiber_columns(config['database'],winding_run,ids,required=('selected_winding','component'))
    paths=PackedPaths(root/'geometry')
    from .fiber_wgraph_stages import _graph,_base_database
    graph=_graph(config,_base_database(config))
    paths.bind_graph(graph)
    selected=config.get('component_selection','all')
    if selected not in ('all','largest'):raise ValueError('component_selection must be all or largest')
    node_mask=None if selected=='all' else winding['component']==np.bincount(winding['component']).argmax()
    phase=time.monotonic()
    d,chosen=prepare_events(paths,graph,workers=int(config.get('workers',32)),node_mask=node_mask)
    print(json.dumps(dict(stage='uv_native_events',seconds=time.monotonic()-phase,
                         fibers=len(chosen),events=len(d['i']))),flush=True)
    parameters=dict(config.get('parameters',{}),write_geometry_store=False,packed_geometry=str(root/'geometry'),
                    workers=int(config.get('workers',32)),output_root=str(root),
                    result_store=config['result_store'],input_run_uuid=winding_run,input_store=config['database'],
                    input_source=input_source)
    prepared,selection=prepare_data(parameters,d,winding['selected_winding'][chosen],paths,supported_input=True)
    selection['input_fibers']=len(ids)
    write(root/'selection.json',selection)
    write(root/'uv-preparation-timings.json',dict(seconds=time.monotonic()-started,
        fibers=len(prepared['source_id']),events=len(prepared['i']),tensor_staging_bytes=0))
    # Release the source graph and construction-only columns before allocating
    # GPU buffers; prepared arrays retain native ownership through capsules.
    del d,graph,paths,catalog,winding
    return solve_uv(parameters,prepared,selection)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--unmanaged',action='store_true');parser.add_argument('--devices')
    args=parser.parse_args()
    if not args.unmanaged:raise RuntimeError('Use the managed workflow')
    solve(json.loads(Path(args.config).read_text()))
