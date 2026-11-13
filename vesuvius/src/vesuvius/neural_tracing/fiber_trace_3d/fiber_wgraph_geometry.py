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
    raise RuntimeError('File-staged geometry preparation was removed; use the direct managed stage API')


def solve(config):
    """Managed CUDA-child entry; exact historic linear IRLS implementation."""
    if config.get('stage_mode')=='uv_direct':
        return _solve_uv_direct(config)
    if config.get('stage_mode')=='winding_direct':
        return _solve_winding_direct(config)
    raise ValueError('Use stage_mode=winding_direct or uv_direct; disk-staged solver inputs are unsupported')


def run_stage(stage,config):
    from .fiber_normal_sign import launch_tensor_solve
    destination=Path(config['result_store'])
    if destination.exists():raise FileExistsError(destination)
    root=Path(config.get('output_root',str(destination)+'.work'))
    settings=dict(config,output_root=str(root),devices=config.get('devices','1'))
    manager=dict(settings.get('gpu_manager',{}));manager.setdefault('exclusive',False);settings['gpu_manager']=manager
    if stage=='uv':return _run_uv(settings)
    if stage!='winding':raise ValueError(stage)
    if settings.get('optimizer','irls')!='irls':raise ValueError('Winding currently supports the tested linear IRLS backend')
    summary=launch_tensor_solve(dict(settings,stage_mode='winding_direct'),__name__,__file__)
    return dict(result_store=str(destination),run_uuid=summary['run_uuid'],solver_summary=summary)


def _read_winding_inputs(config,input_source,selected,*,stage):
    """Shared native graph/result/geometry preparation for winding and UV."""
    from .fiber_wgraph_stages import _graph,_base_database,_fiber_columns
    from .fiber_min_flow_full import PackedPaths
    from .fiber_graph_store import workflow_volume
    config=dict(config,classification_run_uuid=selected['classification'],normal_sign_run_uuid=selected['normal_sign'])
    timings={};phase=time.monotonic()
    base=_base_database(config)
    graph=_graph(config,base)
    timings['native_graph_read']=time.monotonic()-phase;phase=time.monotonic()
    classified=_fiber_columns(config['database'],selected['classification'],graph.ids,
                              required=('family','enabled'),exact=True)
    signs=_fiber_columns(config['database'],selected['normal_sign'],graph.ids,
                        required=('normal_sign',),exact=True)
    family=classified['family'];enabled=classified['enabled'];normal_sign=signs['normal_sign']
    if not np.isin(family,[0,1,2]).all() or not np.isin(enabled,[0,1]).all():
        raise ValueError('Invalid classification readout')
    if not np.isin(normal_sign,[-1,1]).all():raise ValueError('Invalid normal sign')
    keep=(enabled!=0)&(family!=2)
    if not keep.any():raise ValueError('No enabled H/V fibers remain')
    ids=graph.ids[keep];family=family[keep].astype(np.int64);normal_sign=normal_sign[keep].astype(np.int64)
    timings['native_result_columns']=time.monotonic()-phase;phase=time.monotonic()
    print(json.dumps(dict(stage=stage+'_native_readout',
        seconds=timings['native_graph_read']+timings['native_result_columns'],
        fibers=len(ids),constraints=len(graph.event_ids))),flush=True)
    # Return owned native buffers directly. No geometry-cache, numeric-graph,
    # or solver-tensor file is created or opened at this boundary.
    paths=PackedPaths.from_store(dict(config,input_store=config['database'],input_source=input_source,
        working_volume=workflow_volume(dict(config,database=str(base)))),ids)
    if not np.array_equal(paths.ids,ids):raise ValueError('Native geometry fiber identities differ')
    paths.catalog.update(family=family,normal_sign=normal_sign)
    paths.bind_graph(graph)
    timings['native_geometry_read_and_transport']=time.monotonic()-phase
    print(json.dumps(dict(stage=stage+'_native_geometry',
        seconds=timings['native_geometry_read_and_transport'],fibers=len(ids),points=len(paths.xyz))),flush=True)
    return graph,paths,ids,family,normal_sign,timings


def _solve_winding_direct(config):
    """Read, prepare, optimize and persist in one managed process.

    Native geometry/graph arrays own their RAM; neither intermediate arrays nor
    the solution are serialized for another process to reload.
    """
    import torch
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from vc_object_store import CollectionReader
    from .fiber_graph_store import write_fiber_result_columns
    from .fiber_packed_priors import _geometry
    from .fiber_linear_irls import solve_l1,round_components
    from .fiber_min_flow_experiment import write
    started=time.monotonic()
    root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    with CollectionReader(config['database']) as source:input_source=source.source
    selected=_input_lineage(config)
    graph,paths,ids,family,normal_sign,timings=_read_winding_inputs(config,input_source,selected,stage='winding')
    phase=time.monotonic()
    g=_geometry(paths).prepare_winding_graph(
        graph.i,graph.j,np.ascontiguousarray(paths.graph_index,dtype=np.int64),family,normal_sign,
        graph.arc_a,graph.arc_b,graph.distance,graph.parallel,
        graph.normal_a,graph.normal_b,graph.connector,graph.valid_a,graph.valid_b,graph.valid_connector,
        lasagna_winding_scale=float(config.get('lasagna_winding_scale',1.)),
        perpendicular_offset=float(config.get('perpendicular_offset',.5)),
        perpendicular_relative_weight=float(config.get('perpendicular_relative_weight',1.)),
        constraint_batch=int(config.get('constraint_batch',500000)),
        read_batch=int(config.get('read_batch',32768)),
        workers=int(config.get('workers',32)))
    timings['native_winding_graph']=time.monotonic()-phase
    with CollectionReader(config['database']) as source:
        if source.source!=input_source:raise ValueError('Source changed during winding preparation')
    n=len(ids)
    if not len(g['a']):raise ValueError('No usable winding constraints')
    del graph,paths
    phase=time.monotonic()
    _,component=connected_components(coo_matrix((np.ones(len(g['a'])),(g['a'],g['b'])),shape=(n,n)),directed=False)
    timings['components']=time.monotonic()-phase;phase=time.monotonic()
    torch.set_num_threads(config.get('torch_threads',4))
    x,residual=solve_l1(n,g['a'],g['b'],g['target'],g['weight'],np.zeros(n),
        config.get('prior_weight',1e-6),dev='cuda',irls=config.get('irls',15),
        cg_iters=config.get('cg',500),eps=config.get('eps',.05),log=lambda x:print(x,flush=True))
    if not np.isfinite(x).all():raise RuntimeError('IRLS returned nonfinite winding potentials')
    labels=round_components(x,component)
    timings['solve']=time.monotonic()-phase
    summary=dict(fibers=n,arcs=len(g['a']),seconds=timings['solve'],
        weighted_l1=float(np.dot(g['weight'],residual)),mean_residual=float(residual.mean()),
        max_residual=float(residual.max()),expanded_arcs_within_half_turn_percent=float(100*np.mean(residual<=.5)))
    columns=dict(enabled=np.ones(n,np.int64),family=family,selected_winding=labels.astype(np.int64),
                 winding_mean=x,component=component.astype(np.int64))
    metrics=[dict(iteration=int(config.get('irls',15)),scope='all',name=name,value=float(summary[name]),
                  units='turns' if name.endswith('residual') else 'weighted_turns')
             for name in ('weighted_l1','mean_residual','max_residual')]
    summary['run_uuid']=write_fiber_result_columns(config['result_store'],config['database'],input_source,
        'winding','linear_irls',ids,columns,metrics=metrics,
        settings={k:config.get(k,v) for k,v in dict(irls=15,cg=500,eps=.05,prior_weight=1e-6).items()},
        input_run=selected['normal_sign'],workers=int(config.get('workers',32)),
        timings_path=root/'result-export-timings.json')
    summary.update(preparation_timings=timings,tensor_staging_bytes=0,wall_seconds=time.monotonic()-started)
    write(root/'summary.json',summary)
    return summary


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
    from .fiber_uv_native_prepare import prepare_events
    from .fiber_flow_uv import prepare_data,solve as solve_uv
    from .fiber_min_flow_experiment import write
    from vc_object_store import CollectionReader
    root=Path(config['output_root'])
    root.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    with CollectionReader(config['database']) as source:
        input_source=source.source
    winding_run=_selected_run(config['database'],'winding',config.get('input_run_uuid'))
    lineage=_input_lineage(config,winding_run)
    graph,paths,ids,family,normal_sign,timings=_read_winding_inputs(config,input_source,lineage,stage='uv')
    phase=time.monotonic()
    winding=_fiber_columns(config['database'],winding_run,ids,required=('selected_winding','component'))
    timings['winding_result_columns']=time.monotonic()-phase
    selected=config.get('component_selection','all')
    if selected not in ('all','largest'):raise ValueError('component_selection must be all or largest')
    node_mask=None if selected=='all' else winding['component']==np.bincount(winding['component']).argmax()
    phase=time.monotonic()
    d,chosen=prepare_events(paths,graph,workers=int(config.get('workers',32)),node_mask=node_mask)
    print(json.dumps(dict(stage='uv_native_events',seconds=time.monotonic()-phase,
                         fibers=len(chosen),events=len(d['i']))),flush=True)
    parameters=dict(config.get('parameters',{}),write_geometry_store=False,write_solver_checkpoints=False,
                    workers=int(config.get('workers',32)),output_root=str(root),
                    result_store=config['result_store'],input_run_uuid=winding_run,input_store=config['database'],
                    input_source=input_source)
    parameters.pop('packed_geometry',None)
    prepared,selection=prepare_data(parameters,d,winding['selected_winding'][chosen],paths,supported_input=True)
    with CollectionReader(config['database']) as source:
        if source.source!=input_source:raise ValueError('UV input snapshot changed during native preparation')
    selection['input_fibers']=len(ids)
    write(root/'selection.json',selection)
    write(root/'uv-preparation-timings.json',dict(seconds=time.monotonic()-started,stages=timings,
        fibers=len(prepared['source_id']),events=len(prepared['i']),tensor_staging_bytes=0))
    # Release the source graph and construction-only columns before allocating
    # GPU buffers; prepared arrays retain native ownership through capsules.
    del d,graph,paths,winding,family,normal_sign
    if parameters.get('visualize',False):
        from .fiber_flow_uv import visualize
        snapshots={}
        result=solve_uv(parameters,prepared,selection,snapshots=snapshots)
        visualize(parameters,prepared,snapshots=snapshots)
        return result
    return solve_uv(parameters,prepared,selection)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--unmanaged',action='store_true');parser.add_argument('--devices')
    args=parser.parse_args()
    if not args.unmanaged:raise RuntimeError('Use the managed workflow')
    solve(json.loads(Path(args.config).read_text()))
