"""Immutable vc_wgraph result stages, backed by the existing graph solvers.

The selected solver arrays are work products, not a second graph format. Only
typed result rows are persisted; geometry and measurements remain referenced.
"""
import json
from pathlib import Path
import uuid

import numpy as np
from vc_object_store import CollectionReader

from .fiber_graph_store import FiberGraph, PREFIX, write_results


def _uuid(value):
    return str(uuid.UUID(bytes=value)) if isinstance(value,bytes) else str(uuid.UUID(str(value)))


def _selected_run(database, stage, selected=None):
    """Require an unambiguous explicit stage run, never silently pick latest."""
    with CollectionReader(str(database)) as store:
        definitions={row['name']:row for row in store.extension_tables()}
        table=definitions.get(PREFIX+'runs')
        if table is None:raise ValueError(f'Input has no {stage} result run')
        columns=[c['name'] for c in table['columns']]
        rows=[dict(zip(columns,row)) for row in store.extension_rows(PREFIX+'runs')]
    candidates=[_uuid(row['run_uuid']) for row in rows if row['stage']==stage]
    if selected is not None:
        selected=_uuid(selected)
        if selected not in candidates:raise ValueError(f'Selected run is not an input {stage} run')
        return selected
    if len(candidates)!=1:raise ValueError(f'Expected one {stage} run; specify input_run_uuid')
    return candidates[0]


def _fiber_rows(database, run):
    with CollectionReader(str(database)) as store:
        table=next(t for t in store.extension_tables() if t['name']==PREFIX+'fibers')
        columns=[c['name'] for c in table['columns']]
        rows=store.extension_rows(PREFIX+'fibers')
    return {_uuid(row[0]):dict(zip(columns,row)) for row in rows if _uuid(row[1])==run}


def _run_inputs(database, run):
    with CollectionReader(str(database)) as store:
        table=next(t for t in store.extension_tables() if t['name']==PREFIX+'inputs')
        columns=[c['name'] for c in table['columns']]
        rows=store.extension_rows(PREFIX+'inputs',[uuid.UUID(run).bytes])
    return [dict(zip(columns,row)) for row in rows]


def _base_database(config, prior_source=None):
    """Resolve a pinned graph source via the existing collection member map."""
    database=Path(config['database']).resolve()
    explicit=config.get('base_database')
    if explicit:
        candidate=Path(explicit).resolve()
        with CollectionReader(str(candidate)) as store:
            source=store.source
            if prior_source is not None and source!=prior_source:
                raise ValueError('base_database does not match the prior snapshot')
        if candidate!=database:
            with CollectionReader(str(database)) as store:
                if not any(all(m[k]==source[k] for k in ('store_uuid','generation','commit_sequence'))
                           for m in store.members if not m['partition']):
                    raise ValueError('base_database is not a pinned input dependency')
        return candidate
    if prior_source is None:
        with CollectionReader(str(database)) as store:
            names={table['name'] for table in store.extension_tables()}
            runs=store.extension_rows(PREFIX+'runs') if PREFIX+'runs' in names else []
        if not any(row[1]=='classification' for row in runs):return database
        classified=_selected_run(database,'classification',config.get('classification_run_uuid'))
        inputs=[row for row in _run_inputs(database,classified) if row['role']=='input_0']
        if len(inputs)!=1:raise ValueError('Classification must pin its extraction input')
        source=inputs[0]
        prior_source=dict(store_uuid=_uuid(source['store_uuid']),generation=_uuid(source['generation_uuid']),
                          commit_sequence=source['commit_sequence'])
    with CollectionReader(str(database)) as store:
        if store.source==prior_source:return database
        matches=[Path(m['container']) for m in store.members if not m['partition']
                 and all(m[k]==prior_source[k] for k in ('store_uuid','generation','commit_sequence'))]
    if len(matches)!=1:raise ValueError('Cannot resolve unique prior source in input store references')
    return matches[0]


def _graph(config, base):
    root=Path(config.get('graph_cache',str(base)+'.graph-cache'))
    if (root/'manifest.json').exists():return FiberGraph.cached(base,root)
    graph=FiberGraph(base)
    graph.save_cache(root)
    return FiberGraph.cached(base,root)


def _solver_config(config, root):
    settings=dict(config)
    if settings.get('optimizer','wrapped_irls')!='wrapped_irls':
        raise ValueError('vc_wgraph hvb/sign currently use the existing wrapped_irls backend')
    manager=dict(settings.get('gpu_manager',{}))
    manager.setdefault('exclusive',False)
    settings.update(optimizer='wrapped_irls',gpu_manager=manager,devices=settings.get('devices','1'),
                    output_root=str(root),write_result_store=False)
    # This is a solve, not a benchmark sweep: its output must identify one run.
    if len(settings.get('broken_penalties',[.1]))!=1 or len(settings.get('cost_modes',['uniform']))!=1:
        raise ValueError('Choose one broken_penalty/cost_mode for a result stage')
    settings.setdefault('broken_penalties',[.1]);settings.setdefault('cost_modes',['uniform'])
    settings.setdefault('entropy_weights',[0.])
    return settings


def run_stage(stage, config):
    """Execute a stage with ``database`` input and ``result_store`` output."""
    if stage in ('winding','uv'):
        from .fiber_wgraph_geometry import run_stage as geometry_stage
        return geometry_stage(stage,config)
    if stage not in ('hvb','sign'):raise ValueError(f'Unknown downstream stage {stage}')
    from . import fiber_hv_stage as hv, fiber_normal_sign as normal
    destination=Path(config['result_store'])
    if destination.exists():raise FileExistsError(destination)
    root=Path(config.get('output_root',str(destination)+'.work'))
    settings=_solver_config(config,root)
    root.mkdir(parents=True,exist_ok=True)
    selected=None
    if stage=='hvb':
        base=_base_database(config)
        graph=_graph(config,base)
        settings.update(database=str(base),tensor_cache=str(root/'tensors.npz'))
        if config.get('stage1'):raise ValueError('vc_wgraph hvb runs before normal-sign alignment')
        if not settings.get('features'):
            if not config.get('native_config'):
                raise ValueError('hvb needs axial features, or native_config to generate them with the existing extractor')
            native=(dict(config['native_config']) if isinstance(config['native_config'],dict)
                    else json.loads(Path(config['native_config']).read_text()))
            features=root/'axial_features.json'
            native.pop('database',None)
            for key in ('normal_prior_output','normal_curves_output','orient_database'):
                native.pop(key,None)
            native.update(fibers=str(base),axial_prior_output=str(features),axial_prior_only=True,orient_normals=False,
                          output=str(root/'axial_report.json'),progress_file=str(root/'axial_progress.jsonl'))
            hv.run_features(dict(settings,native_config=native))
            settings['features']=str(features)
        hv.prepare(settings,graph=graph)
        summary=normal.launch_tensor_solve(settings,hv.__name__,hv.__file__)
        mode=settings['cost_modes'][0];penalty=settings['broken_penalties'][0]
        solution=root/f'solution_{mode}_b{penalty:g}_e0.npz'
        with np.load(solution) as saved:
            rows=[dict(fiber_uuid=str(fid),family=int(label),enabled=int(label!=2),
                       p_h=float(p[0]),p_v=float(p[1]),p_broken=float(p[2]))
                  for fid,label,p in zip(saved['ids'],saved['labels'],saved['probabilities'])]
        result_stage='classification'
    else:
        selected=_selected_run(config['database'],'classification',config.get('input_run_uuid'))
        prior=json.loads(Path(config['priors']).read_text())
        base=_base_database(config,prior['source_store'])
        graph=_graph(config,base)
        classifications=_fiber_rows(config['database'],selected)
        if set(classifications)!=set(graph.ids):raise ValueError('Classification does not cover the extracted graph')
        keep=np.asarray([classifications[fid]['enabled'] for fid in graph.ids],np.float32)
        if not np.isin(keep,[0,1]).all():raise ValueError('Classification enabled mask is not binary')
        settings.update(database=str(base),tensor_cache=str(root/'tensors.npz'),fixed_keep=True)
        normal.prepare(settings,graph=graph)
        cache=Path(settings['tensor_cache'])
        with np.load(cache) as prepared:arrays={key:prepared[key] for key in prepared.files}
        arrays['keep']=keep
        np.savez(cache,**arrays)
        summary=normal.launch_tensor_solve(settings,normal.__name__,normal.__file__)
        solution=root/f"solution_{settings['broken_penalties'][0]:g}.npz"
        with np.load(solution) as saved:
            if not np.array_equal(saved['broken'],keep<=.5):raise RuntimeError('Sign alignment changed classification')
            rows=[dict(fiber_uuid=str(fid),normal_sign=int(sign),sign_probability=float(p))
                  for fid,sign,p in zip(saved['ids'],saved['normal_sign'],saved['sign_probability'])]
        result_stage='normal_sign'
    run=write_results(destination,[config['database']],result_stage,'wrapped_irls',{'fibers':rows},
                      settings=dict(fixed_keep=stage=='sign',prior_weight=float(settings.get('prior_weight',.001))),
                      input_runs=[selected])
    return dict(stage=stage,result_store=str(destination),run_uuid=run,input_run_uuid=selected,
                solver_summary=summary)
