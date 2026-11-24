"""Immutable vc_wgraph result stages, backed by the existing graph solvers.

The selected solver arrays are work products, not a second graph format. Only
typed result rows are persisted; geometry and measurements remain referenced.
"""
import json
from pathlib import Path
import time
import uuid

import numpy as np
from vc_object_store import CollectionReader

from .fiber_graph_store import FiberGraph, PREFIX


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
        rows=store.extension_rows(PREFIX+'fibers',key_filter={'run_uuid':uuid.UUID(run).bytes})
    return {_uuid(row[0]):dict(zip(columns,row)) for row in rows}


def _fiber_columns(database, run, ids=None, *, required=(), exact=False):
    """One run's native numeric projection, without per-fiber Python records.

    Selected run identity remains explicit. Source input pins are carried by
    the run's ordinary input rows; these arrays do not invent new references.
    """
    with CollectionReader(str(database)) as store:
        columns=store.fiber_result_columns(run)
    source_ids=np.frombuffer(columns['uuid_ascii'],dtype='S36').astype('U36')
    values=columns['values'];valid=columns['valid']
    if ids is not None:
        ids=np.asarray(ids)
        aligned=np.array_equal(source_ids,ids)
        if exact and not aligned:
            raise ValueError('Result rows must cover exactly the extracted graph')
        if not aligned:
            index=np.searchsorted(source_ids,ids)
            if np.any(index>=len(source_ids)) or not np.array_equal(source_ids[index],ids):
                raise ValueError('Result rows do not cover the selected fibers')
            values=values[index];valid=valid[index]
        source_ids=ids
    names=columns['columns'];out={'ids':source_ids}
    if set(required)-set(names):raise ValueError('Unknown numeric result field')
    for i,name in enumerate(names):
        if name in required and not valid[:,i].all():
            raise ValueError(f'Result run has missing {name} values')
        out[name]=values[:,i]
        out[name+'_valid']=valid[:,i]
    return out


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
    """Read native owned columns directly; solver inputs never pass via files."""
    started=time.monotonic()
    print(json.dumps(dict(stage='native_graph_read_started')),flush=True)
    graph=FiberGraph(base,workers=int(config.get('workers',32)),progress=True)
    read_seconds=time.monotonic()-started
    print(json.dumps(dict(stage='native_graph_read_complete',seconds=read_seconds,
                          fibers=len(graph.ids),constraints=len(graph.event_ids))),flush=True)
    return graph


def _solver_config(config, root):
    settings=dict(config)
    settings.setdefault('sqlite_memory_statistics',False)
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


def solve_residuals(database, run):
    """One scaffold run's violated constraints, as the dock ranks them.

    The rows are what `vc_fiber_graph_solve` persisted (ext_fiber_graph_solve_
    residuals): one per constraint the winding solve could not satisfy, with
    the shortest cycle it lies on. `run` is the explicit run UUID; nothing
    here picks a run for the caller.
    """
    with CollectionReader(str(database)) as store:
        table=next((t for t in store.extension_tables() if t['name']==PREFIX+'residuals'),None)
        if table is None:raise ValueError('Input has no scaffold residual rows')
        columns=[c['name'] for c in table['columns']]
        rows=store.extension_rows(PREFIX+'residuals',key_filter={'run_uuid':uuid.UUID(_uuid(run)).bytes})
    out=[]
    for row in rows:
        entry=dict(zip(columns,row))
        for name in ('run_uuid','fiber_a','fiber_b','edge_uuid'):
            if entry.get(name) is not None:entry[name]=_uuid(entry[name])
        out.append(entry)
    out.sort(key=lambda entry:entry['residual_index'])
    return out


def solve_residual_pieces(database, run):
    """The fiber stretches those residuals accuse, keyed by residual index.

    Each stretch is a piece of one fiber between two constraints of the cycle,
    with the share of the violated cost it carries and the control range a
    piece opened for it covers (null when the fiber has no controls there).
    """
    with CollectionReader(str(database)) as store:
        table=next((t for t in store.extension_tables() if t['name']==PREFIX+'residual_pieces'),None)
        if table is None:raise ValueError('Input has no scaffold residual piece rows')
        columns=[c['name'] for c in table['columns']]
        rows=store.extension_rows(PREFIX+'residual_pieces',key_filter={'run_uuid':uuid.UUID(_uuid(run)).bytes})
    pieces={}
    for row in rows:
        entry=dict(zip(columns,row))
        for name in ('run_uuid','fiber_uuid'):
            if entry.get(name) is not None:entry[name]=_uuid(entry[name])
        pieces.setdefault(entry['residual_index'],[]).append(entry)
    for members in pieces.values():members.sort(key=lambda entry:entry['member_index'])
    return pieces


def run_stage(stage, config):
    """Start one managed process before preparing any solver arrays."""
    if stage in ('winding','uv'):
        from .fiber_wgraph_geometry import run_stage as geometry_stage
        return geometry_stage(stage,config)
    if stage not in ('hvb','sign'):raise ValueError(f'Unknown downstream stage {stage}')
    from .fiber_normal_sign import launch_tensor_solve
    destination=Path(config['result_store'])
    if destination.exists():raise FileExistsError(destination)
    with CollectionReader(str(config['database'])) as source:input_source=source.source
    root=Path(config.get('output_root',str(destination)+'.work'))
    settings=_solver_config(config,root)
    return launch_tensor_solve(dict(settings,workflow_stage=stage,input_source=input_source),__name__,__file__)


def _run_direct_stage(stage, config):
    """Execute a stage with ``database`` input and ``result_store`` output."""
    if stage not in ('hvb','sign'):raise ValueError(f'Unknown downstream stage {stage}')
    from . import fiber_hv_stage as hv, fiber_normal_sign as normal
    from .fiber_wrapped_stages import solve_stage
    from .fiber_min_flow_experiment import write
    destination=Path(config['result_store'])
    if destination.exists():raise FileExistsError(destination)
    with CollectionReader(str(config['database'])) as source:
        input_source=source.source
    if input_source!=config['input_source']:raise ValueError('Stage input snapshot changed before preparation')
    root=Path(config.get('output_root',str(destination)+'.work'))
    settings=_solver_config(config,root)
    root.mkdir(parents=True,exist_ok=True)
    selected=None
    if stage=='hvb':
        base=_base_database(config)
        graph=_graph(config,base)
        settings.update(database=str(base))
        if config.get('stage1'):raise ValueError('vc_wgraph hvb runs before normal-sign alignment')
        if not settings.get('features'):
            if not config.get('native_config'):
                raise ValueError('hvb needs axial features, or native_config to generate them with the existing extractor')
            native=(dict(config['native_config']) if isinstance(config['native_config'],dict)
                    else json.loads(Path(config['native_config']).read_text()))
            feature=hv.axial_evidence(dict(settings,native_config=native),graph=graph)
            data,stats=hv.prepare_evidence(settings,feature,graph=graph)
        else:
            feature=None
            data,stats=hv.prepare(settings,graph=graph,return_arrays=True)
        expected_ids=graph.ids;del graph,feature
        summary,saved=solve_stage(settings,'classification',data=data,stats=stats,return_solution=True)
        ids=saved['ids'];labels=saved['labels'];probabilities=saved['probabilities']
        if not np.array_equal(ids,expected_ids):raise ValueError('Classification output fiber identities differ')
        columns=dict(family=labels.astype(np.int64),enabled=(labels!=2).astype(np.int64),
                     p_h=probabilities[:,0],p_v=probabilities[:,1],p_broken=probabilities[:,2])
        result_stage='classification'
    else:
        prior=json.loads(Path(config['priors']).read_text())
        base=_base_database(config,prior['source_store'])
        graph=_graph(config,base)
        classifications=None
        if config.get('classification','run')=='record':
            # Stored annotation tags select the H/V fibers; no classification run is pinned.
            from .fiber_graph_store import fiber_families_from_record
            if config.get('input_run_uuid') is not None:raise ValueError('classification record takes no input run selector')
            family,tag_counts=fiber_families_from_record(base,graph.ids,config.get('record_family','any'))
            keep=(family>=0).astype(np.float32)
            print(json.dumps(dict(stage='sign_record_family',record_family=tag_counts)),flush=True)
        elif config.get('classification','run')=='run':
            selected=_selected_run(config['database'],'classification',config.get('input_run_uuid'))
            classifications=_fiber_columns(config['database'],selected,graph.ids,required=('enabled',),exact=True)
            keep=np.asarray(classifications['enabled'],np.float32)
        else:raise ValueError('classification must be run or record')
        if not np.isin(keep,[0,1]).all():raise ValueError('Classification enabled mask is not binary')
        if not keep.any():raise ValueError('No enabled H/V fibers for sign alignment')
        settings.update(database=str(base),fixed_keep=True)
        data,stats=normal.prepare(settings,graph=graph,return_arrays=True,prior=prior)
        data['keep']=keep
        expected_ids=graph.ids
        first_pass=None
        if settings.get('crossing_filter'):
            # Two passes. The first has no signs to go by and only drops what the normal surfaces
            # call undecidable; its signs then choose, per disagreeing crossing, the closest
            # crossing of the encounter that agrees.
            first_pass,first=solve_stage(dict(settings,output_root=str(root/'first-pass')),'normal_sign',
                                         data=data,stats=stats,return_solution=True)
            if not np.array_equal(first['ids'],expected_ids):raise ValueError('First-pass fiber identities differ')
            signs=np.where(keep>.5,first['normal_sign'],0)
            data,stats=normal.prepare(settings,graph=graph,return_arrays=True,prior=prior,signs=signs)
            data['keep']=keep
            print(json.dumps(dict(stage='sign_second_pass',crossing_filter_dropped=stats['crossing_filter_dropped'],
                                  crossing_filter_replaced=stats['crossing_filter_replaced'])),flush=True)
        del graph,prior,classifications
        summary,saved=solve_stage(settings,'normal_sign',data=data,stats=stats,return_solution=True)
        if first_pass is not None:summary=dict(summary,first_pass=first_pass)
        if not np.array_equal(saved['broken'],keep<=.5):raise RuntimeError('Sign alignment changed classification')
        ids=saved['ids']
        if not np.array_equal(ids,expected_ids):raise ValueError('Normal-sign output fiber identities differ')
        columns=dict(normal_sign=saved['normal_sign'].astype(np.int64),sign_probability=saved['sign_probability'])
        result_stage='normal_sign'
    from .fiber_graph_store import write_fiber_result_columns
    run=write_fiber_result_columns(destination,config['database'],input_source,result_stage,'wrapped_irls',ids,columns,
                      settings=dict(fixed_keep=stage=='sign',prior_weight=float(settings.get('prior_weight',.001))),
                      input_run=selected,workers=int(config.get('workers',32)),timings_path=root/'result-export-timings.json')
    result=dict(stage=stage,result_store=str(destination),run_uuid=run,input_run_uuid=selected,solver_summary=summary)
    write(root/'summary.json',result)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--unmanaged',action='store_true');parser.add_argument('--devices')
    args=parser.parse_args()
    if not args.unmanaged:raise RuntimeError('Use the managed workflow entry point')
    config=json.loads(Path(args.config).read_text())
    from .fiber_graph_store import configure_solver_sqlite
    configure_solver_sqlite(config)
    _run_direct_stage(config['workflow_stage'],config)
