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

WEIGHT_TYPES=('parallel','perpendicular','enclosure','sign','manual')
DEFAULT_WEIGHTS=dict(parallel=1.,perpendicular=1.,enclosure=0.,sign=0.,manual=0.)
LEMON_BACKENDS=('network_simplex','cost_scaling')


def winding_settings(config):
    """Validated winding-stage settings: backend, typed weights, input modes.

    The legacy single ``perpendicular_relative_weight`` was replaced by the
    per-type ``weights`` object and is rejected rather than aliased.
    """
    if 'perpendicular_relative_weight' in config:
        raise ValueError('perpendicular_relative_weight was replaced by weights.perpendicular (relative to weights.parallel)')
    optimizer=config.get('optimizer','irls')
    if optimizer not in ('irls','lemon'):raise ValueError('Winding optimizer must be irls or lemon')
    weights=dict(DEFAULT_WEIGHTS)
    supplied=config.get('weights') or {}
    unknown=set(supplied)-set(WEIGHT_TYPES)
    if unknown:raise ValueError('Unknown constraint weight types: '+', '.join(sorted(unknown)))
    for name,value in supplied.items():
        value=float(value)
        if not np.isfinite(value) or value<0:raise ValueError(f'weights.{name} must be a finite non-negative number')
        weights[name]=value
    lemon=dict(config.get('lemon') or {})
    unknown=set(lemon)-{'backend','capacity_scale'}
    if unknown:raise ValueError('Unknown lemon settings: '+', '.join(sorted(unknown)))
    lemon.setdefault('backend','network_simplex');lemon.setdefault('capacity_scale',1000)
    if lemon['backend'] not in LEMON_BACKENDS:raise ValueError('lemon.backend must be network_simplex or cost_scaling')
    if int(lemon['capacity_scale'])!=lemon['capacity_scale'] or lemon['capacity_scale']<1:
        raise ValueError('lemon.capacity_scale must be a positive integer')
    classification=config.get('classification','run');record_family=config.get('record_family','any');normal_sign=config.get('normal_sign','run')
    # run: the pinned classification run decides every fiber. record: the
    # stored H/V tags do. run+record: the run decides, except for the fibers
    # that carry a tag - a hand-tagged scaffold imported into a traced volume
    # keeps its own families and stays enabled, whatever the classifier made
    # of its few short fibers.
    if classification not in ('run','record','run+record'):
        raise ValueError('classification must be run, record or run+record')
    if record_family not in ('any','manual','automatic'):raise ValueError('record_family must be any, manual or automatic')
    if normal_sign not in ('run','none'):raise ValueError('normal_sign must be run or none')
    automatic=any(weights[name]>0 for name in ('parallel','perpendicular','enclosure','sign'))
    if normal_sign=='none' and automatic:
        raise ValueError('normal_sign none requires zero parallel, perpendicular, enclosure and sign weights (manual links only)')
    if normal_sign=='none' and classification!='record':
        raise ValueError('normal_sign none needs classification record; there is no run to pin')
    if not automatic and weights['manual']<=0:raise ValueError('All constraint weights are zero')
    # How much each crossing counts as evidence on top of its type weight, in
    # the sign stage's own terms: |dot|^normal_dot_weight_power, damped by
    # min(1, 1/d^2) of its measured separation when winding_distance_weight is
    # inverse_square. The defaults are the flat per-type weights.
    dot_power=float(config.get('normal_dot_weight_power',0.))
    if not np.isfinite(dot_power) or dot_power<0:
        raise ValueError('normal_dot_weight_power must be a finite non-negative number')
    distance_weight=config.get('winding_distance_weight','none')
    if distance_weight not in ('inverse_square','none'):
        raise ValueError('winding_distance_weight must be inverse_square or none')
    from .fiber_wgraph_links import link_settings
    return dict(optimizer=optimizer,weights=weights,lemon=dict(backend=lemon['backend'],capacity_scale=int(lemon['capacity_scale'])),
                classification=classification,record_family=record_family,normal_sign=normal_sign,automatic=automatic,
                dot_weight_power=dot_power,winding_distance_weight=distance_weight,
                crossing_filter=bool(config.get('crossing_filter',False)),
                fixed_object=fixed_object_settings(config),
                manual_links=link_settings(config) if weights['manual']>0 else None)


def fixed_object_settings(config):
    """Rigid groups of fibers the solve must keep as they are, or None.

    ``fixed_object.windings`` names a solve's windings JSON (``ids``,
    ``selected_winding`` and ``component``, as ``vc_scaffold_solve`` writes
    it). Every fiber of this store listed there joins its component; a
    component with two or more fibers present is contracted into one node, so
    the global solve may move the group but never its members against each
    other.
    """
    section=config.get('fixed_object')
    if section is None:return None
    if not isinstance(section,dict):raise ValueError('fixed_object must be an object')
    unknown=set(section)-{'windings'}
    if unknown:raise ValueError('Unknown fixed_object settings: '+', '.join(sorted(unknown)))
    path=section.get('windings')
    if not path:raise ValueError('fixed_object needs windings: the solve whose groups are fixed')
    return dict(windings=str(path))


def _input_lineage(config, winding_run=None, *, settings=None):
    """Resolve exact stage ancestors; explicit selectors may not override pins.

    With ``classification: record`` the normal-sign run pins the geometry store
    directly; with ``normal_sign: none`` nothing is pinned.
    """
    from .fiber_wgraph_stages import _selected_run,_run_inputs,_uuid
    settings=settings or dict(classification='run',normal_sign='run')
    database=config['database']
    def parent(run,stage):
        rows=[row for row in _run_inputs(database,run) if row['role']=='input_0']
        if len(rows)!=1 or rows[0]['input_run_uuid'] is None:
            raise ValueError(f'{stage} ancestor is not pinned by selected input run')
        return _selected_run(database,stage,_uuid(rows[0]['input_run_uuid']))
    if settings['normal_sign']=='none':
        if winding_run is not None:raise ValueError('A winding run without a normal-sign run has no pinned ancestors')
        if config.get('normal_sign_run_uuid') is not None or config.get('classification_run_uuid') is not None:
            raise ValueError('Run selectors are meaningless without a normal-sign run')
        return dict(classification=None,normal_sign=None)
    if winding_run is None:
        sign=_selected_run(database,'normal_sign',config.get('normal_sign_run_uuid'))
    else:
        sign=parent(winding_run,'normal_sign')
        explicit=config.get('normal_sign_run_uuid')
        if explicit is not None and _uuid(explicit)!=sign:
            raise ValueError('Selected normal-sign run differs from winding input ancestry')
    if settings['classification']=='record':
        if config.get('classification_run_uuid') is not None:
            raise ValueError('classification record takes no classification run selector')
        return dict(classification=None,normal_sign=sign)
    # run+record pins the same classification run as run: the tags only
    # override its answer for the fibers that carry one.
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
    settings=dict(config,output_root=str(root))
    settings.setdefault('sqlite_memory_statistics',False)
    if stage=='uv':
        settings['devices']=config.get('devices','1')
        manager=dict(settings.get('gpu_manager',{}));manager.setdefault('exclusive',False);settings['gpu_manager']=manager
        return _run_uv(settings)
    if stage!='winding':raise ValueError(stage)
    validated=winding_settings(config)
    if validated['optimizer']=='lemon':
        summary=_run_winding_cpu(dict(settings,stage_mode='winding_direct'))
    else:
        settings['devices']=config.get('devices','1')
        manager=dict(settings.get('gpu_manager',{}));manager.setdefault('exclusive',False);settings['gpu_manager']=manager
        summary=launch_tensor_solve(dict(settings,stage_mode='winding_direct'),__name__,__file__)
    return dict(result_store=str(destination),run_uuid=summary['run_uuid'],solver_summary=summary)


def _run_winding_cpu(config):
    """CPU solve in a plain child process: no GPU-manager grant, no torch.

    The child is the same unmanaged entry the managed launcher uses, so the
    frozen ``config.json`` (with code provenance), the SQLite startup options
    and ``summary.json`` are identical to a managed run; only the grant is gone.
    """
    import subprocess
    import sys
    from .fiber_normal_sign import freeze_solver_config
    frozen,path=freeze_solver_config(config,__file__)
    root=Path(frozen['output_root'])
    completed=subprocess.run([sys.executable,'-m',__name__,'--config',str(path),'--unmanaged'])
    if completed.returncode:raise RuntimeError(f'CPU winding solver exited {completed.returncode}')
    return json.loads((root/'summary.json').read_text())


def _read_winding_inputs(config,input_source,selected,*,stage,modes=None):
    """Shared native graph/result/geometry preparation for winding and UV.

    Families come from a classification run or from the fiber records' family column; signs from a
    normal-sign run or none (manual links only). Without automatic constraints
    the nodes are the geometry store's active fibers and no constraint table
    is needed.
    """
    from .fiber_wgraph_stages import _graph,_base_database,_fiber_columns,_run_inputs,_uuid
    from .fiber_graph_store import workflow_volume,fiber_families_from_record
    from .fiber_min_flow_full import PackedPaths
    from vc_object_store import CollectionReader
    modes=modes or dict(classification='run',normal_sign='run',record_family='any',automatic=True)
    config=dict(config,classification_run_uuid=selected['classification'],normal_sign_run_uuid=selected['normal_sign'])
    timings={};phase=time.monotonic()
    prior_source=None
    if selected['classification'] is None and selected['normal_sign'] is not None:
        rows=[row for row in _run_inputs(config['database'],selected['normal_sign']) if row['role']=='input_0']
        if len(rows)!=1:raise ValueError('Normal-sign run must pin its geometry input')
        prior_source=dict(store_uuid=_uuid(rows[0]['store_uuid']),generation=_uuid(rows[0]['generation_uuid']),
                          commit_sequence=rows[0]['commit_sequence'])
    base=_base_database(config,prior_source)
    graph=None
    if modes['automatic']:
        graph=_graph(config,base);ids=graph.ids
        volume=workflow_volume(dict(config,database=str(base)))
    else:
        with CollectionReader(str(base)) as store:ids=np.array(sorted(r['uuid'] for r in store.objects('fiber')),dtype='U36')
        volume=workflow_volume(dict(working_volume=config.get('working_volume'),fibers=str(base)))
    timings['native_graph_read']=time.monotonic()-phase;phase=time.monotonic()
    tag_counts=None
    if selected['classification'] is not None:
        classified=_fiber_columns(config['database'],selected['classification'],ids,required=('family','enabled'),exact=True)
        family=classified['family'];enabled=classified['enabled']
        if modes['classification']=='run+record':
            # A tagged fiber keeps its tag and stays in the solve; the rest is
            # the classification run's answer, untouched.
            tagged,tag_counts=fiber_families_from_record(base,ids,modes['record_family'])
            override=tagged>=0
            family=np.where(override,tagged,family).astype(np.int64)
            enabled=np.where(override,1,enabled).astype(np.int64)
            tag_counts=dict(tag_counts,overridden=int(override.sum()),
                            overridden_enabled=int((override&(classified['enabled']==0)).sum()),
                            overridden_family=int((override&(classified['family']!=tagged)).sum()))
    else:
        family,tag_counts=fiber_families_from_record(base,ids,modes['record_family'])
        enabled=(family>=0).astype(np.int64);family=np.where(family>=0,family,2)
    if selected['normal_sign'] is not None:
        signs=_fiber_columns(config['database'],selected['normal_sign'],ids,required=('normal_sign',),exact=True)
        normal_sign=signs['normal_sign']
    else:
        normal_sign=np.ones(len(ids),np.int64)
    if not np.isin(family,[0,1,2]).all() or not np.isin(enabled,[0,1]).all():
        raise ValueError('Invalid classification readout')
    if not np.isin(normal_sign,[-1,1]).all():raise ValueError('Invalid normal sign')
    keep=(enabled!=0)&(family!=2)
    if not keep.any():raise ValueError('No enabled H/V fibers remain')
    ids=ids[keep];family=family[keep].astype(np.int64);normal_sign=normal_sign[keep].astype(np.int64)
    timings['native_result_columns']=time.monotonic()-phase;phase=time.monotonic()
    print(json.dumps(dict(stage=stage+'_native_readout',
        seconds=timings['native_graph_read']+timings['native_result_columns'],
        fibers=len(ids),constraints=len(graph.event_ids) if graph is not None else 0,record_family=tag_counts)),flush=True)
    # Return owned native buffers directly. No geometry-cache, numeric-graph,
    # or solver-tensor file is created or opened at this boundary.
    paths=PackedPaths.from_store(dict(config,input_store=config['database'],input_source=input_source,
        working_volume=volume),ids)
    if not np.array_equal(paths.ids,ids):raise ValueError('Native geometry fiber identities differ')
    paths.catalog.update(family=family,normal_sign=normal_sign)
    if graph is not None:paths.bind_graph(graph)
    # The resolved geometry store and tag provenance travel with the geometry.
    paths.source_store=str(base);paths.tag_counts=tag_counts
    timings['native_geometry_read_and_transport']=time.monotonic()-phase
    print(json.dumps(dict(stage=stage+'_native_geometry',
        seconds=timings['native_geometry_read_and_transport'],fibers=len(ids),points=len(paths.xyz))),flush=True)
    return graph,paths,ids,family,normal_sign,timings


def fixed_object_groups(settings,ids):
    """Per fiber: its rigid group's root index and its offset inside it.

    Returns ``(root, offset, report)`` with ``root`` -1 for a fiber in no
    group. A group is a component of the fixed solve with at least two of its
    fibers in this store; a component present once fixes nothing, so it is
    left free and counted.
    """
    section=settings.get('fixed_object')
    if section is None:return None,None,None
    fixed=json.loads(Path(section['windings']).read_text())
    for key in ('ids','selected_winding','component'):
        if key not in fixed:raise ValueError(f'fixed_object windings need {key}')
    winding={str(u):int(w) for u,w in zip(fixed['ids'],fixed['selected_winding'])}
    component={str(u):int(c) for u,c in zip(fixed['ids'],fixed['component'])}
    members={}
    for index,uuid in enumerate(ids):
        key=str(uuid)
        if key in component:members.setdefault(component[key],[]).append((index,winding[key]))
    root=np.full(len(ids),-1,np.int64);offset=np.zeros(len(ids),np.int64)
    groups=0;fibers=0;single=0
    for _,entries in sorted(members.items()):
        if len(entries)<2:
            single+=len(entries)
            continue
        entries.sort()
        base,baseWinding=entries[0]
        groups+=1
        for index,value in entries:
            root[index]=base;offset[index]=value-baseWinding;fibers+=1
    report=dict(solve_fibers=len(winding),solve_components=len(set(component.values())),
                present=sum(len(entries) for entries in members.values()),
                groups=groups,fixed_fibers=fibers,single_fiber_components=single)
    if groups==0:return None,None,report
    return root,offset,report


def _contract_fixed_groups(merged,root,offset,report):
    """Move every arc of a fixed group to its root and drop the internal ones.

    An arc between two members of one group asserts what the group already
    fixes: it is dropped and counted, with the turns it disagrees by, so a
    scaffold that contradicts the constraints cannot hide.
    """
    a=merged['a'];b=merged['b'];target=merged['target'];weight=merged['weight'];side=merged['side']
    rootA=root[a];rootB=root[b]
    inA=rootA>=0;inB=rootB>=0
    internal=inA&inB&(rootA==rootB)
    if internal.any():
        residual=arc_residual(offset[b[internal]]-offset[a[internal]],target[internal],side[internal])
        report.update(internal_arcs=int(internal.sum()),
                      internal_weight=float(weight[internal].sum()),
                      internal_disagreeing_arcs=int((residual>0).sum()),
                      internal_disagreement_turns=float(residual.sum()))
    else:
        report.update(internal_arcs=0,internal_weight=0.,internal_disagreeing_arcs=0,internal_disagreement_turns=0.)
    keep=~internal
    a=a[keep];b=b[keep];target=target[keep];weight=weight[keep];side=side[keep]
    rootA=root[a];rootB=root[b]
    # k_member = k_root + offset, so an arc that leaves the group takes the
    # member's offset into its target.
    movedA=rootA>=0;movedB=rootB>=0
    target=target+np.where(movedA,offset[a],0)-np.where(movedB,offset[b],0)
    a=np.where(movedA,rootA,a);b=np.where(movedB,rootB,b)
    report.update(rewritten_arcs=int((movedA|movedB).sum()),remaining_arcs=int(len(a)))
    return dict(a=a,b=b,target=target,weight=weight,side=side)


def arc_residual(step,target,side):
    """What a constraint costs per unit weight: |step - target| for an equation (side 0); for a
    one-sided sign constraint only the part on the wrong side (step at least the target for side +1,
    at most it for side -1)."""
    difference=np.asarray(step,float)-np.asarray(target,float)
    return np.where(side==0,np.abs(difference),np.maximum(-np.asarray(side,float)*difference,0.))


def aggregate_arcs(a,b,target,weight,side):
    """Sum identical constraints in node order (a < b): equations by (a, b, target), one-sided ones by
    (a, b, side, target). A one-sided constraint stated against node order is turned round."""
    a,b,target,weight,side=(np.asarray(x) for x in (a,b,target,weight,side))
    swap=a>b
    a,b=np.minimum(a,b),np.maximum(a,b);target=np.where(swap,-target,target);side=np.where(swap,-side,side)
    if not len(a):return dict(a=a,b=b,target=target,weight=weight,side=side)
    order=np.lexsort((target,side,b,a));a=a[order];b=b[order];target=target[order];side=side[order]
    starts=np.r_[0,np.flatnonzero((np.diff(a)!=0)|(np.diff(b)!=0)|(np.diff(side)!=0)|(np.diff(target)!=0))+1]
    return dict(a=a[starts],b=b[starts],target=target[starts],side=side[starts],
                weight=np.add.reduceat(np.asarray(weight,dtype=np.float64)[order],starts))


def _assemble_arcs(config,settings,graph,paths,ids,family,normal_sign):
    """Automatic, enclosure, sign and manual constraints with their type weights, aggregated.

    Every constraint is (a, b, target, weight, side) in node order: side 0 states the step b - a
    equals the target, +1 that it is at least the target, -1 at most (the one-sided sign
    constraints)."""
    from .fiber_packed_priors import _geometry
    weights=settings['weights'];parts=[];counts={}
    def part(a,b,target,weight,side=None):
        a=np.asarray(a,np.int64)
        parts.append(dict(a=a,b=np.asarray(b,np.int64),target=np.asarray(target,np.int64),weight=np.asarray(weight,np.float64),
                          side=np.zeros(len(a),np.int64) if side is None else np.asarray(side,np.int64)))
    if settings['automatic']:
        findings=alternative=None
        if settings['crossing_filter']:
            # The shared rule for the crossings' stored normal-surface findings, applied by the native
            # builder under the signs this stage has.
            if not graph.has_crossing_findings:
                raise ValueError('crossing_filter needs an extraction with crossing_findings; set crossing_filter to false to run without')
            findings=np.ascontiguousarray(graph.crossing_findings,dtype=np.uint8)
            alternative=np.ascontiguousarray(graph.crossing_alternative,dtype=np.float64)
        g=_geometry(paths).prepare_winding_graph(
            graph.i,graph.j,np.ascontiguousarray(paths.graph_index,dtype=np.int64),family,normal_sign,
            graph.arc_a,graph.arc_b,graph.distance,graph.parallel,
            graph.normal_a,graph.normal_b,graph.connector,graph.valid_a,graph.valid_b,graph.valid_connector,
            crossing_findings=findings,crossing_alternative=alternative,
            lasagna_winding_scale=float(config.get('lasagna_winding_scale',1.)),
            perpendicular_offset=float(config.get('perpendicular_offset',.5)),
            parallel_weight=weights['parallel'],perpendicular_weight=weights['perpendicular'],
            dot_weight_power=settings['dot_weight_power'],
            damp_by_winding_distance=settings['winding_distance_weight']=='inverse_square',
            sign_weight=weights['sign'],enclosures=weights['enclosure']>0,
            constraint_batch=int(config.get('constraint_batch',500000)),
            read_batch=int(config.get('read_batch',32768)),
            workers=int(config.get('workers',32)))
        counts['automatic']=dict(read=int(g['read']),used=int(g['used']),expanded=int(g['expanded']),arcs=len(g['a']),
                                 without_distance=int(g.get('without_distance',0)),
                                 crossing_filtered=int(g.get('crossing_filtered',0)))
        part(g['a'],g['b'],g['target'],g['weight'])
        if weights['sign']>0:
            part(g['sign_a'],g['sign_b'],g['sign_target'],g['sign_weight'],g['sign_side'])
            counts['sign']=dict(constraints=int(g['sign_constraints']),arcs=len(g['sign_a']))
        if weights['enclosure']>0:
            h=np.asarray(g['enclosure_a'],np.int64)
            part(h,g['enclosure_b'],g['enclosure_target'],np.full(len(h),weights['enclosure']))
            counts['enclosure']={k:int(v) for k,v in g['enclosure_counts'].items()}
    if weights['manual']>0:
        from .fiber_wgraph_links import read_links,manual_link_arcs
        base=getattr(paths,'source_store',None)
        if base is None:raise ValueError('Manual links need the resolved geometry store')
        links,report=manual_link_arcs(read_links(base),ids,family,paths,settings['manual_links'])
        counts['manual']=report
        part(links['a'],links['b'],links['target'],links['weight']*weights['manual'])
    merged={key:np.concatenate([entry[key] for entry in parts]) for key in ('a','b','target','weight','side')}
    positive=merged['weight']>0
    merged={key:value[positive] for key,value in merged.items()}
    if not len(merged['a']):raise ValueError('No usable winding constraints')
    root,offset,fixed_report=fixed_object_groups(settings,ids)
    if fixed_report is not None:counts['fixed_object']=fixed_report
    if root is not None:
        merged=_contract_fixed_groups(merged,root,offset,fixed_report)
        if not len(merged['a']):raise ValueError('No winding constraints left outside the fixed groups')
    g=aggregate_arcs(merged['a'],merged['b'],merged['target'],merged['weight'],merged['side'])
    return g,counts,(root,offset)


def _flat_settings(settings,counts,config):
    """Numeric run settings for the result store; names and modes live in the run method and summary."""
    flat=dict(classification_from_record=settings['classification']=='record',normal_sign_from_run=settings['normal_sign']=='run',
              handedness=int(counts.get('handedness',0)))
    for name in WEIGHT_TYPES:flat['weight_'+name]=float(settings['weights'][name])
    flat['normal_dot_weight_power']=float(settings['dot_weight_power'])
    flat['winding_distance_damped']=settings['winding_distance_weight']=='inverse_square'
    flat['fixed_object']=settings.get('fixed_object') is not None
    if settings['optimizer']=='lemon':
        flat.update(lemon_capacity_scale=int(settings['lemon']['capacity_scale']))
    else:
        flat.update({k:config.get(k,v) for k,v in dict(irls=15,cg=500,eps=.05,prior_weight=1e-6).items()})
    if settings['manual_links'] is not None:
        flat.update(manual_include_auto=bool(settings['manual_links']['include_auto']))
    for group,values in counts.items():
        if not isinstance(values,dict):continue
        for key,value in values.items():
            if isinstance(value,(bool,int,float)):flat[f'{group}_{key}']=value
    return flat


def _solve_winding_direct(config):
    """Read, prepare, optimize and persist in one process.

    Native geometry/graph arrays own their RAM; neither intermediate arrays nor
    the solution are serialized for another process to reload. IRLS runs in
    the managed CUDA child; LEMON runs in the calling process on the CPU.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from vc_object_store import CollectionReader
    from .fiber_graph_store import write_fiber_result_columns
    from .fiber_min_flow_experiment import write
    settings=winding_settings(config)
    started=time.monotonic()
    root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    with CollectionReader(config['database']) as source:input_source=source.source
    selected=_input_lineage(config,settings=settings)
    graph,paths,ids,family,normal_sign,timings=_read_winding_inputs(config,input_source,selected,stage='winding',modes=settings)
    phase=time.monotonic()
    g,counts,(fixed_root,fixed_offset)=_assemble_arcs(config,settings,graph,paths,ids,family,normal_sign)
    if getattr(paths,'tag_counts',None) is not None:counts['record_family']=paths.tag_counts
    counts['handedness']=int(getattr(getattr(paths,'seam',None),'handedness',0))
    timings['native_winding_graph']=time.monotonic()-phase
    with CollectionReader(config['database']) as source:
        if source.source!=input_source:raise ValueError('Source changed during winding preparation')
    n=len(ids)
    del graph,paths
    phase=time.monotonic()
    _,component=connected_components(coo_matrix((np.ones(len(g['a'])),(g['a'],g['b'])),shape=(n,n)),directed=False)
    timings['components']=time.monotonic()-phase;phase=time.monotonic()
    if settings['optimizer']=='lemon':
        from vc_object_store import solve_winding_min_cost_flow,winding_min_cost_flow_available
        if not winding_min_cost_flow_available():raise RuntimeError('vc_object_store was built without LEMON (VC_OBJECT_STORE_WITH_LEMON)')
        print(json.dumps(dict(stage='lemon_solve_started',backend=settings['lemon']['backend'],nodes=n,arcs=len(g['a']))),flush=True)
        flow=solve_winding_min_cost_flow(n,g['a'],g['b'],g['target'],g['weight'],
            backend=settings['lemon']['backend'],capacity_scale=settings['lemon']['capacity_scale'],
            side=g['side'] if g['side'].any() else None)
        labels=np.asarray(flow['winding'],np.int64);x=labels.astype(float)
        residual=arc_residual(labels[g['b']]-labels[g['a']],g['target'],g['side'])
        counts['quantization']=dict(flow['quantization']);counts['lemon']=dict(flow['timing'],objective=float(flow['objective']))
        method='lemon_'+settings['lemon']['backend']
    else:
        import torch
        from .fiber_linear_irls import solve_l1,round_components
        torch.set_num_threads(config.get('torch_threads',4))
        # The solver states a one-sided constraint as "step at least the target": an at-most one is
        # the same constraint read from b to a.
        most=g['side']<0
        x,_=solve_l1(n,np.where(most,g['b'],g['a']),np.where(most,g['a'],g['b']),np.where(most,-g['target'],g['target']),
            g['weight'],np.zeros(n),
            config.get('prior_weight',1e-6),dev='cuda',irls=config.get('irls',15),
            cg_iters=config.get('cg',500),eps=config.get('eps',.05),log=lambda x:print(x,flush=True),
            ineq=g['side']!=0 if g['side'].any() else None)
        if not np.isfinite(x).all():raise RuntimeError('IRLS returned nonfinite winding potentials')
        labels=round_components(x,component)
        residual=arc_residual(x[g['b']]-x[g['a']],g['target'],g['side'])
        method='linear_irls'
    if fixed_root is not None:
        # The group moved as one: its members sit where their root does, each
        # at the offset the fixed solve gave it, in the root's component.
        members=fixed_root>=0
        labels=np.asarray(labels,np.int64).copy()
        x=np.asarray(x,float).copy()
        component=np.asarray(component).copy()
        labels[members]=labels[fixed_root[members]]+fixed_offset[members]
        x[members]=x[fixed_root[members]]+fixed_offset[members]
        component[members]=component[fixed_root[members]]
        residual=arc_residual(labels[g['b']]-labels[g['a']],g['target'],g['side'])
    timings['solve']=time.monotonic()-phase
    summary=dict(fibers=n,arcs=len(g['a']),one_sided_arcs=int((g['side']!=0).sum()),seconds=timings['solve'],method=method,
        weighted_l1=float(np.dot(g['weight'],residual)),mean_residual=float(residual.mean()),
        max_residual=float(residual.max()),expanded_arcs_within_half_turn_percent=float(100*np.mean(residual<=.5)),
        # Distinct labels, not the largest plus one: a fixed group's members
        # take their root's component, which leaves holes in the numbering.
        components=int(np.unique(component).size),counts=counts,settings=settings)
    columns=dict(enabled=np.ones(n,np.int64),family=family,selected_winding=labels.astype(np.int64),
                 winding_mean=x,component=component.astype(np.int64))
    metrics=[dict(iteration=int(config.get('irls',15)) if method=='linear_irls' else 0,scope='all',name=name,
                  value=float(summary[name]),units='turns' if name.endswith('residual') else 'weighted_turns')
             for name in ('weighted_l1','mean_residual','max_residual')]
    summary['run_uuid']=write_fiber_result_columns(config['result_store'],config['database'],input_source,
        'winding',method,ids,columns,metrics=metrics,settings=_flat_settings(settings,counts,config),
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
    config=json.loads(Path(args.config).read_text())
    from .fiber_graph_store import configure_solver_sqlite
    configure_solver_sqlite(config)
    solve(config)
