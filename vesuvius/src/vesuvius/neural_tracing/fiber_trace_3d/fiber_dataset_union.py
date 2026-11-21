"""Combine independently classified fiber cohorts, adding cross-source evidence.

Keep original within-source events and frozen GT evaluation events unchanged.
Only new cross-source events use fresh native extraction/local normal sweeps.
No reference fibers or saved winding labels enter the combined graph solve.
"""
import json
from pathlib import Path
import numpy as np

from .fiber_min_flow_experiment import arrays, write, digest


def combine_graphs(parts):
    """Preserve global UUIDs; concatenate and UUID-sort unsigned whole-fiber graphs."""
    node_keys=('source_id','family','start','end')
    edge_keys=('i','j','ds_a','ds_b','distance','sign','parallel')
    out={k:[] for k in node_keys+edge_keys};groups=[];original=[];offset=0
    for group,d in enumerate(parts):
        n=len(d['source_id'])
        if len(d['links']) or np.any(d['start']!=0):raise ValueError('Whole unsplit fibers required')
        original.extend(map(str,d['source_id']));groups.extend([group]*n)
        for key in node_keys:
            out[key].append(d[key])
        for key in edge_keys:out[key].append(d[key]+offset if key in ('i','j') else d[key])
        offset+=n
    result={k:np.concatenate(v) for k,v in out.items()}
    result['node_ds']=np.column_stack((np.zeros(offset),result['end']))
    result['links']=np.empty((0,4))
    if len(np.unique(result['source_id']))!=offset:
        raise ValueError('Duplicate object UUID across source cohorts')
    order=np.argsort(result['source_id'],kind='stable')
    remap=np.empty(offset,np.int64);remap[order]=np.arange(offset)
    for key in (*node_keys,'node_ds'): result[key]=result[key][order]
    for key in ('i','j'):result[key]=remap[result[key]]
    return result,np.asarray(groups,np.int64)[order],np.asarray(original,dtype='U36')[order]


def prepare(config):
    from .fiber_flow_uv import saved_assignment_inputs
    from .fiber_normal_convention import inward_solution
    from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import read_polylines,write_initial_fibers,fiber_volume,fibers_digest
    root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    parts=[];frames=[];roots=[];signs=[];source_hashes={}
    for source in config['sources']:
        d,_=saved_assignment_inputs(arrays(source['tensor_cache']),arrays(source['solution']))
        parts.append(d)
        normal=inward_solution(arrays(source['normal_solution']))
        lookup=dict(zip(map(str,normal['ids']),normal['normal_sign']))
        signs.extend(lookup[str(s)] for s in d['source_id'])
        frames.append(fiber_volume(source['fibers']))
        for key in ('tensor_cache','solution','normal_solution','events'):
            source_hashes[source[key]]=digest(source[key])
        source_hashes[source['fibers']]=fibers_digest(source['fibers'])
    if any(f!=frames[0] for f in frames):raise ValueError('Source coordinate frames differ')
    d,groups,original=combine_graphs(parts)
    original_order=np.concatenate([p['source_id'] for p in parts])
    signs=np.asarray(signs)[np.argsort(original_order,kind='stable')]
    np.savez(root/'within_tensors.npz',**d)
    np.savez(root/'source_mapping.npz',source_id=d['source_id'],group=groups,original_source_id=original,normal_sign=signs)
    curves=[];events=[];offset=0
    for group,(source,part) in enumerate(zip(config['sources'],parts)):
        mapping={str(s):int(np.searchsorted(d['source_id'],s)) for s in part['source_id']}
        for row,xyz in read_polylines(source['fibers'],part['source_id']):
            node=mapping[row['id']]
            curves.append((dict(row,family='H' if d['family'][node]==0 else 'V'),xyz))
        source_events=json.loads(Path(source['events']).read_text())
        if isinstance(source_events,dict):source_events=source_events['events']
        for e in source_events:
            if e['test_source'] in mapping:
                events.append(dict(e,test_piece=mapping[e['test_source']]))
    if not (root/'selected.vcobj').exists():
        write_initial_fibers(root/'selected.vcobj',curves,volume=frames[0],packed=True)
    else:
        np.testing.assert_array_equal([r['id'] for r,_ in read_polylines(root/'selected.vcobj')],d['source_id'])
    (root/'cross').mkdir(exist_ok=True)
    write(root/'cross/cross_events.json',dict(events=events))
    write(root/'prepared.json',dict(fibers=len(original),within_events=len(d['i']),gt_events=len(events),
        source_fibers={s['name']:len(p['source_id']) for s,p in zip(config['sources'],parts)},
        source_sha256=source_hashes,gt_in_solve=False))


def run(config):
    from .fiber_native_workflow import run as native_run
    from .fiber_min_flow_selection import EdgeStore
    from .fiber_selected_solver_comparison import run as compare
    root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    request=root/'request.json'
    if request.exists() and json.loads(request.read_text())!=config:
        raise ValueError('Union request changed; choose a new output root')
    write(request,config)
    if not (root/'prepared.json').exists():prepare(config)
    for stage,template,done in [('extract','extract_template','extraction.json'),('orient','orient_template','orientation.json')]:
        if (root/done).exists():continue
        cfg=json.loads(Path(config[template]).read_text())
        cfg.update(fibers=str(root/'selected.vcobj'),full_volume=True,output=str(root/done),
            database=str(root/('raw.vcobj' if stage=='extract' else 'oriented.vcobj')),
            progress_file=str(root/(stage+'_progress.jsonl')))
        if stage=='orient':cfg['orient_database']=str(root/'raw.vcobj')
        path=root/(stage+'_config.json');write(path,cfg)
        native_run(dict(output_root=str(root/'native'),native_binary=config['native_binary'],
            input_files=[str(path)],commands=[dict(name=stage,binary='vc_fiber_constraints',args=[str(path)])]))
    if not (root/'selected_tensors.npz').exists():
        d=arrays(root/'within_tensors.npz');mapping=arrays(root/'source_mapping.npz');ids=d['source_id']
        store=EdgeStore(dict(database=str(root/'oriented.vcobj')),ids,
            dict(zip(ids,d['family'])),dict(zip(ids,mapping['normal_sign'])),dict(zip(ids,mapping['group'])))
        store.load(ids);cross=store.edges()
        for col,key in enumerate(('i','j','ds_a','ds_b','distance','sign','parallel')):
            values=np.searchsorted(ids,cross[:,col]) if key in ('i','j') else cross[:,col].astype(d[key].dtype)
            d[key]=np.concatenate((d[key],values))
        np.savez(root/'selected_tensors.npz',**d)
        write(root/'union.json',dict(fibers=len(ids),events=len(d['i']),cross_events=len(cross),
            cross_perpendicular=int((cross[:,6]==0).sum()),cross_parallel=int((cross[:,6]!=0).sum()),
            perpendicular=int((~d['parallel']).sum()),parallel=int(d['parallel'].sum()),
            retained_within_events=len(d['i'])-len(cross),deduplicated=False,gt_in_solve=False))
    comparison=dict(config['comparison'],selection_root=str(root),output_root=str(root/'benchmark'))
    result=compare(comparison)
    write(root/'results.json',result)
    return result
