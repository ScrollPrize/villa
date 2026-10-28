"""Combine independently classified fiber cohorts, adding cross-source evidence.

Keep original within-source events and frozen GT evaluation events unchanged.
Only new cross-source events use fresh native extraction/local normal sweeps.
No reference fibers or saved winding labels enter the combined graph solve.
"""
import json
from pathlib import Path
import sqlite3
import uuid
import numpy as np

from .fiber_min_flow_experiment import arrays, write, digest


def combine_graphs(parts):
    """Assign collision-free IDs and concatenate unsigned whole-fiber graphs."""
    node_keys=('source_id','family','start','end')
    edge_keys=('i','j','ds_a','ds_b','distance','sign','parallel')
    out={k:[] for k in node_keys+edge_keys};groups=[];original=[];offset=0
    for group,d in enumerate(parts):
        n=len(d['source_id'])
        if len(d['links']) or np.any(d['start']!=0):raise ValueError('Whole unsplit fibers required')
        original.extend(map(int,d['source_id']));groups.extend([group]*n)
        for key in node_keys:
            out[key].append(np.arange(offset,offset+n,dtype=np.int64) if key=='source_id' else d[key])
        for key in edge_keys:out[key].append(d[key]+offset if key in ('i','j') else d[key])
        offset+=n
    result={k:np.concatenate(v) for k,v in out.items()}
    result['node_ds']=np.column_stack((np.zeros(offset),result['end']))
    result['links']=np.empty((0,4))
    return result,np.asarray(groups,np.int64),np.asarray(original,np.int64)


def prepare(config):
    from .fiber_flow_uv import saved_assignment_inputs
    from .fiber_normal_convention import inward_solution
    from vesuvius.neural_tracing.tracers.shared.annotations.afv import read_afv_polylines,write_afv
    root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    parts=[];frames=[];roots=[];signs=[];source_hashes={}
    for source in config['sources']:
        d,_=saved_assignment_inputs(arrays(source['tensor_cache']),arrays(source['solution']))
        parts.append(d)
        normal=inward_solution(arrays(source['normal_solution']))
        lookup=dict(zip(map(int,normal['ids']),normal['normal_sign']))
        signs.extend(lookup[int(s)] for s in d['source_id'])
        with sqlite3.connect(Path(source['afv']).resolve().as_uri()+'?mode=ro',uri=True) as db:
            frames.append(json.loads(db.execute("SELECT value FROM metadata WHERE key='frame'").fetchone()[0]))
            roots.append(json.loads(db.execute("SELECT value FROM metadata WHERE key='root'").fetchone()[0]))
        for key in ('tensor_cache','solution','normal_solution','events','afv'):
            source_hashes[source[key]]=digest(source[key])
    if any(f!=frames[0] for f in frames):raise ValueError('Source coordinate frames differ')
    d,groups,original=combine_graphs(parts)
    np.savez(root/'within_tensors.npz',**d)
    np.savez(root/'source_mapping.npz',source_id=d['source_id'],group=groups,original_source_id=original,normal_sign=signs)
    curves=[];events=[];offset=0
    for group,(source,part) in enumerate(zip(config['sources'],parts)):
        mapping={int(s):offset+k for k,s in enumerate(part['source_id'])}
        for row,xyz in read_afv_polylines(source['afv'],part['source_id']):
            annotation=json.loads(row['annotation'])
            annotation['combined_source']=dict(dataset=source['name'],fiber_id=int(row['id']))
            new_id=mapping[int(row['id'])]
            curves.append((dict(row,id=new_id,family='H' if d['family'][new_id]==0 else 'V',
                annotation=json.dumps(annotation)),xyz))
        source_events=json.loads(Path(source['events']).read_text())
        if isinstance(source_events,dict):source_events=source_events['events']
        for e in source_events:
            if int(e['test_source']) in mapping:
                sid=mapping[int(e['test_source'])]
                events.append(dict(e,test_source=sid,test_piece=sid))
        offset+=len(part['source_id'])
    if not (root/'selected.afv').exists():
        write_afv(None,root/'selected.afv',curves,dict(frame=frames[0],root=roots[0],
            uuid=str(uuid.uuid5(uuid.NAMESPACE_URL,json.dumps(source_hashes,sort_keys=True))),
            source='Classified whole-fiber union; no deduplication or geometry changes',
            sources=config['sources'],source_sha256=source_hashes))
    else:
        with sqlite3.connect((root/'selected.afv').as_uri()+'?mode=ro',uri=True) as db:
            metadata={k:json.loads(v) for k,v in db.execute('SELECT key,value FROM metadata')}
            if metadata.get('complete') is not True or metadata.get('sources')!=config['sources']:
                raise ValueError('Existing union AFV provenance mismatch')
            np.testing.assert_array_equal([r[0] for r in db.execute('SELECT id FROM fibers ORDER BY id')],d['source_id'])
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
        cfg.update(afv=str(root/'selected.afv'),full_volume=True,output=str(root/done),
            database=str(root/('raw.afc.sqlite' if stage=='extract' else 'oriented.afc.sqlite')),
            progress_file=str(root/(stage+'_progress.jsonl')))
        if stage=='orient':cfg['orient_database']=str(root/'raw.afc.sqlite')
        path=root/(stage+'_config.json');write(path,cfg)
        native_run(dict(output_root=str(root/'native'),native_binary=config['native_binary'],
            input_files=[str(path)],commands=[dict(name=stage,binary='vc_fiber_constraints',args=[str(path)])]))
    if not (root/'selected_tensors.npz').exists():
        d=arrays(root/'within_tensors.npz');mapping=arrays(root/'source_mapping.npz');ids=d['source_id']
        store=EdgeStore(dict(database=str(root/'oriented.afc.sqlite')),ids,
            dict(zip(ids,d['family'])),dict(zip(ids,mapping['normal_sign'])),dict(zip(ids,mapping['group'])))
        try:store.load(ids);cross=store.edges()
        finally:store.db.close()
        for col,key in enumerate(('i','j','ds_a','ds_b','distance','sign','parallel')):
            d[key]=np.concatenate((d[key],cross[:,col].astype(d[key].dtype)))
        np.savez(root/'selected_tensors.npz',**d)
        write(root/'union.json',dict(fibers=len(ids),events=len(d['i']),cross_events=len(cross),
            cross_perpendicular=int((cross[:,6]==0).sum()),cross_parallel=int((cross[:,6]!=0).sum()),
            perpendicular=int((~d['parallel']).sum()),parallel=int(d['parallel'].sum()),
            retained_within_events=len(d['i'])-len(cross),deduplicated=False,gt_in_solve=False))
    comparison=dict(config['comparison'],selection_root=str(root),output_root=str(root/'benchmark'))
    result=compare(comparison)
    write(root/'results.json',result)
    return result
