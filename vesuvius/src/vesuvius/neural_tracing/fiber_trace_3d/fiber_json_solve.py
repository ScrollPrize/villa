"""Small reviewed-JSON dataset adapter; all extraction/solving stays shared."""
import json
import hashlib
import sqlite3
from pathlib import Path
import numpy as np
from lasagna.inference_provenance import atomic_write, sha256_file


def convert(source_dir, template, destination):
    from vesuvius.neural_tracing.fiber_follow.data.afv import write_afv, read_afv_polylines
    with sqlite3.connect(Path(template).resolve().as_uri()+'?mode=ro', uri=True) as db:
        metadata = {k: json.loads(v) for k, v in db.execute('SELECT key,value FROM metadata')}
    fibers=[]; records=[]
    for i, path in enumerate(sorted(Path(source_dir).glob('*.json')), 1):
        doc=json.loads(path.read_text()); xyz=np.asarray(doc['line_points'], dtype=np.float64)
        shape=metadata['root']['coordinate_base_shape_zyx']
        if doc.get('coordinate_base_shape_zyx',shape) != shape:
            raise ValueError('Coordinate shape mismatch: '+str(path))
        if xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)<2 or not np.isfinite(xyz).all():
            raise ValueError('Invalid curve: '+str(path))
        if doc.get('branches') or doc.get('adjacent_branches'):
            raise ValueError('This pilot does not import explicit links')
        manifests=[' '.join(str(c.get('segment_to_next',{}).get(k,'')) for k in ('normal_manifest','fiber_manifest')) for c in doc['control_points']]
        if not manifests or any('PHercParis4' not in m or '20260411134726' not in m for m in manifests[:-1]):
            raise ValueError('Unverified source scan: '+str(path))
        if np.any(xyz<0) or np.any(xyz>=np.asarray(shape)[::-1]):
            raise ValueError('Curve outside verified source scan: '+str(path))
        row=dict(id=i,name=path.name,family='unknown',annotation=json.dumps(doc))
        fibers.append((row,xyz))
        records.append(dict(id=i,file=str(path),sha256=sha256_file(path),points=len(xyz),
                            saved_hv=doc['hv_classification'],
                            segment_modes=[c.get('segment_to_next',{}).get('interp_mode') for c in doc['control_points'][:-1]]))
    if not fibers: raise ValueError('No input fibers')
    manifest=Path(destination).with_suffix('.sources.json')
    if Path(destination).exists():
        if json.loads(manifest.read_text())!=records: raise ValueError('Changed source fibers')
    else:
        write_afv(template,destination,fibers,dict(uuid='reviewed-json-'+hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()[:24],
                  source='reviewed_vc3d_json', source_files=records))
        atomic_write(manifest,records,durable=False)
    actual=read_afv_polylines(destination)
    if len(actual)!=len(fibers): raise ValueError('Roundtrip fiber count')
    for (row,a),(other,b) in zip(fibers,actual):
        if row['id']!=other['id'] or not np.array_equal(a,b): raise ValueError('AFV roundtrip differs')
    return records


def run(config):
    from .crop_benchmark import run as dispatch
    base=Path(config['templates']); root=Path(config['output_root']); root.mkdir(parents=True,exist_ok=True)
    templates=('extract_raw_winding.json','orient_normals_full.json','extract_axial_features.json',
               'extract_outward_priors.json','run_sign_resolution.json','run_hv_recommended.json',
               'run_h_direction_breakage_v2.json','run_v_direction.json',
               'run_uv_current_baseline.json','run_uv_joint100_fresh_low_width_v2.json')
    signature=dict(config=config,templates={name:sha256_file(base/name) for name in templates})
    manifest=root/'pipeline_inputs.json'
    if manifest.exists() and json.loads(manifest.read_text())!=signature:
        raise ValueError('Changed pipeline inputs: use a new output directory')
    atomic_write(manifest,signature,durable=False)
    def load(name,key=None):
        d=json.loads((base/name).read_text()); return d[key] if key else d
    afv=str(root/'fibers.afv'); database=str(root/'oriented.afc.sqlite')
    convert(config['source_dir'],config['afv_template'],afv)
    common=dict(afv=afv,database=database,umbilicus=str(base/'data/Paris4_umbilicus.json'),
                features=str(root/'features.json'),stage1=str(root/'normal/solution_0.1.npz'),
                hv_solution=str(root/'hv/solution_degree_mix_b0.1_e0.01.npz'))
    manager=dict(exclusive=False,priority=100,cpu_cores=4,vram_bytes_per_gpu=4294967296,label='bbox-four-fibers')
    def native(name,template,**updates):
        cfg=load(template); cfg.update(afv=afv,threads=4,output=str(root/(name+'.json')),**updates)
        path=root/(name+'_config.json'); atomic_write(path,cfg,durable=False)
        if Path(cfg['output']).exists(): return
        dispatch(dict(output_root=str(root/(name+'_runner')),reference_dir=config['source_dir'],
                      input_files=[str(path)],commands=[dict(name=name,binary='vc_fiber_trace_chunk',args=['afv-constraints',str(path)])]))
    native('extract','extract_raw_winding.json',database=str(root/'raw.afc.sqlite'))
    native('orient','orient_normals_full.json',orient_database=str(root/'raw.afc.sqlite'),database=database)
    # These wrappers stamp source hashes required by the shared prepare functions.
    for name,template,key,updates in [
        ('features','extract_axial_features.json','fiber_axial_features',dict(axial_prior_output=common['features'])),
        ('priors','extract_outward_priors.json','fiber_normal_priors',dict(orient_database=database,normal_prior_output=str(root/'priors.json')))]:
        if (root/(name+'.json')).exists(): continue
        cfg=load(template);cfg.update(afv=afv,threads=4,output=str(root/(name+'_native.json')),**updates)
        path=root/(name+'_config.json');atomic_write(path,cfg,durable=False)
        dispatch({key:dict(native_config=str(path),output_root=str(root/(name+'_runner')))})
    for name,template,key in [('normal','run_sign_resolution.json','fiber_normal_sign'),
                               ('hv','run_hv_recommended.json','fiber_hv_stage'),
                               ('h','run_h_direction_breakage_v2.json','fiber_h_direction'),
                               ('v','run_v_direction.json','fiber_direction')]:
        if (root/name/'summary.json').exists(): continue
        cfg=load(template,key);cfg.update(common,output_root=str(root/name),tensor_cache=str(root/(name+'_tensors.npz')),
                                         devices=config.get('devices','0'),gpu_manager=manager)
        cfg.pop('warm_start',None);cfg.pop('baseline_first',None)
        if name=='normal': cfg.update(priors=str(root/'priors.json'),broken_penalties=[.1])
        if name=='h': cfg.update(breakage_weights=[.05],baseline_first=True)
        if name=='v': cfg.update(breakage_weights=[.1],baseline_first=True,inherited_h_solution=str(root/'h/penalty_0.05/solution.npz'))
        dispatch({key:cfg})
    cfg=load('run_uv_current_baseline.json','fiber_uv_layout')
    for key in ('reference_subset','reference_projection','reference_widths','benchmark_updates','benchmark_log_every','benchmark_warmup'):
        cfg.pop(key,None)
    uv_name='uv_whole' if config.get('trusted_gt',False) else 'uv'
    cfg.update(common,h_direction_solution=str(root/'h/penalty_0.05/solution.npz'),
               v_direction_solution=str(root/'v/penalty_0.1/solution.npz'),
               tensor_cache=str(root/(uv_name+'_tensors.npz')),output_root=str(root/uv_name),
               keep_whole_fibers=config.get('trusted_gt',False),force_keep_fibers=config.get('trusted_gt',False),
               initialization_geometry_cache=load('run_uv_joint100_fresh_low_width_v2.json','fiber_uv_layout')['tensor_cache'],
               distributed_world_size=1,devices=config.get('devices','0'),gpu_manager=manager,
               traversal='full_pass',passes=10080,batch_size=24576,log_every=1000,
               benchmark_updates=10080,benchmark_log_every=1000)
    cfg['bucket_shared_phase']=config.get('bucket_shared_phase',False)
    if not (root/uv_name/'summary.json').exists(): dispatch({'fiber_uv_layout':cfg})
    return report(root,uv_name)


def report(root,uv_name='uv'):
    from . import fiber_uv_layout as uv
    root=Path(root); summary=json.loads((root/uv_name/'summary.json').read_text()); cfg=summary['config']
    with np.load(cfg['tensor_cache']) as a: d={key:a[key] for key in a.files}
    with np.load(root/uv_name/'solution.npz') as a: state=a['state']; widths=a['widths']
    if not np.isfinite(state).all() or not np.isfinite(widths).all(): raise ValueError('Nonfinite solution')
    p,labels,keep,pick,u,v,b=uv.hard_geometry(d,state,widths,cfg)
    geometry=(p,labels,keep,pick,u,v,b)
    q,_=uv.geometry_points(d,u,v,b,widths,np.arange(len(keep)),np.zeros(len(keep)))
    q_end,_=uv.geometry_points(d,u,v,b,widths,np.arange(len(keep)),d['node_ds'][:,-1])
    sources=json.loads((root/'fibers.sources.json').read_text())
    names={r['id']:Path(r['file']).name for r in sources}
    pieces=[dict(piece=i,source_id=int(d['source_id'][i]),file=names[int(d['source_id'][i])],
                 family='HV'[d['family'][i]],start_arc=float(d['start'][i]),end_arc=float(d['end'][i]),
                 retained=bool(keep[i]),bucket=int(pick[i]),anchor_q=float(q[i]),end_q=float(q_end[i]),
                 u=float(u[i]),v=float(v[i]),broken_probability=float(p[i,-1]),
                 chosen_joint_probability=float(p[i,pick[i]])) for i in range(len(keep))]
    with sqlite3.connect(Path(cfg['database']).resolve().as_uri()+'?mode=ro',uri=True) as db:
        pairs=[dict(a=int(a),b=int(b),events=int(n),winding_mean=mean,winding_min=lo,winding_max=hi)
               for a,b,n,mean,lo,hi in db.execute('SELECT fiber_a,fiber_b,count(*),avg(winding_distance),min(winding_distance),max(winding_distance) FROM constraints GROUP BY fiber_a,fiber_b')]
    e=np.arange(len(d['i']));qa,qb,delta,active=uv.edge_readout(d,geometry,widths,e,cfg,state)
    positional=qb-qa-d['phase_delta']+.5*(d['family'][d['j']]-d['family'][d['i']])
    constraints=[dict(event_id=int(d['event_ids'][n]),a=int(d['source_id'][d['i'][n]]),
                      b=int(d['source_id'][d['j'][n]]),measured_distance=float(d['distance'][n]),
                      sign=int(d['sign'][n]),bucket_delta=float(delta[n]),position_delta=float(positional[n]),
                      sign_satisfied=bool(delta[n]*d['sign'][n]>=-1e-4)) for n in e]
    result=dict(final=summary['final'],pieces=pieces,constraint_pairs=pairs,
                constraints=constraints,physical_sign_violations=int(np.sum(positional*d['sign'] < -1e-4)),
                solution=str(root/uv_name/'solution.npz'),interpretation='Inferred layout, not annotated adjacency or absolute winding GT')
    atomic_write(root/'report.json',result,durable=False)
    print(json.dumps(result),flush=True)
    return result
