"""Reviewed-JSON adapter with caller-configured extraction and solver stages."""
import json
import hashlib
import sqlite3
from pathlib import Path
import numpy as np
from lasagna.inference_provenance import atomic_write, sha256_file


def convert(source_dir, template, destination, *, expected_manifest_tokens):
    from vesuvius.neural_tracing.tracers.shared.annotations.afv import write_afv, read_afv_polylines
    with sqlite3.connect(Path(template).resolve().as_uri()+'?mode=ro', uri=True) as db:
        metadata = {k: json.loads(v) for k, v in db.execute('SELECT key,value FROM metadata')}
    if not expected_manifest_tokens or any(not str(token) for token in expected_manifest_tokens):
        raise ValueError('Provide expected_manifest_tokens to verify the source scan')
    fibers=[]; records=[]
    for i, path in enumerate(sorted(Path(source_dir).glob('*.json')), 1):
        doc=json.loads(path.read_text()); xyz=np.asarray(doc['line_points'], dtype=np.float64)
        shape=metadata['root']['coordinate_base_shape_zyx']
        if doc.get('coordinate_base_shape_zyx',shape) != shape:
            raise ValueError('Coordinate shape mismatch: '+str(path))
        if xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)<2 or not np.isfinite(xyz).all():
            raise ValueError('Invalid curve: '+str(path))
        if doc.get('branches') or doc.get('adjacent_branches'):
            raise ValueError('This adapter does not import explicit links')
        manifests=[' '.join(str(c.get('segment_to_next',{}).get(k,'')) for k in ('normal_manifest','fiber_manifest')) for c in doc['control_points']]
        if len(manifests)<2 or any(any(str(token) not in m for token in expected_manifest_tokens) for m in manifests[:-1]):
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
    from .fiber_workflows import run as dispatch
    root=Path(config['output_root']); root.mkdir(parents=True,exist_ok=True)
    if any(key in config for key in ('templates','template_files','solve_parameters','trusted_gt')):
        raise ValueError('Use explicit stages and stage parameters; implicit pipeline presets are not supported')
    afv=str(root/'fibers.afv')
    stages=stage_requests(config.get('stages',[]),root,afv)
    signature=dict(config=config,stages=stages,afv_template_sha256=sha256_file(config['afv_template']))
    manifest=root/'pipeline_inputs.json'
    if manifest.exists() and json.loads(manifest.read_text())!=signature:
        raise ValueError('Changed pipeline inputs: use a new output directory')
    atomic_write(manifest,signature,durable=False)
    records=convert(config['source_dir'],config['afv_template'],afv,
                    expected_manifest_tokens=config['expected_manifest_tokens'])
    for stage in stages:
        path=root/(stage['name']+'_request.json')
        atomic_write(path,stage['request'],durable=False)
        # Each workflow owns its provenance checks and resume protocol. Do not
        # infer completion from an arbitrary file left by an earlier experiment.
        dispatch(stage['request'])
    if config.get('layout_output'):
        return report(root,config['layout_output'])
    return dict(afv=afv,sources=records,stages=[stage['name'] for stage in stages])


def stage_requests(stages,root,afv):
    """Resolve explicit workflow requests without injecting model/run settings."""
    names=set();result=[]
    def expand(value):
        if isinstance(value,str):return value.replace('{output_root}',str(root)).replace('{afv}',str(afv))
        if isinstance(value,list):return [expand(v) for v in value]
        if isinstance(value,dict):return {k:expand(v) for k,v in value.items()}
        return value
    for stage in stages:
        name=stage['name']
        if not name or Path(name).name!=name or name in ('.','..') or name in names:
            raise ValueError('Stage names must be unique path components')
        names.add(name)
        if ('config_file' in stage)==('request' in stage):
            raise ValueError('Each stage needs exactly one config_file or request')
        request=json.loads(Path(stage['config_file']).read_text()) if 'config_file' in stage else stage['request']
        if not isinstance(request,dict) or not request or ('commands' not in request and len(request)!=1):
            raise ValueError('Each stage must select exactly one workflow or native command list')
        result.append(dict(name=name,request=expand(request)))
    return result


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
