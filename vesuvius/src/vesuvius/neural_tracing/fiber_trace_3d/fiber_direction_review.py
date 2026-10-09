"""CPU audit and existing-renderer exports for H/V direction breakage sweeps."""
import json
import csv
from pathlib import Path
import numpy as np
from lasagna.inference_provenance import atomic_write, sha256_file
from .fiber_direction import _ro, breakage_readout, family_settings,label_metrics
from .fiber_normal_sign_visualization import export, spatial_subset, obj_fiber, arrow_mesh
from vesuvius.neural_tracing.fiber_follow.data.afv import read_afv_polylines
from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength, interp_at, tangents_at


def audit(root, d):
    root=Path(root);summary=json.loads((root/'summary.json').read_text())
    family,_,(positive,negative)=family_settings(summary['config'])
    with np.load(root/'solution.npz') as f:s={k:f[k] for k in f.files}
    if not np.array_equal(s['ids'],d['ids']) or not np.array_equal(s['active'],d['active']):raise ValueError('IDs changed')
    metrics,joint,rejected=breakage_readout(s[f'sign_logit_{positive}'],s['broken_logit'],d['prior'],d['i'],d['j'],d['dot'],d['component'],d['lengths'])
    label_metrics(metrics,family)
    np.testing.assert_allclose(joint,s[f'joint_probability_{positive}'],atol=1e-7)
    np.testing.assert_allclose(joint[:,[1,0,2]],s[f'joint_probability_{negative}'],atol=1e-7)
    np.testing.assert_array_equal(s[f'direction_sign_{negative}'],-s[f'direction_sign_{positive}'])
    expected=np.zeros(len(d['ids']),np.int8);expected[d['active']]=np.where(rejected,0,np.where(s[f'sign_logit_{positive}']>=0,1,-1))
    np.testing.assert_array_equal(expected,s[f'direction_sign_{positive}'])
    for key in d:
        if key.startswith('inherited_h_direction_sign_'):np.testing.assert_array_equal(s[key],d[key])
    full_rejected=np.zeros(len(d['ids']),bool);full_rejected[d['active']]=rejected
    np.testing.assert_array_equal(full_rejected,s['newly_broken'])
    labels=d['labels'].copy();labels[full_rejected]=2
    np.testing.assert_array_equal(labels,s['labels'])
    np.testing.assert_array_equal(labels==2,s['broken'])
    for k in ('disagreeing_events','new_rejections','retained_events'):
        if metrics[k]!=summary['final'][k]:raise ValueError('Readout mismatch: '+k)
    degree=np.bincount(np.r_[d['i'],d['j']],minlength=len(d['active']))
    metrics['isolated_rejected']=int(((degree==0)&rejected).sum())
    with np.load(summary['config']['warm_start']) as w:oldsign=w[f'direction_sign_{positive}'][d['active']]
    oldbad=d['dot']*oldsign[d['i']]*oldsign[d['j']]<0
    touched=np.zeros(len(d['active']),bool);touched[d['i'][oldbad]]=True;touched[d['j'][oldbad]]=True
    metrics['rejected_without_original_conflict']=int((rejected&~touched).sum())
    sign=np.where(s[f'sign_logit_{positive}']>=0,1,-1);i,j=d['i'],d['j']
    wrong=d['dot']*sign[i]*sign[j]<0
    restored_conflicts=np.bincount(i[wrong&~rejected[j]],minlength=len(rejected))+np.bincount(j[wrong&~rejected[i]],minlength=len(rejected))
    metrics['individually_restorable_without_conflicts']=int((rejected&(restored_conflicts==0)).sum())
    opposite_wrong=d['dot']*sign[i]*sign[j]>0
    opposite_conflicts=np.bincount(i[opposite_wrong&~rejected[j]],minlength=len(rejected))+np.bincount(j[opposite_wrong&~rejected[i]],minlength=len(rejected))
    metrics['individually_restorable_either_sign']=int((rejected&((restored_conflicts==0)|(opposite_conflicts==0))).sum())
    metrics.update(solution_sha256=sha256_file(root/'solution.npz'),earlier_labels_preserved=True)
    atomic_write(root/'breakage_audit.json',metrics,durable=False)
    return s,metrics


def export_views(config,cfg,s,d,selected_root):
    family,label,(positive,negative)=family_settings(cfg);short=family.lower()
    root=Path(config['visualization_root']);survivors=s['ids'][s['labels']==label]
    with _ro(cfg['afv']) as db:
        rows=db.execute('SELECT id,(min_x+max_x)/2.,(min_y+max_y)/2.,(min_z+max_z)/2.,length FROM fibers ORDER BY length DESC,id').fetchall()
    allowed=set(survivors.tolist());pool=[r for r in rows if r[0] in allowed][:int(config.get('longest_candidate_count',10000))]
    selected=sorted(spatial_subset([r[0] for r in pool],[r[1:4] for r in pool],int(config.get('sample_count',1000))))
    # Reuse the established normal-arrow and MeshLab-compatible curve exporter.
    viscfg=dict(output_root=str(root),solution=cfg['stage1'],database=cfg['database'],afv=cfg['afv'],
        hv_solution=cfg['hv_solution'],umbilicus=cfg['umbilicus'],fiber_ids=selected,
        arrow_spacing=1024,arrow_length=640,arrow_radius=20,max_arrows=100000,max_points=2000000)
    export(viscfg)
    rejected_ids=s['ids'][s['newly_broken']].tolist();records=[];first=1
    with np.load(cfg['warm_start']) as old:oldsign=old[f'direction_sign_{positive}'][d['active']]
    bad=d['dot']*oldsign[d['i']]*oldsign[d['j']]<0
    conflicts=np.bincount(np.r_[d['i'][bad],d['j'][bad]],minlength=len(d['active']))
    with (root/f'disabled_{short}.obj').open('w') as out:
        out.write(f'# ALL newly rejected {family} fibers; native L0 XYZ, full original curves\nmtllib fibers.mtl\n')
        for row,points in read_afv_polylines(cfg['afv'],rejected_ids):
            out.write(obj_fiber(row['id'],points,first,True))
            a=int(np.searchsorted(s['ids'][s['active']],row['id']))
            records.append(dict(id=row['id'],first_vertex=first,vertices=len(points),edges=len(points)-1,
                length_base=float(d['lengths'][a]),baseline_incident_conflicts=int(conflicts[a]),
                broken_probability=float(s['broken_probability'][a])))
            first+=len(points)
    with (root/f'disabled_{short}.csv').open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=['id','first_vertex','vertices','edges','length_base','baseline_incident_conflicts','broken_probability'])
        writer.writeheader();writer.writerows(records)
    arrow_records=[];first=1
    with (root/f'surviving_{short}_directions.obj').open('w') as out:
        out.write(f'# Same surviving {family} subset, with {positive.upper()} tangent arrows; native L0 XYZ\nmtllib fibers.mtl\n')
        for row,points in read_afv_polylines(cfg['afv'],selected):
            fid=row['id'];sign=int(s[f'direction_sign_{positive}'][np.searchsorted(s['ids'],fid)])
            if sign not in (-1,1):raise ValueError('Disabled fiber in surviving view')
            out.write(obj_fiber(fid,points,first,False,material=short));first+=len(points)
            arc=arclength(points);spacing=float(config.get('tangent_spacing',1024))
            locations=np.arange(spacing/2,arc[-1],spacing)
            if not len(locations):locations=np.array([arc[-1]/2])
            xyz=interp_at(points,arc,locations);directions=sign*tangents_at(points,arc,locations,half=16)
            for t,p,n in zip(locations,xyz,directions):
                if np.linalg.norm(n)<.99:continue
                length=float(config.get('tangent_length',256));radius=float(config.get('tangent_radius',8))
                vv,ff=arrow_mesh(p,n,length,radius)
                out.write(f'o tangent_{fid}_{t:.3f}\nusemtl {short}\n')
                out.writelines('v '+' '.join(f'{v:.9f}' for v in x)+'\n' for x in vv)
                out.writelines('f '+' '.join(str(int(v)+first) for v in f)+'\n' for f in ff)
                arrow_records.append(dict(id=fid,arc=float(t),sign=sign,point=p.tolist(),tangent=n.tolist(),
                                          first_vertex=first,vertices=len(vv),length=length,radius=radius))
                first+=len(vv)
    files=[f'disabled_{short}.obj',f'disabled_{short}.csv',f'{short}_fibers.obj',f'{short}_normals.ply',f'surviving_{short}_directions.obj','umbilicus.ply','fibers.mtl']
    manifest=dict(config=config,solution=str(selected_root/'solution.npz'),solution_sha256=sha256_file(selected_root/'solution.npz'),
        disabled_count=len(rejected_ids),disabled_records=records,surviving_ids=selected,arrows=arrow_records,
        source_afv_sha256=sha256_file(cfg['afv']),
        selection=f'All newly disabled {family}; farthest-point spatial sample from longest surviving {family}; whole original curves',
        direction=f'{positive.upper()} relative to positive-Z umbilicus axis; saved sign times stored tangent; NOT sheet-normal arrows',
        files={name:dict(bytes=(root/name).stat().st_size,sha256=sha256_file(root/name)) for name in files})
    atomic_write(root/f'{short}_direction_index.json',manifest,durable=False)
    with (root/'README.txt').open('a') as out:
        out.write(f'\n{family} direction pass: disabled_{short}.obj is ALL new {family} rejections (earlier rejections excluded).\n'
                  f'{short}_fibers.obj + {short}_normals.ply are the surviving {family} subset with outward display normals.\n'
                  f'surviving_{short}_directions.obj contains the SAME curves plus {positive.upper()} TANGENT arrows (not normals).\n'
                  f'{short}_direction_index.json records original IDs, signs, selection and arrow positions.\n')
    return dict(root=str(root),disabled=len(rejected_ids),survivors=len(selected),arrows=len(arrow_records))


def run(config):
    root=Path(config['sweep_root']);sweep=json.loads((root/'summary.json').read_text());cfg=sweep['config']
    family,_,(positive,negative)=family_settings(cfg)
    stamp=json.loads(Path(cfg['tensor_cache']).with_suffix('.json').read_text())
    keys=['database','afv','features','stage1','hv_solution','umbilicus']
    if cfg.get('inherited_h_solution'):keys.append('inherited_h_solution')
    for key in keys:
        if sha256_file(cfg[key])!=stamp['signature'][key]:raise ValueError('Changed input '+key)
    with np.load(cfg['tensor_cache']) as f:d={k:f[k] for k in f.files}
    rows=[];chosen=None;chosen_root=Path(sweep['selected']['output_root'])
    for row in sweep['results']:
        solution,metric=audit(row['output_root'],d);rows.append(dict(penalty=row['penalty'],**metric))
        if Path(row['output_root'])==chosen_root:chosen=solution
    if chosen is None:raise ValueError('Selected solution not audited')
    lines=[f'# {family} direction breakage sweep','',
           'Whole-fiber rejection; fixed input evidence. Global U/V remains paused.','',
           f'| Penalty | Newly disabled {family} | Disabled {family} % | Retained arc % | Retained events % | Remaining conflicts | Conflict % of retained events |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"|{r['penalty']:g}|{r['new_rejections']}|{r['new_rejections_percent']:.6f}|{r['retained_arc_percent']:.6f}|{r['retained_events_percent']:.6f}|{r['disagreeing_events']}|{r['disagreement_percent']:.6f}|")
    lines+=['',f"Selected: `{chosen_root}`. Selection favors zero conflicts, then retained arc and fiber count.",
            'Rejection denotes inconsistent direction evidence, not proven physical fiber breakage. All earlier labels/rejections are preserved.',
            f'{negative.upper()} is exact sign negation of {positive.upper()}; pair costs and broken masks are unchanged.']
    if cfg.get('baseline_first'):
        base=json.loads((Path(cfg['warm_start']).parent/'summary.json').read_text())
        lines+=['',f"Sign-only baseline: {base['final']['disagreeing_events']:,} conflicts ({base['final']['disagreement_percent']:.6f}%)."]
    Path(config['report']).write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)
    vis=export_views(config,cfg,chosen,d,chosen_root) if config.get('visualization_root') else None
    result=dict(rows=rows,selected=str(chosen_root),visualization=vis)
    atomic_write(root/'review.json',result,durable=False)
    if config.get('recommendation'):
        selected_metrics=next(r for r in rows if r['solution_sha256']==sha256_file(chosen_root/'solution.npz'))
        recommendation=dict(family=family,convention=positive.upper(),solution=str(chosen_root/'solution.npz'),
            solution_sha256=selected_metrics['solution_sha256'],breakage_weight=sweep['selected']['penalty'],
            newly_broken=selected_metrics['new_rejections'],retained_family=len(d['active'])-selected_metrics['new_rejections'],
            retained_events=selected_metrics['retained_events'],conflicts=selected_metrics['disagreeing_events'],
            retained_arc_percent=selected_metrics['retained_arc_percent'],visualization=vis,
            inherited_h_solution=cfg.get('inherited_h_solution'),inherited_h_sha256=stamp['signature'].get('inherited_h_solution'))
        atomic_write(Path(config['recommendation']),recommendation,durable=False)
    print(json.dumps(result),flush=True);return result
