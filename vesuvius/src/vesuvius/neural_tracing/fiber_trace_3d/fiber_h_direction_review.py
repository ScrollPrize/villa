"""CPU audit and existing-renderer exports for the H direction breakage sweep."""
import json
import csv
from pathlib import Path
import numpy as np
from lasagna.inference_provenance import atomic_write, sha256_file
from .fiber_h_direction import _ro, breakage_readout
from .fiber_normal_sign_visualization import export, spatial_subset, obj_fiber, arrow_mesh
from vesuvius.neural_tracing.fiber_follow.data.afv import read_afv_polylines
from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength, interp_at, tangents_at


def audit(root, d):
    root=Path(root);summary=json.loads((root/'summary.json').read_text())
    with np.load(root/'solution.npz') as f:s={k:f[k] for k in f.files}
    if not np.array_equal(s['ids'],d['ids']) or not np.array_equal(s['active'],d['active']):raise ValueError('IDs changed')
    metrics,joint,rejected=breakage_readout(s['sign_logit_ccw'],s['broken_logit'],d['prior'],d['i'],d['j'],d['dot'],d['component'],d['lengths'])
    np.testing.assert_allclose(joint,s['joint_probability_ccw'],atol=1e-7)
    np.testing.assert_allclose(joint[:,[1,0,2]],s['joint_probability_cw'],atol=1e-7)
    np.testing.assert_array_equal(s['direction_sign_cw'],-s['direction_sign_ccw'])
    expected=np.zeros(len(d['ids']),np.int8);expected[d['active']]=np.where(rejected,0,np.where(s['sign_logit_ccw']>=0,1,-1))
    np.testing.assert_array_equal(expected,s['direction_sign_ccw'])
    full_rejected=np.zeros(len(d['ids']),bool);full_rejected[d['active']]=rejected
    np.testing.assert_array_equal(full_rejected,s['newly_broken'])
    labels=d['labels'].copy();labels[full_rejected]=2
    np.testing.assert_array_equal(labels,s['labels'])
    np.testing.assert_array_equal(labels==2,s['broken'])
    for k in ('disagreeing_events','new_rejections','retained_events'):
        if metrics[k]!=summary['final'][k]:raise ValueError('Readout mismatch: '+k)
    degree=np.bincount(np.r_[d['i'],d['j']],minlength=len(d['active']))
    metrics['isolated_rejected']=int(((degree==0)&rejected).sum())
    with np.load(summary['config']['warm_start']) as w:oldsign=w['direction_sign_ccw'][d['active']]
    oldbad=d['dot']*oldsign[d['i']]*oldsign[d['j']]<0
    touched=np.zeros(len(d['active']),bool);touched[d['i'][oldbad]]=True;touched[d['j'][oldbad]]=True
    metrics['rejected_without_original_conflict']=int((rejected&~touched).sum())
    sign=np.where(s['sign_logit_ccw']>=0,1,-1);i,j=d['i'],d['j']
    wrong=d['dot']*sign[i]*sign[j]<0
    restored_conflicts=np.bincount(i[wrong&~rejected[j]],minlength=len(rejected))+np.bincount(j[wrong&~rejected[i]],minlength=len(rejected))
    metrics['individually_restorable_without_conflicts']=int((rejected&(restored_conflicts==0)).sum())
    metrics.update(solution_sha256=sha256_file(root/'solution.npz'),earlier_labels_preserved=True)
    atomic_write(root/'breakage_audit.json',metrics,durable=False)
    return s,metrics


def export_views(config,cfg,s,d,selected_root):
    root=Path(config['visualization_root']);survivors=s['ids'][s['labels']==0]
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
    with np.load(cfg['warm_start']) as old:oldsign=old['direction_sign_ccw'][d['active']]
    bad=d['dot']*oldsign[d['i']]*oldsign[d['j']]<0
    conflicts=np.bincount(np.r_[d['i'][bad],d['j'][bad]],minlength=len(d['active']))
    with (root/'disabled_h.obj').open('w') as out:
        out.write('# ALL newly rejected H fibers; native L0 XYZ, full original curves\nmtllib fibers.mtl\n')
        for row,points in read_afv_polylines(cfg['afv'],rejected_ids):
            out.write(obj_fiber(row['id'],points,first,True))
            a=int(np.searchsorted(s['ids'][s['active']],row['id']))
            records.append(dict(id=row['id'],first_vertex=first,vertices=len(points),edges=len(points)-1,
                length_base=float(d['lengths'][a]),baseline_incident_conflicts=int(conflicts[a]),
                broken_probability=float(s['broken_probability'][a])))
            first+=len(points)
    with (root/'disabled_h.csv').open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=['id','first_vertex','vertices','edges','length_base','baseline_incident_conflicts','broken_probability'])
        writer.writeheader();writer.writerows(records)
    arrow_records=[];first=1
    with (root/'surviving_h_directions.obj').open('w') as out:
        out.write('# Same surviving H subset, with CCW tangent arrows; native L0 XYZ\nmtllib fibers.mtl\n')
        for row,points in read_afv_polylines(cfg['afv'],selected):
            fid=row['id'];sign=int(s['direction_sign_ccw'][np.searchsorted(s['ids'],fid)])
            if sign not in (-1,1):raise ValueError('Disabled fiber in surviving view')
            out.write(obj_fiber(fid,points,first,False,material='h'));first+=len(points)
            arc=arclength(points);spacing=float(config.get('tangent_spacing',1024))
            locations=np.arange(spacing/2,arc[-1],spacing)
            if not len(locations):locations=np.array([arc[-1]/2])
            xyz=interp_at(points,arc,locations);directions=sign*tangents_at(points,arc,locations,half=16)
            for t,p,n in zip(locations,xyz,directions):
                if np.linalg.norm(n)<.99:continue
                length=float(config.get('tangent_length',256));radius=float(config.get('tangent_radius',8))
                vv,ff=arrow_mesh(p,n,length,radius)
                out.write(f'o tangent_{fid}_{t:.3f}\nusemtl h\n')
                out.writelines('v '+' '.join(f'{v:.9f}' for v in x)+'\n' for x in vv)
                out.writelines('f '+' '.join(str(int(v)+first) for v in f)+'\n' for f in ff)
                arrow_records.append(dict(id=fid,arc=float(t),sign=sign,point=p.tolist(),tangent=n.tolist(),
                                          first_vertex=first,vertices=len(vv),length=length,radius=radius))
                first+=len(vv)
    files=['disabled_h.obj','disabled_h.csv','h_fibers.obj','h_normals.ply','surviving_h_directions.obj','umbilicus.ply','fibers.mtl']
    manifest=dict(config=config,solution=str(selected_root/'solution.npz'),solution_sha256=sha256_file(selected_root/'solution.npz'),
        disabled_count=len(rejected_ids),disabled_records=records,surviving_ids=selected,arrows=arrow_records,
        source_afv_sha256=sha256_file(cfg['afv']),
        selection='All newly disabled H; farthest-point spatial sample from 10000 longest surviving H; whole original curves',
        direction='CCW right-handed about positive-Z umbilicus axis; saved sign times stored tangent; NOT sheet-normal arrows',
        files={name:dict(bytes=(root/name).stat().st_size,sha256=sha256_file(root/name)) for name in files})
    atomic_write(root/'h_direction_index.json',manifest,durable=False)
    with (root/'README.txt').open('a') as out:
        out.write('\nH direction pass: disabled_h.obj is ALL new H rejections (earlier rejections excluded).\n'
                  'h_fibers.obj + h_normals.ply are the surviving H subset with outward display normals.\n'
                  'surviving_h_directions.obj contains the SAME curves plus CCW TANGENT arrows (not normals).\n'
                  'h_direction_index.json records original IDs, signs, selection and arrow positions.\n')
    return dict(root=str(root),disabled=len(rejected_ids),survivors=len(selected),arrows=len(arrow_records))


def run(config):
    root=Path(config['sweep_root']);sweep=json.loads((root/'summary.json').read_text());cfg=sweep['config']
    stamp=json.loads(Path(cfg['tensor_cache']).with_suffix('.json').read_text())
    for key in ('database','afv','features','stage1','hv_solution','umbilicus'):
        if sha256_file(cfg[key])!=stamp['signature'][key]:raise ValueError('Changed input '+key)
    with np.load(cfg['tensor_cache']) as f:d={k:f[k] for k in f.files}
    rows=[];chosen=None;chosen_root=Path(sweep['selected']['output_root'])
    for row in sweep['results']:
        solution,metric=audit(row['output_root'],d);rows.append(dict(penalty=row['penalty'],**metric))
        if Path(row['output_root'])==chosen_root:chosen=solution
    if chosen is None:raise ValueError('Selected solution not audited')
    lines=['# H direction breakage sweep','',
           'Whole-fiber rejection; fixed input evidence. Global U/V remains paused.','',
           '| Penalty | Newly disabled H | Disabled H % | Retained arc % | Retained events % | Remaining conflicts | Conflict % of retained events |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"|{r['penalty']:g}|{r['new_rejections']}|{r['new_rejections_percent']:.6f}|{r['retained_arc_percent']:.6f}|{r['retained_events_percent']:.6f}|{r['disagreeing_events']}|{r['disagreement_percent']:.6f}|")
    lines+=['',f"Selected: `{chosen_root}`. Selection favors zero conflicts, then retained arc and fiber count.",
            'Rejection denotes inconsistent direction evidence, not proven physical fiber breakage. All earlier labels/rejections are preserved.',
            'CW is exact sign negation; pair costs and broken masks are unchanged.']
    Path(config['report']).write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)
    vis=export_views(config,cfg,chosen,d,chosen_root) if config.get('visualization_root') else None
    result=dict(rows=rows,selected=str(chosen_root),visualization=vis)
    atomic_write(root/'review.json',result,durable=False)
    print(json.dumps(result),flush=True);return result
