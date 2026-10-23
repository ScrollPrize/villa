"""Small-graph optimizer comparison using the unchanged complete solver loss."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from lasagna.inference_provenance import atomic_write
from .fiber_gt_constraint_score import arrays
from .fiber_bucket_uv import BucketUV
from .fiber_bucket_uv_visualization import current_eligible_edges
from .fiber_affine_geometry import direction_plan
from . import fiber_soft_winding as soft
from . import fiber_implicit_winding as density


def angle_statistics(ext,state,widths):
    """Physical UV angles for each displayed MAP segment, not winding-space angles."""
    with torch.no_grad():
        _,lp=ext.local_hypotheses(state);col=lp.argmax(1)
        r,l,f,di,gap,o=direction_plan(ext.owner,ext.v_offset_arc,ext.curve.family,ext.curve.direction)
        result={}
        for name,pos in [('shared',ext.shared_positions(state,widths)),
                         ('intermediate',ext.intermediate.physical(widths)),('bucket',ext.positions())]:
            p=pos.double()[torch.arange(len(ext.owner)),col[ext.owner]]
            delta=p[r]-p[l];signed=delta.gather(1,f[:,None]).squeeze(1)*di
            angle=torch.rad2deg(torch.acos((signed/delta.norm(dim=1).clamp_min(1e-12)).clamp(-1,1)))
            result[name]=dict(mean=float(angle.mean()),median=float(angle.median()),
                p95=float(angle.quantile(.95)),max=float(angle.max()),reversals=int((signed<0).sum()))
        return result


def plateau(history,minimum=10000,window=5):
    """Require stationary block-median loss AND geometry, not one lucky iterate."""
    if len(history)<window or history[-1]['step']<minimum:return False
    recent=history[-window:]
    losses=[r['block_median_loss'] for r in recent]
    stable_loss=max(losses)-min(losses)<=max(1e-4,.01*min(losses))
    lengths=[r['length_max_pct'] for r in recent]
    angles=[r['angles']['bucket']['max'] for r in recent]
    return (stable_loss and max(lengths)-min(lengths)<.05 and max(angles)-min(angles)<.05
            and all(r['meetings']==recent[0]['meetings'] and r['meetings']>0 for r in recent))


def no_improvement_plateau(history,minimum=10000,window=5):
    if len(history)<=window or history[-1]['step']<minimum:return False
    prior=history[:-window];recent=history[-window:]
    best=min(r['block_median_loss'] for r in prior)
    improved=min(r['block_median_loss'] for r in recent)<best-max(1e-4,.01*best)
    length=[r['length_max_pct'] for r in recent]
    angle=[r['angles']['bucket']['max'] for r in recent]
    return (not improved and max(length)-min(length)<.05
            and abs(float(np.median(angle[:3]))-float(np.median(angle[-3:])))<.05
            and all(r['meetings']==recent[0]['meetings'] and r['meetings']>0 for r in recent))


def run(settings):
    torch.set_num_threads(4)
    src=Path(settings['source']);root=Path(settings['output_root'])
    root.mkdir(parents=True,exist_ok=True)
    if settings.get('report_only'):return summarize(root)
    cfg=json.loads((src/'config.json').read_text())
    d=arrays(cfg['tensor_cache']);saved=arrays(settings.get('checkpoint',src/'solution.npz'))
    samples=arrays(cfg['soft_samples']);data,nodes=soft.tensor_inputs(d,samples,cfg,'cpu')
    _,initial=soft.uv.radial_initial_geometry(d,cfg)
    initial=torch.as_tensor(initial,dtype=torch.float32)
    results=[]
    for trial in settings['trials']:
        out=root/trial['name']
        if (out/'report.json').exists():
            results.append(json.loads((out/'report.json').read_text()));continue
        out.mkdir(exist_ok=True)
        joint=trial.get('joint',False)
        state=torch.nn.Parameter(torch.as_tensor(saved['state']).clone(),requires_grad=joint)
        w=torch.as_tensor(saved['widths'])-1e-4
        raw=torch.nn.Parameter(w+torch.log(-torch.expm1(-w)),requires_grad=joint)
        if joint and cfg.get('force_keep_fibers'):
            state.register_hook(lambda g:torch.cat((g[:,:3],torch.zeros_like(g[:,3:])),1))
        ext=BucketUV(d,cfg,'cpu');ext.prepare_fields(d,samples);ext.restore(saved)
        ext.cavity.values=torch.as_tensor(saved['winding_sigma'])
        groups=ext.optimizer_groups()
        if joint:groups += [dict(params=[state],lr=cfg['lr']),dict(params=[raw],lr=cfg['width_lr'])]
        local_ids={id(ext.values),id(ext.curve.u),id(ext.curve.v),id(ext.intermediate.values)}
        if trial.get('local_scope')=='bucket':local_ids={id(ext.values)}
        for g in groups:
            g['lr']*=trial.get('lr_factor',1.)
            if all(id(p) in local_ids for p in g['params']):g['lr']*=trial.get('local_lr_factor',1.)
        kind=trial['optimizer']
        if kind=='adam':opt=torch.optim.Adam(groups)
        elif kind=='sgd':opt=torch.optim.SGD(groups,momentum=trial.get('momentum',.9))
        elif kind=='lbfgs':
            if joint:raise ValueError('L-BFGS probe requires a fixed window')
            opt=torch.optim.LBFGS([p for g in groups for p in g['params']],lr=1.,max_iter=10,line_search_fn='strong_wolfe')
        else:raise ValueError(kind)
        ext.bind_optimizer(opt)
        if settings.get('optimizer_state'):opt.load_state_dict(torch.load(settings['optimizer_state'],weights_only=True))
        history=json.loads(Path(settings['history']).read_text()) if settings.get('history') else []
        initial_step=history[-1]['step'] if history else 0
        elapsed_offset=history[-1]['seconds'] if history else 0.
        started=time.monotonic();block=[];converged=False
        r,l,fam,direction,gap,_=direction_plan(ext.owner,ext.v_offset_arc,ext.curve.family,ext.curve.direction)
        for step in range(initial_step,trial['steps']+1):
            widths=torch.nn.functional.softplus(raw)+1e-4
            if joint:ext.update(state,widths.detach(),data,step)
            opt.zero_grad(set_to_none=True)
            loss,terms=soft.objective(state,raw,initial,data,nodes,cfg,pair_cost=density.pair_cost,extension=ext)
            block.append(float(loss.detach()))
            if step%trial.get('log_every',100)==0 or step==trial['steps']:
                with torch.no_grad():
                    _,lp=ext.local_hypotheses(state);col=lp.argmax(1)
                    prob=(lp.exp().amax(1)*state[:,3].sigmoid()).numpy()
                    edges=current_eligible_edges(d,ext,state,widths,col,prob)
                    p=ext.positions()[torch.arange(len(ext.owner)),col[ext.owner]]
                    delta=p[r]-p[l];length=delta.norm(dim=1)
                    err=(length/gap-1).abs()*100
                    forward=delta.gather(1,fam.long()[:,None]).squeeze(1)*direction
                    angle=torch.rad2deg(torch.acos((forward/length.clamp_min(1e-12)).clamp(-1,1)))
                    meeting=(p[ext.v_offset_b[edges]]-p[ext.v_offset_a[edges]]).abs()
                    row=dict(step=step,loss=float(loss),seconds=elapsed_offset+time.monotonic()-started,
                        block_median_loss=float(np.median(block)),angles=angle_statistics(ext,state,widths),
                        meetings=len(edges),length_mean_pct=float(err.mean()),length_max_pct=float(err.max()),
                        angle_max_deg=float(angle.max()),meeting_mean=meeting.mean(0).tolist() if len(edges) else None,
                        meeting_max=meeting.amax(0).tolist() if len(edges) else None,
                        terms={k:float(v) for k,v in terms.items()})
                    if history and history[-1]['step']==step:
                        row['block_median_loss']=history[-1]['block_median_loss']
                        history.pop()
                    history.append(row)
                    block=[]
                    atomic_write(out/'progress.json',history,durable=False)
                    print(json.dumps(dict(trial=trial['name'],**{k:v for k,v in row.items() if k!='terms'})),flush=True)
                    if trial.get('until_converged'):
                        checkpoint=dict(saved,**ext.arrays(),state=state.detach().numpy(),widths=widths.detach().numpy())
                        np.savez(out/'latest.npz',**checkpoint)
                        torch.save(opt.state_dict(),out/'optimizer.pt')
                        criterion=no_improvement_plateau if trial.get('plateau_mode')=='no_improvement' else plateau
                        converged=step>initial_step and criterion(history,trial.get('minimum_steps',10000),trial.get('plateau_blocks',5))
            if converged:break
            if not torch.isfinite(loss):break
            if step==trial['steps']:break
            if kind=='lbfgs':
                def closure():
                    opt.zero_grad(set_to_none=True)
                    value,_=soft.objective(state,raw,initial,data,nodes,cfg,pair_cost=density.pair_cost,extension=ext)
                    value.backward()
                    return value
                opt.step(closure)
            else:
                loss.backward();opt.step()
        result=dict(trial=trial,history=history,completed_steps=step,
                    converged=converged,
                    finite=bool(torch.isfinite(loss)),terminal_loss=float(loss.detach()))
        atomic_write(out/'report.json',result,durable=False)
        sol=dict(saved,**ext.arrays());sol.update(state=state.detach().numpy(),widths=(torch.nn.functional.softplus(raw)+1e-4).detach().numpy())
        if not joint:
            if not np.array_equal(sol['state'],saved['state']):raise RuntimeError('Frozen winding state changed')
            if not np.allclose(sol['widths'],saved['widths'],rtol=1e-7,atol=1e-5):raise RuntimeError('Frozen widths changed')
        np.savez(out/'solution.npz',**sol)
        atomic_write(out/'config.json',dict(cfg,output_root=str(out)),durable=False)
        results.append(result)
    atomic_write(root/'report.json',results,durable=False)
    return results


def summarize(root):
    from .fiber_shared_curve import saved_points
    from .fiber_winding_order import order_margin
    from .fiber_current_gt_solve import enclosure_brackets
    rows=[]
    for path in sorted(root.glob('*/report.json')):
        report=json.loads(path.read_text());s=arrays(path.parent/'solution.npz')
        row=dict(name=path.parent.name,**report['history'][-1])
        row['finite']=bool(np.isfinite(s['state']).all() and np.isfinite(s['bucket_uv_positions']).all())
        if row['finite']:
            cfg=json.loads((path.parent/'config.json').read_text());d=arrays(cfg['tensor_cache'])
            ext=BucketUV(d,cfg,'cpu');ext.restore(s)
            samples=arrays(cfg['soft_samples']);ext.prepare_fields(d,samples)
            data,nodes=soft.tensor_inputs(d,samples,cfg,'cpu')
            _,initial=soft.uv.radial_initial_geometry(d,cfg)
            w=torch.as_tensor(s['widths'])-1e-4
            with torch.no_grad():
                total,_=soft.diagnostic_objective(torch.as_tensor(s['state']),w+torch.log(-torch.expm1(-w)),
                    torch.as_tensor(initial,dtype=torch.float32),data,nodes,cfg,density.pair_cost,ext)
            row['finite']=bool(torch.isfinite(total))
            row['reevaluated_loss']=float(total)
        if row['finite']:
            with torch.no_grad():
                state=torch.as_tensor(s['state']);_,lp=ext.local_hypotheses(state)
                row['angles']=angle_statistics(ext,state,torch.as_tensor(s['widths']))
                col=lp.argmax(1);p=ext.positions()[torch.arange(len(ext.owner)),col[ext.owner]]
                r,l,_,_,gap,owners=direction_plan(ext.owner,ext.v_offset_arc,ext.curve.family,ext.curve.direction)
                length=(p[r]-p[l]).norm(dim=1);worst=int((length/gap-1).abs().argmax())
                row['worst_length_segment']=dict(fiber=int(d['source_id'][owners[worst]]),
                    reference_length=float(gap[worst]),fitted_length=float(length[worst]))
                prob=(lp.exp().amax(1)*state[:,3].sigmoid()).numpy()
                row['meeting_events']=current_eligible_edges(d,ext,state,torch.as_tensor(s['widths']),col,prob).tolist()
                row['MAP_probability_mean']=float(prob.mean())
            for layer,shared in [('shared',True),('intermediate',False)]:
                a,_,_=saved_points(d,s,s['state'],s['widths'],d['i'],d['ds_a'],shared_only=shared)
                b,_,_=saved_points(d,s,s['state'],s['widths'],d['j'],d['ds_b'],shared_only=shared)
                margin=order_margin(b-a,d['sign'],d['family'][d['j']]-d['family'][d['i']],d['parallel'],.5)
                for kind,mask in [('perp',~d['parallel']),('par',d['parallel'])]:
                    mask=mask&(d['sign']!=0)
                    row[layer+'_'+kind+'_sign']=[int((margin[mask]<0).sum()),int(mask.sum())]
                diff=np.where(d['family'][d['i']]==0,b-a,a-b)
                errors=[v for _,_,inner,outer in enclosure_brackets(d) for v in (abs(diff[inner]),abs(-diff[outer]-1))]
                row[layer+'_enclosure_l1']=float(np.mean(errors))
        rows.append(row)
    atomic_write(root/'summary.json',rows,durable=False)
    header='|Run|Updates|Full loss|Meetings|Gap mean U/V|Gap max U/V|Length mean/max %|Angle max|Enclosure L1|Perp/par sign violations|Seconds|\n|---|---:|---:|---:|---|---|---|---:|---:|---|---:|'
    lines=[header]
    for row in rows:
        if not row['finite']:
            lines.append('|'+row['name']+'|—|Diverged|—|—|—|—|—|—|—|—|');continue
        pair=lambda x:'/'.join(f'{v:.3g}' for v in x) if x is not None else '—'
        lines.append(f"|{row['name']}|{row['step']}|{row['loss']:.5g}|{row['meetings']}|{pair(row['meeting_mean'])}|{pair(row['meeting_max'])}|{row['length_mean_pct']:.3g}/{row['length_max_pct']:.3g}|{row['angle_max_deg']:.3g}|{row['shared_enclosure_l1']:.4g}|{pair(row['shared_perp_sign'])}; {pair(row['shared_par_sign'])}|{row['seconds']:.1f}|")
    text='\n'.join(lines)+'\n'
    (root/'summary.md').write_text(text)
    print(text,flush=True)
    return rows
