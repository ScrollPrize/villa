"""Second-stage whole-fiber H/V assignment after fixed normal-stage exclusions."""
import argparse
import json
from pathlib import Path
import sqlite3
import time
import numpy as np
from lasagna.inference_provenance import atomic_write, sha256_file
from .fiber_normal_sign import launch_tensor_solve


def probabilities(h,b):
    import torch
    p=h.sigmoid();q=b.sigmoid()
    return torch.stack(((1-q)*(1-p),(1-q)*p,q),dim=-1)  # H,V,broken


def losses(h,b,a,i,j,relation,cost,prior_weight,entropy_weight):
    import torch
    p=probabilities(h,b);s=2*h.sigmoid()-1;keep=1-b.sigmoid()
    pair=(keep[i]*keep[j]*(1-relation*s[i]*s[j])*.5).mean() if i.numel() else h.sum()*0
    prior=(keep*(1-a*s)*.5).mean()
    rejection=(p[:,2]*cost).mean()
    entropy=-(p*p.clamp_min(1e-12).log()).sum(-1).mean()
    return pair+prior_weight*prior+rejection+entropy_weight*entropy,pair,prior,rejection,entropy


def hard_metrics(probs,a,i,j,relation,lengths):
    """CPU readout also independently validates saved outputs; no vacuous success."""
    labels=np.argmax(probs,axis=1);kept=labels!=2
    edge_kept=kept[i]&kept[j];wrong=((labels[i]==labels[j])!=(relation>0))
    result=dict(h_fibers=int((labels==0).sum()),v_fibers=int((labels==1).sum()),
        newly_broken=int((labels==2).sum()),newly_broken_percent=float(100*(labels==2).mean()),
        retained_arc_percent=float(100*lengths[kept].sum()/lengths.sum()),
        ambiguous_percent=float(100*(probs.max(1)<.9).mean()),
        axial_agreement=float(np.mean(a[kept]*np.where(labels[kept]==1,1,-1))) if kept.any() else None)
    for name,mask in [('all',np.ones(len(i),bool)),('parallel',relation>0),('perpendicular',relation<0)]:
        total=int(mask.sum());retained=int((mask&edge_kept).sum());bad=int((mask&edge_kept&wrong).sum())
        result[name]=dict(events=total,retained=retained,violations=bad,
            retained_percent=100*retained/total if total else None,
            violation_percent=100*bad/retained if retained else None,
            violation_original_percent=100*bad/total if total else None)
    return result


def run_features(config):
    from .crop_benchmark import run as run_crop
    native=json.loads(Path(config['native_config']).read_text())
    paths={key:Path(native[key]) for key in ('afv','umbilicus')}
    for key,path in paths.items():native[key+'_sha256']=sha256_file(path)
    root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    frozen=root/'native_config.json';atomic_write(frozen,native,durable=False)
    run_crop(dict(output_root=str(root),reference_dir=str(paths['umbilicus'].parent),input_files=[str(frozen)],
        commands=[dict(name='tests',binary='test_fiber_trace_winding_bp',args=[]),
                  dict(name='features',binary='vc_fiber_trace_chunk',args=['afv-constraints',str(frozen)])]))
    for key,path in paths.items():
        if sha256_file(path)!=native[key+'_sha256']:
            target=Path(native['axial_prior_output']);target.rename(target.with_suffix('.invalid'))
            raise RuntimeError('Feature input changed during export')


def verify_feature_source(database,feature):
    """Bind cached scores to the actual AFV used by the immutable constraint DB."""
    with sqlite3.connect(Path(database).resolve().as_uri()+'?mode=ro',uri=True) as db:
        provenance=json.loads(db.execute("SELECT value FROM metadata WHERE key='provenance'").fetchone()[0])
        source=Path(provenance['config']['afv'])
        if sha256_file(source)!=feature['afv_sha256']:raise ValueError('Axial/source AFV content mismatch')
        with sqlite3.connect(source.resolve().as_uri()+'?mode=ro',uri=True) as afv:
            source_rows=np.asarray(afv.execute('SELECT id,length FROM fibers ORDER BY id').fetchall())
        rows=np.asarray(db.execute('SELECT id,length_base FROM fibers ORDER BY source_index').fetchall())
        if not np.array_equal(rows[:,0],source_rows[:,0]):raise ValueError('Constraint/AFV IDs differ')
        if not np.allclose(rows[:,1],source_rows[:,1],rtol=1e-10,atol=1e-6):raise ValueError('Constraint/AFV geometry lengths differ')
    return dict(afv_sha256=feature['afv_sha256'],fiber_lengths_checked=len(rows))


def prepare(config):
    start=time.monotonic();database=Path(config['database']);features=Path(config['features']);stage1=Path(config['stage1'])
    signature=dict(version=1,database=sha256_file(database),features=sha256_file(features),stage1=sha256_file(stage1))
    cache=Path(config['tensor_cache']);stamp=cache.with_suffix('.json')
    feature=json.loads(features.read_text())
    if feature['format']!='fiber-axial-priors' or feature['version']!=1:raise ValueError('Wrong axial features')
    verify_feature_source(database,feature)
    if cache.exists() and stamp.exists() and json.loads(stamp.read_text())['signature']==signature:return
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as db:
        meta={k:json.loads(v) for k,v in db.execute('SELECT key,value FROM metadata')}
        if not meta['complete'] or meta['provenance']['afv_uuid']!=feature['afv_uuid']:raise ValueError('Feature/DB identity mismatch')
        prior_summary=json.loads((stage1.parent/'summary.json').read_text())
        if prior_summary['input']['signature']['database_sha256']!=signature['database']:raise ValueError('Stage1/DB mismatch')
        rows=np.asarray(db.execute('SELECT id,length_base FROM fibers ORDER BY source_index').fetchall())
        ids=rows[:,0].astype(np.int64);lengths=rows[:,1]
        if not len(ids) or not np.all(np.diff(ids)>0):raise ValueError('Expected nonempty ascending source IDs')
        with np.load(stage1,allow_pickle=False) as initial:
            if not np.array_equal(ids,initial['ids']):raise ValueError('Stage1 IDs mismatch')
            inherited=initial['broken'].copy()
        if not np.array_equal(ids,feature['source_ids']):raise ValueError('Axial IDs mismatch')
        a=np.asarray(feature['agreement'],np.float32)
        if a.shape!=ids.shape or not np.isfinite(a).all() or np.any(abs(a)>1.00001):raise ValueError('Invalid axial scores')
        cursor=db.execute('SELECT fiber_a,fiber_b,parallel_score,perpendicular_score FROM constraints ORDER BY id')
        chunks=[];original_parallel=original_perp=ties=0
        while batch:=cursor.fetchmany(65536):
            x=np.asarray(batch);relation=np.sign(x[:,2]-x[:,3]);ties+=int((relation==0).sum())
            original_parallel+=int((relation>0).sum());original_perp+=int((relation<0).sum())
            idx=np.searchsorted(ids,x[:,:2].astype(np.int64))
            if np.any(idx>=len(ids)) or not np.array_equal(ids[idx],x[:,:2]):raise ValueError('Bad endpoint IDs')
            keep=~inherited[idx].any(1)&(relation!=0)
            chunks.append(np.column_stack([idx[keep],relation[keep]]))
    edges=np.concatenate(chunks) if chunks else np.empty((0,3))
    active=np.flatnonzero(~inherited)
    if not len(active):raise ValueError('No active fibers remain after stage1')
    remap=np.full(len(ids),-1,np.int64);remap[active]=np.arange(len(active))
    i=remap[edges[:,0].astype(np.int64)];j=remap[edges[:,1].astype(np.int64)]
    degree=np.bincount(np.concatenate([i,j]),minlength=len(active)).astype(np.float32)
    cache.parent.mkdir(parents=True,exist_ok=True)
    temporary=cache.with_suffix('.partial.npz')
    np.savez(temporary,ids=ids,active=active,inherited=inherited,a=a[active],lengths=lengths[active],i=i,j=j,
             relation=edges[:,2].astype(np.float32),degree=degree)
    temporary.replace(cache)
    stats=dict(signature=signature,feature_identity=dict(afv_sha256=feature['afv_sha256'],umbilicus_sha256=feature['umbilicus_sha256']),
               all_fibers=len(ids),inherited_broken=int(inherited.sum()),active_fibers=len(active),active_events=len(i),
               original_parallel=original_parallel,original_perpendicular=original_perp,tied_events_skipped=ties,
               isolated_active_fibers=int((degree==0).sum()),seconds=time.monotonic()-start)
    atomic_write(stamp,stats,durable=False);print('H/V preparation '+json.dumps(stats),flush=True)


def solve(config):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Production H/V optimization requires CUDA')
    torch.set_num_threads(4);torch.manual_seed(config.get('seed',0));torch.use_deterministic_algorithms(True)
    d=np.load(config['tensor_cache']);device='cuda:0'
    a,i,j,r,degree=[torch.from_numpy(d[k]).to(device) for k in ('a','i','j','relation','degree')]
    root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    if (root/'summary.json').exists():raise FileExistsError('Completed H/V output exists')
    initial=torch.where(a==0,torch.full_like(a,.001),2*a)
    stats=json.loads(Path(config['tensor_cache']).with_suffix('.json').read_text())
    def readout(p):
        m=hard_metrics(p,d['a'],d['i'],d['j'],d['relation'],d['lengths'])
        m['total_broken']=stats['inherited_broken']+m['newly_broken'];m['total_broken_percent']=100*m['total_broken']/stats['all_fibers']
        for name in ('parallel','perpendicular'):
            original=stats['original_'+name]
            m[name]['retained_of_pre_stage1_percent']=100*m[name]['retained']/original if original else None
        return m
    baseline=readout(probabilities(initial,torch.full_like(a,-30)).cpu().numpy())
    runs=[];steps=int(config.get('steps',10000));every=int(config.get('log_every',1000))
    for cost_mode in config.get('cost_modes',['uniform','degree_mix']):
      if cost_mode not in ('uniform','degree_mix'):raise ValueError('Unknown cost mode')
      weights=torch.ones_like(a) if cost_mode=='uniform' else .5+.5*degree/degree.mean().clamp_min(1e-12)
      for penalty in config.get('broken_penalties',[.1,1.]):
       for entropy_max in config.get('entropy_weights',[0.,.01]):
        name=f'{cost_mode}_b{penalty:g}_e{entropy_max:g}'
        h=torch.nn.Parameter(initial.clone());b=torch.nn.Parameter(torch.full_like(a,float(np.log(.01/.99))))
        optimizer=torch.optim.Adam([h,b],lr=config.get('lr',.05));history=[]
        torch.cuda.synchronize();started=time.monotonic()
        for step in range(steps+1):
            ramp=min(1.,max(0.,(step/steps-.5)/.3));entropy_weight=entropy_max*ramp
            terms=losses(h,b,a,i,j,r,weights*penalty,config.get('prior_weight',.001),entropy_weight)
            if step%every==0 or step==steps:
                torch.cuda.synchronize()
                record=dict(step=step,seconds=time.monotonic()-started,loss=float(terms[0].detach()),entropy_weight=entropy_weight,
                            **readout(probabilities(h,b).detach().cpu().numpy()))
                if not np.isfinite(record['loss']):raise RuntimeError('Nonfinite H/V loss')
                history.append(record);atomic_write(root/f'history_{name}.json',history,durable=False)
                print('H/V '+name+' '+json.dumps(record),flush=True)
            if step==steps:break
            optimizer.zero_grad(set_to_none=True);terms[0].backward();optimizer.step()
        p=probabilities(h,b).detach().cpu().numpy()
        full=np.zeros((len(d['ids']),3),np.float32);full[:,2]=1;full[d['active']]=p
        labels=np.argmax(full,axis=1).astype(np.uint8)
        np.savez(root/f'solution_{name}.npz',ids=d['ids'],probabilities=full,labels=labels,
                 inherited_broken=d['inherited'],hv_logit=h.detach().cpu().numpy(),broken_logit=b.detach().cpu().numpy(),active=d['active'])
        runs.append(dict(name=name,cost_mode=cost_mode,penalty=penalty,entropy_max=entropy_max,final=history[-1]))
    summary=dict(config=config,input=stats,baseline=baseline,runs=runs,gpu=torch.cuda.get_device_name(),
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(),labels=['H','V','broken'],
                 interpretation='Model relation consistency, not verified family-switch detection; fixed stage1 exclusions')
    atomic_write(root/'summary.json',summary,durable=False)


def run(config):
    prepare(config)
    return launch_tensor_solve(config,__name__,Path(__file__))


def report(config):
    """Audit all saved hard assignments against immutable stage2 inputs."""
    root=Path(config['output_root']);summary=json.loads((root/'summary.json').read_text())
    verified_source=verify_feature_source(summary['config']['database'],json.loads(Path(summary['config']['features']).read_text()))
    with np.load(summary['config']['tensor_cache']) as d:
        rows=[]
        for run in summary['runs']:
            with np.load(root/f"solution_{run['name']}.npz") as s:
                assert np.array_equal(s['ids'],d['ids'])
                assert np.array_equal(s['inherited_broken'],d['inherited'])
                assert np.all(s['labels'][d['inherited']]==2)
                np.testing.assert_allclose(s['probabilities'][d['inherited']],np.tile([0,0,1],(int(d['inherited'].sum()),1)))
                np.testing.assert_allclose(s['probabilities'].sum(1),1,atol=2e-7)
                assert np.array_equal(s['labels'],s['probabilities'].argmax(1))
                label=s['labels'][d['active']];i=d['i'];j=d['j'];r=d['relation']
                kept=(label[i]!=2)&(label[j]!=2)
                violation=((label[i]==label[j])!=(r>0))&kept
                final=run['final'];assert int(violation.sum())==final['all']['violations']
                assert int(kept.sum())==final['all']['retained']
                assert int((label==2).sum())==final['newly_broken']
                rows.append(dict(name=run['name'],**final))
    lines=['# H/V stage2 results','',
        'Whole original fibers; stage1 penalty0.1 exclusions fixed. V=axial, H=perpendicular to the umbilicus axis.',
        'Constraint consistency is not anatomical ground truth or proof of a family-switching fiber.',
        'New broken % uses stage1-retained fibers; total broken % uses all original fibers. Event/arc retention uses stage1-retained input.',
        'Entropy is minimized late; hard readout is argmax(H,V,broken). Shared managed GPU timings are not isolated benchmarks.','',
        '| Cost / penalty / entropy | New broken | Total broken | Retained arc | Retained events | Parallel errors | Perpendicular errors | All errors | Ambiguous | Seconds |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def percent(x):return '—' if x is None else f'{x:.4f}%'
    b=summary['baseline']
    for name,m in [('Axis-only baseline',b)]+[(r['name'],r) for r in rows]:
        vals=[m['newly_broken_percent'],m['total_broken_percent'],m['retained_arc_percent'],m['all']['retained_percent'],
              m['parallel']['violation_percent'],m['perpendicular']['violation_percent'],m['all']['violation_percent'],m['ambiguous_percent']]
        lines.append('|'+name+'|'+'|'.join(map(percent,vals))+'|'+(f"{m['seconds']:.2f}" if 'seconds' in m else '—')+'|')
    lines+=['',f"Inherited broken: {summary['input']['inherited_broken']:,}; active fibers: {summary['input']['active_fibers']:,}; active events: {summary['input']['active_events']:,}.",
            f"GPU: {summary['gpu']}; peak tensor allocation: {summary['peak_allocated_bytes']/1024**2:.1f}MiB.",
            '', 'Saved ternary arrays were audited against input IDs, frozen exclusions and independently recomputed retained/violated events.']
    destination=Path(config['report']);destination.write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)
    return dict(report=str(destination),verified_runs=len(rows),verified_source=verified_source)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--unmanaged',action='store_true');parser.add_argument('--devices')
    args=parser.parse_args()
    if not args.unmanaged:raise RuntimeError('Use approved managed runner')
    solve(json.loads(args.config.read_text()))
