"""GPU calibration of causal embedding integration, using shared fixed-route data."""
import json
import math
from pathlib import Path
import subprocess
import time

from lasagna.inference_provenance import atomic_write, json_digest, sha256_file
from .embedding_fixed_path import prepare_fit_data, pair_loss, pair_metrics, ema_distance_sequence


def integration_flags(family, spacing, values, window=512):
    weight=values[0];half=values[1] if family!='window' else 512
    args=['--embedding-reference-mode','ema','--embedding-weight',str(weight),
          '--embedding-ema-half-life-base-voxels',str(half),
          '--embedding-integration',family,'--embedding-update-spacing-base-voxels',str(spacing)]
    extras={}
    if family=='window': extras={'integration-window-base-voxels':window}
    if family=='gate': extras=dict(zip(['update-gate-bias','update-gate-presence','update-gate-distance'],values[2:]))
    if family=='angular': extras={'update-cap-degrees-per-8-base-voxels':values[2]}
    if family=='dual': extras={'update-long-half-life-base-voxels':half+values[2],'update-mix-bias':values[3],'update-mix-slope':values[4]}
    for flag,value in extras.items(): args+=['--embedding-'+flag,str(value)]
    return args


def integrate(embedding,distance,presence,params,family,spacing,mask=None,initial=None,collect=True):
    """Candidate-major differentiable states; physical cadence clocks are FP64."""
    import torch
    k=params.shape[0];batch=embedding.shape[0]
    if initial is None:
        fast=embedding[:,0].unsqueeze(0).expand(k,-1,-1)
        slow=fast
        elapsed=torch.zeros(k,batch,device=embedding.device,dtype=torch.float64)
    else:fast,slow,elapsed=initial
    half=params[:,1,None,None]
    outputs=[]
    def normalize(value,fallback):
        norm2=value.square().sum(-1,keepdim=True)
        return torch.where(norm2>0,value/norm2.clamp_min(1e-30).sqrt(),fallback)
    for i in range(1,embedding.shape[1]):
        value=embedding[None,:,i]
        disagreement=1-(fast*value).sum(-1).clamp(-1,1)
        if collect:
            if family=='dual':
                slow_dist=1-(slow*value).sum(-1).clamp(-1,1)
                mix=torch.sigmoid(params[:,3,None]+params[:,4,None]*(slow_dist-disagreement))
                outputs.append(mix*disagreement+(1-mix)*slow_dist)
            else:outputs.append(disagreement)
        valid=torch.ones(batch,device=embedding.device,dtype=torch.bool) if mask is None else mask[:,i]
        elapsed=elapsed+torch.where(valid,distance[:,i].double(),0)[None]
        admit=(elapsed>=spacing[:,None])&valid[None]
        decay=torch.exp2(-elapsed[:,:,None]/half.double()).to(embedding.dtype)
        if family=='gate':
            gate=torch.sigmoid(params[:,2,None]+params[:,3,None]*(presence[None,:,i]-.5)-params[:,4,None]*disagreement)
            decay=1-(1-decay)*gate[:,:,None]
        proposed=normalize(decay*fast+(1-decay)*value,fast)
        if family=='angular':
            dot=(proposed*fast).sum(-1)
            angle=torch.acos(dot.clamp(-1+1e-7,1-1e-7))
            cap=params[:,2,None]*(math.pi/180)*elapsed.to(params.dtype)/8
            limited=torch.minimum(angle,cap)
            a=torch.sin(angle-limited)/torch.sin(angle).clamp_min(1e-8)
            b=torch.sin(limited)/torch.sin(angle).clamp_min(1e-8)
            capped=normalize(a[:,:,None]*fast+b[:,:,None]*proposed,fast)
            proposed=torch.where((angle>cap)[:,:,None],capped,proposed)
            proposed=torch.where((dot<=-1)[:,:,None],fast,proposed)
        if family=='dual':
            long_half=half+params[:,2,None,None]
            long_decay=torch.exp2(-elapsed[:,:,None]/long_half.double()).to(embedding.dtype)
            updated=normalize(long_decay*slow+(1-long_decay)*value,slow)
            slow=torch.where(admit[:,:,None],updated,slow)
        fast=torch.where(admit[:,:,None],proposed,fast)
        elapsed=torch.where(admit,0.,elapsed)
    return (torch.stack(outputs,dim=-1) if collect else None),(fast,slow,elapsed)


def window_distances(embedding,distance,mask,spacings,ranges,n):
    """Uniform finite-history cosine mean, with exact physical admission/expiry."""
    import torch
    k=len(spacings);batch,t,channels=embedding.shape
    arcs=distance.double().cumsum(-1)
    elapsed=torch.zeros(k,batch,device=embedding.device,dtype=torch.float64)
    admitted=torch.zeros(k,batch,t,device=embedding.device,dtype=torch.bool)
    admitted[:,:,0]=True
    for i in range(1,t):
        elapsed=elapsed+torch.where(mask[:,i],distance[:,i].double(),0)[None]
        valid=(elapsed>=spacings[:,None])&mask[None,:,i]
        admitted[:,:,i]=valid
        elapsed=torch.where(valid,0.,elapsed)
    result=[]
    for i in range(t-n,t):
        include=admitted[:,:,:i]&((arcs[:,i:i+1]-arcs[:,:i])[None]<=ranges[:,None,None])
        count=include.sum(-1)
        # Match per-reference clamping rather than assuming a normalized mean.
        dots=(embedding[:,:i]*embedding[:,i:i+1]).sum(-1).clamp(-1,1)
        result.append(torch.where(count>0,((1-dots)[None]*include).sum(-1)/count.clamp_min(1),0))
    return torch.stack(result,dim=-1)


def matrix():
    ema=[dict(spacing=s,start=[.045401283,h]) for s in (8,16,32,64) for h in (64,128,256,512,1024)]
    window=[dict(spacing=s,window=w,start=[.045401283]) for s in (8,16,32,64) for w in (256,512,1024,2048)]
    return dict(ema=ema,window=window,
        gate=[dict(spacing=8,start=[.045401283,h,3,2,2]) for h in (64,256,1024)],
        angular=[dict(spacing=8,start=[.045401283,h,cap]) for h,cap in ((64,1),(256,5),(1024,15))],
        dual=[dict(spacing=8,start=[.045401283,h,gap,0,2]) for h,gap in ((64,512),(128,1024),(256,2048))])


def calibrate(settings):
    import torch
    data=prepare_fit_data(settings)
    if any('presence' not in s for c,w in data.rows for s in w['good']):
        raise ValueError('integration calibration requires native presence export')
    output=Path(settings['integration_output']);output.mkdir(parents=True,exist_ok=True)
    def t(x):return data.tensor(x)
    with torch.no_grad():
        _,seed=ema_distance_sequence(data.we,data.wd,t(512),data.wm)
        distances,_=ema_distance_sequence(data.be,data.bd,t(512),initial=seed.repeat_interleave(2,dim=0))
        baseline=(data.bc+.25*distances).reshape(len(data.rows),2,data.n)
        parity=float((baseline.reshape(-1,data.n)-data.native).abs().max())
        if parity>2e-5:raise ValueError(f'baseline native parity failed:{parity}')
        baseline_gap=(baseline[:,0]-baseline[:,1]).mean(-1)
    def metrics(cost):return pair_metrics(cost,data.splits,data.failure,data.weights,baseline_gap)
    bounds={
        'ema':([.001,32],[2,2048]),'window':([.001],[2]),
        'gate':([.001,32,-8,-8,0],[2,2048,8,8,12]),
        'angular':([.001,32,.01],[2,2048,90]),
        'dual':([.001,32,32,-8,0],[2,2048,4096,8,12])}
    evidence=dict(dataset_sha256=sha256_file(Path(settings['output'])/'dataset.json'),
        implementation_sha256=sha256_file(Path(__file__)),
        shared_fit_sha256=sha256_file(Path(__file__).with_name('embedding_fixed_path.py')),
        iterations=settings.get('integration_iterations',150),matrix=matrix())
    key=json_digest(evidence)
    report=dict(evidence=evidence,baseline=metrics(baseline),baseline_native_parity=parity,candidates=[],winners={})
    # Explicit previous fitted control on this longer/common-prefix dataset.
    with torch.no_grad():
        _,seed=ema_distance_sequence(data.we,data.wd,t(510.432189941),data.wm)
        d,_=ema_distance_sequence(data.be,data.bd,t(510.432189941),initial=seed.repeat_interleave(2,dim=0))
        report['previous_linear']=metrics((data.bc+.0454012826085*d).reshape(len(data.rows),2,data.n))
    for family,variants in matrix().items():
        checkpoint=output/(family+'.json')
        if checkpoint.exists():
            cached=json.loads(checkpoint.read_text())
            if cached.get('key')==key:
                report['candidates']+=cached['candidates'];report['winners'][family]=cached['winner'];continue
        started=time.monotonic()
        lower,upper=map(t,bounds[family]);starts=t([v['start'] for v in variants]);k=len(variants)
        raw=torch.nn.Parameter(torch.logit((starts-lower)/(upper-lower)))
        optimizer=torch.optim.Adam([raw],lr=.03)
        spacing=t([v['spacing'] for v in variants]).double()
        if family=='window':
            ranges=t([v['window'] for v in variants]).double()
            e=torch.cat([data.we.repeat_interleave(2,dim=0),data.be[:,1:]],dim=1)
            d=torch.cat([data.wd.repeat_interleave(2,dim=0),data.bd[:,1:]],dim=1)
            mask=torch.cat([data.wm.repeat_interleave(2,dim=0),torch.ones_like(data.bd[:,1:],dtype=torch.bool)],dim=1)
            with torch.no_grad():fixed=window_distances(e,d,mask,spacing,ranges,data.n)
        def scores(params):
            if family=='window':dist=fixed
            else:
                with torch.no_grad():
                    _,state=integrate(data.we,data.wd,data.wp,params,family,spacing,data.wm,collect=False)
                    state=tuple(s.repeat_interleave(2,dim=1).detach() for s in state)
                dist,_=integrate(data.be,data.bd,data.bp,params,family,spacing,initial=state)
            return (data.bc[None]+params[:,0,None,None]*dist).reshape(k,len(data.rows),2,data.n)
        best=[None]*k;history=[]
        for iteration in range(settings.get('integration_iterations',150)+1):
            params=lower+(upper-lower)*raw.sigmoid()
            cost=scores(params)
            fitloss=torch.stack([pair_loss(c,data.splits['fit'],data.failure,data.weights,baseline_gap) for c in cost])
            val=torch.stack([pair_loss(c,data.splits['selection'],data.failure,data.weights,baseline_gap) for c in cost]).detach().cpu().tolist()
            values=params.detach().cpu().tolist()
            for j in range(k):
                if best[j] is None or val[j]<best[j]['selection_loss']:
                    best[j]=dict(iteration=iteration,selection_loss=val[j],parameters=values[j])
            if iteration%25==0:
                row=dict(iteration=iteration,fit_loss=fitloss.detach().cpu().tolist(),selection_loss=val)
                history.append(row)
                print(f'integration family={family} step={iteration} best_selection={min(val):.6f} elapsed={time.monotonic()-started:.1f}s',flush=True)
            if iteration==settings.get('integration_iterations',150):break
            regularizer=.005*(params[:,0]/.25).log().square()
            if family!='window':regularizer=regularizer+.005*(params[:,1]/512).log().square()
            optimizer.zero_grad();(fitloss+regularizer).sum().backward()
            if not torch.isfinite(raw.grad).all():raise ValueError('nonfinite integration gradient')
            optimizer.step()
        with torch.no_grad():selected=scores(t([b['parameters'] for b in best]))
        candidates=[dict(family=family,index=j,variant=v,best=best[j],metrics=metrics(selected[j]),
                         trace_args=integration_flags(family,v['spacing'],best[j]['parameters'],v.get('window',512))) for j,v in enumerate(variants)]
        winner=min(candidates,key=lambda c:c['best']['selection_loss'])
        saved=dict(key=key,candidates=candidates,winner=winner,history=history,seconds=time.monotonic()-started)
        atomic_write(checkpoint,saved)
        report['candidates']+=candidates;report['winners'][family]=winner
        atomic_write(output/'matrix.json',report)
    atomic_write(output/'matrix.json',report)
    print(json.dumps(report['winners'],indent=2),flush=True)
    return report


def audit(settings):
    """Replay real exported vectors through both native and Torch integration."""
    import torch
    data=prepare_fit_data(settings)
    output=Path(settings['integration_output'])
    report=json.loads((output/'matrix.json').read_text())
    # Include every candidate setting, not just family winners.
    requests=[];expected=[]
    cases=[w for c,w in data.rows if w['kind']=='failure'][:3]
    for candidate in report['candidates']:
        family=candidate['family'];v=candidate['variant'];p=candidate['best']['parameters']
        for w in cases:
            for branch in ('good','bad'):
                seq=w[branch];e=data.tensor([[s['embedding'] for s in seq]])
                d=data.tensor([[s['distance'] for s in seq]]);presence=data.tensor([[s['presence'] for s in seq]])
                with torch.no_grad():
                    if family=='window':
                        costs=window_distances(e,d,torch.ones_like(d,dtype=torch.bool),data.tensor([v['spacing']]).double(),data.tensor([v['window']]).double(),len(seq)-1)
                    else:costs,_=integrate(e,d,presence,data.tensor([p]),family,data.tensor([v['spacing']]).double())
                expected.append((costs[0,0]*p[0]).cpu().tolist())
                requests.append(dict(trace_args=candidate['trace_args'],samples=seq))
    request=output/'parity_queries.json';atomic_write(request,requests)
    observed=json.loads(subprocess.check_output([settings['native_binary'],'integrate-embeddings',str(request)],text=True))
    import numpy as np
    errors=[float(np.max(np.abs(np.asarray(a)-np.asarray(b[1:])))) for a,b in zip(expected,observed)]
    if len(expected)!=len(observed) or max(errors)>2e-5:raise ValueError(f'native integration parity failed:{max(errors)}')
    result=dict(paths=len(errors),max_abs=max(errors),errors=errors,native_sha256=sha256_file(Path(settings['native_binary'])),
                library_sha256=sha256_file(Path(settings['native_binary']).parent.parent/'core/libvc_fiber_tracer.so'))
    atomic_write(output/'parity.json',result);print(json.dumps({k:v for k,v in result.items() if k!='errors'}),flush=True)
    return result
