"""Offline frozen-route EMA calibration; native sampling and GPU-only fitting."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from lasagna.inference_provenance import atomic_write, json_digest, sha256_file
from .embedding_route_analysis import arc_lengths, line_point


def fiber_split(cases, seed=20260915):
    """Content-identity grouping precedes mining; all directions/spans stay together."""
    by_scroll = defaultdict(set)
    identities = {}
    for c in cases:
        p = c['provenance']
        identity = sha256_file(Path(p['fiber_path']))
        identities[p['fiber_path']] = identity
        by_scroll[p['coordinate_space'].split('/')[0]].add(identity)
    assignment = {}
    for scroll, ids in sorted(by_scroll.items()):
        ids = sorted(ids, key=lambda s: hashlib.sha256(f'{seed}:{s}'.encode()).hexdigest())
        # Small scrolls cannot support three independent groups: reserve a test
        # fiber, and use remaining fibers for fitting. Selection uses larger ones.
        ntest = max(1, math.ceil(len(ids)*.2))
        nselect = math.ceil(len(ids)*.2) if len(ids) >= 5 else 0
        for i, identity in enumerate(ids):
            assignment[identity] = 'test' if i < ntest else 'selection' if i < ntest+nselect else 'fit'
    return {path: dict(identity=identity, split=assignment[identity]) for path,identity in identities.items()}


def resample(points, count, step):
    arcs = arc_lengths(points)
    if arcs[-1] < (count-1)*step:
        raise ValueError('insufficient_path_length')
    return np.stack([line_point(points, arcs, i*step) for i in range(count)])


def make_windows(case, settings):
    """GT connector is a Hermite displacement correction, not a jump onto GT."""
    gt = np.asarray(case['reference_points_base_xyz'], dtype=float)
    ga = arc_lengths(gt)
    step = float(case['provenance']['trace_to_base'])*float(case['provenance']['effective_trace']['step_voxels'])
    n = round(settings.get('window_base',256)/step)
    warm = round(settings.get('warmup_base',1024)/step)
    rejected = Counter()
    windows = []
    failures = defaultdict(list)
    for f in case['replay']['failures']:
        if f['segment_point_index'] is not None:
            failures[f['segment_index']].append(f['segment_point_index'])
    for si, seg in enumerate(case['replay']['segments']):
        path = np.asarray(seg['trace_points_base_xyz'], dtype=float)
        if len(path) < n+3:
            rejected['short_segment'] += 1
            continue
        matches = {m['trace_point_index']:m for m in seg['matches']}
        proposals = []
        for end in failures.get(si,[]):
            # Locate the last well-aligned point before this departure, then
            # include a short pre-departure portion. Never cross a restart.
            good = [i for i in range(max(1,end-4*n),end) if i in matches and matches[i]['euclidean_error_base_voxels'] <= 8]
            if not good:
                rejected['no_shared_near_reference_start'] += 1
                continue
            start = max(1, good[-1]-4)
            if start+n >= len(path):
                start = max(1,len(path)-n-2)
            proposals.append((start,'failure'))
        # Successful controls are not given a GT-better ranking label.
        control_start=round(settings.get('control_start_base',1024)/step)
        if not failures.get(si) and len(path) > n+control_start+2:
            proposals.append((min(control_start,len(path)-n-2),'control'))
        for start,kind in proposals[:settings.get('max_windows_per_segment',2)]:
            if start not in matches:
                rejected['missing_match'] += 1
                continue
            arc = matches[start]['matched_reference_arc_base']
            if arc+(n+4)*step > ga[-1]:
                rejected['reference_end'] += 1
                continue
            begin = max(0,start-warm)
            prefix = path[begin:start+1]
            try:
                bad = resample(path[start:],n+1,step)
                # Preserve starting position AND tangent; correction decays to
                # zero by 64 base voxels, then follow the reference.
                t = np.arange(n+5)*step
                curve = np.stack([line_point(gt,ga,arc+x) for x in t])
                tangent = (path[start]-path[start-1]); tangent /= np.linalg.norm(tangent)
                gt_tangent = (curve[1]-curve[0])/step
                u = np.minimum(t/64.,1.)[:,None]
                delta = path[start]-curve[0]
                curve += (2*u**3-3*u**2+1)*delta + (u**3-2*u**2+u)*64*(tangent-gt_tangent)
                good = resample(curve,n+1,step)
                good[0] = bad[0] = path[start]
                # Equal step count, approximately identical Euclidean distances.
                # Reject tight curvature rather than exploiting shortened steps.
                if min(np.linalg.norm(np.diff(good,axis=0),axis=1)) < .98*step:
                    raise ValueError('connector_curvature')
                windows.append(dict(segment=si,start_index=start,kind=kind,score_start=len(prefix),
                    good=np.concatenate([prefix,good[1:]]).tolist(),
                    bad=np.concatenate([prefix,bad[1:]]).tolist(),
                    connector_offset_base=float(np.linalg.norm(delta)), step_base=step))
            except ValueError as e:
                rejected[str(e)] += 1
    return windows, dict(rejected)


def extract(settings):
    root, output = Path(settings['evaluation']), Path(settings['output'])
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((root/'summary.json').read_text())
    if not summary['completed']:
        raise ValueError('completed source evaluation required')
    paths = sorted((root/'cases').glob('*/trace.json'),key=lambda p:int(p.parent.name))
    cases = [json.loads(p.read_text()) for p in paths]
    split = fiber_split(cases)
    atomic_write(output/'split.json',split)
    binary = settings['native_binary']
    binary_sha = sha256_file(Path(binary))
    lib = Path(binary).parent.parent/'core/libvc_fiber_tracer.so'
    library_sha = sha256_file(lib) if lib.exists() else None

    def one(item):
        path,c = item
        idx = c['provenance']['case_index']
        target = output/f'case_{idx}.json'
        manifest = Path(c['provenance']['fiber_manifest'])
        key = json_digest(dict(trace_sha=sha256_file(path),settings=settings,binary=binary_sha,extractor=sha256_file(Path(__file__)),
                               library=library_sha,manifest=sha256_file(manifest)))
        if target.exists():
            saved = json.loads(target.read_text())
            if saved['key'] == key:
                return saved
        windows,rejected = make_windows(c,settings)
        query = dict(fiber_manifest=str(manifest),normal_manifest=c['provenance']['normal_manifest'],
                     trace_to_base_scale=c['provenance']['trace_to_base'],
                     trace_args=summary['config']['trace_args'],
                     paths=[w[k] for w in windows for k in ('good','bad')])
        query_path = output/f'query_{idx}.json'
        atomic_write(query_path,query)
        sampled = []
        if windows:
            with (output/f'native_{idx}.log').open('w') as stderr:
                result = subprocess.check_output([binary,'score-fixed-paths',str(query_path)],stderr=stderr,text=True,timeout=600)
            sampled = json.loads(result)
            if len(sampled) != 2*len(windows):
                raise ValueError('native path count mismatch')
        accepted = []
        for wi,w in enumerate(windows):
            pair = sampled[2*wi:2*wi+2]
            if any(p['status'] != 'ok' for p in pair):
                rejected['native_unavailable'] = rejected.get('native_unavailable',0)+1
                continue
            first = w['score_start']
            # Rectangular cone grid extends beyond the nominal 25 degree cone.
            limit = math.degrees(math.atan(math.sqrt(2)*math.tan(math.radians(25))))+.1
            if any(max(s['prediction_angle'] for s in p['samples'][first:]) > limit for p in pair):
                rejected['outside_prediction_cone'] = rejected.get('outside_prediction_cone',0)+1
                continue
            row = {k:v for k,v in w.items() if k not in ('good','bad')}
            row.update(good=pair[0]['samples'],bad=pair[1]['samples'])
            accepted.append(row)
        saved = dict(key=key,case=idx,fiber=c['provenance']['fiber_path'],
                     scroll=c['provenance']['coordinate_space'].split('/')[0],
                     **split[c['provenance']['fiber_path']],windows=accepted,rejected=rejected,
                     requested=len(windows),native_reasons=[r.get('reason') for r in sampled if r['status']!='ok'])
        atomic_write(target,saved)
        print(f"extract case={idx} accepted={len(accepted)}/{len(windows)} rejected={rejected}",flush=True)
        return saved
    with ThreadPoolExecutor(max_workers=settings.get('workers',8)) as pool:
        exported = list(pool.map(one,zip(paths,cases)))
    atomic_write(output/'dataset.json',dict(settings=settings,split=split,cases=exported,
        source_summary_sha256=sha256_file(root/'summary.json'),native_binary_sha256=binary_sha,
        native_library_sha256=library_sha))
    counts = Counter((c['split'],w['kind']) for c in exported for w in c['windows'])
    print('dataset '+str(counts),flush=True)
    return exported


def ema_distance_sequence(embedding, distance, half_life, mask=None, initial=None):
    """Differentiable parent-before-update normalized EMA, native cancellation rule."""
    import torch
    state = embedding[:,0] if initial is None else initial
    costs = []
    for i in range(1,embedding.shape[1]):
        value = embedding[:,i]
        costs.append(1-(state*value).sum(-1).clamp(-1,1))
        decay = torch.exp2(-distance[:,i:i+1]/half_life)
        mixed = decay*state+(1-decay)*value
        norm2 = mixed.square().sum(-1,keepdim=True)
        norm = norm2.clamp_min(1e-30).sqrt()
        updated = torch.where(norm2>0,mixed/norm,state)
        state = updated if mask is None else torch.where(mask[:,i:i+1],updated,state)
    return torch.stack(costs,dim=1),state


def mapped_distance(distance, threshold, temperature):
    import torch.nn.functional as F
    return temperature*(F.softplus((distance-threshold)/temperature)-F.softplus(-threshold/temperature))


def prepare_fit_data(settings):
    import torch
    import torch.nn.functional as F
    if not torch.cuda.is_available():
        raise RuntimeError('real fitting requires a manager-granted CUDA GPU; no CPU fallback')
    torch.manual_seed(20260915)
    torch.set_num_threads(4)
    output = Path(settings['output'])
    dataset = json.loads((output/'dataset.json').read_text())
    rows = [(c,w) for c in dataset['cases'] for w in c['windows']]
    if not rows:
        raise ValueError('no usable windows')
    device = torch.device('cuda:0')
    maxwarm = max(w['score_start'] for c,w in rows)
    n = len(rows[0][1]['good'])-rows[0][1]['score_start']
    channels = len(rows[0][1]['good'][0]['embedding'])
    warm_e = np.zeros((len(rows),maxwarm,channels),np.float32)
    warm_d = np.zeros((len(rows),maxwarm),np.float32)
    warm_m = np.zeros((len(rows),maxwarm),bool)
    branch_e,branch_d,branch_p,base,native = [],[],[],[],[]
    warm_p = np.zeros((len(rows),maxwarm),np.float32)
    for j,(c,w) in enumerate(rows):
        k = w['score_start']
        seq = w['good'][:k]
        warm_e[j,:k] = [s['embedding'] for s in seq]
        warm_e[j,k:] = warm_e[j,k-1]
        warm_d[j,:k] = [s['distance'] for s in seq]
        warm_p[j,:k] = [s.get('presence',1.) for s in seq]
        warm_p[j,k:] = warm_p[j,k-1]
        warm_m[j,:k] = True
        branch_e.extend([[s['embedding'] for s in w[b][k-1:]] for b in ('good','bad')])
        branch_d.extend([[s['distance'] for s in w[b][k-1:]] for b in ('good','bad')])
        branch_p.extend([[s.get('presence',1.) for s in w[b][k-1:]] for b in ('good','bad')])
        base.extend([[s['base_cost'] for s in w[b][k:]] for b in ('good','bad')])
        native.extend([[s['total_cost'] for s in w[b][k:]] for b in ('good','bad')])
    def tensor(x,dtype=torch.float32): return torch.as_tensor(np.asarray(x),device=device,dtype=dtype)
    we,wd,wm,be,bd,bc = tensor(warm_e),tensor(warm_d),tensor(warm_m,torch.bool),tensor(branch_e),tensor(branch_d),tensor(base)
    splits = {s:tensor([c['split']==s for c,w in rows],torch.bool) for s in ('fit','selection','test')}
    failure = tensor([w['kind']=='failure' for c,w in rows],torch.bool)
    if any(not bool((splits[s]&failure).any()) for s in splits):
        raise ValueError('each split needs covered failure windows; inspect exclusions')
    # Equal total influence per fiber, not per number of failures/windows.
    counts = Counter(c['identity'] for c,w in rows)
    weights = tensor([1/counts[c['identity']] for c,w in rows])

    from types import SimpleNamespace
    return SimpleNamespace(output=output, dataset=dataset, rows=rows, n=n, we=we, wd=wd, wm=wm,
        be=be, bd=bd, bc=bc, wp=tensor(warm_p), bp=tensor(branch_p), native=tensor(native), splits=splits, failure=failure, weights=weights, tensor=tensor)


def pair_loss(cost, split, failure, weights, baseline_gap):
    import torch.nn.functional as F
    gap = (cost[:,0]-cost[:,1]).mean(-1)
    f, ctrl = split&failure, split&~failure
    rank = (F.softplus((gap[f]+.005)/.05)*weights[f]).sum()/weights[f].sum()
    control = ((gap[ctrl]-baseline_gap[ctrl]).square()*weights[ctrl]).sum()/weights[ctrl].sum().clamp_min(1e-8)
    return rank+10*control


def pair_metrics(cost, splits, failure, weights, baseline_gap):
    gap = (cost[:,0]-cost[:,1]).mean(-1)
    report = {}
    for s,mask in splits.items():
        f=mask&failure
        report[s]=dict(windows=int(mask.sum()),failure_windows=int(f.sum()),
            ranking_loss=float(pair_loss(cost,mask,failure,weights,baseline_gap)),
            gt_preferred_percent=float((gap[f]<0).float().mean()*100),mean_gap=float(gap[f].mean()),
            short_two_step_gt_preferred_percent=float(((cost[f,0,:2]-cost[f,1,:2]).mean(-1)<0).float().mean()*100))
    return report


def fit(settings):
    import torch
    data = prepare_fit_data(settings)
    output,rows,n,tensor = data.output,data.rows,data.n,data.tensor
    we,wd,wm,be,bd,bc = data.we,data.wd,data.wm,data.be,data.bd,data.bc
    splits,failure,weights = data.splits,data.failure,data.weights
    def score(params,smooth=False):
        weight,half,threshold,temp = params
        with torch.no_grad():
            _,state = ema_distance_sequence(we,wd,half,wm)
        distances,_ = ema_distance_sequence(be,bd,half,initial=state.repeat_interleave(2,dim=0).detach())
        penalty = mapped_distance(distances,threshold,temp) if smooth else distances
        return (bc+weight*penalty).reshape(len(rows),2,n),distances
    initial = [tensor(x) for x in (.25,512.,0.,.1)]
    baseline,_ = score(initial)
    parity = float((baseline.reshape(-1,n)-data.native).abs().max())
    if parity > 2e-5:
        raise ValueError(f'native EMA score parity failed: {parity}')
    baseline_gap = (baseline[:,0]-baseline[:,1]).mean(-1).detach()

    def losses(cost,split):
        return pair_loss(cost,split,failure,weights,baseline_gap)
    def metrics(cost):
        return pair_metrics(cost,splits,failure,weights,baseline_gap)
    report = dict(dataset_sha256=sha256_file(output/'dataset.json'),native_parity_max_abs=parity,
                  baseline=metrics(baseline),fits=[],notes='Calibration-heldout, not pristine test; historical cohort already used for tuning. Detached finite warm-up; fixed routes are not beam candidates.')
    atomic_write(output/'fit_report.json',report)
    if settings.get('evaluate_parameters_from'):
        learned = json.loads(Path(settings['evaluate_parameters_from']).read_text())
        for previous in learned['fits']:
            selected,_ = score([tensor(p) for p in previous['best']['parameters']],previous['smooth'])
            report['fits'].append(dict(smooth=previous['smooth'],best=previous['best'],
                metrics=metrics(selected.detach()),fitted_on=settings['evaluate_parameters_from']))
        atomic_write(output/'fit_report.json',report)
        print(json.dumps(report,indent=2),flush=True)
        return report
    for smooth in (False,True):
        # Bounds prevent huge embedding penalties or an effectively frozen EMA.
        lower = tensor([.001,32.,0.,.01]); upper = tensor([2.,2048.,1.,.5])
        start = tensor([.25,512.,.05,.1])
        raw = torch.nn.Parameter(torch.logit((start-lower)/(upper-lower)))
        optimizer = torch.optim.Adam([raw],lr=settings.get('learning_rate',.03))
        best = None
        history = []
        for it in range(settings.get('iterations',300)+1):
            params = lower+(upper-lower)*raw.sigmoid()
            cost,_ = score(params,smooth)
            fitloss = losses(cost,splits['fit'])
            reg = .005*((params[0]/.25).log().square()+(params[1]/512).log().square())
            loss = fitloss+reg
            # Selection subset only; final test metrics not used for selection.
            val = float(losses(cost,splits['selection']).detach())
            if best is None or val < best['selection_loss']:
                best = dict(iteration=it,selection_loss=val,parameters=[float(p) for p in params.detach()])
            if it%25==0:
                row=dict(iteration=it,fit_loss=float(fitloss.detach()),selection_loss=val,parameters=[float(p) for p in params.detach()])
                history.append(row)
                print(f"fit smooth={smooth} "+json.dumps(row),flush=True)
            if it==settings.get('iterations',300): break
            optimizer.zero_grad();loss.backward();optimizer.step()
        selected,_ = score([tensor(p) for p in best['parameters']],smooth)
        report['fits'].append(dict(smooth=smooth,best=best,metrics=metrics(selected.detach()),history=history))
        atomic_write(output/'fit_report.json',report)
    print(json.dumps(report,indent=2),flush=True)
    return report


def report(settings):
    """Saved-data-only comparison, including fit/selection/heldout trace metrics."""
    from .replay_metrics import complete_report, split_report
    output=Path(settings['output'])
    data=json.loads((output/'dataset.json').read_text())
    fitted=json.loads((output/'fit_report.json').read_text())
    base=Path(settings['evaluation']).parents[1]
    runs=[('Classic','classic_thickness40_grace160_20260914_v2'),
          ('u11800 embedding off','eligible_ema999_final_vis_20260914_v1'),
          ('u11800 + u42000 EMA baseline','mixed_current_ema_only_20260915_v1'),
          ('Fitted linear EMA','fixed_path_ema_linear_20260915_v1'),
          ('Fitted smooth EMA','fixed_path_ema_smooth_20260915_v1')]
    result=dict(fit=fitted,rows=[],pending=[],split=data['split'])
    for label,name in runs:
        path=base/name/'full/summary.json'
        if not path.exists():result['pending'].append(label);continue
        summary=json.loads(path.read_text())
        if not summary.get('completed'):result['pending'].append(label);continue
        metrics=complete_report(summary)
        grouped=split_report(summary,data['split'])
        result['rows'].append(dict(name=label,path=str(path),summary_sha256=sha256_file(path),
                                  metrics=metrics,splits=grouped))
    exclusions=Counter()
    distances=[]
    counts=Counter()
    for c in data['cases']:
        exclusions.update(c['rejected'])
        for w in c['windows']:
            counts[(c['scroll'],c['split'],w['kind'])]+=1
            k=w['score_start']
            distances.append([sum(s['distance'] for s in w[b][k:]) for b in ('good','bad')])
    result['exclusions']=dict(exclusions)
    result['window_counts']=[dict(scroll=k[0],split=k[1],kind=k[2],count=v) for k,v in sorted(counts.items())]
    dist=np.asarray(distances)
    result['branch_distance_base']={b:dict(min=float(dist[:,i].min()),median=float(np.median(dist[:,i])),max=float(dist[:,i].max())) for i,b in enumerate(('good','bad'))}
    long=base/'fixed_path_ema_fit_long_20260915_v1/fit_report.json'
    result['long_window']=json.loads(long.read_text()) if long.exists() else None
    result['implementation_sha256']={str(p):sha256_file(p) for p in [Path(settings['native_binary']),
        Path(settings['native_binary']).parent.parent/'core/libvc_fiber_tracer.so',Path(__file__)]}
    atomic_write(output/'comparison.json',result)
    lines=['# Fixed-path EMA calibration — 2026-09-15','',
        'Frozen u11800 presence/direction + paired current u42000 embeddings. No anchors or neural-model training.',
        'Distances in mm; means = combined directed reference distance / (counted restarts + 1).',
        'Whole-fiber split:30 fit /9 model-selection /11 calibration-heldout. This cohort was used historically for tuning; heldout is not pristine unseen test data.','']
    def table(title,rows):
        lines.extend([f'## {title}','','| Model | Restarts | Mean mm | Median mm | P05 mm | P95 mm |',
            '|---|---:|---:|---:|---:|---:|'])
        for label,m in rows:
            lines.append(f"| {label} | {m['failures']} | {m['mean_segment_length_mm']:.3f} | {m['median_segment_length_mm']:.3f} | {m['p05_segment_length_mm']:.3f} | {m['p95_segment_length_mm']:.3f} |")
        lines.append('')
    table('Full cohort',[(r['name'],r['metrics']['global_result']) for r in result['rows']])
    for split in ('fit','selection','test'):
        table('Trace results: '+split,[(r['name'],r['splits'][split]) for r in result['rows']])
    for scroll in sorted(result['rows'][0]['metrics']['scrolls']):
        table(scroll,[(r['name'],r['metrics']['scrolls'][scroll]) for r in result['rows']])
    lines.extend(['## Fitted parameters','',
        '| Mapping | Weight | Half-life base voxels | Threshold | Temperature | Selection iteration |',
        '|---|---:|---:|---:|---:|---:|','| Baseline linear | 0.25 | 512 | — | — | — |'])
    for f in fitted['fits']:
        p=f['best']['parameters'];smooth=f['smooth']
        lines.append(f"| {'Smooth' if smooth else 'Linear'} | {p[0]:.9f} | {p[1]:.6f} | {p[2] if smooth else '—'} | {p[3] if smooth else '—'} | {f['best']['iteration']} |")
    lines.extend(['','## Fixed-path objective','',
        'GPU Adam,300steps/family,LR0.03. Learn bounded weight[.001,2], half-life[32,2048], optional threshold[0,1] and temperature[.01,.5].',
        'Mean per-step native base loss + embedding penalty, pairwise softplus ranking (margin.005,temperature.05); successful controls preserve baseline relative cost. Log-weight/half-life regularizer. Inverse-total-window-per-fiber weighting, separately normalized failure/control losses.',
        'Smooth penalty = weight*temperature*[softplus((distance-threshold)/temperature)-softplus(-threshold/temperature)], so penalty(0)=0.',
        'Shared1024-base-voxel maximum warm-up, detached after recomputation at each current half-life;32-step/256-base-voxel branches, including native parent-before-update normalized EMA.',
        f"Native vs GPU exported-score max absolute discrepancy: {fitted['native_parity_max_abs']:.9g}.",'',
        '| Mapping | Subset | Failure windows | GT preferred % | Ranking loss |','|---|---|---:|---:|---:|'])
    for label,m in [('Baseline',fitted['baseline'])]+[('Smooth' if f['smooth'] else 'Linear',f['metrics']) for f in fitted['fits']]:
        for split,v in m.items():lines.append(f"| {label} | {split} | {v['failure_windows']} | {v['gt_preferred_percent']:.2f} | {v['ranking_loss']:.6f} |")
    lines.extend(['','## Coverage and limitations','',
        f"Accepted {len(distances)} windows; exclusions: `{dict(exclusions)}`.",
        f"Actual branch distances (base voxels): `{result['branch_distance_base']}`.",
        'No windows were replaced with zeros. Native extraction is CPU-only with existing local cache and readiness checks.',
        'Hermite GT connector is continuous and limited by curvature/circumscribed cone checks, NOT guaranteed on the discrete beam grid. Equal sample counts are not exactly equal Euclidean lengths.',
        'Finite warm-up reseeds embedding and smoothness state; parity is against native exported finite-prefix scoring, not original full ancestry. Long-window sums are not depth2 beam search. Report also contains first-two-step diagnostics.',
        'Initial short dataset accidentally recorded null library hash (wrong lib/ location); current native core library hash is included in comparison.json, and subsequent long extraction records it correctly. Binary/scoring implementation did not change between those exports.',
        'Long-window diagnostic, when present in comparison.json, applies the same learned parameters without fitting. It mines512-base-voxel windows independently, so it is not a strictly paired truncation ablation.',
        '', '## Artifacts',''])
    for r in result['rows']:lines.append(f"- {r['name']}: {r['path']}")
    lines.extend(['',f"Pending evaluations: {result['pending']}", 'No automatic default change. No visualization run.',''])
    (output/'comparison.md').write_text('\n'.join(lines))
    print(str(output/'comparison.md'),flush=True)
    return result
