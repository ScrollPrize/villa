"""Length-biased held-out diagnostic rollouts with recorded decisions and horizon scoring.

Use monitor + calibration fibers only; leave final fibers untouched. Sampling weights
are fiber lengths, without replacement. No weights depend on model outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from .evaluate import dataset_sources
from .seeds import EvaluationAudit, evaluate, monitor_coverage, score_trace, summarize_outcomes
from .recovery_scoring import recovery_score
from ..data.observations import FiberTracer
from ..shared.experiment import jsonable
from ..shared.geometry import arclength, interp_at, tangent_at
from ..tracing.heading import oriented_seed_heading, SeedHeadingError, SEED_HEADING_POLICY
from ..tracing.policy import checkpoint_policy
from ..tracing.trace import TraceParams
from ..train.train import load_checkpoint
from ..train.runloop import raise_open_file_limit


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(value, default=jsonable, indent=1)+'\n')
    temporary.replace(path)


def length_weighted_order(lengths, count, rng):
    lengths = np.asarray(lengths, float)
    if lengths.ndim != 1 or not len(lengths) or not np.isfinite(lengths).all() or (lengths <= 0).any():
        raise ValueError('Finite positive fiber lengths required')
    return rng.choice(len(lengths), min(count, len(lengths)), replace=False, p=lengths/lengths.sum())


def truncate_path(path, length):
    path = np.asarray(path, float)
    arc = arclength(path) if len(path) > 1 else np.zeros(len(path))
    if arc[-1] <= length:
        return path.copy()
    keep = arc < length
    return np.concatenate([path[keep], interp_at(path, arc, [length])])


def prepare_seeds(source, count, rng):
    fibers, manifest, vol = source['fibers'], source['manifest'], source['volume']
    allowed = manifest['calibration_fibers']
    eligible = [i for i in allowed if fibers[i].length >= 128.]
    lengths = np.asarray([fibers[i].length for i in eligible])
    # Draw the whole weighted order to replace CT-heading rejections without choosing by model outcome.
    order = length_weighted_order(lengths, len(lengths), rng)
    seeds, skipped = [], []
    for j in order:
        fi = eligible[j]
        f = fibers[fi]
        sign = float(rng.choice([-1, 1]))
        offset = float(rng.uniform(32., max(32., .2*f.length)))
        t = offset if sign > 0 else f.length-offset
        pos = interp_at(f.points, f.s, [t])[0]
        try:
            axis = oriented_seed_heading(vol, pos, f.tag, tangent_at(f.points, f.s, t))
        except SeedHeadingError as error:
            skipped.append(dict(fiber=fi, reason=str(error)))
            continue
        seeds.append(dict(fiber=int(fi), t=t, sign=sign, pos=pos, heading=axis*sign,
                          family=f.tag, seed_heading_policy=SEED_HEADING_POLICY))
        if len(seeds) >= count:
            break
    return dict(monitor=manifest['monitor'], length_weighted=seeds, skipped=skipped,
                eligible_count=len(eligible), eligible_length_quantiles=np.quantile(lengths, [0,.25,.5,.75,1]),
                selected_length_quantiles=np.quantile([fibers[s['fiber']].length for s in seeds], [0,.25,.5,.75,1]),
                original_manifest_sha256=manifest['sha256'])


class DetailedAudit(EvaluationAudit):
    """Add evidence recording to the canonical evaluation audit without changing decisions."""
    def start_batch(self, fibers, seeds):
        super().start_batch(fibers, seeds)
        self.details = [[] for _ in seeds]
        self.states = [[] for _ in seeds]

    def __call__(self, index, state):
        before = self.records[index].copy()
        super().__call__(index, state)
        record, labeler = self.records[index], self.labelers[index]
        distance = float(labeler.vertex_distances[-1]) if len(labeler.vertex_distances) else None
        self.details[index].append(dict(travelled=state['travelled'], confidence_first=float(state['confidence'][0]),
            confidence_last=float(state['confidence'][-1]), commit=int(state['n_commit']),
            stopped=bool(state['would_stop']), blocked=bool(state['recovery_blocked']),
            match_distance=distance, matched_t=labeler.t, departure=labeler.departure_distance,
            boundary=labeler.boundary_distance, switched=labeler.switch is not None,
            accepted_unsafe=record['accepted_unsafe']-before['accepted_unsafe'],
            premature_stop=record['premature_stop']-before['premature_stop'],
            rejected_safe=record['rejected_safe']-before['rejected_safe'],
            rejected_unsafe=record['rejected_unsafe']-before['rejected_unsafe']))
        self.states[index].append({k: np.asarray(state[k]).copy() for k in
            ('pos','frame','hist','hmask','points','confidence','travelled','n_commit','heading_start')})


def summary(rows):
    result = summarize_outcomes(rows)
    for horizon in ('400','1000','2000'):
        members = [r['horizons'][horizon] for r in rows if horizon in r['horizons']]
        if members:
            result.setdefault('horizons', {})[horizon] = summarize_outcomes(members)
    local_correct = sum(r['local']['local_correct_length'] for r in rows)
    local_scored = sum(r['local']['local_scored_length'] for r in rows)
    result['local_precision'] = local_correct/max(local_scored, 1e-9)
    result['local_correct_length'] = local_correct
    result['local_scored_length'] = local_scored
    result['local_unknown_length'] = sum(r['local']['local_unknown_length'] for r in rows)
    recovered = sum(r['local']['recovered_coverage_length'] for r in rows)
    available = sum(r['local']['recovered_available'] for r in rows)
    result['recovered_coverage_length'] = recovered
    result['recovered_length_weighted_coverage'] = recovered/max(available, 1e-9)
    result['reason_counts'] = {reason:sum(r['reason']==reason for r in rows) for reason in sorted({r['reason'] for r in rows})}
    result['available_quantiles'] = np.quantile([r['available_full'] for r in rows], [0,.25,.5,.75,1])
    result['startup_departures'] = sum(r['diverged'] and r['correct'] < 32 for r in rows)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed-manifest')
    ap.add_argument('--count', type=int, default=64)
    ap.add_argument('--limit', type=int, help='Use only the first N frozen seeds per cohort for paired controls')
    ap.add_argument('--sources', nargs='+')
    ap.add_argument('--cohorts', nargs='+', default=['monitor','length_weighted'])
    from ..tracing.policy import DEFAULT_CONFIDENCE, DEFAULT_GATE, DEFAULT_N_COMMIT, GATES
    ap.add_argument('--confidence', type=float, default=DEFAULT_CONFIDENCE)
    ap.add_argument('--n-commit', type=int, default=DEFAULT_N_COMMIT)
    ap.add_argument('--gate', choices=GATES, default=DEFAULT_GATE)
    ap.add_argument('--max-len', type=float, default=2000.)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--seed', type=int, default=20261003)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--precision', choices=('bf16', 'fp32'), default='bf16',
                    help='model arithmetic while tracing; fp32 (TF32 off) makes traces independent of batch composition')
    ap.add_argument('--sampler-seed', type=int, default=0,
                    help="seed of the flow model's sampled proposals (TraceParams.seed; keyed per trace and decision)")
    args = ap.parse_args()
    torch.set_num_threads(4)
    if args.precision == 'fp32':
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    raise_open_file_limit()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'summary.json').exists():
        raise FileExistsError('Completed evaluation already exists')
    print('Loading checked held-out sources', flush=True)
    sources = dataset_sources(SimpleNamespace(checkpoint=args.checkpoint, out=str(out/'results.json'), command='run', sources=args.sources))
    frozen = json.loads(Path(args.seed_manifest).read_text()) if args.seed_manifest else dict(
        protocol='length-proportional sampling without replacement; calibration fibers only; random direction; seed 32 to 20% of length from starting end',
        seed=args.seed, count=args.count, sources={})
    for i, source in enumerate(sources):
        if source['name'] not in frozen['sources']:
            frozen['sources'][source['name']] = prepare_seeds(source,args.count,np.random.default_rng(args.seed+i))
        selection = frozen['sources'][source['name']]
        assert selection['original_manifest_sha256'] == source['manifest']['sha256']
        selected = {s['fiber'] for s in selection['length_weighted']}
        assert selected <= set(source['manifest']['calibration_fibers'])
        assert not selected & set(source['manifest']['final_fibers'])
        print(source['name'], 'selected length quantiles', selection['selected_length_quantiles'], flush=True)
    write(out/'seeds.json', frozen)
    model, crop, nh, _, ck = load_checkpoint(args.checkpoint, args.device)
    policy = checkpoint_policy(ck,model.cfg,confidence=args.confidence,n_commit=args.n_commit,gate=args.gate)
    provenance = dict(checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        step=ck['step'], operating_policy=policy.to_dict(), max_len=args.max_len, batch=args.batch,
        seeds_sha256=hashlib.sha256((out/'seeds.json').read_bytes()).hexdigest(), final_fibers_used=False,
        note='Length-biased diagnostic population; not a uniform-population estimate. Local agreement does not establish fiber identity.',
        args=vars(args))
    write(out/'protocol.json', provenance)
    all_rows = []
    started = time.monotonic()
    for source in sources:
        name = source['name']
        fibers = source['fibers']
        tracer = FiberTracer(model,source['volume'],crop,nh,TraceParams.from_policy(policy,max_len=args.max_len,forward_chunk=args.batch,precision=args.precision,seed=args.sampler_seed),device=args.device)
        try:
            for cohort in args.cohorts:
                seeds = frozen['sources'][name][cohort]
                if args.limit is not None:
                    seeds = seeds[:args.limit]
                destination = out/name/cohort
                destination.mkdir(parents=True,exist_ok=True)
                for start in range(0,len(seeds),args.batch):
                    if (destination/f'rows_{start:03d}.json').exists():
                        all_rows.extend(json.loads((destination/f'rows_{start:03d}.json').read_text()))
                        continue
                    chunk = seeds[start:start+args.batch]
                    audit = DetailedAudit(tracer,float(ck['tolerance']),source.get('detector'))
                    paths=[]
                    rows,_ = evaluate(tracer,fibers,chunk,batch=args.batch,history_audit=audit,
                        coverage_max_len=args.max_len,on_trace=lambda seed,path,reason:paths.append(path))
                    for j,(row,seed,path) in enumerate(zip(rows,chunk,paths)):
                        f=fibers[seed['fiber']]
                        row.update(source=name,cohort=cohort,seed_index=start+j,available_full=f.length-seed['t'] if seed['sign']>0 else seed['t'],
                                   fiber_length=f.length,family=f.tag,kink_repairs=f.kink_repairs,foldbacks=list(f.foldbacks),
                                   decisions_detail=audit.details[j],seed=seed,horizons={})
                        for horizon in (400,1000,2000):
                            if horizon>args.max_len:continue
                            row['horizons'][str(horizon)]=monitor_coverage(score_trace(truncate_path(path,horizon),f,seed['t'],seed['sign']),horizon)
                        row['local']=recovery_score(path,f,seed['t'],seed['sign'],max_len=args.max_len,tol=3.,sample_step=.5,persistence=1.5)
                        arrays = dict(path=np.asarray(path),annotation=f.points,annotation_s=f.s)
                        states=audit.states[j]
                        if states:
                            arrays.update({k:np.stack([s[k] for s in states]) for k in states[0]})
                        np.savez_compressed(destination/f'trace_{start+j:03d}.npz',**arrays)
                    write(destination/f'rows_{start:03d}.json',rows)
                    all_rows.extend(rows)
                    print(json.dumps(dict(source=name,cohort=cohort,completed=min(start+args.batch,len(seeds)),total=len(seeds),
                        seconds=round(time.monotonic()-started,1),precision=summarize_outcomes(rows)['length_precision'])),flush=True)
        finally:
            tracer.close()
    groups={}
    for row in all_rows:
        groups.setdefault(row['source']+'/'+row['cohort'],[]).append(row)
    result=dict(provenance=provenance,seconds=time.monotonic()-started,groups={k:summary(v) for k,v in groups.items()})
    write(out/'summary.json',result)
    print(json.dumps(result,default=jsonable),flush=True)


if __name__=='__main__':
    main()
