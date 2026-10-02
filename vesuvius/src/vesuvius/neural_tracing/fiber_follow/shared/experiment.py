"""Frozen monitor/calibration/final manifests and paired rollout reporting."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from vesuvius.neural_tracing.fiber_follow.evaluation.seeds import summarize


def jsonable(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    raise TypeError(type(x).__name__)


def rollout_summary(rows):
    result=summarize(rows)
    wrong=[float(r['offtrack']) for r in rows if r['offtrack']>0]
    result.update(divergence_count=sum(r['diverged'] for r in rows),
                  wrong_continuation_lengths=wrong,
                  wrong_continuation_quantiles=dict(zip(('p50','p90','p95','max'),np.quantile(wrong,[.5,.9,.95,1]).tolist())) if wrong else None)
    return result


def paired_bootstrap(baseline, current, repeats=2000, seed=0):
    """Resample paired fibers, retaining every seed/direction within a fiber."""
    key=lambda r:(r['fiber'],r['t0'],r['sign'],r.get('sampling_seed',0))
    b={key(r):r for r in baseline}; c={key(r):r for r in current}
    if len(b)!=len(baseline) or len(c)!=len(current):
        raise ValueError('Duplicate trace identities; record sampling_seed for repeated rollouts')
    if b.keys()!=c.keys(): raise ValueError('Paired bootstrap requires identical seed identities')
    names=sorted({r['fiber'] for r in baseline})
    groups={name:sorted(k for k in b if k[0]==name) for name in names}
    rng=np.random.default_rng(seed)
    diffs=[]
    for _ in range(repeats):
        keys=[k for name in rng.choice(names,len(names)) for k in groups[name]]
        pair=[rollout_summary([pool[k] for k in keys]) for pool in (b,c)]
        diffs.append([pair[1][metric]-pair[0][metric] for metric in ('length_weighted_coverage','length_precision','diverged','wrong_len_mean','wrong_length')])
    return {metric:np.quantile(np.asarray(diffs)[:,i],[.025,.5,.975]).tolist()
            for i,metric in enumerate(('length_weighted_coverage','length_precision','diverged','wrong_len_mean','wrong_length'))}


def read_manifest(path):
    """Check the frozen payload hash, split membership and seed identities."""
    saved=json.loads(Path(path).read_text())
    payload={k:v for k,v in saved.items() if k!='sha256'}
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,default=jsonable).encode()).hexdigest()
    if digest!=saved['sha256']:
        raise ValueError('Frozen seed manifest was modified')
    sets=[set(saved[name+'_fibers']) for name in ('monitor','calibration','final')]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Evaluation splits overlap')
    for name,ids in zip(('monitor','calibration','final'),sets):
        if any(s['fiber'] not in ids for s in saved[name]):
            raise ValueError('Seed appears outside its frozen split')
    return saved
