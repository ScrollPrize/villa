"""Frozen monitor/calibration/final manifests and rollout reporting."""
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
