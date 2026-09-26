"""Frozen monitor/calibration/final manifests and paired rollout reporting."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from vesuvius.neural_tracing.fiber_follow.data import fiber_manifest
from vesuvius.neural_tracing.fiber_follow.evaluate import make_seeds, summarize
from vesuvius.neural_tracing.fiber_follow.volume import FiberVolumeSpec


def jsonable(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    raise TypeError(type(x).__name__)


def freeze_manifest(path, fibers, vol, assessment, original_config):
    path=Path(path)
    original=json.loads(Path(original_config).read_text())
    if [f.name for f in fibers] != original['val']:
        raise ValueError('Validation ordering differs from assessment run')
    identities=fiber_manifest(fibers)
    original_lookup={f['name']:f for f in original['fiber_manifest']}
    if any(f != original_lookup.get(f['name']) for f in identities):
        raise ValueError('Annotation geometry differs from the assessment')
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['fibers'] != identities or FiberVolumeSpec(**saved['volume']) != vol.spec:
            raise ValueError('Frozen evaluation manifest differs from current geometry/volume')
        return saved
    calibration=json.loads(Path(assessment).read_text())
    calibration_ids={s['fiber'] for s in calibration}
    if len(calibration_ids)!=48:
        raise ValueError('Expected the original 48 assessment fibers')
    monitor_ids=np.random.default_rng(123).choice(len(fibers),min(original['diag_seeds'],len(fibers)),replace=False).tolist()
    if calibration_ids & set(monitor_ids):
        raise ValueError('Monitor/calibration overlap')
    monitors=make_seeds([fibers[i] for i in monitor_ids],vol,per_fiber=1,seed=123)[::2][:original['diag_seeds']]
    for s in monitors: s['fiber']=monitor_ids[s['fiber']]
    final_ids=[i for i in range(len(fibers)) if i not in calibration_ids | set(monitor_ids)]
    final=make_seeds([fibers[i] for i in final_ids],vol,per_fiber=2,seed=20260926)
    for s in final: s['fiber']=final_ids[s['fiber']]
    saved=dict(version=1,fibers=identities,volume=vol.spec.to_dict(),monitor=monitors,
               calibration=calibration,final=final,monitor_fibers=monitor_ids,
               calibration_fibers=sorted(calibration_ids),final_fibers=final_ids)
    saved['sha256']=hashlib.sha256(json.dumps(saved,sort_keys=True,default=jsonable).encode()).hexdigest()
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(saved,indent=2,default=jsonable))
    return saved


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
