"""Frozen angle/enclosure winding reference, independent of fitted UV geometry."""
import json
from pathlib import Path
import numpy as np
import torch
from lasagna.inference_provenance import atomic_write
from .fiber_gt_constraint_score import arrays
from .fiber_shared_curve import saved_points
from .fiber_implicit_winding import map_state


def statistics(values):
    values=np.abs(values)
    return dict(count=len(values),mean=float(values.mean()),p95=float(np.quantile(values,.95)),max=float(values.max()))


def compare(d,s,reference):
    offsets={int(k):float(v) for k,v in reference['fiber_offsets'].items()}
    targets=[];actual=[];owners=[]
    realized=map_state(torch.as_tensor(s['state'])).numpy()
    for side in ('a','b'):
        ids=d['i' if side=='a' else 'j'];ds=d['ds_'+side]
        q,_,_=saved_points(d,s,realized,s['widths'],ids,ds,shared_only=True)
        for event,record in enumerate(reference['events']):
            e=record['index'];source=int(d['source_id'][ids[e]])
            if source!=record['source_'+side] or not np.isclose(d['phase_'+side][e],record['phase_'+side],rtol=0,atol=1e-6):
                raise ValueError('GT winding reference endpoint identity/phase mismatch')
            targets.append(offsets[source]+record['phase_'+side]);actual.append(q[e]);owners.append(source)
    residual=np.asarray(actual)-np.asarray(targets);shift=float(np.median(residual))
    error=residual-shift;owners=np.asarray(owners)
    return dict(definition='MAP integer winding plus endpoint phase; lower label wins exact ties; one global additive shift removed, no affine or per-fiber fit',
        realized_labels=realized[:,0].tolist(),
        global_shift=shift,error_windings=statistics(error),
        per_fiber={str(i):statistics(error[owners==i]) for i in sorted(offsets)},
        excluded_source_ids=reference['excluded_source_ids'])


def run(root,reference_path):
    root=Path(root);cfg=json.loads((root/'config.json').read_text())
    result=compare(arrays(cfg['tensor_cache']),arrays(root/'solution.npz'),json.loads(Path(reference_path).read_text()))
    result['reference_path']=str(reference_path)
    atomic_write(root/'gt_winding_reference_report.json',result,durable=False)
    return result
