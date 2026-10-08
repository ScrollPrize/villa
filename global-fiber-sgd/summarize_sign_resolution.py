"""Publish recorded sign-solve measurements without rerunning optimization."""
import argparse
import json
from pathlib import Path
import numpy as np


def summarize(root, output):
    full=json.loads((root/'summary.json').read_text())
    rows=[]
    for name,m in full['baselines'].items():
        rows.append((name,0,m['retained_events_percent'],m['weighted_disagreement_all_percent'],
                     m['weighted_disagreement_retained_percent'],None))
    checked=[]
    with np.load(full['config']['tensor_cache']) as data:
        ids=data['ids'];i=data['i'];j=data['j'];dot=data['dot'].astype(np.float64)
        for run in full['runs']:
            label=f"{run['broken_penalty']:g}"
            with np.load(root/f'solution_{label}.npz') as solution:
                assert np.array_equal(ids,solution['ids'])
                s=solution['normal_sign'];b=solution['broken']
                assert np.isin(s,[-1,1]).all()
                assert np.isfinite(solution['sign_probability']).all()
                assert np.isfinite(solution['broken_probability']).all()
                inward='normal_convention' in solution and str(solution['normal_convention'].item())=='inward'
                expected=np.where(solution['sign_logit']>0,1,-1) if inward else np.where(solution['sign_logit']>=0,1,-1)
                assert np.array_equal(s,expected)
                assert np.array_equal(b,solution['broken_logit']>=0)
                kept=~b[i]&~b[j];wrong=dot*s[i]*s[j]<0;w=abs(dot)
                m=run['final'];retained=100*kept.mean()
                error=100*w[kept&wrong].sum()/w[kept].sum() if kept.any() else None
                assert abs(retained-m['retained_events_percent'])<2e-5
                assert error is None or abs(error-m['weighted_disagreement_retained_percent'])<2e-5
                checked.append(dict(penalty=run['broken_penalty'],broken_fibers=int(b.sum()),
                    retained_events=int(kept.sum()),opposed_retained_events=int((kept&wrong).sum()),
                    weighted_retained_disagreement_percent=error))
            rows.append((f"Adam penalty {label}",m['broken_fibers_percent'],retained,
                         m['weighted_disagreement_all_percent'],error,m['seconds']))
    text=['# Full-volume fiber normal sign results','',
          f"{full['input']['fibers']:,} whole fibers; {full['input']['used_events']:,} perpendicular events.",
          f"{full['config']['steps']:,} full-batch Adam steps, lr={full['config']['lr']}, weak prior={full['config']['prior_weight']}.",'',
          'Disagreement is dot-strength-weighted sign opposition—not anatomical error. Broken means model-selected exclusion.',
          'All-event disagreement includes excluded edges; retained disagreement uses only edges whose two fibers remain enabled.','',
          '| Method | Broken fibers | Retained events | All-event disagreement | Retained disagreement | Solve time |',
          '|---|---:|---:|---:|---:|---:|']
    for name,broken,retained,all_error,error,seconds in rows:
        error_text='—' if error is None else f'{error:.6f}%'
        seconds_text='—' if seconds is None else f'{seconds:.2f}s'
        text.append(f'|{name}|{broken:.4f}%|{retained:.4f}%|{all_error:.6f}%|{error_text}|{seconds_text}|')
    text.extend(['',f"GPU: {full['gpu']}; peak tensor allocation {full['peak_allocated_bytes']/1024**2:.2f}MiB.",
                 'One managed4GiB allocation; GPU was shared with training, so timings are observed runs, not isolated throughput benchmarks.',
                 'No graph splitting or winding optimization. Source database unchanged.','',
                 '| Penalty | Broken fibers | Retained events | Opposed retained events |',
                 '|---|---:|---:|---:|'])
    for row in checked:
        text.append(f"|{row['penalty']:g}|{row['broken_fibers']:,}|{row['retained_events']:,}|{row['opposed_retained_events']:,}|")
    text.extend(['',f'Artifacts: `{root}`.',
                 'Independent NumPy readout validated every saved solution against the logged hard-readout metrics.',
                 'See [model, schema and exact commands](sign_resolution.md).'])
    output.write_text('\n'.join(text)+'\n')
    output.with_suffix('.json').write_text(json.dumps(dict(summary=full,validation=checked),indent=2)+'\n')
    print('\n'.join(text))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();summarize(a.root,a.output)
