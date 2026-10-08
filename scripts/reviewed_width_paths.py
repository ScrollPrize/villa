"""Read-only native sampling and analysis of saved unweighted u11000 paths."""
import csv
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import time

import numpy as np
from lasagna.inference_provenance import atomic_write, sha256_file
from reviewed_fiber_evaluation import REPO, ROOT
from vesuvius.neural_tracing.fiber_trace_3d.embedding_route_analysis import Queries, arc_lengths

BASE=ROOT/'normalwidth_u11000_off_no_reverse_20261006_v1'
OUT=ROOT/'normalwidth_u11000_width_paths_20261006_v1'
BINARY=REPO/'volume-cartographer/build/fiber-evaluation/bin/vc_fiber_trace_metric'
SHA='2408ece6ffa2251655709cf3987357f1892a573828929934266145f09f9d2eba'
REPORT_NAME='reviewed_u11000_width_paths_20261006'
MODEL_LABEL='raw u11000'


def stats(values):
    a=np.asarray(values,dtype=float)
    a=a[np.isfinite(a)]
    return {'n':len(a),**({k:float(v) for k,v in zip(
        ('mean','p05','median','p95','max'),(a.mean(),*np.percentile(a,[5,50,95]),a.max()))} if len(a) else {})}


def prior_means(width):
    """Seed then mean strictly before candidate; any missing prefix invalidates it."""
    w=np.asarray(width,float)
    out=np.full(len(w),np.nan)
    if not len(w): return out
    out[0]=w[0]
    if len(w)>1: out[1:]=np.cumsum(w[:-1])/np.arange(1,len(w))
    return out


def final_excursion(ratios, failure_index):
    """Oracle-localized final departure: equality is still inside."""
    good=np.flatnonzero(np.asarray(ratios[:failure_index+1])<=1)
    return min(failure_index,int(good[-1]+1)) if len(good) else 0


def variation(w):
    w=np.asarray(w,float)
    valid=w[np.isfinite(w)&(w>0)]
    if not len(valid): return {'samples':len(w),'valid':0}
    median=float(np.median(valid))
    return dict(samples=len(w),valid=len(valid),mean=float(valid.mean()),median=median,
                cv_percent=float(100*valid.std()/valid.mean()),
                p95_abs_median_deviation_percent=float(np.percentile(100*np.abs(valid/median-1),95)),
                **{f'outside_{t}_percent':float(100*np.mean(np.abs(valid/median-1)>t/100)) for t in (10,20,40)})


def sample_queries(item):
    source,manifest,query=item
    folder=OUT/'samples'/source
    folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    for batch,begin in enumerate(range(0,len(query.points),200000)):
        points=query.points[begin:begin+200000]
        request=folder/f'points_{batch}.json'
        response=folder/f'values_{batch}.json'
        identity={'manifest_sha256':sha256_file(manifest),'native_sha256':sha256_file(BINARY),
                  'points':points}
        # Resume requires exact points and sampler binary, not only filenames.
        meta=folder/f'identity_{batch}.json'
        encoded=json.dumps(identity,sort_keys=True)
        if not (response.exists() and meta.exists() and meta.read_text()==encoded):
            atomic_write(request,points,durable=False)
            with response.open('w') as output,(folder/f'native_{batch}.log').open('w') as log:
                subprocess.run([str(BINARY),'sample-width',str(manifest),str(request)],stdout=output,stderr=log,
                               check=True,timeout=600)
            meta.write_text(encoded)
        result=json.loads(response.read_text())
        if len(result)!=len(points): raise ValueError('native sample count differs')
        rows.extend(result)
    print(f'width-query {source}: {len(rows)} points; missing/invalid={sum(r["status"]!="ok" for r in rows)}',flush=True)
    return source,rows


def save_csv(path,rows):
    if not rows: return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    started=time.monotonic()
    OUT.mkdir(parents=True,exist_ok=True)
    summary=json.loads((BASE/'full/summary.json').read_text())
    if not summary['completed'] or summary['cases']!=182:
        raise ValueError('unexpected baseline')
    workload=json.loads((BASE/'workload.json').read_text())
    if workload['checkpoint_sha256']!=SHA: raise ValueError('wrong checkpoint')
    if sha256_file(Path(workload['checkpoint']))!=SHA: raise ValueError('changed frozen checkpoint')
    paths=sorted((BASE/'full/cases').glob('*/trace.json'),key=lambda p:int(p.parent.name))
    if len(paths)!=182: raise ValueError('incomplete trace artifacts')
    with ThreadPoolExecutor(16) as pool:
        cases=list(pool.map(lambda p:(p,json.loads(p.read_text())),paths))
    queries={}; manifests={}; data=[]; hashes={}
    for path,case in cases:
        prov=case['provenance']; source=prov['source_id']; index=prov['case_index']
        if prov['effective_trace']['width_relative_tolerance']!=0 or prov['effective_trace']['replay_reverse_interval_base']!=0:
            raise ValueError('baseline must be unweighted/no reverse')
        manifest=Path(prov['fiber_manifest'])
        if source in manifests and manifests[source]!=manifest: raise ValueError('source manifest mismatch')
        manifests[source]=manifest
        q=queries.setdefault(source,Queries())
        gt=np.asarray(case['reference_points_base_xyz'],float); ga=arc_lengths(gt)
        ref_arc=np.r_[np.arange(0,ga[-1],8.),ga[-1]]
        ref_points=np.stack([np.interp(ref_arc,ga,gt[:,d]) for d in range(3)],axis=1)
        ref_ids=[q.add(p) for p in ref_points]
        segments=[]
        for si,s in enumerate(case['replay']['segments']):
            points=np.asarray(s['trace_points_base_xyz'],float)
            if not len(points): continue
            ids=[q.add(p) for p in points]
            matched=np.full(len(points),-1,dtype=int); ratios=np.full(len(points),np.nan)
            ref_arcs=np.full(len(points),np.nan)
            for m in s['matches']:
                i=m['trace_point_index']
                expected=np.array([np.interp(m['matched_reference_arc_base'],ga,gt[:,d]) for d in range(3)])
                if np.linalg.norm(expected-np.asarray(m['matched_reference_point_base_xyz']))>.1:
                    raise ValueError(f'reference arc/XYZ orientation mismatch in case {index}')
                matched[i]=q.add(m['matched_reference_point_base_xyz'])
                ratios[i]=m['threshold_error_ratio']; ref_arcs[i]=m['matched_reference_arc_base']
            # Seed is on reference; do not synthesize match XYZ for any other point.
            ratios[0]=0; ref_arcs[0]=s['start_reference_arc_base']
            failures=[f for f in case['replay']['failures'] if f['segment_index']==si]
            segments.append(dict(si=si,ids=np.array(ids),matched=matched,ratios=ratios,
                                 ref_arcs=ref_arcs,arcs=arc_lengths(points),failures=failures))
        data.append(dict(case=index,source=source,reverse=prov['reverse'],fiber=prov['fiber_path'],
                         ref_ids=np.array(ref_ids),ref_arcs=ref_arc,segments=segments))
        hashes[str(path)]=sha256_file(path)
    with ThreadPoolExecutor(5) as pool:
        native_results=dict(pool.map(sample_queries,[(s,manifests[s],q) for s,q in queries.items()]))
    values={s:np.array([r.get('width',np.nan) if r['status']=='ok' else np.nan for r in rows]) for s,rows in native_results.items()}
    presence={s:np.array([r.get('presence',np.nan) for r in rows]) for s,rows in native_results.items()}
    fibers=[]; segments=[]; events=[]; windows=[]; point_groups=defaultdict(list); plots=[]
    reference_deviations=[]; trace_deviations=[]; reference_seed_deviations=[]; trace_seed_deviations=[]
    reference_presence_groups=defaultdict(list)
    for case in data:
        source=case['source']; w=values[source]; p=presence[source]
        if not case['reverse']:
            rw=w[case['ref_ids']]
            fibers.append(dict(case=case['case'],source=source,fiber=case['fiber'],**variation(rw)))
            reference_deviations.extend((100*np.abs(rw/prior_means(rw)-1))[1:])
            reference_seed_deviations.extend((100*np.abs(rw/rw[0]-1))[1:])
            rd=100*np.abs(rw/prior_means(rw)-1)
            rp=p[case['ref_ids']]
            for label,mask in [('presence_ge_05',rp>=.5),('presence_lt_05',rp<.5)]:
                mask[0]=False
                reference_presence_groups[label].extend(rd[mask])
        for seg in case['segments']:
            tw=w[seg['ids']]; tp=p[seg['ids']]; mean=prior_means(tw)
            valid=seg['matched']>=0
            rw=np.full(len(tw),np.nan); rp=rw.copy()
            rw[valid]=w[seg['matched'][valid]]; rp[valid]=p[seg['matched'][valid]]
            dev=100*np.abs(tw/mean-1)
            trace_deviations.extend(dev[1:])
            trace_seed_deviations.extend((100*np.abs(tw/tw[0]-1))[1:])
            for name,mask in [('inside',seg['ratios']<=1),('outside',seg['ratios']>1),
                              ('near_reference',seg['ratios']<=.5),('failure_free_segment',np.full(len(tw),not seg['failures']))]:
                mask[0]=False
                point_groups[name].extend(dev[mask])
            segments.append(dict(case=case['case'],source=source,segment=seg['si'],
                                 reverse=case['reverse'],failures=len(seg['failures']),**variation(tw)))
            for failure in seg['failures']:
                fi=failure['segment_point_index']
                if fi is None or fi>=len(tw): raise ValueError('failure point outside segment')
                onset=final_excursion(seg['ratios'],fi)
                history=mean[onset] if onset>0 else np.nan
                common=dict(case=case['case'],source=source,reverse=case['reverse'],fiber=case['fiber'],
                    failure=failure['index'],segment=seg['si'],onset_trace_index=onset,failure_trace_index=fi,
                    onset_reference_arc_base=float(seg['ref_arcs'][onset]),
                    failure_reference_arc_base=failure['reference_arc_base'],pre_excursion_mean=history,
                    pre_excursion_points=onset,excursion_length_base=float(seg['arcs'][fi]-seg['arcs'][onset]))
                for offset in [-256,-128,-64,0,64,128,256,'failure']:
                    at=seg['arcs'][fi] if offset=='failure' else seg['arcs'][onset]+offset
                    if at<0 or at>seg['arcs'][fi]: continue
                    j=int(np.argmin(np.abs(seg['arcs'][:fi+1]-at)))
                    row=dict(common,offset=offset,actual_offset_base=float(seg['arcs'][j]-seg['arcs'][onset]),
                        trace_width=tw[j],reference_width=rw[j],trace_presence=tp[j],reference_presence=rp[j],
                        threshold_ratio=seg['ratios'][j],trace_prior_deviation_percent=dev[j],
                        trace_deviation_percent=100*abs(tw[j]/history-1),
                        reference_deviation_percent=100*abs(rw[j]/history-1))
                    row['reference_advantage_pp']=row['trace_deviation_percent']-row['reference_deviation_percent']
                    windows.append(row)
                    if offset=='failure': events.append(row)
                plots.append((common,case['ref_arcs'],w[case['ref_ids']],p[case['ref_ids']],
                              seg['ref_arcs'][:fi+1],tw[:fi+1],tp[:fi+1],seg['ratios'][:fi+1]))
    expected_events=sum(len(c['failures']) for s in summary['sources'] for c in s['cases'])
    if len(fibers)!=91 or len(events)!=expected_events: raise ValueError(f'unexpected fibers/events {len(fibers)}/{len(events)}')
    report=dict(checkpoint_sha256=SHA,baseline=str(BASE),cases=len(data),fibers=len(fibers),failures=len(events),
        failing_cases=len({e['case'] for e in events}),failing_fibers=len({e['fiber'] for e in events}),
        units={'width':'network-input voxels (not base voxels)','distance':'base voxels'},
        coverage={s:{'requested':len(v),'valid':int(np.isfinite(v).sum())} for s,v in values.items()},
        reference_fiber_cv_percent=stats([f['cv_percent'] for f in fibers if f['valid']]),
        reference_p95_median_deviation_percent=stats([f['p95_abs_median_deviation_percent'] for f in fibers if f['valid']]),
        reference_prior_deviation_percent=stats(reference_deviations),trace_prior_deviation_percent=stats(trace_deviations),
        reference_seed_deviation_percent=stats(reference_seed_deviations),trace_seed_deviation_percent=stats(trace_seed_deviations),
        reference_by_presence={k:stats(v) for k,v in reference_presence_groups.items()},
        point_groups={},windows={},scrolls={})
    for name,v in point_groups.items():
        a=np.asarray(v); a=a[np.isfinite(a)]
        report['point_groups'][name]={**stats(a),**{f'above_{t}_percent':float(100*np.mean(a>t)) for t in (10,20,40,80)}}
    for offset in [-256,-128,-64,0,64,128,256,'failure']:
        rows=[r for r in windows if r['offset']==offset]
        paired=[r for r in rows if np.isfinite(r['reference_advantage_pp'])]
        report['windows'][str(offset)]=dict(requested_available=len(rows),paired=len(paired),
            trace_deviation=stats([r['trace_deviation_percent'] for r in paired]),
            reference_deviation=stats([r['reference_deviation_percent'] for r in paired]),
            advantage=stats([r['reference_advantage_pp'] for r in paired]),
            reference_preferred_percent=float(100*np.mean([r['reference_advantage_pp']>0 for r in paired])) if paired else None,
            **{f'threshold_{t}':{'wrong_only_rejected':int(sum(r['trace_deviation_percent']>t>=r['reference_deviation_percent'] for r in paired)),
                               'reference_only_rejected':int(sum(r['reference_deviation_percent']>t>=r['trace_deviation_percent'] for r in paired))} for t in (10,20,40)})
    for source in values:
        fs=[f for f in fibers if f['source']==source]; es=[e for e in events if e['source']==source]
        report['scrolls'][source]=dict(fibers=len(fs),events=len(es),
            width=stats([f['median'] for f in fs if f['valid']]),cv=stats([f['cv_percent'] for f in fs if f['valid']]),
            failure_width_advantage=stats([e['reference_advantage_pp'] for e in es]))
    save_csv(OUT/'fibers.csv',fibers); save_csv(OUT/'segments.csv',segments)
    save_csv(OUT/'failures.csv',events); save_csv(OUT/'failure_windows.csv',windows)
    render_plots(plots)
    report.update(wall_seconds=time.monotonic()-started,trace_sha256=hashes,
                  manifest_sha256={str(m):sha256_file(m) for m in manifests.values()},native_sha256=sha256_file(BINARY))
    atomic_write(OUT/'summary.json',report,durable=False)
    write_report(report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('trace_sha256','manifest_sha256')},indent=2),flush=True)


def render_plots(plots):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=OUT/'plots'; folder.mkdir(exist_ok=True)
    for event,ra,rw,rp,ta,tw,tp,ratio in plots:
        center=event['onset_reference_arc_base']; stop=event['failure_reference_arc_base']
        fig,axes=plt.subplots(3,1,figsize=(10,6),sharex=True)
        axes[0].plot(ra-center,rw,label='reference'); axes[0].plot(ta-center,tw,label='unweighted trace')
        axes[0].axhline(event['pre_excursion_mean'],color='k',ls=':',label='pre-excursion mean')
        axes[0].set_ylabel('Width (input vx)'); axes[0].legend(loc='best',fontsize=8)
        axes[1].plot(ra-center,rp); axes[1].plot(ta-center,tp); axes[1].set_ylabel('Presence')
        axes[2].plot(ta-center,ratio,color='purple'); axes[2].axhline(1,color='gray',ls=':')
        axes[2].set_ylabel('Ellipse error ratio'); axes[2].set_xlabel('Matched reference arc relative to excursion onset (base vx)')
        for ax in axes:
            ax.axvline(0,color='gray',ls='--'); ax.axvline(stop-center,color='red',ls='--')
            ax.set_xlim(-512,max(256,stop-center+256)); ax.grid(alpha=.2)
        fig.suptitle(f'{event["source"]} case {event["case"]} failure {event["failure"]} reverse={event["reverse"]}')
        fig.tight_layout(); fig.savefig(folder/f'case_{event["case"]:03d}_failure_{event["failure"]:03d}.png',dpi=120)
        plt.close(fig)


def write_report(r):
    target=REPO/'vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/benchmarks'/REPORT_NAME
    atomic_write(target.with_suffix('.json'),r,durable=False)
    lines=['# Width variation on reference/unweighted trace paths','',
        f'Artifacts, CSVs and {r["failures"]} failure plots: `{OUT}`. Baseline: `{BASE}`; frozen {MODEL_LABEL}; 91 fibers/182 directions. No retracing, no inference. Native scalar sampling shared with embedding-query code; coherent channel coverage checked.', '',
        'Width is predicted width in network-input voxels, not annotated true width. Reference sampled every8base voxels; trace samples are original committed points. Reference fiber summaries count only forward cases once; trace point summaries include both directions and weight points equally. Fibers/directions/events are not independent.', '',
        'Prior means contain seed and all strictly preceding points, reset per restart. Missing earlier widths invalidate the mean. This is a committed-path diagnostic, not exact speculative beam-state reconstruction. Failure baseline is frozen strictly before the final departure from the acceptance ellipse (>1); locating that departure uses future outcome. Inside-ellipse is not proof of correct branch. Reference comparisons use stored matchedXYZ; offsets stay within the same observed segment. Reference continuation after failure is shown, but no nonexistent continued trace is invented.', '',
        '## Reference variation (one value per fiber)','',
        f'CV% distribution: `{r["reference_fiber_cv_percent"]}`',
        f'Within-fiber P95 absolute deviation from median (%): `{r["reference_p95_median_deviation_percent"]}`','',
        '## Causal prior-width deviations (point-weighted)','',
        '| Point group | Valid points | Median deviation % | P95 % | >10% | >20% | >40% |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,s in r['point_groups'].items():
        lines.append(f'| {name} | {s["n"]} | {s.get("median",0):.2f} | {s.get("p95",0):.2f} | {s["above_10_percent"]:.2f}% | {s["above_20_percent"]:.2f}% | {s["above_40_percent"]:.2f}% |')
    lines+=['','## Failure windows (event-weighted)','',
        '| Offset from final excursion (base vx) | Paired events | Trace median deviation % | GT median deviation % | GT preferred % | 10% wrong-only / GT-only rejected | 40% wrong-only / GT-only rejected |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for offset,s in r['windows'].items():
        t10=s['threshold_10'];t40=s['threshold_40']
        lines.append(f'| {offset} | {s["paired"]}/{s["requested_available"]} | {s["trace_deviation"].get("median",0):.2f} | {s["reference_deviation"].get("median",0):.2f} | {s["reference_preferred_percent"]} | {t10["wrong_only_rejected"]}/{t10["reference_only_rejected"]} | {t40["wrong_only_rejected"]}/{t40["reference_only_rejected"]} |')
    lines+=['','Coverage: `'+json.dumps(r['coverage'])+'`. Missing samples never become zeros. '+
            f'{r["failures"]} events from {r["failing_cases"]} directed cases / {r["failing_fibers"]} distinct fibers.', '',
            'Reference deviations stratified by presence (0.5 diagnostic threshold, not a new tracer setting): `'+json.dumps(r['reference_by_presence'])+'`.', '',
            'Seed-only deviation: `'+json.dumps(r['reference_seed_deviation_percent'])+'` on GT; `'+json.dumps(r['trace_seed_deviation_percent'])+'` on traces. Seed itself excluded from all point-deviation summaries.', '',
            '## Interpretation and next candidates','',
            '- These are retrospective samples, not predicted restart improvements. See paired denominators and threshold counts above; missing coverage and short histories must not be read as successful discrimination.',
            '- A physically constant fiber width need not give a constant predicted width. Soft penalties or confidence-qualified global-width updates are hypotheses requiring controlled reruns.',
            '- Post-excursion offsets are omitted when the trace has already ended. No continuation was fabricated.', '',
            '## Reproduce','','Use `reviewed_width_paths` for historical u11000; `reviewed_ema999_width` for the pinned current EMA999 benchmark and diagnostic.', '',
            '```bash','bash /home/hendrik/villa3/scripts/run_training_failure_mining','```']
    target.with_suffix('.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
