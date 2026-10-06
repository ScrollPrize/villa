"""Summarize long_trace_audit artifacts and render selected CT failure evidence."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .long_trace_audit import summary, write
from .evaluation import paired_report
from .evaluate import dataset_sources
from .seeds import matched_profile
from ..shared.geometry import tangent_at, arclength


def load_rows(directory):
    return [r for p in sorted(Path(directory).glob('*/*/rows_*.json')) for r in json.loads(p.read_text())]


def key(row):
    return (row['source'],row['cohort'],row['fiber'],row['t0'],row['sign'])


def comparison(base, new, base_protocol, new_protocol):
    a,b={key(r):r for r in base},{key(r):r for r in new}
    keys=sorted(a.keys()&b.keys())
    def report(rows,protocol):
        return dict(checkpoint=protocol['checkpoint'],operating_policy=protocol['operating_policy'],manifest_sha256='same-explicit-keys',
                    rows=[dict(rows[k],split=rows[k]['cohort']) for k in keys])
    return paired_report(report(a,base_protocol),report(b,new_protocol),repeats=2000,seed=20261003)


def paired_length_metrics(base, new, repeats=2000):
    """Paired fiber bootstrap of ratios, rather than ratios of bootstrap bounds."""
    a,b={key(r):r for r in base},{key(r):r for r in new}
    keys=sorted(a.keys()&b.keys())
    groups={'all': keys}
    for k in keys:
        groups.setdefault(k[0]+'/'+k[1],[]).append(k)
    def values(rows):
        return np.array([[r['correct'],r['offtrack'],r['local']['local_correct_length'],
                          r['local']['local_scored_length'],r['local']['recovered_coverage_length'],
                          r['local']['recovered_available']] for r in rows])
    def metrics(total):
        c,w,lc,ls,rc,available=total
        return np.array([c/max(c+w,1e-9),lc/max(ls,1e-9),lc,rc/max(available,1e-9)])
    names=('strict_precision','local_precision','local_correct_length','recovered_coverage')
    rng=np.random.default_rng(20261003);out={}
    for group,kk in groups.items():
        fibers=sorted({k[:3] for k in kk})
        members=[np.array([i for i,k in enumerate(kk) if k[:3]==f]) for f in fibers]
        aa,bb=values([a[k] for k in kk]),values([b[k] for k in kk])
        va,vb=metrics(aa.sum(0)),metrics(bb.sum(0));draws=[]
        for _ in range(repeats):
            ii=np.concatenate([members[j] for j in rng.integers(len(fibers),size=len(fibers))])
            draws.append(metrics(bb[ii].sum(0))-metrics(aa[ii].sum(0)))
        lo,hi=np.quantile(draws,[.025,.975],axis=0)
        out[group]={n:dict(base=float(va[i]),new=float(vb[i]),delta=float(vb[i]-va[i]),ci95=[float(lo[i]),float(hi[i])]) for i,n in enumerate(names)}
    return out


def diagnose(rows):
    failed=[r for r in rows if r['diverged']]
    result=dict(n=len(rows),diverged=len(failed),
        failure_onset_bins={f'{lo}-{hi}':sum(lo<=r['correct']<hi for r in failed) for lo,hi in [(0,32),(32,128),(128,400),(400,1000),(1000,float('inf'))]},
        diverged_with_sustained_return=sum(r['excursion_returns']>0 for r in failed),
        diverged_but_local_precision_above_90=sum(r['local']['local_precision'] is not None and r['local']['local_precision']>.9 for r in failed),
        local_correct_beyond_strict=sum(r['local']['local_correct_length']-r['correct'] for r in rows),
        reached_annotation_boundary=sum(r['unknown']>0 for r in rows),
        length_strata={},annotation_strata={})
    for label,lo,hi in [('under400',0,400),('400to1000',400,1000),('1000to2000',1000,2000),('over2000',2000,float('inf'))]:
        members=[r for r in rows if lo<=r['available_full']<hi]
        if members:result['length_strata'][label]=summary(members)
    for label,test in [('foldback',lambda r:bool(r['foldbacks'])),('no_foldback',lambda r:not r['foldbacks']),('kink_repaired',lambda r:r['kink_repairs']>0),('no_kink_repair',lambda r:r['kink_repairs']==0)]:
        members=[r for r in rows if test(r)]
        if members:result['annotation_strata'][label]=dict(n=len(members),diverged=sum(r['diverged'] for r in members),precision=summary(members)['length_precision'])
    unsafe=[d for r in rows for d in r['decisions_detail'] if d['accepted_unsafe']]
    result['unsafe_decisions']=dict(n=len(unsafe),
        committed_confidence_above_08=sum(d.get('confidence_committed',0)>=.8 for d in unsafe),
        committed_confidence_above_09=sum(d.get('confidence_committed',0)>=.9 for d in unsafe),
        starting_within_3=sum(d['match_distance'] is not None and d['match_distance']<=3 for d in unsafe),
        starting_beyond_3=sum(d['match_distance'] is not None and d['match_distance']>3 for d in unsafe))
    premature=[d for r in rows for d in r['decisions_detail'] if d['premature_stop']]
    result['premature_stop_proposals']=dict(n=len(premature),
        rejected_safe=sum(d['rejected_safe'] for d in premature),
        rejected_unsafe=sum(d['rejected_unsafe'] for d in premature),
        unknown_first_prefix=sum(not(d['rejected_safe'] or d['rejected_unsafe']) for d in premature))
    result['startup_heading_bins']={}
    for lo,hi in [(0,10),(10,20),(20,40),(40,181)]:
        members=[r for r in rows if lo<=r.get('startup_heading_error_deg',-1)<hi]
        if members:result['startup_heading_bins'][f'{lo}-{hi}']=dict(n=len(members),diverged=sum(r['diverged'] for r in members),early_departure=sum(r['diverged'] and r['correct']<32 for r in members))
    return result


def choose_cases(rows, count=6):
    # Deterministic evidence selection covers severe failure, return, early failure and stop.
    ranked=[]
    for group in ([r for r in rows if r['diverged'] and r['correct']<32],
                  sorted([r for r in rows if r['diverged']],key=lambda r:-r['offtrack']),
                  sorted([r for r in rows if r['excursion_returns']],key=lambda r:-r['excursion_returns']),
                  [r for r in rows if r['premature_stop']],
                  sorted(rows,key=lambda r:-r['correct'])):
        if group:ranked.append(group[0])
    ranked+=sorted(rows,key=lambda r:-r['offtrack'])
    out=[]
    for row in ranked:
        if key(row) not in {key(r) for r in out}:out.append(row)
        if len(out)==count:break
    return out


def render_case(root, row, source, destination):
    folder=root/'primary'/row['source']/row['cohort']
    data=np.load(folder/f"trace_{row['seed_index']:03d}.npz")
    path=data['path'];annotation=data['annotation'];ss=data['annotation_s']
    travelled=data['travelled'];details=row['decisions_detail']
    event=row['correct'] if row['diverged'] else row['length']
    j=int(np.clip(np.searchsorted(travelled,event,side='right')-1,0,len(travelled)-1))
    pos=data['pos'][j];frame=data['frame'][j]
    local_gt=(annotation-pos)@frame
    local_path=(path-pos)@frame
    f=source['fibers'][row['fiber']]
    matched=details[j]['matched_t']
    tangent=tangent_at(f.points,f.s,matched)*row['sign']
    angle=float(np.degrees(np.arccos(np.clip(frame[:,2]@tangent,-1,1))))
    fig,axes=plt.subplots(2,4,figsize=(16,9),gridspec_kw={'width_ratios':[1,1,1,1.5]})
    lateral=np.arange(-16,16.001,.5);forward=np.arange(-32,64.001,.5)
    aa,ff=np.meshgrid(lateral,forward)
    for dim in (0,1):
        other=1-dim
        for col,offset in enumerate((-2.,0.,2.)):
            loc=np.zeros((*aa.shape,3));loc[...,dim]=aa;loc[...,other]=offset;loc[...,2]=ff
            world=pos+loc@frame.T
            image=source['volume'].sample_image_nearest(world[...,::-1])
            lo,hi=np.percentile(image,[2,98])
            ax=axes[dim,col];ax.imshow(image,origin='lower',extent=(-16,16,-32,64),aspect='auto',cmap='gray',vmin=lo,vmax=hi)
            for points,color,label in [(local_gt,'lime','annotation'),(local_path,'orange','trace')]:
                visible=(np.abs(points[:,other]-offset)<=2)&(points[:,2]>=-32)&(points[:,2]<=64)&(np.abs(points[:,dim])<=16)
                ax.plot(np.where(visible,points[:,dim],np.nan),np.where(visible,points[:,2],np.nan),color=color,lw=1.2,label=label)
            proposed=data['points'][j];ax.plot(proposed[:,dim],proposed[:,2],color='cyan',ls='--',lw=1,label='proposal projection')
            ax.scatter([0],[0],s=20,color='red');ax.set_title(f"{'u' if dim==0 else 'v'} section; other axis {offset:+g}")
            ax.set_xlim(-16,16);ax.set_ylim(-32,64)
    arc=arclength(path) if len(path)>1 else np.zeros(1)
    # Arclength-matched distance (as strict scoring): the annotation's other windings near the path do not count.
    distance=matched_profile(path,np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))],f,row['t0'],row['sign'])[0]
    ax=axes[0,3];ax.plot(arc,distance);ax.axhline(3,color='red',ls='--');ax.axvline(event,color='black',ls=':')
    ax.set_ylim(0,min(30,max(8,float(np.nanmax(np.where(np.isfinite(distance),distance,30.))))));ax.set_title('Matched annotation distance');ax.set_xlabel('Travel (trace voxels)')
    ax=axes[1,3];ax.plot(travelled,data['confidence'][:,0],label='first prefix');ax.plot(travelled,data['confidence'][:,-1],label='full prefix')
    unsafe=np.array([d['accepted_unsafe'] for d in details],bool)
    ax.scatter(travelled[unsafe],data['confidence'][unsafe,0],color='red',s=12,label='accepted unsafe')
    ax.axhline(.5,color='gray',ls='--');ax.axvline(event,color='black',ls=':');ax.set_ylim(0,1.05);ax.legend(fontsize=8);ax.set_xlabel('Travel (trace voxels)')
    fig.suptitle(f"{row['source']} / {row['cohort']} / {row['seed_index']} / {row['fiber_name']}\n"
        f"strict correct {row['correct']:.0f}, wrong {row['offtrack']:.0f}; local precision {row['local']['local_precision'] or 0:.1%}; "
        f"stop {row['reason']}; inspected at {travelled[j]:.1f}; frame/annotation tangent angle {angle:.1f} deg\n"
        'Fixed sections in actual decision frame. Green/orange overlays within 2 voxels of section; cyan proposal is projected.',fontsize=10)
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(destination,dpi=120);plt.close(fig)
    return dict(source=row['source'],cohort=row['cohort'],seed_index=row['seed_index'],file=destination.name,
                frame_heading_angle=angle,decision_index=j,decision=details[j],fiber_name=row['fiber_name'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory');ap.add_argument('--render',action='store_true');args=ap.parse_args()
    root=Path(args.directory);base=load_rows(root/'primary');protocol=json.loads((root/'primary/protocol.json').read_text())
    for row in base:
        data=np.load(root/'primary'/row['source']/row['cohort']/f"trace_{row['seed_index']:03d}.npz")
        for j,d in enumerate(row['decisions_detail']):
            d['confidence_committed']=float(data['confidence'][j,max(0,int(data['n_commit'][j])-1)])
        if 'frame' in data:
            tangent=tangent_at(data['annotation'],data['annotation_s'],row['t0'])*row['sign']
            row['startup_heading_error_deg']=float(np.degrees(np.arccos(np.clip(data['frame'][0,:,2]@tangent,-1,1))))
    groups={}
    for row in base:groups.setdefault(row['source']+'/'+row['cohort'],[]).append(row)
    report=dict(groups={k:dict(summary=summary(v),diagnosis=diagnose(v)) for k,v in groups.items()},comparisons={},paired_length_metrics={})
    for name in ('confidence_030','confidence_080','commit_04','checkpoint_055000'):
        if (root/name/'summary.json').exists():
            other=load_rows(root/name);other_protocol=json.loads((root/name/'protocol.json').read_text())
            report['comparisons'][name]=comparison(base,other,protocol,other_protocol)
            report['paired_length_metrics'][name]=paired_length_metrics(base,other)
    write(root/'analysis.json',report)
    # Plot how the same paths score when clipped at different travel horizons.
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for name,group in report['groups'].items():
        horizons=group['summary']['horizons'];xx=[int(k) for k in horizons]
        axs[0].plot(xx,[100*horizons[str(k)]['length_precision'] for k in xx],marker='o',label=name)
        axs[1].plot(xx,[100*horizons[str(k)]['unknown_length_fraction'] for k in xx],marker='o',label=name)
    axs[0].set_ylabel('Strict scored precision (%)');axs[1].set_ylabel('Unknown length (%)')
    for ax in axs:ax.set_xlabel('Travel horizon (trace voxels)');ax.grid(alpha=.2)
    axs[1].legend(fontsize=7);fig.tight_layout();fig.savefig(root/'horizons.png',dpi=150);plt.close(fig)
    cases=[]
    if args.render:
        sources=dataset_sources(SimpleNamespace(checkpoint=protocol['checkpoint'],out=str(root/'render.json'),command='run',sources=None))
        lookup={s['name']:s for s in sources}
        folder=root/'cases';folder.mkdir(exist_ok=True)
        for name,rows in groups.items():
            for row in choose_cases(rows,count=6 if row_source(name)=='paris4' else 2):
                dest=folder/f"{row['source']}_{row['cohort']}_{row['seed_index']:03d}.png"
                case=render_case(root,row,lookup[row['source']],dest);cases.append(case)
                print('Rendered',dest.name,flush=True)
        write(root/'cases.json',cases)
    elif (root/'cases.json').exists():cases=json.loads((root/'cases.json').read_text())
    columns=['Cohort','N','Strict precision','Local geometric precision','Coverage','Diverged','Sustained returns','Unknown length']
    body=['<!doctype html><meta charset="utf-8"><title>79k long trace audit</title><style>body{font:16px system-ui;max-width:1200px;margin:35px auto}table{border-collapse:collapse}td,th{padding:9px;border:1px solid #ccc}img{max-width:100%}</style>',
          '<h1>79k long trace audit</h1><p>Fixed monitor seeds and length-weighted calibration fibers. Final-test fibers excluded. Local geometric agreement does not establish identity. Distances are trace-grid voxels.</p>',
          '<table><tr>'+''.join('<th>'+x+'</th>' for x in columns)+'</tr>']
    for name,g in report['groups'].items():
        s=g['summary'];d=g['diagnosis'];values=[name,s['n'],f"{s['length_precision']:.1%}",f"{s['local_precision']:.1%}",f"{s['coverage_mean']:.1%}",s['divergence_count'],d['diverged_with_sustained_return'],f"{s['unknown_length_fraction']:.1%}"]
        body.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in values)+'</tr>')
    body+=['</table><p>Strict precision keeps all scored travel after the first sustained departure wrong. Local precision measures current geometric agreement and separately censors annotation endings, including after a return. It cannot certify fiber identity. Coverage above is mean strict coverage; length-weighted and recovered coverage are in the detailed metrics. Paris held-out traces have no certified neighbor-bank identity coverage.</p>',
           '<img src="horizons.png"><p><a href="analysis.json">Detailed metrics and paired fiber bootstrap intervals</a> · <a href="primary/protocol.json">Protocol</a> · <a href="primary/seeds.json">Frozen seeds</a> · <a href="annotation_proximity.json">Other-annotation proximity audit</a></p>']
    if report['comparisons']:
        body.append('<h2>Paired controls</h2><p>Each cell shows 79k at threshold 0.5 / commit 16, then the named alternative on identical seeds. Controls use only the first 32 length-weighted seeds; the 55k comparison also includes all 32 monitor seeds. These diagnostic controls do not use final-test seeds.</p>')
        body.append('<table><tr><th>Alternative / cohort</th><th>N</th><th>Strict precision</th><th>Local precision</th><th>Local correct length</th><th>Diverged</th><th>Premature stops</th></tr>')
        for name,comparisons in report['comparisons'].items():
            for cohort,counts in comparisons.items():
                if '/' not in cohort:continue
                ratios=report['paired_length_metrics'][name][cohort]
                cells=[name+' / '+cohort,str(counts['traces'])]
                for metric in ('strict_precision','local_precision'):
                    m=ratios[metric];cells.append(f"{m['base']:.1%} → {m['new']:.1%}")
                m=ratios['local_correct_length'];cells.append(f"{m['base']:.0f} → {m['new']:.0f}")
                for metric in ('diverged','premature_stop'):
                    m=counts[metric];cells.append(f"{m['base']:.0f} → {m['new']:.0f}")
                body.append('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in cells)+'</tr>')
        body.append('</table><p>95% paired fiber-bootstrap intervals for differences are in analysis.json. Confidence changes can alter refinement and the subsequent trajectory; these are complete reruns, not rescored copies of the original path. Decision counts also change with commit size, so use length metrics for comparisons across commit policies.</p>')
    findings=root/'findings.html'
    if findings.exists():body.insert(2,findings.read_text())
    notes=json.loads((root/'case_notes.json').read_text()) if (root/'case_notes.json').exists() else {}
    for case in cases:
        body.append(f'<h2>{html.escape(case["file"])}</h2>')
        if case['file'] in notes:body.append('<p>'+html.escape(notes[case['file']])+'</p>')
        body.append(f'<img loading="lazy" src="cases/{case["file"]}">')
    (root/'report.html').write_text('\n'.join(body))
    print('Wrote',root/'report.html',flush=True)


def row_source(name):return name.split('/')[0]


if __name__=='__main__':main()
