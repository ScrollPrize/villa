"""Evaluate cached whole-volume flow labels; references never modify the solve."""
import json
import csv
import time
from pathlib import Path
import numpy as np

from .fiber_min_flow_experiment import arrays,write
from .fiber_min_flow_full import PackedPaths
from .fiber_min_flow_benchmark import Paths
from .fiber_min_flow_evaluation import large_reference
from .fiber_reference_comparison import Surface,winding_at,grouped_weighted_median,winding_error_metrics


def sample_arcs(lengths,spacing):
    lengths=np.asarray(lengths)
    if spacing<=0 or np.any(lengths<=0):raise ValueError('Positive spacing and lengths required')
    counts=np.maximum(2,np.ceil(lengths/spacing).astype(np.int64)+1)
    offsets=np.r_[0,np.cumsum(counts)]
    owner=np.repeat(np.arange(len(lengths)),counts)
    local=np.arange(offsets[-1])-np.repeat(offsets[:-1],counts)
    step=lengths[owner]/(counts[owner]-1)
    weights=step*np.where((local==0)|(local==counts[owner]-1),.5,1.)
    return owner,local*step,weights,offsets


def continuous_winding(seam,xyz,turns,labels):
    rel=seam.relative(xyz)
    phase=np.mod(np.arctan2(rel[:,1],rel[:,0])-seam.angle,2*np.pi)/(2*np.pi)
    return labels+turns+phase


def error_summary(error):
    error=np.asarray(error);absolute=abs(error)
    if not len(error):return dict(fibers=0)
    median=float(np.median(error))
    edges=[0,.25,.5,1,2,5,10,20,40,np.inf]
    counts=np.histogram(absolute,edges)[0]
    return dict(fibers=len(error),l1=float(absolute.mean()),signed_mean=float(error.mean()),
        signed_median=median,local_centered_l1=float(abs(error-median).mean()),
        absolute_quantiles=dict(zip(('p10','p25','p50','p75','p90','p95','p99','max'),
            map(float,np.quantile(absolute,[.1,.25,.5,.75,.9,.95,.99,1])))),
        within_percent={str(t):float(100*np.mean(absolute<=t)) for t in [.25,.5,1,2,5,10,20]},
        histogram=[dict(low=a,high=b if np.isfinite(b) else None,count=int(n),percent=float(100*n/len(error)))
            for a,b,n in zip(edges[:-1],edges[1:],counts)])


def analyze(config):
    """Read-only drift diagnostics; never rescale or modify the saved solution."""
    source=Path(config['evaluation_root']);out=Path(config['output_root']);out.mkdir(parents=True,exist_ok=True)
    d=arrays(source/'fiber_reference.npz');summary=json.loads((source/'summary.json').read_text())
    shift=summary['spiral']['all']['global_offset']
    valid=np.isfinite(d['reference_minus_fit'])
    error=shift-d['reference_minus_fit']  # positive = fitted winding too high
    sizes=np.bincount(d['component']);largest=int(sizes.argmax())
    cohorts={'all':valid,'main_component':valid&(d['component']==largest),
             'other_components':valid&(d['component']!=largest),
             'at_least_90pct_coverage':valid&(d['coverage']>=.9)}
    report=dict(global_offset=shift,error_sign='fitted minus reference; after existing global offset',
        cohorts={name:error_summary(error[mask]) for name,mask in cohorts.items()},bands={})
    width=float(config.get('winding_band_width',10))
    if width<=0:raise ValueError('Positive band width required')
    labels=np.floor(np.nan_to_num(d['reference'],nan=-1e9)/width).astype(np.int64)
    for name in ('all','main_component'):
        use=cohorts[name];bands=[]
        for label in np.unique(labels[use]):
            mask=use&(labels==label);row=error_summary(error[mask])
            row.update(reference_min=float(label*width),reference_max=float((label+1)*width),
                       percent_of_matched=float(100*mask.sum()/valid.sum()))
            bands.append(row)
        report['bands'][name]=bands
    # Independent held-out diagnostic of systematic radial drift, not a new score.
    # UUID integer parity gives a stable, order-independent held-out split.
    even=np.fromiter((int(str(s)[-1],16)%2==0 for s in d['source_id']),bool,len(d['source_id']))
    use=cohorts['main_component'];train=use&even;test=use&~train
    x=d['reference'];design=np.column_stack([np.ones(train.sum()),x[train]])
    coeff=np.linalg.lstsq(design,error[train],rcond=None)[0]
    residual=error[test]-(coeff[0]+coeff[1]*x[test])
    report['linear_drift_diagnostic']=dict(fit_cohort='even UUID integer parity, main component',
        evaluation_cohort='odd UUID integer parity, main component',intercept=float(coeff[0]),slope=float(coeff[1]),
        implied_fitted_winding_scale=float(1+coeff[1]),
        before=error_summary(error[test]),after=error_summary(residual),
        warning='Diagnostic trend removal only; no change to the solve or headline reference score')
    fine=np.floor(np.nan_to_num(d['reference'],nan=-1e9)).astype(np.int64)
    regions=[]
    for label in np.unique(fine[use]):
        mask=use&(fine==label)
        if mask.sum()<1000:continue
        row=error_summary(error[mask]);row.update(reference_min=int(label),reference_max=int(label+1))
        regions.append(row)
    report['best_one_winding_regions']=sorted(regions,key=lambda r:r['l1'])[:5]
    report['error_selected_subsets']={str(t):dict(error_summary(error[valid&(abs(error)<=t)]),
        percent_of_matched=float(100*np.mean(abs(error[valid])<=t))) for t in [.25,.5,1,2]}
    # Same nearest-integer convention as the reference projection's winding IDs.
    winding=np.floor(np.nan_to_num(d['reference'],nan=-1e9)+.5).astype(np.int64)
    report['per_reference_winding']={}
    for name in ('all','main_component'):
        rows=[]
        for label in np.unique(winding[cohorts[name]]):
            mask=cohorts[name]&(winding==label);e=error[mask];median=float(np.median(e))
            residual=e-median
            rows.append(dict(winding=int(label),fibers=int(mask.sum()),
                additional_shift=-median,total_shift=shift-median,
                l1_before=float(abs(e).mean()),l1_after=float(abs(residual).mean()),
                residual_p90=float(np.quantile(abs(residual),.9)),
                shift_iqr=float(np.quantile(e,.75)-np.quantile(e,.25)),
                within_one_after_percent=float(100*np.mean(abs(residual)<=1))))
        report['per_reference_winding'][name]=rows
        with (out/f'per_winding_{name}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Optimal shift per reference winding','',
        'Largest connected component only; equal weight per matched fiber. Reference IDs are nearest integers of the projected reference winding.',
        '',f'The existing global shift is {shift:.6f}. Additional shift is added to that aligned solution; total shift applies to the raw saved winding.',
        'Each shift is the median reference-minus-fit and minimizes L1 within that winding. This is a diagnostic, not a modified solution.',
        '', '|Reference winding|Fibers|Additional shift|Total shift|L1 before|L1 after|Residual p90|',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for r in report['per_reference_winding']['main_component']:
        lines.append(f"|{r['winding']}|{r['fibers']}|{r['additional_shift']:+.2f}|{r['total_shift']:.2f}|{r['l1_before']:.2f}|{r['l1_after']:.2f}|{r['residual_p90']:.2f}|")
    (out/'per_winding.md').write_text('\n'.join(lines)+'\n')
    write(out/'summary.json',report)
    print(json.dumps(report),flush=True)
    return report


def solution_catalog_indices(catalog, solution):
    """Map a solved component to its immutable full-volume geometry catalog."""
    ids=np.asarray(solution['source_id']);source=np.asarray(catalog['source_id'])
    ix=np.searchsorted(source,ids)
    if not len(ids) or np.any(ids[1:]<=ids[:-1]) or np.any(ix>=len(source)):
        raise ValueError('Invalid solution source IDs')
    if not np.array_equal(source[ix],ids):raise ValueError('Unknown solution source IDs')
    if not np.array_equal(catalog['family'][ix],solution['family']):
        raise ValueError('Fiber classification changed')
    return ix


def continuous_gt_metrics(paths, solution, gt, gc, gd, events):
    """Shared continuous-endpoint GT protocol for fresh and cached projections."""
    from .fiber_irls_benchmark import metrics
    from .fiber_uv_arc_phase import event_arc_points
    from .fiber_uv_layout import hard_assignment
    gs=arrays(Path(gt['config']).parent/'solution.npz')
    _,_,_,pick=hard_assignment(gs['state'],len(gs['widths']),gc,best_nonbroken=True)
    rows=events['events'] if isinstance(events,dict) else events
    index={str(s):i for i,s in enumerate(solution['source_id'])}
    rows=[e for e in rows if e['test_source'] in index]
    ids=np.array([index[e['test_source']] for e in rows],dtype=np.int64)
    for e in rows:
        if gd['source_id'][e['gt_piece']]!=e['gt_source']:
            raise ValueError('GT piece/source mismatch')
    xyz,turns=paths.at_many(solution['source_id'][ids],np.array([e['test_arc'] for e in rows]))
    q=continuous_winding(paths.seam,xyz,turns,solution['winding'][ids])+paths.seam.angle/(2*np.pi)
    truth=event_arc_points(gd,gs['state'],pick,gs['widths'],
        np.array([e['gt_piece'] for e in rows],dtype=np.int64),np.array([e['gt_arc'] for e in rows]))
    return metrics(q,truth,rows,np.ones(len(rows),bool))


def rescore(config):
    """Exact reuse of reference projections when only integer labels/subset change."""
    source=Path(config['evaluation_root']);out=Path(config['output_root']);out.mkdir(parents=True,exist_ok=True)
    old_config=json.loads((source/'request.json').read_text())['config']
    cached=arrays(source/'fiber_reference.npz');old=arrays(old_config['solution']);new=arrays(config['solution'])
    if config.get('largest_component_only'):
        keep=new['component']==np.bincount(new['component']).argmax()
        new={k:new[k][keep] for k in ('source_id','family','winding','component')}
    if not np.array_equal(cached['source_id'],old['source_id']):raise ValueError('Projection/baseline identity mismatch')
    ix=np.searchsorted(old['source_id'],new['source_id'])
    if np.any(ix>=len(old['source_id'])) or not np.array_equal(old['source_id'][ix],new['source_id']):
        raise ValueError('New solve is not a subset of the projected fibers')
    paths=PackedPaths(old_config['geometry'])
    solution_catalog_indices(paths.catalog,new)
    gt=old_config['gt'];gc=json.loads(Path(gt['config']).read_text());gd=arrays(gc['tensor_cache']);gp=Paths(gc,gd)
    reference=json.loads(Path(gt['reference']).read_text());events=json.loads(Path(gt['events']).read_text())
    d={'source_id':new['source_id']};report={'fibers':len(ix),'cohort':'identical retained fibers for both solutions'}
    comparison=arrays(config['comparison_solution']) if config.get('comparison_solution') else old
    ci=np.searchsorted(comparison['source_id'],new['source_id'])
    if np.any(ci>=len(comparison['source_id'])) or not np.array_equal(comparison['source_id'][ci],new['source_id']):
        raise ValueError('Comparison solution does not cover retained cohort')
    report['comparison_solution']=config.get('comparison_solution',old_config['solution'])
    for name,solution,delta in [('baseline_same_cohort',dict(winding=comparison['winding'][ci],component=comparison['component'][ci]),
                                cached['reference_minus_fit'][ix]-(comparison['winding'][ci]-old['winding'][ix])),
                               ('perpendicular_only',new,cached['reference_minus_fit'][ix]-(new['winding']-old['winding'][ix]))]:
        gt_score=large_reference(d,solution,paths,reference,events,gd,gp)
        for row in gt_score['single_global_gauge']['signs'].values():
            row['percent']=100*row['violations']/row['events'] if row['events'] else None
        valid=np.isfinite(delta);shift=float(np.median(delta[valid]));error=shift-delta
        sizes=np.bincount(solution['component']);main=solution['component']==sizes.argmax()
        stats=error_summary(error[valid]);stats['global_offset']=shift
        report[name]=dict(gt=gt_score,spiral=stats,main_component_spiral=error_summary(error[valid&main]),
                         components=int(np.count_nonzero(sizes)),main_component_fibers=int(main.sum()))
        if config.get('continuous_gt_protocol'):
            report[name]['continuous_gt']=continuous_gt_metrics(paths,dict(solution,source_id=new['source_id']),gt,gc,gd,events)
        if name=='perpendicular_only':
            np.savez(out/'fiber_reference.npz',source_id=new['source_id'],component=new['component'],
                reference_minus_fit=delta,reference=cached['reference'][ix],
                fitted=cached['fitted'][ix]+new['winding']-old['winding'][ix],coverage=cached['coverage'][ix])
    write(out/'request.json',config);write(out/'summary.json',report)
    print(json.dumps(report),flush=True)
    return report


def run(config):
    root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    signature=dict(config=config,solution=dict(size=Path(config['solution']).stat().st_size,
        mtime_ns=Path(config['solution']).stat().st_mtime_ns))
    if (root/'request.json').exists() and json.loads((root/'request.json').read_text())!=signature:
        raise ValueError('Evaluation inputs changed')
    write(root/'request.json',signature)
    started=time.monotonic();paths=PackedPaths(config['geometry']);d=paths.catalog
    solution=arrays(config['solution'])
    catalog_ix=solution_catalog_indices(d,solution)
    if not config.get('allow_solution_subset',False) and len(catalog_ix)!=len(d['source_id']):
        raise ValueError('Source identities differ; enable allow_solution_subset for component solves')
    ids=solution['source_id'];d=dict(source_id=ids,family=solution['family'])
    sizes=np.bincount(solution['component']);largest=int(sizes.argmax())
    report=dict(fibers=len(ids),components=len(sizes),largest_component_fibers=int(sizes[largest]),
        largest_component_percent=float(100*sizes[largest]/len(ids)),singletons=int(np.sum(sizes==1)))
    if 'gt' in config:
        gt=config['gt'];gc=json.loads(Path(gt['config']).read_text());gd=arrays(gc['tensor_cache'])
        gp=Paths(gc,gd);reference=json.loads(Path(gt['reference']).read_text())
        events=json.loads(Path(gt['events']).read_text())
        scored=large_reference(d,solution,paths,reference,events,gd,gp)
        for stats in [scored['signs'],scored['single_global_gauge']['signs']]:
            for row in stats.values():
                row['percent']=100*row['violations']/row['events'] if row['events'] else None
        report['gt']=scored
        if config.get('continuous_gt_protocol'):
            report['continuous_gt']=continuous_gt_metrics(paths,solution,gt,gc,gd,events)
        write(root/'gt_metrics.json',scored);write(root/'summary.json',report)
        print(json.dumps(dict(stage='gt_complete',result=scored)),flush=True)
    settings=config['spiral'];surface=Surface(settings['reference'],settings.get('xyz_scale',4.),
        settings.get('surface_stride',2),settings.get('workers',32),settings.get('candidates',4))
    spacing=settings.get('sample_spacing',200.);maximum=settings.get('max_distance',200.)
    lengths=paths.keys['arc'][paths.offset[catalog_ix+1]-1]
    count=len(ids);delta=np.full(count,np.nan);fit=np.full(count,np.nan);truth=np.full(count,np.nan)
    coverage=np.zeros(count);sample_count=0;matched_count=0
    batch=settings.get('fiber_batch',2048);parts=root/'parts';parts.mkdir(exist_ok=True)
    for first in range(0,count,batch):
        last=min(first+batch,count);path=parts/f'{first:08d}.npz'
        if path.exists():values=arrays(path)
        else:
            owner,arc,weight,offset=sample_arcs(lengths[first:last],spacing)
            xyz,turns=paths.at_many(ids[first:last][owner],arc)
            rc,distance=surface.project(xyz)
            ref=winding_at(rc[:,1],surface.x,surface.q)
            predicted=continuous_winding(paths.seam,xyz,turns,solution['winding'][first:last][owner])
            valid=(distance<=maximum)&np.isfinite(ref)
            use_weight=np.where(valid,weight,0.)
            values=dict(delta=grouped_weighted_median(ref-predicted,use_weight,offset),
                fit=grouped_weighted_median(predicted,use_weight,offset),
                truth=grouped_weighted_median(ref,use_weight,offset),
                coverage=np.add.reduceat(use_weight,offset[:-1])/lengths[first:last],
                samples=np.array(len(arc)),matched=np.array(valid.sum()))
            np.savez(path,**values)
        for name,target in [('delta',delta),('fit',fit),('truth',truth),('coverage',coverage)]:target[first:last]=values[name]
        sample_count+=int(values['samples']);matched_count+=int(values['matched'])
        elapsed=time.monotonic()-started
        if first//batch%10==0 or last==count:
            progress=dict(stage='spiral_projection',fibers=last,total=count,samples=sample_count,
                seconds=elapsed,eta_seconds=elapsed*(count-last)/last)
            write(root/'progress.json',progress);print(json.dumps(progress),flush=True)
    valid=np.isfinite(delta);shift=float(np.median(delta[valid])) if valid.any() else None
    def metrics(mask):
        use=valid&mask;e=abs(delta[use]-shift) if shift is not None else np.array([])
        return dict(fibers=int(use.sum()),global_offset=shift,l1=float(e.mean()) if len(e) else None,
            within_two_windings_percent=float(100*np.mean(e<=2)) if len(e) else None,
            max=float(e.max()) if len(e) else None,
            length_weighted_l1=float(np.average(e,weights=lengths[use]*coverage[use])) if len(e) else None)
    report['spiral']=dict(all=metrics(np.ones(count,bool)),largest_component=metrics(solution['component']==largest),
        minimum_half_length_covered=metrics(coverage>=.5),
        median_coordinate_comparison=winding_error_metrics(fit,truth),
        matched_fibers=int(valid.sum()),matched_fiber_percent=float(100*valid.mean()),
        samples=sample_count,matched_samples=matched_count,
        covered_arc_length_percent=float(100*np.sum(lengths*coverage)/lengths.sum()),
        definition='Per-fiber arc-weighted median of pointwise reference-minus-fit; one global median offset, no component alignment',
        sample_spacing_base=spacing,maximum_projection_distance_base=maximum)
    np.savez(root/'fiber_reference.npz',source_id=ids,reference_minus_fit=delta,
        fitted=fit,reference=truth,coverage=coverage,component=solution['component'])
    report.update(stage='complete',seconds=time.monotonic()-started)
    write(root/'summary.json',report);write(root/'progress.json',dict(stage='complete'))
    print(json.dumps(report),flush=True)
    return report
