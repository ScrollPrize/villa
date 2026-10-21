"""Frozen-solution cross-constraint votes against intact reviewed GT fibers."""
import json
from pathlib import Path
import sqlite3

import numpy as np
from lasagna.inference_provenance import atomic_write, sha256_file
from . import fiber_uv_layout as uv


def arrays(path):
    with np.load(path) as src:return {k:src[k] for k in src.files}


def offset_votes(required):
    """One additive test-layout offset; ties round toward positive infinity."""
    required=np.asarray(required,dtype=float)
    if not len(required) or not np.isfinite(required).all():raise ValueError('No finite winding votes')
    votes=np.floor(required+.5).astype(np.int64)
    offsets,counts=np.unique(votes,return_counts=True)
    candidates=offsets[counts==counts.max()]
    offset=min(candidates,key=lambda x:(float(np.abs(required-x).mean()),abs(int(x)),int(x)))
    return int(offset),votes==offset,votes


def piece_at(d,source,arc):
    choices=np.flatnonzero((d['source_id']==source)&(d['start']<=arc)&(d['end']>=arc))
    # At a shared cut take the later piece, just as native preparation does.
    return int(choices[np.argmax(d['start'][choices])]) if len(choices) else None


def coarse_at(d,state,pick,piece,angle,k):
    anchor_turn=np.rint(float(d['node_phase'][piece,0])-float(d['phase'][piece]))
    return float(pick[piece]+angle-anchor_turn+state[piece,k+1])


def event_residuals(events,d,state,pick,k,gd,gs,gpick,config=None,gt_config=None,widths=None,saved_geometry=None):
    """Required additive test winding correction, shared by both GT metrics."""
    if config and config.get('soft_winding_model',False):
        import torch
        from .fiber_soft_winding import points
        from .fiber_uv_arc_phase import event_arc_points,event_signed_arcs
        if widths is None:raise ValueError('Soft winding GT metric needs fitted widths')
        ids=np.array([e['test_piece'] for e in events])
        ds=event_signed_arcs(d,ids,np.array([e['test_arc'] for e in events]))
        if config.get('shared_curve_geometry'):
            from .fiber_shared_curve import saved_points
            if saved_geometry is None:raise ValueError('Shared-curve GT scoring requires saved geometry')
            q=saved_points(d,saved_geometry,state,widths,ids,ds,
                shared_only=config.get('gt_coordinate_layer')=='predicted')[0]
        else:
          with torch.no_grad():
            q=points(torch.as_tensor(state[ids],dtype=torch.float64),
                torch.as_tensor(d['family'][ids]),torch.as_tensor(ds,dtype=torch.float64),
                torch.as_tensor(widths,dtype=torch.float64))[0].numpy()
        gq=event_arc_points(gd,gs['state'],gpick,gs['widths'],
            np.array([e['gt_piece'] for e in events]),np.array([e['gt_arc'] for e in events]))
        return gq-q+.5*np.array([e['family_delta'] for e in events])-np.array([e['signed_distance'] for e in events])
    if config and config.get('unified_winding_loss',False):
        from .fiber_uv_arc_phase import event_arc_points
        if widths is None:raise ValueError('Unified winding GT metric needs fitted widths')
        def read(meta,st,pk,w,prefix):
            return event_arc_points(meta,st,pk,w,np.array([e[prefix+'_piece'] for e in events]),
                                    np.array([e[prefix+'_arc'] for e in events]))
        return (read(gd,gs['state'],gpick,gs['widths'],'gt')-read(d,state,pick,widths,'test')
                +.5*np.array([e['family_delta'] for e in events])-np.array([e['signed_distance'] for e in events]))
    if config and config.get('arc_phase_constraints',False):
        from .fiber_uv_arc_phase import event_arc_points
        from .fiber_uv_integer_sheets import target
        if widths is None:raise ValueError('Arc-phase GT metric needs fitted widths')
        sd=np.array([e['signed_distance'] for e in events]);fd=np.array([e['family_delta'] for e in events])
        par=np.array([e['parallel'] for e in events],dtype=bool)
        def read(meta,st,pk,w,prefix):
            return event_arc_points(meta,st,pk,w,np.array([e[prefix+'_piece'] for e in events]),
                                    np.array([e[prefix+'_arc'] for e in events]))
        return read(gd,gs['state'],gpick,gs['widths'],'gt')-read(d,state,pick,widths,'test')-target(
            abs(sd),np.sign(sd),fd,par,config.get('perpendicular_layer_offset',.5))
    if config and config.get('integer_sheet_constraints',False):
        from .fiber_uv_integer_sheets import target
        if not (gt_config or {}).get('integer_sheet_constraints',False):
            gpick=gpick+np.rint(gs['state'][:,len(gs['widths'])+1]).astype(np.int64)
        if config.get('legacy_integer_score_conversion',False):
            pick=pick+np.rint(state[:,k+1]).astype(np.int64)
        sd=np.array([e['signed_distance'] for e in events]);fd=np.array([e['family_delta'] for e in events])
        par=np.array([e['parallel'] for e in events],dtype=bool)
        targets=target(abs(sd),np.sign(sd),fd,par,config.get('perpendicular_layer_offset',.5))
        # Trusted GT's selected winding labels retained; remove both saved phase
        # adjustments because new coarse integer topology contains no such term.
        raw=np.asarray([coarse_at(gd,gs['state'],gpick,e['gt_piece'],e['gt_angle'],len(gs['widths']))
            -gs['state'][e['gt_piece'],len(gs['widths'])+1]
            -coarse_at(d,state,pick,e['test_piece'],e['test_angle'],k)+state[e['test_piece'],k+1]
            -e['phase_delta'] for e in events])
        return np.rint(raw)-targets
    return np.asarray([
        coarse_at(gd,gs['state'],gpick,e['gt_piece'],e['gt_angle'],len(gs['widths']))
        -coarse_at(d,state,pick,e['test_piece'],e['test_angle'],k)
        -e['phase_delta']+.5*e['family_delta']-e['signed_distance']
        for e in events],dtype=float)


def enabled_l1(required,enabled,parallel):
    """Equal enabled-event L1, with a single L1-optimal integer gauge."""
    required=np.asarray(required,dtype=float)
    enabled=np.asarray(enabled,dtype=bool);parallel=np.asarray(parallel,dtype=bool)
    if required.shape!=enabled.shape or parallel.shape!=enabled.shape:
        raise ValueError('Event masks must match residuals')
    values=np.sort(required[enabled])
    if not len(values) or not np.isfinite(required).all():
        raise ValueError('No finite enabled GT constraints')
    # Every real point in the median interval minimizes L1. If it contains
    # integers, choose the one closest to zero; otherwise test its neighbors.
    lo,hi=values[(len(values)-1)//2],values[len(values)//2]
    first,last=int(np.ceil(lo)),int(np.floor(hi))
    if first<=last:
        offset=min(max(0,first),last)
    else:
        offset=min((int(np.floor(lo)),int(np.ceil(hi))),
                   key=lambda x:(float(np.abs(values-x).mean()),abs(x),x))
    error=np.abs(required-offset)
    correct=np.floor(required+.5)==offset
    def part(mask):
        return dict(events=int(mask.sum()),l1=float(error[mask].mean()) if mask.any() else None,
            correct=int(correct[mask].sum()),correct_percent=float(100*correct[mask].mean()) if mask.any() else None)
    return dict(gt_enabled_l1=float(error[enabled].mean()),gt_global_offset=offset,
        gt_enabled_events=int(enabled.sum()),gt_total_events=len(required),
        gt_enabled_percent=float(100*enabled.mean()),gt_parallel=part(enabled&parallel),
        gt_perpendicular=part(enabled&~parallel),gt_disabled=part(~enabled),
        gt_enabled_correct=int(correct[enabled].sum()),gt_enabled_correct_percent=float(100*correct[enabled].mean()))


def gt_solve_width_metrics(widths, gt_widths, gt_intervals, global_offset):
    """Equal-weight occupied GT intervals; align test q + offset = GT q."""
    intervals=np.unique(np.asarray(gt_intervals,dtype=np.int64))
    test=intervals-int(global_offset)
    valid=(intervals>=0)&(intervals<len(gt_widths))&(test>=0)&(test<len(widths))
    intervals=intervals[valid];test=test[valid]
    fit=np.asarray(widths)[test];reference=np.asarray(gt_widths)[intervals]
    if np.any(reference<=0) or not np.isfinite(reference).all():raise ValueError('Invalid GT solve widths')
    ratios=fit/reference
    return dict(global_offset=int(global_offset),compared_windings=len(intervals),
        gt_intervals=intervals.tolist(),test_intervals=test.tolist(),
        fitted_widths=fit.tolist(),gt_solve_widths=reference.tolist(),
        mean_normalized_factor=float(ratios.mean()) if len(ratios) else None,
        mean_absolute_relative_error_percent=float(100*np.abs(ratios-1).mean()) if len(ratios) else None,
        mean_absolute_error_base_vx=float(np.abs(fit-reference).mean()) if len(ratios) else None,
        max_absolute_error_base_vx=float(np.abs(fit-reference).max()) if len(ratios) else None,
        weighting='equal GT winding intervals occupied by frozen cross-event base points')


class FrozenGTScorer:
    """Replay cached cross events without extracting geometry for each trial."""
    def __init__(self,spec,config):
        root=Path(spec['root']);gtroot=Path(spec['gt_root'])
        self.paths=[root/'inputs.json',root/spec['events'],gtroot/'uv_whole/config.json',
                    gtroot/'uv_whole/solution.npz',Path(config['tensor_cache'])]
        provenance=json.loads(self.paths[0].read_text())
        self.gcfg=json.loads(self.paths[2].read_text())
        self.paths.append(Path(self.gcfg['tensor_cache']))
        for key,path in [('tensors',self.paths[4]),('gt_tensors',self.paths[5]),
                         ('gt_solution',self.paths[3])]:
            if sha256_file(path)!=provenance[key]:raise ValueError('Changed GT scoring '+key)
        if (not (config.get('bucket_shared_phase') or config.get('soft_winding_model'))
                or not self.gcfg.get('bucket_shared_phase') or config['h_convention']!=self.gcfg['h_convention']):
            raise ValueError('GT scoring requires matching corrected coarse convention')
        self.d=arrays(self.paths[4]);self.gd=arrays(self.paths[5]);self.gs=arrays(self.paths[3])
        self.events=json.loads(self.paths[1].read_text())
        _,_,keep,self.gpick=uv.hard_assignment(self.gs['state'],len(self.gs['widths']),self.gcfg,best_nonbroken=True)
        if not keep.all():raise ValueError('Trusted GT must remain enabled')
        for e in self.events:
            if int(self.d['source_id'][e['test_piece']])!=e['test_source']:
                raise ValueError('Cached event piece identity mismatch')
            if int(self.gd['source_id'][e['gt_piece']])!=e['gt_source']:
                raise ValueError('Cached GT event piece identity mismatch')
        self.hashes={str(p):sha256_file(p) for p in self.paths}

    def score(self,solution,config):
        s=arrays(solution);k=len(s['widths'])
        for field in ('source_id','start','end','family'):
            if not np.array_equal(s[field],self.d[field]):raise ValueError('Changed scored solution '+field)
        result=self.score_state(s['state'],s['widths'],config,saved_geometry=s)
        from .fiber_uv_arc_phase import event_arc_points
        q=event_arc_points(self.gd,self.gs['state'],self.gpick,self.gs['widths'],
            np.array([e['gt_piece'] for e in self.events]),np.array([e['gt_arc'] for e in self.events]))
        if result['gt_global_offset'] is not None:
            result['gt_solve_widths']=gt_solve_width_metrics(s['widths'],self.gs['widths'],
                np.floor(q).astype(np.int64),result['gt_global_offset'])
        return result

    def score_state(self,state,widths,config,saved_geometry=None):
        k=len(widths)
        if config.get('soft_winding_model'):
            if state.shape!=(len(self.d['family']),4):raise ValueError('Invalid soft state shape')
            keep=state[:,3]>=0;pick=None
            if config.get('implicit_winding_map'):
                state=state.copy()
                state[:,0]=np.floor(state[:,0]+.5)
        else:
            _,_,keep,pick=uv.hard_assignment(state,k,config,best_nonbroken=True)
        required=event_residuals(self.events,self.d,state,pick,k,self.gd,self.gs,
                               self.gpick,config,self.gcfg,widths,saved_geometry)
        return enabled_l1(required,[keep[e['test_piece']] for e in self.events],
                          [e['parallel'] for e in self.events])


def run(settings):
    if settings.get('frozen_event_cache'):
        # Reuse extracted events for arbitrary matched saved solves, without
        # rerunning extraction or mutating the frozen event-cache provenance.
        configs=[]
        for path in settings['configs']:
            source=json.loads(Path(path).read_text())
            if 'fiber_implicit_winding' in source:
                config=dict(source['fiber_implicit_winding'],soft_winding_model=True,
                            implicit_winding_map=True)
            elif 'fiber_soft_winding' in source:
                config=dict(source['fiber_soft_winding'],soft_winding_model=True)
            else:
                config=dict(source.get('fiber_uv_layout',source))
            if settings.get('metric_geometry')=='continuous_arc':
                config['unified_winding_loss']=True
            elif settings.get('metric_geometry') is not None:
                raise ValueError('Unsupported metric_geometry')
            configs.append(config)
        scorer=FrozenGTScorer(settings['frozen_event_cache'],configs[0])
        results=[]
        for config in configs:
            if config['tensor_cache']!=configs[0]['tensor_cache'] or config['h_convention']!=configs[0]['h_convention']:
                raise ValueError('Frozen scoring requires matching geometry')
            solution=Path(config['output_root'])/'solution.npz'
            row=dict(output_root=config['output_root'],solution_sha256=sha256_file(solution),
                     winding_estimator='MAP integer plus phase' if config.get('implicit_winding_map') else 'continuous/baseline',
                     metric_geometry=settings.get('metric_geometry','configured'),
                     retention_rule='enabled_logit>=0' if config.get('soft_winding_model') else 'saved_bucket_hard_assignment',
                     **scorer.score(solution,config))
            results.append(row);print(json.dumps(row),flush=True)
        root=Path(settings['output_root']);root.mkdir(parents=True,exist_ok=True)
        atomic_write(root/'results.json',dict(results=results,input_hashes=scorer.hashes),durable=False)
        return results
    raise ValueError('Scoring requires an explicit frozen_event_cache')
