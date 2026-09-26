"""Calibration-only selection, then locked 6,000-voxel final rollouts.

Examples:
  python scripts/evaluate_single_path.py calibrate --manifest PREP/seeds.json --out EVAL --checkpoints RUN/ckpt_*.pt
  python scripts/evaluate_single_path.py final --manifest PREP/seeds.json --out FINAL --selection EVAL/selection.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from vesuvius.neural_tracing.fiber_follow.experiment import read_manifest
from vesuvius.neural_tracing.fiber_follow.recovery import recovery_counts
import numpy as np
import torch
from vesuvius.neural_tracing.fiber_follow.data import ZBand,fiber_manifest,load_fibers,split_fibers,label_state
from vesuvius.neural_tracing.fiber_follow.evaluate import evaluate
from vesuvius.neural_tracing.fiber_follow.experiment import jsonable,rollout_summary,paired_bootstrap
from vesuvius.neural_tracing.fiber_follow.history_audit import HistoryAudit
from vesuvius.neural_tracing.fiber_follow.supervision import prefix_labels
from vesuvius.neural_tracing.fiber_follow.trace import ModelTracer,TraceParams
from vesuvius.neural_tracing.fiber_follow.train import load_checkpoint
from vesuvius.neural_tracing.fiber_follow.volume import FiberVolume, FiberVolumeSpec


class RecoveryAudit(HistoryAudit):
    def __init__(self,tracer,tolerance):
        super().__init__(tracer,tolerance);self.rows=[]
    def __call__(self,index,state):
        if not self.active[index]:return
        col=self.collectors[index]
        if col(state) is False:
            self.active[index]=False;self.censored_traces+=1;return
        row=col.rows[-1]
        item=label_state(col.fiber,state['pos'],state['frame'],state['hist'],state['hmask'],self.cfg,
                         t=col.t,reverse=col.sign<0,offtrack=row['offtrack'])
        tensor=lambda a:torch.as_tensor(np.asarray(a),dtype=torch.float32)[None]
        batch={k:tensor(item[k]) for k in ('dense_ab','dense_mask','endpoint_known','end_local','offtrack')}
        target,mask,error=prefix_labels(tensor(state['points']),batch,self.tolerance,self.max_recovery_distance)
        h=min(3,target.shape[1]-1)
        self.rows.append(dict(drift=float(row['drift']),departed=bool(row['offtrack']),
            four_correct=bool(target[0,h]),four_known=bool(mask[0,h]),first_correct=bool(target[0,0]),
            first_known=bool(mask[0,0]),would_stop=bool(state['would_stop']),commit=int(state['n_commit']),
            error=float(error[0]),confidence4=float(state['confidence'][h]),fiber=col.fi,progress=float(col.t)))
        col.rows.clear();col.distances.clear()



def evaluation_sampling_seeds(sampler_mode, requested=None, selection=None):
    """Final repeats are locked with calibration, just like threshold/checkpoint."""
    if selection is not None:
        seeds=selection.get('sampling_seeds',[0])
        if requested is not None and requested!=seeds:
            raise ValueError('Final sampling seeds must match the locked calibration selection')
    else:
        seeds=requested if requested is not None else ([0,1,2] if sampler_mode=='gaussian' else [0])
    if not seeds or len(set(seeds))!=len(seeds) or any(s<0 or s>=2**63 for s in seeds):
        raise ValueError('Sampling seeds must be distinct integers in [0, 2**63)')
    return seeds


def main(argv=None, *, checkpoint_loader=None, model_tracer=None, volume_validator=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=('calibrate','final'))
    ap.add_argument('--checkpoints',nargs='+')
    ap.add_argument('--selection',type=Path)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--thresholds',type=float,nargs='+',default=(.5,.7,.8,.85,.9,.95,.98))
    ap.add_argument('--fibers',default='/mnt/raid_nvme/spiral_dataset_working/fibers')
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--batch',type=int,default=1)
    ap.add_argument('--sampling-seeds',type=int,nargs='+',
                    help='Gaussian default: 0 1 2; deterministic default: 0. Locked at calibration.')
    ap.add_argument('--baseline-rows',type=Path)
    ap.add_argument('--baseline-archive',type=Path,help='Archived v10 implementation, for baseline calibration/final rollouts only')
    args=ap.parse_args(argv);torch.set_num_threads(4)
    manifest=read_manifest(args.manifest);args.out.mkdir(parents=True,exist_ok=True)
    selection=None
    if args.mode=='final':
        if args.selection is None or args.checkpoints:
            ap.error('Final evaluation requires a calibration selection; checkpoint overrides are forbidden')
        selection=json.loads(args.selection.read_text())
        if selection['split']!='calibration' or selection['manifest_sha256']!=manifest['sha256']:
            raise ValueError('Selection was not made on this calibration manifest')
        checkpoint=Path(selection['checkpoint'])
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=selection['checkpoint_sha256']:
            raise ValueError('Selected checkpoint changed')
        if selection.get('baseline_archive') != (str(args.baseline_archive.resolve()) if args.baseline_archive else None):
            raise ValueError('Final baseline implementation must match calibration selection')
        choices=[(str(checkpoint),selection['threshold'])]
    else:
        if not args.checkpoints:ap.error('Calibration needs --checkpoints')
        choices=[(p,t) for p in args.checkpoints for t in args.thresholds]
    split='final' if args.mode=='final' else 'calibration';reports=[]
    for checkpoint,threshold in choices:
        tracer_class=model_tracer or ModelTracer
        if args.baseline_archive:
            from evaluate_recovery import load_archived_baseline,ArchivedTracer
            model,crop,nh,spec,ck=load_archived_baseline(checkpoint,args.baseline_archive,args.device)
            tracer_class=ArchivedTracer
            for key in ('fiber_zarr_dir','ct_zarr','grid_scale'):
                if spec.to_dict()[key]!=manifest['volume'][key]: raise ValueError('Baseline data source differs from manifest')
        else:
            model,crop,nh,spec,ck=(checkpoint_loader or load_checkpoint)(checkpoint,args.device)
            if volume_validator is not None:
                volume_validator(spec,manifest)
            elif spec!=FiberVolumeSpec(**manifest['volume']): raise ValueError('Volume differs from frozen manifest')
        _,val=split_fibers(load_fibers(args.fibers,grid_scale=spec.grid_scale),ZBand(45000/spec.grid_scale,48500/spec.grid_scale))
        if fiber_manifest(val)!=manifest['fibers']: raise ValueError('Geometry differs from frozen manifest')
        sampler_mode=getattr(model.cfg,'sampler_mode','zero')
        sampling_seeds=evaluation_sampling_seeds(sampler_mode,args.sampling_seeds,selection)
        tracer=tracer_class(model,FiberVolume(spec),crop,nh,TraceParams(max_len=6000,confidence=threshold,
            n_commit=ck.get('n_commit',8)),device=args.device)
        start=time.monotonic();rows=[];audit_rows=[];seed_summaries=[]
        try:
            for sampling_seed in sampling_seeds:
                tracer.p.seed=sampling_seed
                audit=RecoveryAudit(tracer,ck.get('tolerance',1.5))
                repeated,_=evaluate(tracer,val,manifest[split],batch=args.batch,history_audit=audit)
                for r in repeated:
                    r['avail']=min(r['avail'],6000);r['followed']=min(r['followed'],6000)
                    r['coverage']=r['followed']/max(r['avail'],1e-6)
                    r['sampling_seed']=sampling_seed
                rows.extend(repeated)
                audit_rows.extend(dict(r,sampling_seed=sampling_seed) for r in audit.rows)
                seed_summaries.append(dict(sampling_seed=sampling_seed,**rollout_summary(repeated)))
        finally:tracer.close()
        report=rollout_summary(rows)
        report.update(checkpoint=str(Path(checkpoint).resolve()),checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            manifest_sha256=manifest['sha256'],split=split,threshold=threshold,max_len=6000,
            architecture=ck.get('architecture'),vol_spec=spec.to_dict(),n_commit=tracer.p.n_commit,
            sampler_mode=sampler_mode,sampling_seeds=sampling_seeds,sampling_seed_summaries=seed_summaries,
            recovery=recovery_counts(audit_rows),seconds=time.monotonic()-start,step=ck['step'],
            baseline_archive=str(args.baseline_archive.resolve()) if args.baseline_archive else None)
        report['coverage_at_95_scored_precision']=report['length_weighted_coverage'] if report['length_precision']>=.95 else None
        if args.baseline_rows:
            base_rows=json.loads(args.baseline_rows.read_text())
            base=rollout_summary(base_rows)
            report['paired_fiber_bootstrap_delta_95ci']=paired_bootstrap(base_rows,rows)
            gain=report['length_weighted_coverage']/max(base['length_weighted_coverage'],1e-12)-1
            report['acceptance_comparison']=dict(relative_coverage_gain=gain,
                both_scored_precision_ge_95=min(base['length_precision'],report['length_precision'])>=.95,
                coverage_gain_ge_20_percent=gain>=.2,wrong_length_delta=report['wrong_length']-base['wrong_length'],
                divergence_delta=report['diverged']-base['diverged'],
                baseline_wrong_continuation_quantiles=base['wrong_continuation_quantiles'],
                current_wrong_continuation_quantiles=report['wrong_continuation_quantiles'])
        name=f'{Path(checkpoint).parent.name}_{Path(checkpoint).stem}_c{threshold:.3f}'
        if sampling_seeds!=[0]: name+='_s'+'-'.join(map(str,sampling_seeds))
        for suffix,values in (('',report),('_rows',rows),('_decisions',audit_rows)):
            (args.out/(name+suffix+'.json')).write_text(json.dumps(values,indent=2,default=jsonable))
        reports.append(report);print(json.dumps({k:report[k] for k in ('checkpoint','threshold','length_precision','length_weighted_coverage','wrong_length')}),flush=True)
    if args.mode=='calibrate':
        eligible=[r for r in reports if r['length_precision']>=.95 and r['scored_length']>0]
        if not eligible: raise RuntimeError('No calibration checkpoint/threshold reaches 95% scored precision; no selection written')
        selected=max(eligible,key=lambda r:r['length_weighted_coverage'])
        path=args.out/'selection.json'
        if path.exists(): raise FileExistsError('Selection already locked; use a new output directory')
        path.write_text(json.dumps(selected,indent=2))

if __name__=='__main__':main()
