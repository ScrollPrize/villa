"""CPU preflight: real source crops, labels and CT-only forward/backward.

Reads only requested remote chunks. Does not launch or modify training.
"""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.regression.datasets import read_dataset_config, build_mixed_dataset,load_primary_dataset,HoldoutFilteredBank
from vesuvius.neural_tracing.fiber_follow.regression.data import IdentityObservationBuilder, IdentitySampling
from vesuvius.neural_tracing.fiber_follow.regression.model import DirectConfig, build_model
from vesuvius.neural_tracing.fiber_follow.regression.supervision import loss_terms
from vesuvius.neural_tracing.fiber_follow.shared.data import FollowDataset, SampleConfig, ZBand
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec
from vesuvius.neural_tracing.fiber_follow.shared.volume import FiberVolumeSpec


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset-config',default=str(Path(__file__).resolve().parents[1]/'configs/mixed_ct_datasets.json'))
    ap.add_argument('--out',default=str(Path(__file__).resolve().parents[1]/'datasets/automated_fiber_volumes/training_preflight.json'))
    ap.add_argument('--prefetch-connections',type=int,default=0)
    args=ap.parse_args()
    if args.prefetch_connections < 0:raise ValueError('Prefetch connections must be nonnegative')
    torch.set_num_threads(2);torch.manual_seed(7349)
    document,digest=read_dataset_config(args.dataset_config)
    cfg=DirectConfig(input_mode='ct',direction_inputs=False,encoder='patch4',token_only=True,
        fine=CropSpec(depth=48,width=33,behind=24),n_future=4,n_history=32,
        hidden=16,heads=2,channels=4,layers=1,decoder_layers=1,recurrent_refinement_steps=1)
    sampling=IdentitySampling(presence_dropout=0.,decision_fraction=.3,bank_following_probability=.2,
        bank_coverage_probability=.2,memory_switch_probability=.3)
    sample=SampleConfig(crop=cfg.fine,n_future=4,n_history=32,recent_history_points=32,
        full_observed_history=True,no_history_prob=.15,short_history_prob=.4)
    source=next(s for s in document['sources'] if s['kind']=='paris4')
    spec=FiberVolumeSpec(source['fiber_zarrs'],ct_zarr=source['ct'],ct_level=source['ct_level'],
        ct_grid_scale=source['ct_grid_scale'],grid_scale=source['grid_scale'],inputs='ct',load_presence=False)
    band=ZBand(*(v/spec.grid_scale for v in source['val_z']))
    print('Loading Paris 4 annotations',flush=True)
    _,fibers,heldout,manifest=load_primary_dataset(document,spec)
    bank=HoldoutFilteredBank(source['negative_bank'],fibers,band,grid_scale=spec.grid_scale,heldout=heldout)
    bank.validate_volume(spec)
    primary=FollowDataset(fibers,spec,sample,None,chunk=2,seed=17,cache_bytes=64<<20,
        batch_builder=IdentityObservationBuilder(cfg,fibers,sampling,augment=True,negative_bank=bank))
    options=SimpleNamespace(microbatch=2,worker_cache_gb=.0625,fresh_fraction=.7)
    mixed,provenance=build_mixed_dataset(primary,document,cfg,sample,sampling,options,seed=17)
    splits=Path(args.out).parent/'splits'
    splits.mkdir(parents=True,exist_ok=True)
    for name,dataset in zip(mixed.names,mixed.datasets):
        validation=getattr(dataset,'validation_fibers',heldout)
        value=dict(dataset_config_sha256=digest,training_fibers=len(dataset.fibers),
            validation_fibers=len(validation),
            validation_ids=sorted(validation.ids) if hasattr(validation,'ids') else [f.name for f in validation],
            manifest=getattr(dataset,'validation_manifest',manifest))
        (splits/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
    prefetch_stats=None
    with ExitStack() as stack:
        service=None
        if args.prefetch_connections:
            from vesuvius.neural_tracing.fiber_follow.shared.remote_prefetch import RemotePrefetcher
            service=stack.enter_context(RemotePrefetcher(args.prefetch_connections))
            for dataset in mixed.datasets:
                if dataset.vol_spec.ct_zarr.startswith(('s3://','http://','https://')):
                    dataset.remote_prefetch=service.client
        results=[]
        for name,dataset in zip(mixed.names,mixed.datasets):
            print('Reading',name,flush=True)
            data=next(iter(dataset));model=build_model(cfg)
            kwargs={'candidates':data['candidate_points']} if 'candidate_points' in data and data['candidate_mask'].any() else {}
            output=model(data['x'],data['hist'],data['hmask'],**kwargs)
            terms=loss_terms(output,data,cfg)
            loss=terms['geometry_per_state'].mean()+.5*terms['confidence_per_state'].mean()
            if 'candidate_per_state' in terms:loss=loss+terms['candidate_per_state'].mean()
            loss.backward()
            assert torch.isfinite(loss) and data['x']['fine'].shape[1]==1
            assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
            row=dict(name=name,shape=list(data['x']['fine'].shape),loss=float(loss.detach()),
                valid_history_slabs=int(data['x']['history_valid'].sum()),sources=data['source'].tolist(),
                foreign_voxels=int(data['foreign'].sum()),forward_backward='passed')
            print(json.dumps(row),flush=True);results.append(row)
        if service:prefetch_stats=service.snapshot()
    report=dict(dataset_config_sha256=digest,model_cfg=cfg.to_dict(),sampling=asdict(sampling),
                probabilities=mixed.weights.tolist(),datasets=provenance,checks=results,
                remote_prefetch=prefetch_stats)
    Path(args.out).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
