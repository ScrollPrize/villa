"""Check real full-size axial batches, optional memory, and model gradients."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from vesuvius.neural_tracing.fiber_follow.regression.model import DirectConfig,build_model,crop_support
from vesuvius.neural_tracing.fiber_follow.regression.data import IdentityObservationBuilder,IdentitySampling
from vesuvius.neural_tracing.fiber_follow.regression.neighbor_bank import NeighborBank
from vesuvius.neural_tracing.fiber_follow.regression.supervision import loss_terms,memory_probe_terms
from vesuvius.neural_tracing.fiber_follow.regression.train import move_batch,conv_memory_format
from vesuvius.neural_tracing.fiber_follow.shared.components import ComponentRule
from vesuvius.neural_tracing.fiber_follow.shared.data import FollowDataset,OnPolicyStates,SampleConfig,ZBand,load_fibers,split_fibers
from vesuvius.neural_tracing.fiber_follow.shared.volume import FiberVolumeSpec


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fibers',default='/mnt/raid_nvme/spiral_dataset_working/fibers')
    ap.add_argument('--fiber-zarrs',default='/mnt/raid_nvme/spiral_dataset_working/fiber_zarrs')
    ap.add_argument('--ct',default='/mnt/raid_nvme/volpkgs/s1_2um_ds2.volpkg/volumes/s1_ds2.zarr')
    ap.add_argument('--bank',default=str(Path(__file__).parents[1]/'output/neighbor_samples_r0_32_l80_160_v2'))
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--microbatch',type=int,default=4)
    ap.add_argument('--batches',type=int,default=4)
    ap.add_argument('--forward',action='store_true')
    ap.add_argument('--direction-inputs',action='store_true')
    ap.add_argument('--memory-version',type=int,choices=(2,3,4,5),default=2)
    ap.add_argument('--memory-slots',type=int,default=0)
    ap.add_argument('--memory-steps',type=int,default=32)
    ap.add_argument('--memory-stride',type=int,default=4)
    ap.add_argument('--memory-patch-size',type=int,default=17)
    ap.add_argument('--memory-grad-steps',type=int,default=32)
    ap.add_argument('--memory-switch-probability',type=float,default=0.)
    ap.add_argument('--memory-switch-tail',type=float,nargs=2,default=(16.,96.))
    ap.add_argument('--onpolicy',nargs='*',default=[],help='Replay caches, e.g. collected with observed tracks')
    args=ap.parse_args(argv)
    torch.set_num_threads(4);torch.manual_seed(0)
    cfg=DirectConfig(direction_inputs=args.direction_inputs,memory_version=args.memory_version,memory_slots=args.memory_slots,memory_steps=args.memory_steps,
                     memory_stride=args.memory_stride,memory_patch_size=args.memory_patch_size,
                     memory_grad_steps=args.memory_grad_steps,correction=args.memory_version not in (4,5),
                     feature_memory_revision=2 if args.memory_version == 5 else 1,
                     recurrent_refinement_steps=2 if args.memory_version == 5 else 0)
    spec=FiberVolumeSpec(args.fiber_zarrs,ct_zarr=args.ct,ct_level=0,ct_grid_scale=4.,inputs='ct+presence')
    band=ZBand(45000/spec.grid_scale,48500/spec.grid_scale)
    fibers,_=split_fibers(load_fibers(args.fibers,grid_scale=spec.grid_scale),band)
    bank=NeighborBank(args.bank,fibers,band,grid_scale=spec.grid_scale)
    bank.validate_volume(spec)
    sampling=IdentitySampling(rule=ComponentRule(lateral_max=32.),decision_fraction=.5,
        negative_near_fraction=.5,negative_near_distance=12.,bank_coverage_probability=.2,bank_following_probability=.1,
        memory_switch_probability=args.memory_switch_probability,memory_switch_tail=tuple(args.memory_switch_tail))
    builder=IdentityObservationBuilder(cfg,fibers,sampling,negative_bank=bank,augment=True)
    sample=SampleConfig(crop=cfg.fine,n_history=cfg.n_history,n_future=cfg.n_future,recent_history_points=cfg.n_history)
    chunk = args.microbatch
    if cfg.feature_memory:
        if args.microbatch % (2*cfg.feature_sequence_length):
            raise ValueError('V4 preflight microbatch must contain an even number of full sequence streams')
        chunk //= cfg.feature_sequence_length
    ds=FollowDataset(fibers,spec,sample,band,chunk=chunk,seed=37,batch_builder=builder,
                     onpolicy=[OnPolicyStates.load(p) for p in args.onpolicy])
    it=iter(ds);args.out.mkdir(parents=True,exist_ok=True)
    model=build_model(cfg).to(args.device,memory_format=conv_memory_format(args.device)) if args.forward else None
    if cfg.feature_memory:
        import copy
        from .feature_sequences import FeatureStreamStates
        from .train import optimizer_update
        states = FeatureStreamStates()
        ema = copy.deepcopy(model) if model is not None else None
        opt = torch.optim.AdamW(model.parameters(), lr=0.) if model is not None else None
    rows=[]
    for index in range(args.batches):
        started=time.perf_counter();cpu=next(it)
        if cfg.feature_memory:
            steps = cpu['feature_sequence']
            for batch in steps:
                assert 'memory_patches' not in batch['x'] and 'feature_seed_x' not in batch['x']
                assert torch.isfinite(batch['x']['fine']).all()
                assert all(torch.isfinite(batch[k]).all() for k in ('memory_target_identity','memory_target_offset'))
            if index == 0:
                torch.save(cpu, args.out/'batch.pt')
            row = dict(batch=index, read_seconds=time.perf_counter()-started,
                       states=sum(len(b['hist']) for b in steps), decisions=len(steps))
            if model is not None:
                metrics = optimizer_update(model, ema, opt, [cpu], index+1, 0., device=args.device,
                                           compute_metrics=False, stream_states=states)
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                assert model.encoder.compress.weight.grad.abs().sum() > 0
                row.update(loss=metrics['loss'], retained_streams=len(states.states))
            rows.append(row)
            print(json.dumps(row), flush=True)
            continue
        assert {'fine','seed','seed_mask','seed_age','seed_tangent'} <= set(cpu['x'])
        if cfg.memory_slots:
            assert cpu['x']['memory_mask'][:,-1].all()
            assert torch.isfinite(cpu['x']['memory_patches']).all()
            if cfg.memory_version >= 2:
                assert all(torch.isfinite(cpu[k]).all() for k in ('memory_target_identity','memory_target_offset'))
        valid=torch.cat((cpu['positive_mask'],cpu['negative_mask'].flatten(1)),1).bool()
        assert crop_support(cpu['identity_points'],cfg.fine)[valid].all()
        if index==0:torch.save(cpu,args.out/'batch.pt')
        row=dict(batch=index,read_seconds=time.perf_counter()-started,states=len(cpu['hist']),
            matched=int((cpu['source']==5).sum()),visible_seeds=int(cpu['seed_present'].sum()),
            observable=int(cpu['identity_observable'].sum()),positive_pairs=int(cpu['positive_mask'].sum()),
            negative_pairs=int(cpu['negative_mask'].sum()))
        if cfg.memory_slots:
            row.update(memory_observations=int(cpu['x']['memory_mask'].sum()),
                       memory_anchors=int(cpu['x']['memory_seed_valid'].sum()),
                       memory_labeled_writes=int(cpu['memory_target_identity_mask'].sum()),
                       memory_departed_writes=int((cpu['memory_target_identity_mask'] & (cpu['memory_target_identity'] < .5)).sum()),
                       memory_switch=int((cpu['location_source'] == 7).sum()),recent_replay=int((cpu['source'] == 2).sum()))
        if model is not None:
            b=move_batch(cpu,args.device);model.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=args.device.startswith('cuda')):
                out=model(b['x'],b['hist'],b['hmask'],queries=b['identity_points'],candidates=b['candidate_points'])
                terms=loss_terms(out,b,cfg)
                loss=terms['geometry_per_state'].mean()+.5*terms['confidence_per_state'].mean()+.5*terms['identity_per_state'].mean()+terms['candidate_per_state'].mean()
                if 'route_per_state' in terms:
                    loss=loss+cfg.route_loss_weight*terms['route_per_state'].mean()
                if 'memory_probe' in out:
                    probe=memory_probe_terms(out,b)
                    loss=loss+.5*(probe['memory_identity_per_state'].mean()+probe['memory_offset_per_state'].mean())
            assert torch.isfinite(loss)
            loss.backward()
            for name,param in model.named_parameters():
                if param.grad is not None:assert torch.isfinite(param.grad).all(),name
            assert model.encoder.compress.weight.grad.abs().sum()>0
            row.update(loss=float(loss.detach()),identity_pairs=int(terms['identity_count']))
        rows.append(row);print(json.dumps(row),flush=True)
    report=dict(config=cfg.to_dict(),sampling=asdict(sampling),bank=str(bank.root),bank_shards=bank.shard_count,rows=rows)
    (args.out/'report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
