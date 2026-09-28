"""Check real full-size axial batches, optional memory, and model gradients."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from vesuvius.neural_tracing.fiber_follow.regression.model import DirectConfig,DirectFollower,crop_support
from vesuvius.neural_tracing.fiber_follow.regression.data import IdentityObservationBuilder,IdentitySampling
from vesuvius.neural_tracing.fiber_follow.regression.neighbor_bank import NeighborBank
from vesuvius.neural_tracing.fiber_follow.regression.supervision import loss_terms,memory_probe_terms
from vesuvius.neural_tracing.fiber_follow.regression.train import move_batch,conv_memory_format,compile_training_model
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
    ap.add_argument('--compile',action='store_true')
    ap.add_argument('--memory-slots',type=int,default=0)
    ap.add_argument('--memory-version',type=int,choices=(2,3),default=3)
    ap.add_argument('--activation-checkpointing',action='store_true')
    ap.add_argument('--memory-steps',type=int,default=32)
    ap.add_argument('--memory-stride',type=int,default=4)
    ap.add_argument('--memory-patch-size',type=int,default=17)
    ap.add_argument('--memory-grad-steps',type=int,default=32)
    ap.add_argument('--memory-switch-probability',type=float,default=0.)
    ap.add_argument('--memory-switch-tail',type=float,nargs=2,default=(16.,96.))
    ap.add_argument('--onpolicy',nargs='*',default=[],help='Replay caches, e.g. collected with observed tracks')
    args=ap.parse_args(argv)
    torch.set_num_threads(4);torch.manual_seed(0)
    cfg=DirectConfig(memory_slots=args.memory_slots,memory_steps=args.memory_steps,
                     memory_stride=args.memory_stride,memory_patch_size=args.memory_patch_size,
                     memory_grad_steps=args.memory_grad_steps,memory_version=args.memory_version,
                     activation_checkpointing=args.activation_checkpointing)
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
    ds=FollowDataset(fibers,spec,sample,band,chunk=args.microbatch,seed=37,batch_builder=builder,
                     onpolicy=[OnPolicyStates.load(p) for p in args.onpolicy])
    it=iter(ds);args.out.mkdir(parents=True,exist_ok=True)
    model=DirectFollower(cfg).to(args.device,memory_format=conv_memory_format(args.device)) if args.forward else None
    if model is not None and args.compile:
        model=compile_training_model(model)
    rows=[]
    for index in range(args.batches):
        started=time.perf_counter();cpu=next(it)
        assert {'fine','seed','seed_mask','seed_age','seed_tangent'} <= set(cpu['x'])
        if cfg.memory_slots:
            assert cpu['x']['memory_mask'][:,-1].all()
            assert torch.isfinite(cpu['x']['history_crops' if cfg.memory_version == 3 else 'memory_patches']).all()
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
                if 'memory_probe' in out:
                    probe=memory_probe_terms(out,b)
                    loss=loss+.5*(probe['memory_identity_per_state'].mean()+probe['memory_offset_per_state'].mean())
            assert torch.isfinite(loss)
            loss.backward()
            for name,param in model.named_parameters():
                if param.grad is not None:assert torch.isfinite(param.grad).all(),name
            assert model.encoder.compress.weight.grad.abs().sum()>0
            row.update(loss=float(loss.detach()),identity_pairs=int(terms['identity_count']))
            if args.device.startswith('cuda'):
                torch.cuda.synchronize()
                row.update(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                           peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            row['total_seconds'] = time.perf_counter()-started
        rows.append(row);print(json.dumps(row),flush=True)
    report=dict(config=cfg.to_dict(),sampling=asdict(sampling),bank=str(bank.root),bank_shards=bank.shard_count,rows=rows)
    (args.out/'report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
