"""Check real full-size axial batches, observation memory, and model gradients."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from vesuvius.neural_tracing.fiber_follow.regression.model import DirectConfig,build_model,crop_support
from vesuvius.neural_tracing.fiber_follow.regression.data import IdentityObservationBuilder,IdentitySampling
from vesuvius.neural_tracing.fiber_follow.regression.neighbor_bank import NeighborBank
from vesuvius.neural_tracing.fiber_follow.regression.supervision import loss_terms
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
    ap.add_argument('--memory-slots',type=int,default=DirectConfig.memory_slots)
    ap.add_argument('--memory-steps',type=int,default=DirectConfig.memory_steps)
    ap.add_argument('--memory-stride',type=int,default=DirectConfig.memory_stride)
    ap.add_argument('--memory-switch-probability',type=float,default=0.)
    ap.add_argument('--memory-switch-tail',type=float,nargs=2,default=(16.,96.))
    ap.add_argument('--onpolicy',nargs='*',default=[],help='Replay caches, e.g. collected with observed tracks')
    args=ap.parse_args(argv)
    torch.set_num_threads(4);torch.manual_seed(0)
    cfg=DirectConfig(direction_inputs=args.direction_inputs,memory_slots=args.memory_slots,
                     memory_steps=args.memory_steps,memory_stride=args.memory_stride)
    spec=FiberVolumeSpec(args.fiber_zarrs,ct_zarr=args.ct,ct_level=0,ct_grid_scale=4.,inputs='ct+presence')
    band=ZBand(45000/spec.grid_scale,48500/spec.grid_scale)
    fibers,_=split_fibers(load_fibers(args.fibers,grid_scale=spec.grid_scale),band)
    bank=NeighborBank(args.bank,fibers,band,grid_scale=spec.grid_scale)
    bank.validate_volume(spec)
    sampling=IdentitySampling(rule=ComponentRule(lateral_max=32.),decision_fraction=.5,
        bank_coverage_probability=.2,bank_following_probability=.1,
        memory_switch_probability=args.memory_switch_probability,memory_switch_tail=tuple(args.memory_switch_tail))
    builder=IdentityObservationBuilder(cfg,fibers,sampling,negative_bank=bank,augment=True)
    sample=SampleConfig(crop=cfg.fine,n_history=cfg.n_history,n_future=cfg.n_future,recent_history_points=cfg.n_history)
    chunk = args.microbatch
    if args.microbatch % (2*cfg.feature_sequence_length):
        raise ValueError('V4 preflight microbatch must contain an even number of full sequence streams')
    chunk //= cfg.feature_sequence_length
    ds=FollowDataset(fibers,spec,sample,band,chunk=chunk,seed=37,batch_builder=builder,
                     onpolicy=[OnPolicyStates.load(p) for p in args.onpolicy])
    it=iter(ds);args.out.mkdir(parents=True,exist_ok=True)
    model=build_model(cfg).to(args.device,memory_format=conv_memory_format(args.device)) if args.forward else None
    import copy
    from .stratified_replay import ObservationHistory
    from .feature_sequences import decision_count
    from .train import optimizer_update
    states = ObservationHistory()
    ema = copy.deepcopy(model) if model is not None else None
    opt = torch.optim.AdamW(model.parameters(), lr=0.) if model is not None else None
    rows=[]
    for index in range(args.batches):
        started=time.perf_counter();cpu=next(it)
        steps = cpu['feature_sequence']
        for batch in steps:
            assert 'memory_patches' not in batch['x'] and 'feature_seed_x' not in batch['x']
            assert torch.isfinite(batch['x']['fine']).all()
        if index == 0:
            torch.save(cpu, args.out/'batch.pt')
        row = dict(batch=index, read_seconds=time.perf_counter()-started,
                   states=sum(len(b['hist']) for b in steps), decisions=sum(decision_count(b) for b in steps))
        if model is not None:
            metrics = optimizer_update(model, ema, opt, [cpu], index+1, 0., device=args.device,
                                       compute_metrics=False, stream_states=states)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            if metrics['optimizer_applied']:
                assert model.encoder.compress.weight.grad.abs().sum() > 0
            row.update(loss=metrics['loss'], retained_streams=len(states.streams))
        rows.append(row)
        print(json.dumps(row), flush=True)
    report=dict(config=cfg.to_dict(),sampling=asdict(sampling),bank=str(bank.root),bank_shards=bank.shard_count,rows=rows)
    (args.out/'report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
