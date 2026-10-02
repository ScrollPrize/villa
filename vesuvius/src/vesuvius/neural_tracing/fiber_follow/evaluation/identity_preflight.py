"""Check real full-size axial batches, historical slabs, and model gradients."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from vesuvius.neural_tracing.fiber_follow.models.model import CoordinateRegressionConfig, build_model, crop_support
from vesuvius.neural_tracing.fiber_follow.data.observations import IdentityObservationBuilder, IdentitySampling
from vesuvius.neural_tracing.fiber_follow.data.neighbor_bank import NeighborBank
from vesuvius.neural_tracing.fiber_follow.train.supervision import loss_terms
from vesuvius.neural_tracing.fiber_follow.train.train import move_batch, conv_memory_format
from vesuvius.neural_tracing.fiber_follow.data.components import ComponentRule
from vesuvius.neural_tracing.fiber_follow.data.data import FollowDataset, OnPolicyStates, SampleConfig, TaskBudget, ZBand, load_fibers, split_fibers
from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolumeSpec


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
    ap.add_argument('--synthetic-tail',type=float,nargs=2,default=(4.,16.))
    ap.add_argument('--onpolicy',nargs='*',default=[],help='Replay caches, e.g. collected with observed tracks')
    args=ap.parse_args(argv)
    torch.set_num_threads(4);torch.manual_seed(0)
    cfg=CoordinateRegressionConfig()
    spec=FiberVolumeSpec(args.fiber_zarrs,ct_zarr=args.ct,ct_level=0,ct_grid_scale=4.,inputs='ct',load_presence=False)
    band=ZBand(45000/spec.grid_scale,48500/spec.grid_scale)
    fibers,_=split_fibers(load_fibers(args.fibers,grid_scale=spec.grid_scale),band)
    bank=NeighborBank(args.bank,fibers,band,grid_scale=spec.grid_scale)
    bank.validate_volume(spec)
    sampling=IdentitySampling(rule=ComponentRule(lateral_max=32.),bank_coverage_probability=.2,
        synthetic_tail=tuple(args.synthetic_tail))
    builder=IdentityObservationBuilder(cfg,fibers,sampling,negative_bank=bank,augment=True)
    sample=SampleConfig(crop=cfg.fine,n_history=cfg.n_history,n_future=cfg.n_future,recent_history_points=cfg.n_history)
    # No live chains outside a trainer; their share goes to fresh traces.
    ds=FollowDataset(fibers,spec,sample,band,chunk=args.microbatch,seed=37,batch_builder=builder,
                     budget=TaskBudget.parse(['live=0','fresh=.65']),
                     onpolicy=[OnPolicyStates.load(p) for p in args.onpolicy])
    if args.out.exists():
        raise FileExistsError('Use a fresh preflight output directory')
    it=iter(ds);args.out.mkdir(parents=True)
    from vesuvius.neural_tracing.fiber_follow.data.ct_normalization import prepare_normalization
    prepare_normalization(args.out, [spec])
    model=build_model(cfg).to(args.device,memory_format=conv_memory_format(args.device)) if args.forward else None
    import copy
    from vesuvius.neural_tracing.fiber_follow.train.train import optimizer_update, prepare_training
    ema = copy.deepcopy(model) if model is not None else None
    opt = torch.optim.AdamW(model.parameters(), lr=0.) if model is not None else None
    if model is not None:
        prepare_training(model, args.microbatch)
    rows=[]
    for index in range(args.batches):
        started=time.perf_counter();cpu=next(it)
        assert torch.isfinite(cpu['x']['fine']).all()
        assert torch.isfinite(cpu['x']['history_slabs']).all()
        if index == 0:
            torch.save(cpu, args.out/'batch.pt')
        row = dict(batch=index, read_seconds=time.perf_counter()-started,
                   states=len(cpu['hist']), decisions=len(cpu['hist']))
        if model is not None:
            metrics = optimizer_update(model, ema, opt, [cpu], index+1, 0., device=args.device,
                                       compute_metrics=False)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            if metrics['optimizer_applied']:
                assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
                assert metrics['history_grad_norm'] > 0
            row.update(loss=metrics['loss'], history_grad_norm=metrics['history_grad_norm'],
                       history_valid_slabs_mean=metrics['history_valid_slabs_mean'])
        rows.append(row)
        print(json.dumps(row), flush=True)
    report=dict(config=cfg.to_dict(),sampling=asdict(sampling),bank=str(bank.root),bank_shards=bank.shard_count,rows=rows)
    (args.out/'report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
