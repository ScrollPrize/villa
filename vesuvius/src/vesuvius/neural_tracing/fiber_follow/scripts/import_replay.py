"""Build the permanent v11 recovery bank from existing training collections."""
import argparse
from pathlib import Path
from vesuvius.neural_tracing.fiber_follow.data import SampleConfig, ZBand, load_fibers, split_fibers
from vesuvius.neural_tracing.fiber_follow.geometry import CropSpec
from vesuvius.neural_tracing.fiber_follow.replay import import_states
from vesuvius.neural_tracing.fiber_follow.volume import FiberVolumeSpec


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('sources',nargs='+')
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--fibers',default='/mnt/raid_nvme/spiral_dataset_working/fibers')
    ap.add_argument('--fiber-zarrs',default='/mnt/raid_nvme/spiral_dataset_working/fiber_zarrs')
    ap.add_argument('--ct',default='/mnt/raid_nvme/volpkgs/s1_2um_ds2.volpkg/volumes/s1_ds2.zarr')
    ap.add_argument('--val-z',nargs=2,type=float,default=(45000,48500))
    ap.add_argument('--limit',type=int,default=20000)
    ap.add_argument('--seed',type=int,default=0)
    args=ap.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(args.out)
    spec=FiberVolumeSpec(args.fiber_zarrs,ct_zarr=args.ct,ct_level=1,ct_grid_scale=8.,inputs='ct+presence')
    band=ZBand(*(z/spec.grid_scale for z in args.val_z))
    fibers,_=split_fibers(load_fibers(args.fibers,grid_scale=spec.grid_scale),band)
    cfg=SampleConfig(crop=CropSpec(depth=176,width=96,behind=128,history_render='segments',history_sigma=.35))
    bank=import_states(args.sources,fibers,cfg,band,spec,limit=args.limit,seed=args.seed)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    bank.save(args.out)
    print(dict(states=len(bank),**bank.provenance['counts'],out=str(args.out)))

if __name__=='__main__':
    main()
