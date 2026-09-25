"""Freeze evaluation splits and holdout recovery states before v11 training."""
import argparse
import json
from pathlib import Path
from vesuvius.neural_tracing.fiber_follow.data import (
    SampleConfig,ZBand,load_fibers,split_fibers,
)
from vesuvius.neural_tracing.fiber_follow.experiment import freeze_manifest
from vesuvius.neural_tracing.fiber_follow.recovery import make_recovery_states
from vesuvius.neural_tracing.fiber_follow.geometry import CropSpec
from vesuvius.neural_tracing.fiber_follow.volume import FiberVolume,FiberVolumeSpec


def main(argv=None):
    root=Path(__file__).resolve().parents[1]
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-run',type=Path,default=root/'output/flow_v10_small_b8_d64')
    ap.add_argument('--out',type=Path,default=root/'output/single_path_v11_preparation')
    args=ap.parse_args(argv)
    old=json.loads((args.source_run/'config.json').read_text())
    spec=FiberVolumeSpec(old['fiber_zarrs'],ct_zarr=old['ct'],ct_level=1,ct_grid_scale=8.,inputs='ct+presence')
    vol=FiberVolume(spec,cache_bytes=1<<30)
    fibers=load_fibers(old['fibers'],grid_scale=spec.grid_scale)
    _,val=split_fibers(fibers,ZBand(*(v/spec.grid_scale for v in old['val_z'])))
    args.out.mkdir(parents=True,exist_ok=True)
    manifest=freeze_manifest(args.out/'seeds.json',val,vol,args.source_run/'assessment_20260925/seeds.json',args.source_run/'config.json')
    print({k:len(manifest[k]) for k in ('monitor','calibration','final')},flush=True)
    cfg=SampleConfig(crop=CropSpec(depth=176,width=96,behind=128,history_render='segments',history_sigma=.35),
                     history_jitter=0.,history_wobble=1.,angle_sigmas_deg=(2.,5.,10.))
    path=args.out/'calibration_recovery.npz'
    if path.exists(): return
    states=make_recovery_states(val,manifest['calibration'],cfg,dict(split='calibration',
        seed_manifest_sha256=manifest['sha256'],volume=spec.to_dict(),
        construction='frozen augmented states; no confirmed departures'))
    states.save(path)
    print(dict(recovery_states=len(states),path=str(path)),flush=True)

if __name__=='__main__': main()
