"""Compare beam crops against the previous cubic-block reader on identical CT."""
import argparse
import json
import time

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.crop_sampling import scalar_crops
from vesuvius.neural_tracing.fiber_follow.data import collate_with_volume, tight_block
from vesuvius.neural_tracing.fiber_follow.geometry import CropSpec, crop_local_grid, frame_from_heading
from vesuvius.neural_tracing.fiber_follow.volume import FiberVolume, FiberVolumeSpec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ct', required=True)
    ap.add_argument('--fiber-zarrs', required=True)
    ap.add_argument('--iterations', type=int, default=10)
    ap.add_argument('--position', type=float, nargs=3, default=(2000., 2000., 5500.))
    args = ap.parse_args()
    torch.set_num_threads(1)
    vol = FiberVolume(FiberVolumeSpec(args.fiber_zarrs, ct_zarr=args.ct, ct_level=1,
                                    ct_grid_scale=8., inputs='ct'), cache_bytes=256 << 20)
    crop = CropSpec(depth=192, width=96, behind=128, history_render='segments', history_sigma=.35)
    grid = torch.from_numpy(crop_local_grid(crop)).float()
    hist = np.c_[np.zeros(128), np.zeros(128), -np.arange(1., 129.)]
    items = [dict(pos=np.asarray(args.position), frame=frame_from_heading(np.array(axis)),
                  hist_local=hist, hmask=np.ones(128, np.float32))
             for axis in ((0.,0.,1.),(.3,.4,.866025403784),(.7,.7,.141421356237))]
    before = lambda item: collate_with_volume([item], vol, crop, grid)['x']
    after = lambda item: scalar_crops([item], vol, crop, history=True).half()
    for item in items:  # warm Numba and the shared OS/reader caches, verify exact values
        assert torch.equal(before(item), after(item))
    report = dict(ct=args.ct, level=1, compressed=vol.ct.codec is not None,
                  crop=[192,96,96], position=args.position, iterations=args.iterations,
                  bit_identical=True, threads=1, reader_cache_bytes=256 << 20)
    for name, run in (('cubic', before), ('tight_shared_direct', after)):
        elapsed=[]
        for i in range(args.iterations):
            started=time.perf_counter(); run(items[i % len(items)])
            elapsed.append(1000*(time.perf_counter()-started))
        report[name]=dict(mean_ms=float(np.mean(elapsed)),p50_ms=float(np.median(elapsed)),
                          p95_ms=float(np.percentile(elapsed,95)))
    report['tight_source_voxels']=[int(np.prod(tight_block(i['pos'],i['frame'],crop)[1])) for i in items]
    report['cubic_source_voxels']=int(crop.block_size**3)
    report['mapped_chunks']=sum(isinstance(v,np.memmap) for v in vol.ct._cache.values())
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
