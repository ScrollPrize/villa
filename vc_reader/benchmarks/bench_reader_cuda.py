"""Public CUDA reader latency, including native planning and output allocation.

Use run_native.py twice to select both extensions. Requires the real CT input
used by bench_sample.py and the existing optional CUDA build. No installs.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from _common import environment
from bench_sample import DEFAULT_ARRAY, REGION
from bench_cuda_sample import stats
from vc_reader.reader import ZarrReader
from vc_reader import cuda, augment as A


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--array',default=DEFAULT_ARRAY)
    ap.add_argument('--reps',type=int,default=11)
    ap.add_argument('--warmup',type=int,default=3)
    ap.add_argument('--output',type=Path,default=Path('/tmp/public-reader-cuda.json'))
    ap.add_argument('--only',default='')
    args=ap.parse_args()
    if args.reps<1 or args.warmup<1: ap.error('positive repetitions/warmup required')
    origin=np.asarray(REGION[0])+[128,128,128]
    axes=np.array([[1,0,0],[0,np.cos(.35),-np.sin(.35)],[0,np.sin(.35),np.cos(.35)]],np.float32)
    points=(origin+np.random.default_rng(7).uniform(0,96,(1_000_000,3))).astype(np.float64)
    integers=np.rint(points).astype(np.int32)
    grids=[((128,128,128),origin+[i*4+.25,33.25,49.25],axes) for i in range(8)]
    with ZarrReader.open(args.array,threads=16,memory_cache=0) as reader, reader.cuda() as gpu:
        # Float32 CPU/GPU calls use the same automatic integer-origin splitting.
        cases={
            'affine128':(lambda:reader.read_affine((128,)*3,origin+.25,axes),lambda source:gpu.read_affine((128,)*3,origin+.25,axes,source=source)),
            'affine256':(lambda:reader.read_affine((256,)*3,origin+.25,axes),lambda source:gpu.read_affine((256,)*3,origin+.25,axes,source=source)),
            'affine_batch8':(lambda:reader.read_affine_many(grids),lambda source:gpu.read_affine_many(grids,source=source)),
            'box256':(lambda:reader.read(origin+[17,33,49],(256,)*3,dtype=np.float32),lambda source:gpu.read(origin+[17,33,49],(256,)*3,dtype=np.float32,source=source)),
            'points_local':(lambda:reader.sample(points),lambda source:gpu.sample(points,source=source)),
            'integer_local':(lambda:reader.read_points(integers),lambda source:gpu.read_points(integers,source=source)),
        }
        report=dict(environment=environment(),array=args.array,threads=16,reps=args.reps,warmup=args.warmup,
                    protocol='warm mmap, seeded shuffled modes; public API planning/allocation included; outputs on GPU; verification download and cache clearing excluded',rows=[])
        rng=np.random.default_rng(12)
        only=set(filter(None,args.only.split(',')))
        for name,(cpu,device) in cases.items():
            if only and name not in only: continue
            expected=cpu(); expected=expected if isinstance(expected,list) else [expected]
            request=gpu.prepare_points(integers if name=='integer_local' else points,integer=name=='integer_local') if name.endswith('_local') else None
            modes=['cpu','cpu_upload','cuda_fresh','cuda_resident','auto_fresh']
            if request: modes.append('prepared_resident')
            times={m:[] for m in modes}
            maximum=0.
            for rep in range(-args.warmup,args.reps):
                order=modes.copy(); rng.shuffle(order)
                for mode in order:
                    if mode in ('cuda_fresh','auto_fresh'): gpu.clear_cache()
                    if mode in ('cuda_resident','prepared_resident'): device('chunks')
                    start=time.perf_counter()
                    if mode=='cpu': out=cpu()
                    elif mode=='cpu_upload': out=device('cpu')
                    elif mode=='prepared_resident': out=request.read(source='chunks')
                    else: out=device('auto' if mode=='auto_fresh' else 'chunks')
                    elapsed=(time.perf_counter()-start)*1000
                    outputs=out if isinstance(out,list) else [out]
                    for actual,reference in zip(outputs,expected):
                        actual=actual.numpy() if isinstance(actual,cuda.DeviceArray) else actual
                        np.testing.assert_allclose(actual,reference,atol=5e-5,rtol=1e-6)
                        maximum=max(maximum,float(np.max(np.abs(actual.astype(float)-reference),initial=0)))
                    if rep>=0: times[mode].append(elapsed)
                    del out,outputs
            row=dict(name=name,timings={m:stats(t) for m,t in times.items()},max_abs=maximum,cache=gpu.stats)
            report['rows'].append(row)
            print(name+' '+ ' '.join(f'{m}={np.median(t):.3f}' for m,t in times.items()),flush=True)
        if not only or 'pipeline' in only:
            transforms=[A.Normalize('minmax'),A.BrightnessMultiply(1.1),A.GaussianNoise(.01)]
            cp=A.Pipeline(transforms,backend='cpu',compact=False); gp=A.Pipeline(transforms,backend='cuda')
            times={m:[] for m in ('cpu_pipeline','cuda_fresh_pipeline','cuda_resident_pipeline')}
            for rep in range(-args.warmup,args.reps):
                order=list(times); rng.shuffle(order)
                for mode in order:
                    if mode=='cuda_fresh_pipeline': gpu.clear_cache()
                    if mode=='cuda_resident_pipeline': cases['affine128'][1]('chunks')
                    start=time.perf_counter()
                    if mode=='cpu_pipeline':
                        out=cp({'image':cases['affine128'][0]()[None]},seed=12,inplace=True)['image']
                    else:
                        out=gp({'image':cases['affine128'][1]('chunks').reshape(1,128,128,128)},seed=12,inplace=True)['image']
                    elapsed=(time.perf_counter()-start)*1000
                    actual=out.numpy() if isinstance(out,cuda.DeviceArray) else out
                    # Compare to the same CUDA pipeline on the CPU sampled crop;
                    # CPU/CUDA RNG implementation has separately tested tolerances.
                    reference=gp({'image':cases['affine128'][0]()[None]},seed=12)['image']
                    np.testing.assert_allclose(actual,reference,atol=2e-6,rtol=2e-6)
                    if rep>=0: times[mode].append(elapsed)
                    del out
            report['rows'].append(dict(name='pipeline',timings={m:stats(t) for m,t in times.items()}))
            print('pipeline '+' '.join(f'{m}={np.median(t):.3f}' for m,t in times.items()),flush=True)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__': main()
