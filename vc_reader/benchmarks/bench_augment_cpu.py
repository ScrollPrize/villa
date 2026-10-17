"""CPU-only catalog timings with exact output digests for before/after checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from vc_reader import _augment, augment as A
from vc_reader.reader import ZarrReader
from augment_catalog import catalog, inputs, describe
from bench_augment import DEFAULT_ARRAY, summary
from bench_augment_native import arrays


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--array',default=DEFAULT_ARRAY)
    parser.add_argument('--center',type=int,nargs=3,default=(9064,3664,3664))
    parser.add_argument('--sizes',type=int,nargs='+',default=[128,256])
    parser.add_argument('--reps',type=int,default=5)
    parser.add_argument('--warmup',type=int,default=1)
    parser.add_argument('--only',required=True,help='Comma-separated catalog names.')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,help='Require exact per-seed output digests from this previous run.')
    parser.add_argument('--export-crops',action='store_true',help='Save raw normalized float32 crops beside the report for the native profiler.')
    args=parser.parse_args()
    if args.reps<1 or args.warmup<0 or min(args.sizes)<8:parser.error('sizes>=8, reps>=1, warmup>=0 required')
    expected=json.loads(args.baseline.read_text()) if args.baseline else None
    report=dict(command=sys.argv,array=args.array,center=list(args.center),reps=args.reps,warmup=args.warmup,
                platform=platform.platform(),extension=str(_augment.__file__),
                binary_sha256=hashlib.sha256(Path(_augment.__file__).read_bytes()).hexdigest(),
                thread_env={k:os.getenv(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
                protocol='CPU synchronous public Pipeline; seed 123+rep; reads, preparation, hashing and destruction excluded; single thread; compact=True',results=[])
    if expected:
        for key in ('array','center','reps','warmup'):
            if expected[key]!=report[key]:raise ValueError('baseline protocol mismatch: '+key)
    selected=set(args.only.split(','))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for size in args.sizes:
        start=tuple(v-size//2 for v in args.center)
        with ZarrReader.open(args.array,threads=1) as reader:
            image=reader.read(start,(size,)*3).astype(np.float32)[None]/np.float32(255)
            reference=reader.read((start[0]+384,*start[1:]),(size,)*3).astype(np.float32)[None]/np.float32(255)
        sources=inputs(image)
        if args.export_crops:image[0].tofile(args.output.parent/f'crop{size}.f32')
        cases=catalog(size,reference)
        if selected-{c.name for c in cases}:parser.error('unknown or utility case')
        for case in cases:
            if case.name not in selected:continue
            pipeline=A.Pipeline(case.transforms,backend='cpu')
            source=sources[case.sample]
            row=dict(size=size,name=case.name,parameters=' → '.join(describe(t) for t in case.transforms),
                     input_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in source.items()},
                     reference_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),digests=[])
            print('START',size,case.name,flush=True)
            measured=[]
            for rep in range(-args.warmup,args.reps):
                begun=time.perf_counter_ns()
                result=pipeline(source,seed=123+max(rep,0))
                elapsed=(time.perf_counter_ns()-begun)*1e-9
                if rep>=0:
                    measured.append(elapsed)
                    row['digests'].append({key:dict(shape=value.shape,dtype=str(value.dtype),sha256=hashlib.sha256(value.tobytes()).hexdigest()) for key,value in arrays(result)})
                del result
            row['timings']=summary(measured)
            if expected:
                old=next(r for r in expected['results'] if r['size']==size and r['name']==case.name)
                # JSON normalizes tuple shapes to lists.
                for key in ('parameters','input_sha256','reference_sha256','digests'):
                    if json.loads(json.dumps(row[key]))!=old[key]:raise AssertionError(f'{size} {case.name}: {key} changed')
                row['bitwise_equal']=True
                row['speedup']=old['timings']['p50_ms']/row['timings']['p50_ms']
            report['results'].append(row)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            print('DONE',size,case.name,row['timings'],flush=True)


if __name__=='__main__':main()
