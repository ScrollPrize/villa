"""Full CPU/CUDA catalog on real CT; checkpoint timings and orthogonal PNGs.

Use run_native.py twice to select both reader and augmentation extensions.
All three modes use the owning, synchronous public Pipeline API. No installs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import numpy as np
from vc_reader import _augment, augment as A, cuda
from vc_reader.reader import ZarrReader
from augment_catalog import catalog, inputs, describe, UTILITY_TRANSFORMS
from augment_preview import preview, gallery, effect_strength
from bench_augment import DEFAULT_ARRAY, summary
from bench_augment_native import arrays, fingerprint


def to_host(output):
    return {k: [v.numpy() for v in a] if isinstance(a,list) else a.numpy() for k,a in output.items()}


def compare(output, reference):
    actual=dict(arrays(output));expected=dict(arrays(reference))
    if actual.keys()!=expected.keys():
        raise AssertionError('output keys differ')
    report={}
    for key,b in expected.items():
        a=actual[key]
        if a.shape!=b.shape or a.dtype!=b.dtype:
            raise AssertionError(f'{key}: shape/dtype mismatch')
        maximum=0.;bad=0
        for start in range(0,a.size,1<<20):
            x=a.ravel()[start:start+(1<<20)];y=b.ravel()[start:start+(1<<20)]
            delta=np.abs(x.astype(np.float64)-y.astype(np.float64))
            maximum=max(maximum,float(delta.max(initial=0)))
            valid=np.isfinite(x) & np.isfinite(y)
            if a.dtype.kind=='f':valid &= delta<=3e-6+3e-6*np.abs(y)
            else:valid &= x==y
            bad+=int(np.count_nonzero(~valid))
        report[key]=dict(max_abs=maximum,mismatched=bad)
    return report


def save(directory, report, *, include_utilities=False):
    path=directory/f"{report['size']}.json";temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(path)
    gallery(directory,[json.loads(p.read_text()) for p in sorted(directory.glob('[0-9]*.json'))],include_utilities=include_utilities)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--array',default=DEFAULT_ARRAY)
    parser.add_argument('--center',nargs=3,type=int,default=(9064,3664,3664))
    parser.add_argument('--size',type=int,required=True)
    parser.add_argument('--reps',type=int,default=5)
    parser.add_argument('--warmup',type=int,default=1)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--build-dir',type=Path,default=Path('/tmp/vc-volume-cuda'))
    parser.add_argument('--only',default='')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--include-utilities',action='store_true',help='Also benchmark and display preprocessing, derived targets and pipeline helpers.')
    args=parser.parse_args()
    selected={name.strip() for name in args.only.split(',') if name.strip()} or None
    if selected and not args.include_utilities and selected & UTILITY_TRANSFORMS:
        parser.error('utility cases require --include-utilities: '+', '.join(sorted(selected & UTILITY_TRANSFORMS)))
    if args.size<8 or args.reps<1 or args.warmup<0:parser.error('size>=8, reps>=1, warmup>=0 required')
    if not cuda.is_available():parser.error('CUDA reader and augmentation builds required')
    args.output.mkdir(parents=True,exist_ok=True)
    start=tuple(v-args.size//2 for v in args.center)
    reference_start=(start[0]+384,*start[1:])
    with ZarrReader.open(args.array,threads=1) as reader:
        image=reader.read(start,(args.size,)*3).astype(np.float32)[None]/np.float32(255)
        reference=reader.read(reference_start,(args.size,)*3).astype(np.float32)[None]/np.float32(255)
    if image.min()==image.max():raise ValueError('constant crop')
    samples=inputs(image)
    binary=Path(_augment.__file__)
    cache=args.build_dir/'CMakeCache.txt'
    flags=args.build_dir/'CMakeFiles/_augment.dir/flags.make'
    cpu=platform.processor()
    if Path('/proc/cpuinfo').exists():cpu=next(v.split(':',1)[1].strip() for v in Path('/proc/cpuinfo').read_text().splitlines() if v.startswith('model name'))
    report=dict(command=sys.argv,size=args.size,start=start,reference_start=reference_start,array=args.array,
                reps=args.reps,warmup=args.warmup,cpu=cpu,platform=platform.platform(),python=sys.version,
                gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip(),
                affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
                extension=str(binary),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                build_cache=cache.read_text() if cache.exists() else None,build_flags=flags.read_text() if flags.exists() else None,
                git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                protocol='one CPU thread; CPU compact=True; one warmup and five reps unless overridden; synchronous owning public calls; shuffled backend order per repetition; seed 123+rep; reads, preparation, hashing, comparison, output destruction and PNG rendering excluded; CUDA host includes transfers; resident excludes initial upload and final download but includes device input copy and validation',
                thread_env={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
                inputs={k:fingerprint(v) for k,v in samples.items()},reference=fingerprint({'reference':reference}),
                crop_stats=dict(low=float(image.min()),high=float(image.max()),mean=float(image.mean()),foreground_fraction=float(np.mean(image>.5))),
                correctness='Every timed output compared against same-seed CPU output; exact integer labels; float atol=rtol=3e-6. Mismatches are reported, never silently discarded.',results=[])
    path=args.output/f'{args.size}.json'
    if args.resume and path.exists():
        old=json.loads(path.read_text())
        for key in ('binary_sha256','inputs','reference','reps','warmup'):
            if old[key]!=report[key]:raise ValueError('resume protocol/input/build mismatch: '+key)
        report['results']=old['results']
    done={r['name'] for r in report['results']}
    workload=catalog(args.size,reference,include_utilities=args.include_utilities)
    if selected and selected-{c.name for c in workload}:raise ValueError('unknown case')
    report['requested_cases']=[c.name for c in workload if not selected or c.name in selected]
    random.Random(21017).shuffle(workload)
    for case in workload:
        if case.name in done or selected and case.name not in selected:continue
        parameters=' → '.join(describe(t) for t in case.transforms)
        row=dict(name=case.name,parameters=parameters,sample=case.sample,output=case.output,note=case.note,timings={},comparisons=[])
        print(f'START {args.size} {case.name}',flush=True)
        begun=time.perf_counter()
        try:
            source=samples[case.sample]
            targets=dict(image=A.Image(),mask=A.Label(),surface=A.CoordinateField())
            pipelines={mode:A.Pipeline(case.transforms,targets=targets,backend='cpu' if mode=='cpu' else 'cuda') for mode in ('cpu','cuda_host','cuda_resident')}
            device={k:cuda.asarray(v) for k,v in source.items()}
            measured={mode:[] for mode in pipelines}
            best_preview=-1
            for rep in range(-args.warmup,args.reps):
                modes=list(pipelines);random.Random(291+rep).shuffle(modes);outputs={}
                for mode in modes:
                    t0=time.perf_counter_ns()
                    value=pipelines[mode](device if mode=='cuda_resident' else source,seed=123+max(0,rep))
                    elapsed=(time.perf_counter_ns()-t0)*1e-9
                    if rep>=0:measured[mode].append(elapsed)
                    outputs[mode]=to_host(value) if mode=='cuda_resident' else value
                    del value
                if rep>=0:
                    detail={mode:compare(outputs[mode],outputs['cpu']) for mode in ('cuda_host','cuda_resident')}
                    row['comparisons'].append(dict(seed=123+rep,**detail))
                    if any(v['mismatched'] for result in detail.values() for v in result.values()):row['error']='CPU/CUDA numerical mismatch; inspect comparisons'
                strength=effect_strength(case,source,outputs['cpu']) if rep>=0 else -1
                if strength>best_preview:
                    best_preview=strength
                    picture=f'previews/{args.size}/{case.name}.png'
                    row['preview_selection']=preview(args.output/picture,case,source,outputs['cpu'],args.size,parameters,123+rep)
                    row['preview']=picture
                del outputs
            if fingerprint(source)!=report['inputs'][case.sample]:raise AssertionError('input mutated')
            row['timings']={mode:summary(values) for mode,values in measured.items()}
            print('DONE',args.size,case.name,{m:round(v['p50_ms'],3) for m,v in row['timings'].items()},row.get('error','ok'),flush=True)
            del pipelines,device
        except Exception as exc:
            row['error']=f'{type(exc).__name__}: {exc}'
            import traceback
            traceback.print_exc()
        row['wall_seconds']=time.perf_counter()-begun
        report['results'].append(row);save(args.output,report,include_utilities=args.include_utilities)
    save(args.output,report,include_utilities=args.include_utilities)
    print(f"COMPLETE {args.size}: {len(report['results'])} cases; {sum('error' in r for r in report['results'])} reported errors",flush=True)


if __name__=='__main__':main()
