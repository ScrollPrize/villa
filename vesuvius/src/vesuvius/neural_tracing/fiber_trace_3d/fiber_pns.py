"""Explicit optional PNS benchmark; reuse the certified integer-capacity graphs."""
import ctypes
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import tempfile
import time

import numpy as np

from .fiber_lemon import quantize, evaluate
from .fiber_min_flow import certify_flow
from .fiber_min_flow_benchmark import Paths, graph
from .fiber_min_flow_experiment import arrays, write, digest

PNS_MAX=(2**31-1)//2


def shifted_problem(nodes, q):
    """y=x-q: -q<=y<=q becomes 0<=x<=2q; supplies are outflow-inflow."""
    if nodes<1 or nodes+len(q['a'])>=PNS_MAX:
        raise ValueError('PNS graph index limit exceeded')
    if np.any(q['a']<0) or np.any(q['b']<0) or np.any(q['a']>=nodes) or np.any(q['b']>=nodes):
        raise ValueError('Node outside graph')
    capacity=q['capacity']
    # Conservative guard also protects artificial-tree flows and int32 updates.
    if np.any(capacity<=0) or sum(map(int,capacity))*2>=PNS_MAX:
        raise ValueError('PNS int32 flow-range guard exceeded; do not truncate capacities')
    if int(np.max(abs(q['target']),initial=0))*(nodes+1)>PNS_MAX//4:
        raise ValueError('PNS int32 potential-range guard exceeded')
    supply=np.zeros(nodes,np.int64)
    np.add.at(supply,q['a'],capacity);np.add.at(supply,q['b'],-capacity)
    return dict(upper=2*capacity,cost=-q['target'],supply=supply,
                objective_offset=sum(int(d)*int(w) for d,w in zip(q['target'],capacity)))


def write_dimacs(path, nodes, q):
    problem=shifted_problem(nodes,q)
    with Path(path).open('w') as f:
        f.write(f'p min {nodes} {len(q["a"])}\n')
        for node,supply in enumerate(problem['supply']):
            if supply:f.write(f'n {node+1} {supply}\n')
        for a,b,upper,cost in zip(q['a'],q['b'],problem['upper'],problem['cost']):
            f.write(f'a {a+1} {b+1} 0 {upper} {cost}\n')
    return problem


def build_native(source, output, *, parallel, avx2=False):
    source=Path(source).resolve();output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    if platform.machine().lower() not in ('x86_64','amd64','i386','i686'):
        raise RuntimeError('Upstream PNS unconditionally includes x86 intrinsics; optional backend unavailable here')
    if avx2:
        from numpy._core._multiarray_umath import __cpu_features__
        if not __cpu_features__.get('AVX2',False):raise RuntimeError('AVX2 not supported on this CPU')
    cxx=shlex.split(os.environ.get('CXX','c++'));wrapper=Path(__file__).with_suffix('.cpp')
    flags=['-O3','-std=c++17','-fPIC','-dynamiclib' if sys.platform=='darwin' else '-shared']
    if parallel:flags+=['-fopenmp','-DOMP_AVX2' if avx2 else '-DOMP']
    elif avx2:flags+=['-DAVX2']
    if avx2:flags+=['-mavx2']
    flags+=[f'-DPNS_SOURCE_FILE="{source}"']
    metadata=dict(upstream_sha256=digest(source),wrapper_sha256=digest(wrapper),flags=flags,
                  compiler=subprocess.check_output([*cxx,'--version'],text=True).splitlines()[0])
    manifest=output.with_suffix('.build.json')
    if output.exists() and manifest.exists() and json.loads(manifest.read_text())==metadata:return output
    with tempfile.NamedTemporaryFile(dir=output.parent,prefix='pns-build-',delete=False) as f:temporary=Path(f.name)
    try:
        subprocess.run([*cxx,*flags,str(wrapper),'-o',str(temporary)],check=True)
        os.replace(temporary,output);write(manifest,metadata)
    finally:temporary.unlink(missing_ok=True)
    return output


def native_solve(library, path, nodes, q, threads, time_limit=20.):
    if threads < 1 or time_limit <= 0 or not np.isfinite(time_limit):
        raise ValueError('Positive thread count and finite positive time limit required')
    shifted_problem(nodes,q)  # enforce numerical range before calling int32 code
    lib=ctypes.CDLL(str(library));fn=lib.fiber_pns_solve
    vec=np.ctypeslib.ndpointer(dtype=np.int64,ndim=1,flags='C_CONTIGUOUS')
    fn.argtypes=[ctypes.c_char_p,ctypes.c_int32,ctypes.c_int32,ctypes.c_int,ctypes.c_double,
                 vec,vec,np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS'),
                 ctypes.c_char_p,ctypes.c_size_t];fn.restype=ctypes.c_int
    k=np.empty(nodes,np.int64);x=np.empty(len(q['a']),np.int64);timing=np.zeros(7)
    error=ctypes.create_string_buffer(2048)
    status=fn(os.fsencode(path),nodes,len(x),threads,time_limit,k,x,timing,error,len(error))
    profile=dict(zip(('input_seconds','solve_seconds','native_total_seconds','iterations',
                      'pricing_seconds','tree_seconds','potential_seconds'),map(float,timing)))
    if status==2:return dict(timeout=True,profile=profile)
    if status:raise RuntimeError(error.value.decode())
    y=x-q['capacity']
    result=certify_flow(nodes,q['a'],q['b'],q['target'],q['capacity'].astype(float),
                       np.zeros(len(x),bool),k,y.astype(float),seconds=timing[2],iterations=int(timing[3]))
    result['certificate']['backend']='PNS'
    return dict(timeout=False,profile=profile,result=result)


def run(config):
    if int(config.get('repeats',3)) < 1:
        raise ValueError('At least one repeat required')
    root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    if (root/'request.json').exists():raise ValueError('Choose a fresh output directory')
    write(root/'request.json',config)
    prior=Path(config['comparison_root']);previous=json.loads((prior/'results.json').read_text())
    prior_config=json.loads((prior/'request.json').read_text());dataset=prior_config['dataset']
    d=arrays(dataset['tensor_cache']);baseline=arrays(prior_config['baseline_solution'])
    paths=Paths(dataset,d);_,transport=graph(dataset,d,paths);g=arrays(prior/'graph.npz')
    extension='.dylib' if sys.platform=='darwin' else '.so'
    libraries={name:build_native(config['source'],root/(name+extension),parallel=parallel,avx2=config.get('avx2',False))
               for name,parallel in [('sequential',False),('parallel',True)]}
    tests=[('sequential',1)]+[('parallel',int(t)) for t in config.get('thread_counts',[1,2,4,8,16,32,64,112])]
    repeats=int(config.get('repeats',3));report=dict(nodes=len(d['source_id']),arcs=len(g['a']),trials=[])
    for scale in config.get('scales',[1000,1]):
        q,quantization=quantize(g,scale);path=root/f'graph_{scale}.min'
        problem=write_dimacs(path,len(d['source_id']),q)
        expected=next(t['certificate']['primal'] for t in previous['trials'] if t['scale']==scale)
        for mode,threads in tests:
            profiles=[];walls=[];metrics=[];outcome=None
            for repeat in range(repeats):
                write(root/'progress.json',dict(stage='solve',scale=scale,mode=mode,threads=threads,repeat=repeat+1))
                t=time.monotonic();outcome=native_solve(libraries[mode],path,len(d['source_id']),q,threads,config.get('time_limit',20.))
                walls.append(time.monotonic()-t);profiles.append(outcome['profile'])
                if outcome['timeout']:break
                result=outcome['result']
                if result['certificate']['primal']!=expected:raise RuntimeError('PNS optimum differs from certified LEMON/HiGHS optimum')
                metrics.append(evaluate(d,g,transport,result,baseline))
            trial=dict(scale=scale,mode=mode,threads=threads,timeout=outcome['timeout'],repeats=len(profiles),
                       profiles=profiles,quantization=quantization,objective_offset=problem['objective_offset'])
            if not outcome['timeout']:
                times=np.array([p['solve_seconds'] for p in profiles])
                trial.update(certificate=result['certificate'],metrics=metrics[-1],all_metrics=metrics,
                    solve_seconds=dict(mean=float(times.mean()),p50=float(np.median(times)),p95=float(np.quantile(times,.95)),
                                       min=float(times.min()),max=float(times.max())),
                    certified_wall_seconds_p50=float(np.median(walls)))
                np.savez(root/f'{mode}_t{threads}_q{scale}.npz',source_id=d['source_id'],
                         winding=result['winding'],component=result['component'],flow=result['flow'])
            report['trials'].append(trial);write(root/'results.json',report)
            print(json.dumps(trial),flush=True)
    write(root/'progress.json',dict(stage='complete'))
    return report
