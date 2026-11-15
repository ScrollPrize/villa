"""Optional native network-simplex/cost-scaling comparison on cached flow graphs."""
import ctypes
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time

import numpy as np

from .fiber_min_flow import aggregate_l1_edges, certify_flow, solve
from .fiber_min_flow_experiment import arrays, digest, write
from .fiber_min_flow_benchmark import Paths, graph
from .fiber_min_flow_evaluation import sign_counts


def instrument_header(text):
    """Insert read-only hooks; fail closed if the installed algorithm changes."""
    start=text.index('    void startAugment(int max_length) {')
    end=text.index('    void startPush() {',start)
    section=text[start:end]
    phase='        ++eps_phase_cnt;'
    loop='        while (true) {'
    if section.count(phase)!=1 or section.count(loop)!=1:
        raise ValueError('Unsupported LEMON cost-scaling header for progress instrumentation')
    hook='::fiber_lemon_progress(*this, double(_res_node_num) * _alpha, _epsilon, _excess);'
    section=section.replace('      int eps_phase_cnt = 0;', '      int eps_phase_cnt = 0;\n      unsigned long long report_tick = 0;')
    section=section.replace(phase,phase+'\n        '+hook)
    section=section.replace(loop,loop+'\n          if ((++report_tick & 65535) == 0) '+hook)
    return text[:start]+section+text[end:]


def build_native(output, compiler=None, *, progress=False):
    """Explicit optional build; never installs packages or modifies environment."""
    output = Path(output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name('fiber_lemon.cpp')
    cxx = shlex.split(compiler or os.environ.get('CXX', 'c++'))
    flags = shlex.split(subprocess.check_output(['pkg-config','--cflags','--libs','lemon'], text=True))
    # Ubuntu's 1.3.1 lemon.pc incorrectly emits -lemon (libemon), although
    # the installed library is liblemon. Preserve all other package flags.
    flags = ['-llemon' if flag=='-lemon' else flag for flag in flags]
    header_hash=None
    if progress:
        include=Path(subprocess.check_output(['pkg-config','--variable=includedir','lemon'],text=True).strip())
        header=include/'lemon/cost_scaling.h'
        header_hash=digest(header)
        generated=output.parent/'instrumented/lemon/cost_scaling.h'
        generated.parent.mkdir(parents=True,exist_ok=True)
        generated.write_text(instrument_header(header.read_text()))
        header_hash+=':'+digest(generated)
        flags=['-I'+str(generated.parent.parent),*flags]
    version = subprocess.check_output([*cxx, '--version'], text=True).splitlines()[0]
    build = dict(source_sha256=digest(source), compiler=version, compiler_command=cxx, flags=flags,
                 mode='Release -O3 -DNDEBUG -std=c++17; no fast-math',
                 lemon=subprocess.check_output(['pkg-config','--modversion','lemon'],text=True).strip(),
                 progress_header_sha256=header_hash)
    metadata = output.with_suffix('.build.json')
    if output.exists() and metadata.exists() and json.loads(metadata.read_text()) == build:
        return output
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix='lemon-build-', delete=False) as f:
        temporary = Path(f.name)
    try:
        command = [*cxx, '-O3', '-DNDEBUG', '-std=c++17', '-fPIC',
                   '-dynamiclib' if sys.platform=='darwin' else '-shared',
                   str(source), '-o', str(temporary), *flags]
        subprocess.run(command, check=True)
        os.replace(temporary, output)
        write(metadata, build)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def quantize(g, scale):
    """Round nonnegative aggregated capacities to nearest integer, half up."""
    if not np.isfinite(scale) or scale < 1 or scale != int(scale):
        raise ValueError('Capacity scale must be a positive integer')
    w = np.asarray(g['weight'], float)
    if w.ndim!=1 or any(np.shape(g[k])!=w.shape for k in ('a','b','target')):
        raise ValueError('Graph arrays must have matching one-dimensional shapes')
    target=np.asarray(g['target'])
    if not np.isfinite(target).all() or np.any(target!=np.rint(target)):
        raise ValueError('Integer arc targets required')
    if not np.isfinite(w).all() or np.any(w<=0) or np.max(w,initial=0)*scale > 2**52:
        raise ValueError('Invalid or excessive capacities')
    capacity = np.floor(w*scale+.5).astype(np.int64)
    keep = capacity > 0
    q = {k:np.ascontiguousarray(g[k][keep],dtype=np.int64) for k in ('a','b','target')}
    q['capacity'] = capacity[keep]
    # Guard integer sums and cost-scaling potential arithmetic conservatively.
    if int(capacity.max(initial=0))*len(capacity) > 2**60:
        raise ValueError('Potential capacity-sum overflow')
    return q, dict(scale=int(scale), original_arcs=len(w), retained_arcs=int(keep.sum()),
                   dropped_arcs=int((~keep).sum()),
                   dropped_original_weight=float(w[~keep].sum()), original_weight=float(w.sum()),
                   capacity_absolute_error=float(np.abs(capacity/scale-w).sum()))


def native_solve(library, nodes, q, backend, *, progress_path=None, original_graph=None,
                 reference_mean=1., progress_interval=20.):
    if backend not in ('network_simplex','cost_scaling'):
        raise ValueError('Unknown LEMON backend')
    shape=np.shape(q['a'])
    if len(shape)!=1 or any(np.shape(q[k])!=shape for k in ('b','target','capacity')):
        raise ValueError('Graph arrays must have matching one-dimensional shapes')
    if nodes < 1 or nodes!=int(nodes):raise ValueError('Positive integer node count required')
    if np.any(q['target'] != np.rint(q['target'])):
        raise ValueError('Integer arc targets required')
    if int(np.max(abs(q['target']), initial=0))*(nodes+1)**2 > 2**60:
        raise ValueError('Potential cost-scaling overflow')
    lib = ctypes.CDLL(str(library)); fn = lib.fiber_lemon_solve
    vec = np.ctypeslib.ndpointer(dtype=np.int64,ndim=1,flags='C_CONTIGUOUS')
    if progress_path is not None:
        if original_graph is None or reference_mean<=0 or progress_interval<0:
            raise ValueError('Progress requires original graph and positive reference mean')
        original={k:np.ascontiguousarray(original_graph[k],dtype=np.int64) for k in ('a','b','target')}
        original['weight']=np.ascontiguousarray(original_graph['weight'],dtype=np.float64)
        setter=lib.fiber_lemon_set_progress
        setter.argtypes=[ctypes.c_char_p,ctypes.c_int64,ctypes.c_int64,vec,vec,vec,
                         np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS'),
                         ctypes.c_double,ctypes.c_double,ctypes.c_double]
        setter.restype=None
        setter(os.fsencode(progress_path),nodes,len(original['a']),original['a'],original['b'],original['target'],
               original['weight'],float(original['weight'].sum()),reference_mean,progress_interval)
    fn.argtypes = [ctypes.c_int,ctypes.c_int64,ctypes.c_int64,vec,vec,vec,vec,vec,vec,
                  np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS'),
                  ctypes.c_char_p,ctypes.c_size_t]
    fn.restype = ctypes.c_int
    winding = np.empty(nodes,np.int64); flow = np.empty(len(q['a']),np.int64)
    timing = np.zeros(5); error = ctypes.create_string_buffer(2048)
    status = fn(0 if backend=='network_simplex' else 1,nodes,len(flow),q['a'],q['b'],q['target'],
                q['capacity'],winding,flow,timing,error,len(error))
    if status: raise RuntimeError(error.value.decode())
    result = certify_flow(nodes,q['a'],q['b'],q['target'],q['capacity'].astype(float),
        np.zeros(len(flow),bool),winding,flow.astype(float),seconds=float(timing[-1]),iterations=0)
    result['certificate']['backend'] = 'LEMON '+backend
    result['certificate']['iterations'] = None  # library does not expose iteration count
    result['native_timing'] = dict(zip(('graph','setup','solve','export','total'),map(float,timing)))
    return result


def evaluate(d, g, transport, solution, baseline):
    # Component origins are arbitrary. Align each trial component to the baseline
    # for comparisons only; if rounding disconnects the graph, disclose this.
    k = solution['winding'].copy()
    for component in np.unique(solution['component']):
        use = solution['component']==component
        k[use] -= int(np.floor(np.median(k[use]-baseline['winding'][use])))
    local = k[d['j']]-k[d['i']]+transport['fiber_b']-transport['fiber_a']-transport['connector']
    margin = (~d['parallel']) & (d['sign']*(d['family'][d['j']]-d['family'][d['i']])<0)
    measured = abs(local-transport['measured_target'])
    valid = transport['valid']
    objective = float(np.sum(g['weight']*abs(k[g['b']]-k[g['a']]-g['target'])))
    return dict(original_weight_l1_sum=objective, original_weight_l1_mean=objective/float(g['weight'].sum()),
        measured_l1={kind:float(measured[valid&mask].mean()) if np.any(valid&mask) else None
                     for kind,mask in [('parallel',d['parallel']),('perpendicular',~d['parallel'])]},
        signs=sign_counts(d['parallel'],d['sign'],local,margin),
        changed_fibers=int(np.count_nonzero(k!=baseline['winding'])),
        changed_fibers_percent=float(100*np.mean(k!=baseline['winding'])),
        components=int(len(np.unique(solution['component']))),
        cross_component_original_events=int(np.sum(valid & (solution['component'][d['i']]!=solution['component'][d['j']]))),
        gauge='per-component median alignment to baseline for evaluation only')


def run(config):
    root = Path(config['output_root']); root.mkdir(parents=True,exist_ok=True)
    if (root/'request.json').exists():raise ValueError('Choose a fresh benchmark output directory')
    write(root/'request.json',config)
    library = build_native(root/('fiber_lemon.dylib' if sys.platform=='darwin' else 'fiber_lemon.so'))
    dataset = config['dataset']; d=arrays(dataset['tensor_cache'])
    paths=Paths(dataset,d); raw,transport=graph(dataset,d,paths)
    g=aggregate_l1_edges(raw['a'],raw['b'],raw['target'],raw['weight'])
    np.savez(root/'graph.npz',**g)
    baseline=arrays(config['baseline_solution'])
    previous=json.loads(Path(config['baseline_certificate']).read_text())
    repeats=int(config.get('repeats',3))
    if repeats<1:raise ValueError('At least one repeat required')
    # Same aggregated graph/order for LP timing; validate against recorded solution.
    baseline_times=[]
    for _ in range(repeats):
        t=time.monotonic(); fractional=solve(len(d['source_id']),**g)
        baseline_times.append(time.monotonic()-t)
        np.testing.assert_allclose(fractional['certificate']['primal'],previous['primal'],rtol=1e-10)
    fractional_seconds=float(np.median(baseline_times))
    report=dict(nodes=len(d['source_id']),used_events=int(transport['valid'].sum()),
        arcs=len(g['a']),baseline=dict(certificate=fractional['certificate'],
        wall_seconds=fractional_seconds,all_wall_seconds=baseline_times,
        metrics=evaluate(d,g,transport,baseline,baseline)),trials=[])
    write(root/'results.json',report)
    for scale in config.get('scales',[1000,1]):
        q,quantization=quantize(g,scale)
        qg=dict(a=q['a'],b=q['b'],target=q['target'],weight=q['capacity'].astype(float))
        # Separate algorithmic speed from objective quantization: also solve the
        # identical integer-capacity graph with the existing LP backend.
        t=time.monotonic(); lp=solve(len(d['source_id']),**qg); lp_seconds=time.monotonic()-t
        for backend in ('network_simplex','cost_scaling'):
            timings=[]; totals=[]
            for repeat in range(repeats):
                write(root/'progress.json',dict(stage='solve',scale=scale,backend=backend,repeat=repeat+1))
                t=time.monotonic(); result=native_solve(library,len(d['source_id']),q,backend)
                totals.append(time.monotonic()-t);timings.append(result['native_timing'])
                np.testing.assert_allclose(result['certificate']['primal'],lp['certificate']['primal'],rtol=1e-12)
            samples=np.array([x['total'] for x in timings]); kernel=np.array([x['solve'] for x in timings])
            trial=dict(scale=scale,backend=backend,quantization=quantization,
                certificate=result['certificate'],metrics=evaluate(d,g,transport,result,baseline),
                native_seconds=dict(mean=float(samples.mean()),p50=float(np.median(samples)),
                                    p95=float(np.quantile(samples,.95)),min=float(samples.min()),max=float(samples.max())),
                algorithm_seconds_p50=float(np.median(kernel)),
                certified_wall_seconds_p50=float(np.median(totals)),repeats=repeats,
                all_timings=timings,lp_same_capacity_seconds=lp_seconds,
                speedup_native_vs_fractional_lp=fractional_seconds/float(np.median(samples)))
            np.savez(root/f'{backend}_{scale}.npz',source_id=d['source_id'],
                     winding=result['winding'],component=result['component'],flow=result['flow'])
            report['trials'].append(trial);write(root/'results.json',report)
            print(json.dumps(trial),flush=True)
    write(root/'progress.json',dict(stage='complete'))
    return report


def run_full(config):
    """Solve an existing aggregated full graph, with no extraction or LP rerun."""
    root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    source=Path(config['source_root'])
    if (root/'request.json').exists():raise ValueError('Choose a fresh full-solve output directory')
    write(root/'request.json',dict(config=config,graph=dict(path=str(source/'graph.npz'),
        size=(source/'graph.npz').stat().st_size,mtime_ns=(source/'graph.npz').stat().st_mtime_ns)))
    if config.get('experiment_link'):Path(config['experiment_link']).symlink_to(root,target_is_directory=True)
    started=time.monotonic()
    write(root/'progress.json',dict(stage='load_graph'))
    g=arrays(source/'graph.npz');catalog=arrays(source/'geometry/catalog.npz')
    nodes=len(catalog['source_id']);scale=config.get('scale',1)
    q,quantization=quantize(g,scale)
    write(root/'quantization.json',quantization)
    library=build_native(root/('fiber_lemon.dylib' if sys.platform=='darwin' else 'fiber_lemon.so'),progress=True)
    write(root/'progress.json',dict(stage='native_setup',nodes=nodes,arcs=len(q['a']),scale=scale))
    result=native_solve(library,nodes,q,'cost_scaling',progress_path=root/'progress.json',
        original_graph=g,reference_mean=config['reference_mean'],progress_interval=config.get('progress_interval',20.))
    k=result['winding'];objective=0.
    for start in range(0,len(g['a']),1000000):
        sl=slice(start,start+1000000)
        objective+=float(np.sum(g['weight'][sl]*abs(k[g['b'][sl]]-k[g['a'][sl]]-g['target'][sl])))
    mean=objective/float(g['weight'].sum())
    np.savez(root/'solution.npz',source_id=catalog['source_id'],family=catalog['family'],
             winding=k,component=result['component'],flow=result['flow'])
    summary=dict(stage='complete',fibers=nodes,arcs=len(q['a']),quantization=quantization,
        certificate=result['certificate'],native_timing=result['native_timing'],seconds=time.monotonic()-started,
        objective_original_weights=objective,normalized_objective=mean,
        factor_vs_small=mean/config['reference_mean'],gt_used=False)
    write(root/'summary.json',summary);write(root/'progress.json',summary)
    print(json.dumps(summary),flush=True)
    return summary
