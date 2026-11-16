"""Native network-simplex/cost-scaling comparison on cached flow graphs.

The solver is the LEMON circulation built into ``vc_object_store``
(``solve_winding_min_cost_flow``); quantization here is the explicit
reference the experiments report against.
"""
import json
from pathlib import Path
import time

import numpy as np

from .fiber_min_flow import aggregate_l1_edges, certify_flow, solve
from .fiber_min_flow_experiment import arrays, digest, write
from .fiber_min_flow_benchmark import Paths, graph
from .fiber_min_flow_evaluation import sign_counts


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


def native_solve(nodes, q, backend):
    """Exact integer solve of a quantized graph through the native LEMON binding."""
    from vc_object_store import solve_winding_min_cost_flow, winding_min_cost_flow_available
    if backend not in ('network_simplex','cost_scaling'):
        raise ValueError('Unknown LEMON backend')
    if not winding_min_cost_flow_available():
        raise RuntimeError('vc_object_store was built without LEMON (VC_OBJECT_STORE_WITH_LEMON)')
    shape=np.shape(q['a'])
    if len(shape)!=1 or any(np.shape(q[k])!=shape for k in ('b','target','capacity')):
        raise ValueError('Graph arrays must have matching one-dimensional shapes')
    if nodes < 1 or nodes!=int(nodes):raise ValueError('Positive integer node count required')
    if np.any(q['target'] != np.rint(q['target'])):
        raise ValueError('Integer arc targets required')
    capacity=np.ascontiguousarray(q['capacity'],dtype=np.int64)
    if np.any(capacity<=0):raise ValueError('Quantized capacities must be positive')
    # Capacities are already integers: scale 1 keeps them exactly.
    native=solve_winding_min_cost_flow(int(nodes),np.ascontiguousarray(q['a'],dtype=np.int64),
        np.ascontiguousarray(q['b'],dtype=np.int64),np.ascontiguousarray(q['target'],dtype=np.int64),
        capacity.astype(np.float64),backend=backend,capacity_scale=1)
    winding=np.asarray(native['winding'],np.int64);flow=np.asarray(native['flow'],np.int64)
    timing=native['timing']
    result = certify_flow(nodes,q['a'],q['b'],q['target'],capacity.astype(float),
        np.zeros(len(flow),bool),winding,flow.astype(float),seconds=float(timing['total']),iterations=0)
    result['certificate']['backend'] = 'LEMON '+backend
    result['certificate']['iterations'] = None  # library does not expose iteration count
    result['native_timing'] = dict(graph=float(timing['graph']),setup=0.,solve=float(timing['solve']),export=0.,total=float(timing['total']))
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
                t=time.monotonic(); result=native_solve(len(d['source_id']),q,backend)
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
    if 'progress_interval' in config:
        raise ValueError('progress_interval was removed with the runtime-compiled adapter; the native solve reports no mid-solve snapshots')
    write(root/'progress.json',dict(stage='native_setup',nodes=nodes,arcs=len(q['a']),scale=scale))
    result=native_solve(nodes,q,'cost_scaling')
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
