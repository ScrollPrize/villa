"""Managed, same-graph comparison against an externally supplied IRLS solver.

Load the upstream implementation rather than maintaining a second copy. No
track extraction, winding-inference evidence, segmentation or field filling.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

from .fiber_min_flow_experiment import arrays, digest, write
from .fiber_graph_store import write_winding_result
from .fiber_linear_irls import round_components


def load_solver(path, expected_hash):
    if digest(path) != expected_hash:
        raise ValueError('Upstream solver changed; update the explicit source binding')
    spec = importlib.util.spec_from_file_location('benchmark_sheetgraph', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.solve_l1


def metrics(q, truth, events, enabled):
    """Frozen-GT continuous endpoint protocol, one global integer gauge."""
    from .fiber_gt_constraint_score import enabled_l1
    from .fiber_winding_order import order_margin
    par = np.array([e['parallel'] for e in events], bool)
    fd = np.array([e['family_delta'] for e in events])
    sd = np.array([e['signed_distance'] for e in events])
    required = truth-q+.5*fd-sd
    result = enabled_l1(required, enabled, par)
    shift = result['gt_global_offset']
    if shift is None:
        return result
    # Geometry ordering, NOT the magnitude of measured Lasagna separation.
    sign = np.sign([e['normal_cosine'] for e in events])
    wrong = order_margin(truth-q-shift, sign, fd, par, .5) < 0
    result['signs'] = {}
    for name, mask in [('parallel', par), ('perpendicular', ~par)]:
        mask = mask & enabled & (sign != 0)
        result['signs'][name] = dict(events=int(mask.sum()), violations=int(wrong[mask].sum()),
            percent=float(100*wrong[mask].mean()) if mask.any() else None)
    return result


def solve(config):
    import torch
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from .fiber_min_flow import aggregate_l1_edges, solve as lp_solve
    from .fiber_min_flow_benchmark import Paths, graph
    from .fiber_lemon import build_native, quantize, native_solve
    from .fiber_shared_curve import saved_points
    from .fiber_uv_arc_phase import event_arc_points, event_signed_arcs
    from .fiber_uv_layout import hard_assignment

    torch.set_num_threads(4)
    root = Path(config['output_root'])
    if (root/'summary.json').exists():
        raise FileExistsError('Choose a new output directory')
    d = arrays(config['dataset']['tensor_cache'])
    saved = arrays(config['sgd_solution']) if config.get('sgd_solution') else None
    if saved is not None:
        for key in ('source_id', 'start', 'end', 'family'):
            np.testing.assert_array_equal(d[key], saved[key])
    paths = Paths(config['dataset'], d)
    raw, transport = graph(config['dataset'], d, paths)
    g = aggregate_l1_edges(raw['a'], raw['b'], raw['target'], raw['weight'])
    n = len(d['source_id'])
    _, component = connected_components(coo_matrix((np.ones(len(g['a'])),
        (g['a'], g['b'])), shape=(n,n)), directed=False)
    np.savez(root/'graph.npz', **g, source_id=d['source_id'])
    solver = load_solver(config['solver_source'], config['solver_sha256'])
    history = []
    def log(message):
        history.append(str(message)); print(message, flush=True)
    torch.cuda.synchronize()
    started = time.monotonic()
    x, residual = solver(n, g['a'], g['b'], g['target'], g['weight'],
        np.zeros(n), config.get('prior_weight', 1e-6), dev='cuda',
        irls=config.get('irls',15), cg_iters=config.get('cg',500),
        eps=config.get('eps',.05), log=log)
    torch.cuda.synchronize()
    elapsed = time.monotonic()-started
    if not np.isfinite(x).all():
        raise RuntimeError('IRLS returned nonfinite potentials')
    labels = round_components(x, component)
    np.savez(root/'irls.npz', winding=labels, continuous=x, component=component,
             source_id=d['source_id'], residual=residual)
    write_winding_result(root/'irls.vcobj',config['dataset']['fibers'],d['source_id'],labels,
                         method='irls',family=d['family'],component=component,continuous=x)
    solutions = {'irls_rounded': labels, 'irls_continuous': x}
    times = {'irls': elapsed}
    certificates = {}
    # Fractional-capacity LP provides an exact common-objective certificate.
    if config.get('exact_lp', True):
        started = time.monotonic(); lp = lp_solve(n, **g)
        times['exact_lp'] = time.monotonic()-started
        solutions['exact_lp'] = lp['winding']; certificates['exact_lp'] = lp['certificate']
    capacity_scales = config.get('capacity_scales',[1000,1])
    library = build_native(root/('lemon.dylib' if sys.platform=='darwin' else 'lemon.so')) if capacity_scales else None
    for scale in capacity_scales:
        q, quantization = quantize(g, scale)
        started = time.monotonic()
        result = native_solve(library, n, q, 'cost_scaling')
        key = 'lemon_'+str(scale)
        times[key] = time.monotonic()-started
        certificates[key] = dict(certificate=result['certificate'], quantization=quantization)
        solutions[key] = result['winding']
        np.savez(root/(key+'.npz'), **{k:result[k] for k in ('winding','component')}, source_id=d['source_id'])
        write_winding_result(root/(key+'.vcobj'),config['dataset']['fibers'],d['source_id'],result['winding'],
                             method=key,family=d['family'],component=result['component'])

    events = json.loads(Path(config['events']).read_text())
    if isinstance(events, dict): events = events['events']
    gt_config = json.loads(Path(config['gt_config']).read_text())
    gd = arrays(gt_config['tensor_cache']); gs = arrays(config['gt_solution'])
    _, _, keep, pick = hard_assignment(gs['state'],len(gs['widths']),gt_config,best_nonbroken=True)
    if not keep.all(): raise ValueError('Disabled trusted reference')
    for e in events:
        if d['source_id'][e['test_piece']] != e['test_source'] or gd['source_id'][e['gt_piece']] != e['gt_source']:
            raise ValueError('Cross-event identity mismatch')
    ids = np.array([e['test_piece'] for e in events])
    arc = np.array([e['test_arc'] for e in events])
    truth = event_arc_points(gd, gs['state'], pick, gs['widths'],
        np.array([e['gt_piece'] for e in events]), np.array([e['gt_arc'] for e in events]))
    xyz, turns = paths.at_many(d['source_id'][ids],arc)
    rel = paths.seam.relative(xyz)
    phase = (np.arctan2(rel[:,1],rel[:,0])-paths.seam.angle)%(2*np.pi)/(2*np.pi)
    # Existing graph seam is -pi/2, whereas SGD phase is atan2/(2pi).
    # Subtract the quarter turn, leaving the arbitrary integer gauge to scoring.
    progression = turns+phase+paths.seam.angle/(2*np.pi)
    if saved is not None:
        sgd_q = saved_points(d,saved,saved['state'],saved['widths'],ids,
            event_signed_arcs(d,ids,arc),shared_only=True)[0]
        enabled = saved['state'][:,3] >= 0
    sizes=np.bincount(component)
    main=component==sizes.argmax()
    report = dict(nodes=n, events=len(d['i']), arcs=len(g['a']), seconds=times,
        certificates=certificates, irls_history=history,
        source_sha256=config['solver_sha256'],
        protocol='Frozen GT continuous endpoint, L1-optimal global integer gauge', results={})
    if saved is not None:
        report.update(sgd_enabled_fibers=int(enabled.sum()), sgd_enabled_percent=float(100*enabled.mean()))
    report['component_sizes']=sizes.tolist()
    report['largest_component_fibers']=int(main.sum())
    for name, k in solutions.items():
        objective=float(np.sum(g['weight']*abs(k[g['b']]-k[g['a']]-g['target'])))
        report['results'][name]=dict(graph_l1=objective/float(g['weight'].sum()),
            graph_objective=objective, all_events=metrics(k[ids]+progression,truth,events,np.ones(len(ids),bool)),
            main_component=metrics(k[ids]+progression,truth,events,main[ids]))
        if saved is not None:
            report['results'][name].update(
                sgd_enabled_cohort=metrics(k[ids]+progression,truth,events,enabled[ids]),
                main_component_sgd_enabled=metrics(k[ids]+progression,truth,events,(enabled&main)[ids]))
    if saved is not None:
        report['results']['sgd_saved']=dict(sgd_enabled_cohort=metrics(sgd_q,truth,events,enabled[ids]),
            all_events_diagnostic=metrics(sgd_q,truth,events,np.ones(len(ids),bool)),
            main_component_sgd_enabled=metrics(sgd_q,truth,events,(enabled&main)[ids]))
    if config.get('sgd_report'):
        historical=json.loads(Path(config['sgd_report']).read_text())
        actual=report['results']['sgd_saved']['sgd_enabled_cohort']
        expected=historical['score']
        for key in ('gt_enabled_l1','gt_global_offset','gt_enabled_events'):
            np.testing.assert_allclose(actual[key],expected[key],rtol=1e-10,atol=1e-10)
        report['historical_sgd_seconds']=historical['final']['seconds']
    inputs=[config['dataset']['tensor_cache'],config['events'],
            config['gt_config'],config['gt_solution'],gt_config['tensor_cache']]
    if saved is not None: inputs.append(config['sgd_solution'])
    report['input_sha256']={str(p):digest(p) for p in inputs}
    write(root/'summary.json',report)
    print(json.dumps(report),flush=True)
    return report


def run(config):
    from .fiber_normal_sign import launch_tensor_solve
    return launch_tensor_solve(config,__name__,Path(__file__))


def solve_full(config):
    """Run the same upstream graph solver on an existing full-volume graph."""
    import torch
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    torch.set_num_threads(4)
    root=Path(config['output_root']);source=Path(config['source_root'])
    g=arrays(source/'graph.npz');catalog=arrays(source/'geometry/catalog.npz')
    count=len(catalog['source_id'])
    _,components=connected_components(coo_matrix((np.ones(len(g['a']),np.int8),
        (g['a'],g['b'])),shape=(count,count)),directed=False)
    keep=components==np.bincount(components).argmax()
    remap=np.cumsum(keep)-1;edge=keep[g['a']]&keep[g['b']]
    g={k:v[edge] for k,v in g.items()}
    g['a']=remap[g['a']];g['b']=remap[g['b']]
    n=int(keep.sum())
    write(root/'selection.json',dict(input_fibers=count,fibers=n,arcs=len(g['a']),
        policy='largest component of original unquantized graph'))
    solver=load_solver(config['solver_source'],config['solver_sha256'])
    started=time.monotonic();history=[]
    def log(message):
        row=dict(seconds=time.monotonic()-started,message=str(message))
        history.append(row);write(root/'progress.json',dict(stage='solve',history=history))
        print(json.dumps(row),flush=True)
    x,residual=solver(n,g['a'],g['b'],g['target'],g['weight'],np.zeros(n),
        config.get('prior_weight',1e-6),dev='cuda',irls=config.get('irls',15),
        cg_iters=config.get('cg',500),eps=config.get('eps',.05),log=log)
    torch.cuda.synchronize();seconds=time.monotonic()-started
    if not np.isfinite(x).all():raise RuntimeError('Nonfinite IRLS solution')
    k=round_components(x,np.zeros(n,np.int64))
    np.savez(root/'solution.npz',source_id=catalog['source_id'][keep],family=catalog['family'][keep],
        winding=k,continuous=x,component=np.zeros(n,np.int64))
    objective=float(np.sum(g['weight']*abs(k[g['b']]-k[g['a']]-g['target'])))
    summary=dict(stage='complete',fibers=n,arcs=len(g['a']),seconds=seconds,
        objective_original_weights=objective,normalized_objective=objective/float(g['weight'].sum()),
        history=history,source_sha256=config['solver_sha256'],gt_used=False)
    write(root/'summary.json',summary);write(root/'progress.json',summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--unmanaged',action='store_true')
    parser.add_argument('--devices')
    args=parser.parse_args()
    if not args.unmanaged: raise RuntimeError('Use the managed workflow')
    config=json.loads(Path(args.config).read_text())
    (solve_full if config.get('full_graph') else solve)(config)
