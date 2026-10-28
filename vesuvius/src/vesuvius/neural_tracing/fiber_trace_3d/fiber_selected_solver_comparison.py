"""Same-cohort independent winding solves and conditional fixed-label UV fit."""
import json
from pathlib import Path
import sqlite3
import numpy as np

from .fiber_min_flow_experiment import arrays, write, digest
from .fiber_min_flow_benchmark import subset_minimal


def prepare(config, solver):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from .fiber_min_flow_benchmark import Paths
    from .fiber_flow_sgd_full import measured_geometry
    from .fiber_selected_dataset_eval import remap_events
    from . import fiber_uv_layout as uv, fiber_soft_winding as soft
    from vesuvius.neural_tracing.tracers.shared.annotations.afv import read_afv_polylines, write_afv
    root=Path(config['output_root']);source=Path(config['selection_root'])
    d=arrays(source/'selected_tensors.npz')
    with sqlite3.connect(f'file:{source.resolve()}/selected.afv?mode=ro',uri=True) as db:
        ids=np.array([r[0] for r in db.execute('SELECT id FROM fibers ORDER BY id')])
    d=subset_minimal(d,np.isin(d['source_id'],ids))
    n=len(d['source_id'])
    _,comp=connected_components(coo_matrix((np.ones(len(d['i'])),(d['i'],d['j'])),shape=(n,n)),directed=False)
    largest=comp==np.bincount(comp).argmax()
    d=subset_minimal(d,largest)
    # All three solvers share the SGD-supported cohort; never invent missing
    # perpendicular endpoint geometry. Preserve the largest component again.
    perp=~d['parallel'];owners=np.unique(np.r_[d['i'][perp],d['j'][perp]])
    if not len(owners):raise ValueError('Selected source component has no perpendicular events')
    keep=np.zeros(len(d['source_id']),bool);keep[owners]=True
    d=subset_minimal(d,keep)
    n_supported=len(d['source_id'])
    _,comp=connected_components(coo_matrix((np.ones(len(d['i'])),(d['i'],d['j'])),shape=(n_supported,n_supported)),directed=False)
    d=subset_minimal(d,comp==np.bincount(comp).argmax())
    write(root/'selection.json',dict(input_sources=n,largest_component=int(largest.sum()),
        perpendicular_supported=n_supported,final_fibers=len(d['source_id']),events=len(d['i']),
        parallel=int(d['parallel'].sum()),perpendicular=int((~d['parallel']).sum()),gt_in_solve=False))
    minimal={k:v.copy() for k,v in d.items()}
    path_cfg=dict(afv=str(source/'selected.afv'),umbilicus=config['umbilicus'])
    paths=Paths(path_cfg,d)
    for side,key in [('a','i'),('b','j')]:
        xyz,turns=paths.at_many(d['source_id'][d[key]],d['ds_'+side])
        rel=paths.seam.relative(xyz)
        d['phase_'+side]=turns+(np.arctan2(rel[:,1],rel[:,0])-paths.seam.angle)%(2*np.pi)/(2*np.pi)+paths.seam.angle/(2*np.pi)
    d['phase_delta']=(d['phase_b']-d['phase_a']+.5)%1-.5
    d['event_ids']=np.arange(len(d['i']))
    d=measured_geometry(d,paths,d['source_id'],d['family'],d['end'].copy())
    d['degree']=np.bincount(np.r_[d['i'],d['j']],minlength=len(d['source_id'])).astype(np.float32)
    centers,widths=uv.radial_initial_geometry(d,solver)
    d.update(radial_initial_centers=centers,radial_initial_widths=widths)
    np.savez(root/'tensors.npz',**d);np.savez(root/'minimal_tensors.npz',**minimal)
    state=np.column_stack((centers,d['phase'],d['z'],np.full(len(centers),4.))).astype(np.float32)
    init=root/'init';init.mkdir(exist_ok=True)
    np.savez(init/'solution.npz',state=state,widths=widths,**{k:d[k] for k in ('source_id','start','end','family')})
    np.savez(init/'samples.npz',**soft.prepare_samples(solver,d))
    write(init/'config.json',dict(tensor_sha256=digest(root/'tensors.npz'),soft_samples=str(init/'samples.npz')))
    events=remap_events(json.loads((source/'cross/cross_events.json').read_text())['events'],d)
    write(root/'gt_events.json',events)
    write_afv(source/'selected.afv',root/'fibers.afv',read_afv_polylines(source/'selected.afv',d['source_id']),
              dict(selection='GT-free largest perpendicular-supported component; same cohort for all solvers'))
    write(root/'prepared.json',dict(complete=True))


def run(config):
    from .fiber_full_slab import clean_config
    from .fiber_implicit_winding import run as solve_density
    from .fiber_irls_benchmark import run as compare
    from .fiber_flow_uv import run as fit_uv
    root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    request=root/'request.json'
    if request.exists() and json.loads(request.read_text())!=config:
        raise ValueError('Comparison request changed; choose a new output root')
    base=json.loads(Path(config['base_config']).read_text())
    solver=clean_config(base.get('fiber_implicit_winding',base))
    solver.update(afv=str(Path(config['selection_root'])/'selected.afv'),umbilicus=config['umbilicus'],
        tensor_cache=str(root/'tensors.npz'),initial_solution=str(root/'init/solution.npz'),
        soft_samples=str(root/'init/samples.npz'),output_root=str(root/'solve'),
        devices=config.get('devices','1'),gpu_manager=config['gpu_manager'],log_every=100)
    solver.pop('gt_benchmark',None)
    write(request,config)
    if not (root/'prepared.json').exists():
        if config.get('prepared_root'):
            prepared=Path(config['prepared_root']).resolve()
            old=json.loads((prepared/'request.json').read_text())
            for key in ('selection_root','umbilicus','gt_root','base_config'):
                if old[key]!=config[key]:raise ValueError('Prepared input mismatch: '+key)
            if not json.loads((prepared/'prepared.json').read_text())['complete']:
                raise ValueError('Incomplete prepared input')
            for name in ('tensors.npz','minimal_tensors.npz','fibers.afv','gt_events.json','selection.json','init'):
                dest=root/name
                if not dest.exists():dest.symlink_to(prepared/name)
            write(root/'prepared.json',dict(complete=True,reused_from=str(prepared)))
        else:prepare(config,solver)
    include_sgd=config.get('include_sgd',True)
    if include_sgd:
        write(root/'solve_config.json',dict(fiber_implicit_winding=solver))
        if not (root/'solve/summary.json').exists():solve_density(solver)
    bench=dict(output_root=str(root/'comparison'),dataset=dict(afv=str(root/'fibers.afv'),
        tensor_cache=str(root/'tensors.npz'),umbilicus=config['umbilicus'],
        perpendicular_relative_weight=solver['perpendicular_relative_weight'],
        perpendicular_offset=solver['perpendicular_offset'],lasagna_winding_scale=solver['lasagna_winding_scale']),
        events=str(root/'gt_events.json'),exact_lp=config.get('exact_lp',True),
        gt_config=str(Path(config['gt_root'])/'uv_whole/config.json'),
        gt_solution=str(Path(config['gt_root'])/'uv_whole/solution.npz'),
        solver_source=config['solver_source'],solver_sha256=config['solver_sha256'],
        irls=15,cg=500,eps=.05,prior_weight=1e-6,capacity_scales=[1000],
        devices=config.get('devices','1'),gpu_manager=config['gpu_manager'])
    if include_sgd:bench['sgd_solution']=str(root/'solve/solution.npz')
    write(root/'comparison_config.json',dict(fiber_irls_benchmark=bench))
    if not (root/'comparison/summary.json').exists():compare(bench)
    # Fixed-label geometry is a separate conditional refinement, not the SGD
    # winding solve above. No selection-time GT winding labels enter this fit.
    for method in config.get('uv_methods',['lemon_1000','irls']):
        parent=root/('fixed_'+method);parent.mkdir(exist_ok=True)
        for name,src in [('selected_with_gt.afv',root/'fibers.afv'),('selected_tensors.npz',root/'minimal_tensors.npz')]:
            dest=parent/name
            if not dest.exists():dest.symlink_to(src.resolve())
        saved=arrays(root/'comparison'/(method+'.npz'))
        np.testing.assert_array_equal(saved['source_id'],arrays(root/'tensors.npz')['source_id'])
        np.savez(parent/'selected_solution.npz',winding=saved['winding'])
        fit=dict(config['uv_parameters'],flow_root=str(parent),output_root=str(root/('uv_'+method)),
            umbilicus=config['umbilicus'],devices=config.get('devices','1'),gpu_manager=config['gpu_manager'])
        write(root/('uv_'+method+'_config.json'),dict(fiber_flow_uv=fit))
        fit_uv(fit)
    report=json.loads((root/'comparison/summary.json').read_text())
    if include_sgd:
        report['sgd_seconds']=json.loads((root/'solve/summary.json').read_text())['final']['seconds']
    write(root/'results.json',report)
    return report
