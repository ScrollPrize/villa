"""Same-cohort independent winding solves and conditional fixed-label UV fit."""
import json
from pathlib import Path
import numpy as np

from .fiber_min_flow_experiment import arrays, write, digest
from .fiber_min_flow_benchmark import subset_minimal
from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import (
    fiber_volume, fiber_catalog, read_polylines, write_initial_fibers)


def catalog_filter(database, config):
    """Filter whole classified sources, without rerunning extraction or H/V.

    Width is the piece-local arc-weighted mean in base voxels, not the parent
    trace's coverage-policy width. Missing widths never silently pass a cut.
    """
    minimum_length = config.get('minimum_length_mm')
    minimum_width = config.get('minimum_width_base')
    for value in (minimum_length, minimum_width):
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError('Fiber thresholds must be finite and nonnegative')
    voxel_mm = config.get('voxel_size_mm')
    if minimum_length is not None and (voxel_mm is None or not np.isfinite(voxel_mm) or voxel_mm <= 0):
        raise ValueError('Length filtering requires explicit base voxel_size_mm')
    rows = fiber_catalog(database,fiber_volume(database))
    ids = []; lengths = []; missing = 0; total = len(rows)
    for row in rows:
        fid,length = row['id'],row['length']
        if minimum_length is not None and length*voxel_mm <= minimum_length:continue
        if minimum_width is not None:
            record = row['meta'].get('fiber_attributes',{}).get('width_mean',{})
            width = record.get('values')
            if width is None:
                missing += 1
                continue
            if record.get('units') != 'base_voxel' or not np.isscalar(width) or not np.isfinite(width):
                raise ValueError('Invalid piece-local width_mean for fiber '+str(fid))
            if width <= minimum_width:continue
        ids.append(fid); lengths.append(length)
    return np.asarray(ids,dtype='U36'), np.asarray(lengths), dict(
        input_fibers=total, passed=len(ids), missing_width=missing,
        minimum_length_mm=minimum_length, minimum_width_base=minimum_width,
        voxel_size_mm=voxel_mm, comparison='strictly greater',
        width_statistic='piece-local length-weighted mean')


def prepare(config, solver):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from .fiber_min_flow_benchmark import Paths
    from .fiber_flow_sgd_full import measured_geometry
    from .fiber_selected_dataset_eval import remap_events
    from . import fiber_uv_layout as uv, fiber_soft_winding as soft
    root=Path(config['output_root']);source=Path(config['selection_root'])
    d=arrays(source/'selected_tensors.npz')
    ids,lengths,filter_stats=catalog_filter(source/'selected.vcobj',config)
    d=subset_minimal(d,np.isin(d['source_id'],ids))
    n=len(d['source_id'])
    def unavailable(reason, **details):
        report=dict(status='not_solvable',reason=reason,catalog_filter=filter_stats,
                    input_sources=n,gt_in_solve=False,**details)
        write(root/'selection.json',report);write(root/'results.json',report)
        return False
    if not n:return unavailable('No fibers pass the requested length/width filters')
    _,comp=connected_components(coo_matrix((np.ones(len(d['i'])),(d['i'],d['j'])),shape=(n,n)),directed=False)
    largest=comp==np.bincount(comp).argmax()
    d=subset_minimal(d,largest)
    # All three solvers share the SGD-supported cohort; never invent missing
    # perpendicular endpoint geometry. Preserve the largest component again.
    perp=~d['parallel'];owners=np.unique(np.r_[d['i'][perp],d['j'][perp]])
    if not len(owners):return unavailable('Largest component has no perpendicular events; no supported UV cohort',
        largest_component=int(largest.sum()),component_sizes=np.bincount(comp).tolist(),
        component_events=len(d['i']))
    keep=np.zeros(len(d['source_id']),bool);keep[owners]=True
    d=subset_minimal(d,keep)
    n_supported=len(d['source_id'])
    _,comp=connected_components(coo_matrix((np.ones(len(d['i'])),(d['i'],d['j'])),shape=(n_supported,n_supported)),directed=False)
    d=subset_minimal(d,comp==np.bincount(comp).argmax())
    final_lengths=lengths[np.searchsorted(ids,d['source_id'])]
    voxel_mm=filter_stats['voxel_size_mm']
    length_stats = dict(mean_mm=float(final_lengths.mean()*voxel_mm),
        median_mm=float(np.median(final_lengths)*voxel_mm),
        total_m=float(final_lengths.sum()*voxel_mm/1000)) if voxel_mm is not None else None
    write(root/'selection.json',dict(input_sources=n,largest_component=int(largest.sum()),
        perpendicular_supported=n_supported,final_fibers=len(d['source_id']),events=len(d['i']),
        parallel=int(d['parallel'].sum()),perpendicular=int((~d['parallel']).sum()),gt_in_solve=False,
        catalog_filter=filter_stats,lengths=length_stats))
    minimal={k:v.copy() for k,v in d.items()}
    path_cfg=dict(fibers=str(source/'selected.vcobj'),umbilicus=config['umbilicus'])
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
    write_initial_fibers(root/'fibers.vcobj',read_polylines(source/'selected.vcobj',d['source_id']),
                         volume=fiber_volume(source/'selected.vcobj'),packed=True)
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
    solver.update(fibers=str(Path(config['selection_root'])/'selected.vcobj'),umbilicus=config['umbilicus'],
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
            for key in ('minimum_length_mm','minimum_width_base'):
                if old.get(key)!=config.get(key):raise ValueError('Prepared filter mismatch: '+key)
            if not json.loads((prepared/'prepared.json').read_text())['complete']:
                raise ValueError('Incomplete prepared input')
            for name in ('tensors.npz','minimal_tensors.npz','fibers.vcobj','gt_events.json','selection.json','init'):
                dest=root/name
                if not dest.exists():dest.symlink_to(prepared/name)
            write(root/'prepared.json',dict(complete=True,reused_from=str(prepared)))
        elif prepare(config,solver) is False:
            return json.loads((root/'results.json').read_text())
    include_sgd=config.get('include_sgd',True)
    if include_sgd:
        write(root/'solve_config.json',dict(fiber_implicit_winding=solver))
        if not (root/'solve/summary.json').exists():solve_density(solver)
    bench=dict(output_root=str(root/'comparison'),dataset=dict(fibers=str(root/'fibers.vcobj'),
        tensor_cache=str(root/'tensors.npz'),umbilicus=config['umbilicus'],
        perpendicular_relative_weight=solver['perpendicular_relative_weight'],
        perpendicular_offset=solver['perpendicular_offset'],lasagna_winding_scale=solver['lasagna_winding_scale']),
        events=str(root/'gt_events.json'),exact_lp=config.get('exact_lp',True),
        gt_config=str(Path(config['gt_root'])/'uv_whole/config.json'),
        gt_solution=str(Path(config['gt_root'])/'uv_whole/solution.npz'),
        solver_source=config['solver_source'],solver_sha256=config['solver_sha256'],
        irls=15,cg=500,eps=.05,prior_weight=1e-6,capacity_scales=config.get('capacity_scales',[1000]),
        devices=config.get('devices','1'),gpu_manager=config['gpu_manager'])
    if include_sgd:bench['sgd_solution']=str(root/'solve/solution.npz')
    write(root/'comparison_config.json',dict(fiber_irls_benchmark=bench))
    if not (root/'comparison/summary.json').exists():compare(bench)
    # Fixed-label geometry is a separate conditional refinement, not the SGD
    # winding solve above. No selection-time GT winding labels enter this fit.
    for method in config.get('uv_methods',['lemon_1000','irls']):
        parent=root/('fixed_'+method);parent.mkdir(exist_ok=True)
        for name,src in [('selected_with_gt.vcobj',root/'fibers.vcobj'),('selected_tensors.npz',root/'minimal_tensors.npz')]:
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
