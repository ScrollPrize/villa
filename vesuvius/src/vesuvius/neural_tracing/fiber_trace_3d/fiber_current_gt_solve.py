"""Current three-layer solver on intact reference fibers, without frozen layout."""
import json
from pathlib import Path
import numpy as np
from lasagna.inference_provenance import atomic_write, sha256_file
from . import fiber_soft_winding as soft, fiber_uv_layout as uv
from .fiber_gt_constraint_score import arrays


from .fiber_enclosure_width_init import enclosure_brackets


def enclosure_subset(d):
    targets={};brackets=enclosure_brackets(d)
    for h,v,inner,outer in brackets:
        for edge,value in [(inner,0.),(outer,-1. if d['i'][outer]==h else 1.)]:
            if edge in targets and targets[edge]!=value:raise ValueError('Conflicting enclosure targets')
            targets[edge]=value
    if not targets:raise ValueError('No enclosure brackets')
    ids=np.array(sorted(targets));result=dict(d)
    for key in ('i','j','ds_a','ds_b','sign','distance','parallel','event_ids','phase_a','phase_b','phase_delta'):
        if key in d:result[key]=d[key][ids]
    # This trial's distance argument contains an EXACT signed qb-qa target,
    # never a Lasagna measurement. The special objective reads it accordingly.
    result['distance']=np.array([targets[e] for e in ids],dtype=np.float32)
    result['degree']=np.bincount(np.r_[result['i'],result['j']],minlength=len(d['family'])).astype(np.float32)
    return result,dict(brackets=brackets,original_events=ids.tolist(),targets=result['distance'].tolist(),
        unconnected_sources=d['source_id'][result['degree']==0].tolist())


def run(settings):
    from .fiber_workflows import run as dispatch
    root = Path(settings['output_root']); root.mkdir(parents=True, exist_ok=True)
    gt = Path(settings['gt_root'])
    old = json.loads((gt/'uv_whole/config.json').read_text())
    cfg = json.loads(Path(settings['base_config']).read_text())['fiber_implicit_winding']
    cfg.update(settings.get('parameters',{}))
    if 'fixed_sigma' in settings:
        cfg.update(cavity_winding_sigma=False,learned_winding_sigma=False,winding_sigma=float(settings['fixed_sigma']))
    if (root/'solve/summary.json').exists():
        raise FileExistsError('Completed GT run exists; use its report/visualization directly')
    d = arrays(old['tensor_cache'])
    if cfg.get('gt_pair_mode')=='enclosure':
        d,audit=enclosure_subset(d)
        atomic_write(root/'enclosure_targets.json',audit,durable=False)
    if len(np.unique(d['source_id'])) != len(d['family']) or len(d['links']):
        raise ValueError('GT must consist of whole, unsplit fibers')
    # Reuse only extraction/family/direction metadata, never the old fitted UV.
    # Initialization may come from the caller's tensor metadata, never fitted GT UV.
    centers, widths = uv.radial_initial_geometry(d, cfg)
    d.update(radial_initial_centers=centers, radial_initial_widths=widths)
    np.savez(root/'tensors.npz', **d)
    cfg.update(fibers=old['fibers'], tensor_cache=str(root/'tensors.npz'),
               output_root=str(root/'solve'), initial_solution=str(root/'init/solution.npz'),
               devices=settings.get('devices',cfg.get('devices','0')),
               force_keep_fibers=True)
    if 'steps' in settings:cfg['steps']=settings['steps']
    cfg['gpu_manager']=dict(cfg['gpu_manager'], exclusive=False,
        label='whole-reference-solver')
    init=root/'init'; init.mkdir(exist_ok=True)
    state=np.column_stack((centers,d['phase'],d['z'],np.full(len(centers),100.))).astype(np.float32)
    np.savez(init/'solution.npz', state=state, widths=widths,
             **{k:d[k] for k in ('source_id','start','end','family')})
    np.savez(init/'samples.npz', **soft.prepare_samples(cfg,d))
    atomic_write(init/'config.json',dict(tensor_sha256=sha256_file(root/'tensors.npz'),
        soft_samples=str(init/'samples.npz')),durable=False)
    atomic_write(root/'solve_config.json',{'fiber_implicit_winding':cfg},durable=False)
    dispatch({'fiber_implicit_winding':cfg})
    sol=arrays(root/'solve/solution.npz')
    if not np.all(sol['state'][:,3]==100):raise ValueError('Trusted fibers lost forced-active state')
    if settings.get('winding_reference'):
        winding_reference=Path(settings['winding_reference'])
        from .fiber_gt_winding_reference import run as check_windings
        check_windings(root/'solve',winding_reference)
    if cfg.get('predicted_only_geometry'):
        return json.loads((root/'solve/summary.json').read_text())
    dispatch({'fiber_bucket_uv_report':dict(solve_roots=[str(root/'solve')], output_root=str(root/'metrics'))})
    if settings.get('visualization') is not None:
        dispatch({'fiber_uv_visualization':dict(settings['visualization'],bucket_comparison=str(root/'solve'),
            output_root=str(root/'visualization'))})
    return json.loads((root/'solve/summary.json').read_text())
