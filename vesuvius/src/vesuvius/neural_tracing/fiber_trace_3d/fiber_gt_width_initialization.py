"""Prepare width-only perturbations of the unfitted intact-GT initialization."""
import json
from pathlib import Path
import numpy as np
from lasagna.inference_provenance import atomic_write, sha256_file
from .fiber_gt_constraint_score import arrays


def perturb_widths(original, trial, radial):
    if trial['kind']=='scale':
        value=original*float(trial['factor'])
    elif trial['kind']=='radial':
        value=np.asarray(radial,dtype=original.dtype).copy()
    else:
        raise ValueError('Unknown width initialization '+trial['kind'])
    if value.shape!=original.shape or not np.isfinite(value).all() or not (value>0).all():
        raise ValueError('Perturbed widths must be positive finite and retain shape')
    return value


def run(settings):
    cfg=json.loads(Path(settings['base_recipe']).read_text())['fiber_implicit_winding']
    original=arrays(cfg['initial_solution'])
    radial=arrays(settings['radial_initial_solution'])['widths']
    root=Path(settings['output_root']);root.mkdir(parents=True,exist_ok=True)
    manifest={}
    for trial in settings['trials']:
        dest=root/trial['name']
        if dest.exists():raise FileExistsError(dest)
        dest.mkdir()
        changed=dict(original,widths=perturb_widths(original['widths'],trial,radial))
        np.savez(dest/'initial.npz',**changed)
        saved=arrays(dest/'initial.npz')
        assert all(np.array_equal(saved[k],v) for k,v in original.items() if k!='widths')
        recipe=dict(cfg,**settings.get('parameters',{}))
        recipe.update(trial.get('parameters',{}))
        recipe.update(initial_solution=str(dest/'initial.npz'),output_root=str(dest/'solve'),devices=trial['device'])
        recipe.pop('initial_solution_sha256',None)
        atomic_write(dest/'config.json',recipe,durable=False)
        atomic_write(dest/'recipe.json',{'fiber_implicit_winding':recipe},durable=False)
        record=dict(trial,initial_solution=str(dest/'initial.npz'),recipe=str(dest/'recipe.json'),
                    solve=str(dest/'solve'),source=cfg['initial_solution'],
                    source_sha256=sha256_file(cfg['initial_solution']),
                    width_factor_min=float((saved['widths']/original['widths']).min()),
                    width_factor_max=float((saved['widths']/original['widths']).max()),
                    unchanged_non_width_arrays=True)
        atomic_write(dest/'perturbation.json',record,durable=False)
        manifest[trial['name']]=record
    atomic_write(root/'manifest.json',manifest,durable=False)
    return manifest
