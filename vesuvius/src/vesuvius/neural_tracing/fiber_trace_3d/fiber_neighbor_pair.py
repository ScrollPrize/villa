"""Fresh paired evaluation of an exported neighbor dataset, then standard review."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from lasagna.inference_provenance import atomic_write, sha256_file
from .fiber_full_slab import clean_config
from .fiber_gt_constraint_score import arrays
from .fiber_anchored_dataset import induced_graph, perpendicular_supported_graph
from .fiber_selected_dataset_eval import remap_events


def run(settings):
    dataset=Path(settings['dataset']);root=Path(settings['output_root'])
    root.mkdir(parents=True,exist_ok=True)
    if settings.get('wait_for_dataset',False):
        print('Waiting for completed dataset export',flush=True)
        while not (dataset/'summary.json').exists():time.sleep(2)
    # Completion and artifact validity matter, not which expansion round or
    # stopping rule produced the caller's dataset.
    json.loads((dataset/'summary.json').read_text())
    cfg=clean_config(json.loads(Path(settings['base_config']).read_text())['fiber_implicit_winding'])
    if settings.get('optimum'):
        cfg.update(json.loads(Path(settings['optimum']).read_text())['parameters'])
    cfg.update(settings.get('parameters',{}))
    for key in ('force_keep_source_ids','force_keep_fibers','freeze_source_ids','freeze_widths'):
        cfg.pop(key,None)
    cfg['gpu_manager']=dict(cfg['gpu_manager'],exclusive=False)
    paths=[Path(settings['base_config']),dataset/'summary.json',
           dataset/'tensors.npz',dataset/'cross/cross_events.json',Path(__file__)]
    if settings.get('optimum'):paths.append(Path(settings['optimum']))
    from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import fibers_digest
    signature=dict(settings=settings,inputs={str(p):sha256_file(p) for p in paths},
                   selected_fibers=fibers_digest(dataset/'selected.vcobj'))
    manifest=root/'inputs.json'
    if manifest.exists() and json.loads(manifest.read_text())!=signature:raise ValueError('Changed paired inputs')
    atomic_write(manifest,signature,durable=False)
    base=root/'base.json';atomic_write(base,dict(fiber_implicit_winding=cfg),durable=False)
    # Reuse exactly the established whole-fiber/perpendicular-support selection.
    parent=arrays(dataset/'tensors.npz')
    d=induced_graph(parent,np.unique(parent['source_id']),cfg)
    d,audit=perpendicular_supported_graph(d,cfg)
    prepared=root/'prepared';prepared.mkdir(exist_ok=True)
    np.savez(prepared/'tensors.npz',**d)
    events=remap_events(json.loads((dataset/'cross/cross_events.json').read_text())['events'],d)
    atomic_write(prepared/'gt_events.json',events,durable=False)
    atomic_write(prepared/'selection.json',audit,durable=False)
    def execute(name,request,done):
        path=root/(name+'.json');atomic_write(path,request,durable=False)
        if not done.exists():
            print('Starting '+name,flush=True)
            with (root/(name+'.log')).open('a') as log:
                subprocess.run([sys.executable,'-m','vesuvius.neural_tracing.fiber_trace_3d.fiber_workflows','--config',str(path)],
                               stdout=log,stderr=subprocess.STDOUT,check=True)
        return json.loads(done.read_text())
    def arm(joint):
        name='joint' if joint else 'free';out=root/name
        args=dict(base_config=str(base),gt_root=settings['gt_root'],output_root=str(out),
                  devices=settings['devices'][int(joint)])
        if joint:
            args['baseline_root']=str(prepared)
            execute(name+'_run',dict(fiber_joint_gt_eval=args),out/'results.json')
            return execute(name+'_report',dict(fiber_joint_gt_eval=dict(args,report_only=True)),out/'comparison.json')
        args['dataset']=str(dataset)
        result=execute(name+'_run',dict(fiber_selected_dataset_eval=args),out/'results.json')
        signs=execute(name+'_sign',dict(fiber_density_sign_report=dict(gt_evaluations=[str(out)],
                      output_root=str(out/'gt_signs'))),out/'gt_signs/summary.json')
        return dict(result,signs=signs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        free=pool.submit(arm,False);joint=pool.submit(arm,True)
        reports=dict(free=free.result(),joint=joint.result())
    atomic_write(root/'results.json',reports,durable=False)
    vis=root/'visualization'
    execute('visualization',dict(paired_winding_review=dict(free=str(root/'free/solve'),
            joint=str(root/'joint/solve'),output_root=str(vis))),vis/'summary.json')
    print(json.dumps(dict(completed=True,output_root=str(root),visualization=str(vis))),flush=True)
    return reports
