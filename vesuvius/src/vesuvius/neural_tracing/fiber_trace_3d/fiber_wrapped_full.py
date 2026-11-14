"""Full-source extraction and stage comparison using the existing native tools.

Inputs, paths, resource limits and stage selection are external configuration.
No inference model, solver loss, extraction thresholds or source data are changed.
Completed steps can be resumed only with identical settings.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import json
from pathlib import Path
import subprocess
import tempfile
import time

from lasagna.inference_provenance import atomic_write, sha256_file
from .constraint_extraction import resolve_native_binary


def native_step(config, name, binary, settings, *, timeout=None):
    root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    frozen=root/(name+'.config.json');record=root/(name+'.run.json')
    if record.exists():
        previous=json.loads(record.read_text())
        if previous['returncode']==0 and json.loads(frozen.read_text())==settings:
            print('REUSE '+name,flush=True)
            return previous
        raise FileExistsError('Existing failed/different step: '+str(record))
    if frozen.exists() and json.loads(frozen.read_text())!=settings:
        raise ValueError('Step configuration changed: '+str(frozen))
    atomic_write(frozen,settings,durable=False)
    executable=resolve_native_binary(binary,config)
    binary_digest=sha256_file(executable)
    started=time.monotonic();timed_out=False
    print('START '+name,flush=True)
    tail=deque(maxlen=128);log_errors=[]
    mirror_root=config.get('native_log_mirror')
    if mirror_root:Path(mirror_root).mkdir(parents=True,exist_ok=True)
    with (root/(name+'.log')).open('w') as log, ThreadPoolExecutor(max_workers=1) as logging:
        mirror=(Path(mirror_root)/(name+'.log')).open('w') if mirror_root else None
        # The child's error channel remains a pipe, even if storage rejects a
        # log write. Retain the final diagnostics independently of that file.
        process=subprocess.Popen([executable,str(frozen)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 text=True,errors='replace',bufsize=1)
        def capture():
            targets=[x for x in (log,mirror) if x is not None]
            try:
                for line in process.stdout:
                    tail.append(line.rstrip('\n'))
                    for target in targets[:]:
                        try:target.write(line);target.flush()
                        except OSError as error:
                            log_errors.append(str(error));targets.remove(target)
                            try:target.close()
                            except OSError:pass
            finally:
                process.stdout.close()
                if mirror:
                    try:mirror.close()
                    except OSError as error:log_errors.append(str(error))
        captured=logging.submit(capture)
        atomic_write(root/'status.json',dict(stage=name,pid=process.pid,started_unix=time.time()),durable=False)
        try:
            rc=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out=True;process.terminate()
            try:rc=process.wait(timeout=2)
            except subprocess.TimeoutExpired:process.kill();rc=process.wait()
        except BaseException:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            raise
        captured.result()
    result=dict(returncode=rc,timed_out=timed_out,wall_seconds=time.monotonic()-started,
                binary_sha256=binary_digest,config=str(frozen),log_errors=log_errors)
    if rc or log_errors:result['output_tail']=list(tail)
    atomic_write(record,result,durable=False)
    print('FINISH '+name+' '+json.dumps(result),flush=True)
    if rc:raise RuntimeError('Native step failed: '+name+'; see '+str(root/(name+'.log')))
    return result


def prepare_sources(config):
    from vc_reader.reader.writer import ObjectStore
    root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    published=config['lasagna']
    remote=ObjectStore(published['root'],auth=None)
    metadata=root/'metadata';metadata.mkdir(exist_ok=True)
    for key in (published['manifest'],'inference.json'):
        target=metadata/key
        if not target.exists() and not remote.read_file(key,target):raise FileNotFoundError(key)
    provenance=json.loads((metadata/'inference.json').read_text())
    compound=Path(config['compound']['root'])
    field=json.loads((compound/'.zattrs').read_text())['lasagna']
    lasagna=json.loads((metadata/published['manifest']).read_text())
    expected=config['coordinate_space'].split('/')
    if [provenance['source'][k] for k in ('sample_id','volume_id')]!=expected:
        raise ValueError('Published Lasagna scan differs')
    if lasagna['base_shape_zyx']!=field['base_shape_zyx']:
        raise ValueError('Normal sources have different base shapes')
    channels={c for g in field['groups'].values() for c in g['channels']}
    if not {'sheet_nx','sheet_ny','sheet_grad_mag'}<=channels:
        raise ValueError('Compound prediction lacks sheet normal channels')
    report=dict(published_run=provenance.get('run_uuid'),base_shape_zyx=lasagna['base_shape_zyx'],
        lasagna_manifest_sha256=sha256_file(metadata/published['manifest']),
        compound_manifest_sha256=sha256_file(compound/'.zattrs'),
        winding_source='published Lasagna in both variants; compound sheet_grad_mag is validity only',
        lasagna_streamed=True)
    atomic_write(root/'sources.json',report,durable=False)
    print('SOURCES '+json.dumps(report),flush=True)
    return report


def import_settings(config, target, report, limit=0):
    return dict(config.get('import',{}),source=config['source'],output=str(target),report=str(report),
        volume_id=config['coordinate_space'].split('/')[1],scale_factor=1.,fiber_limit=limit,
        verify_readback=False)


def source_settings(config, variant):
    root=Path(config['output_root'])
    published=config['lasagna']
    winding=published['root'].rstrip('/')+'/'+published['manifest']
    provenance=str(root/'metadata/inference.json')
    if variant=='lasagna':
        return dict(normal_manifest=winding,normal_provenance=provenance,normal_source='lasagna')
    if variant=='compound':
        return dict(normal_manifest=config['compound']['root'],normal_provenance=config['compound']['provenance'],
            normal_source='fiber',winding_manifest=winding,winding_provenance=provenance)
    raise ValueError('Unknown normal source: '+variant)


def extraction_settings(config, variant, *, joint_normals=True):
    root=Path(config['output_root']);out=root/variant;out.mkdir(parents=True,exist_ok=True)
    settings=dict(config.get('extraction',{}),**source_settings(config,variant),
        fibers=str(root/'fibers.vcobj'),coordinate_space=config['coordinate_space'],
        full_volume=True,repetitions=1,remote_cache=str(out/'remote_cache'),
        progress_interval_seconds=5,progress_file=str(out/'extraction_progress.jsonl'),
        database=str(out/'raw.vcobj'),output=str(out/'extraction.json'))
    settings.setdefault('normal_cache_mode','auto')
    settings.setdefault('normal_cache_fraction',0.5)
    settings.setdefault('chunk_size_base_voxels',2048)
    settings['orient_normals']=joint_normals
    if joint_normals:
        settings.update(orient_normals=True,normal_sample_spacing=config.get('extraction',{}).get('normal_sample_spacing',16.),
            database=str(out/'oriented.vcobj'),normal_prior_output=str(out/'outward_priors.json'),
            umbilicus=config['umbilicus'],umbilicus_sha256=sha256_file(config['umbilicus']))
    return settings


def prepare_geometry(config):
    from .fiber_graph_store import store_digest
    from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import fibers_digest
    root=Path(config['output_root']);target=root/'geometry_identity.json'
    identity=store_digest(root/'fibers.vcobj')
    if target.exists():
        report=json.loads(target.read_text())
        if report['store_digest']!=identity:raise ValueError('Source store changed')
        return report
    started=time.monotonic()
    report=dict(store_digest=identity,fibers_digest=fibers_digest(root/'fibers.vcobj'),
                umbilicus_sha256=sha256_file(config['umbilicus']))
    report['seconds']=time.monotonic()-started
    atomic_write(target,report,durable=False)
    print('GEOMETRY IDENTITY '+json.dumps(report),flush=True)
    return report


def solver_settings(config, variant, backend, stage):
    """One end-to-end pipeline: each H/V solve inherits its own sign exclusions."""
    if backend not in ('adam', 'wrapped_irls') or stage not in ('normal_sign', 'classification'):
        raise ValueError('Unknown solver or stage')
    root = Path(config['output_root'])
    source = root / variant
    pipeline = source / backend
    settings = dict(config['solver_parameters'], database=str(source/'oriented.vcobj'),
        optimizer=backend, output_root=str(pipeline/stage),
        devices=config['solver_devices'][backend], gpu_manager=config['gpu_manager'],
        write_result_store=config.get('write_result_store', True))
    if stage == 'normal_sign':
        settings.update(priors=str(source/'outward_priors.json'),
                        tensor_cache=str(source/'normal_tensors.npz'))
    else:
        penalties = settings['broken_penalties']
        if len(penalties) != 1:
            raise ValueError('Full comparison requires one fixed disabling penalty')
        settings.update(features=str(root/'axial_features.json'),
            stage1=str(pipeline/'normal_sign'/f'solution_{penalties[0]:g}.npz'),
            tensor_cache=str(pipeline/'hv_tensors.npz'), cost_modes=['degree_mix'],
            entropy_weights=[config.get('adam_entropy_weight', .01) if backend == 'adam' else 0.])
    return settings


def managed_solver_pool():
    """Each GPU lease owns a distinct process tree; never fork a loaded graph."""
    return ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn'))


def solve_variant(config, variant):
    from . import fiber_normal_sign, fiber_hv_stage
    from .fiber_graph_store import FiberGraph
    backends = ('adam', 'wrapped_irls')
    rows = []
    print('PREPARE GRAPH '+variant, flush=True)
    prepared = time.monotonic()
    graph = FiberGraph(Path(config['output_root'])/variant/'oriented.vcobj')
    print('PREPARED GRAPH '+variant+' '+str(time.monotonic()-prepared)+'s', flush=True)
    # CPU preparation stays outside GPU grants. Independent optimizers get
    # separate nonexclusive, memory-sized grants, never an exclusive parent.
    with managed_solver_pool() as pool:
        for stage, module in [('normal_sign', fiber_normal_sign), ('classification', fiber_hv_stage)]:
            jobs = []
            for backend in backends:
                settings = solver_settings(config, variant, backend, stage)
                summary = Path(settings['output_root'])/'summary.json'
                if summary.exists():
                    saved = json.loads(summary.read_text())
                    if any(saved['config'].get(k) != v for k, v in settings.items()):
                        raise ValueError('Completed solve has different settings: '+str(summary))
                    jobs.append((backend, settings, saved))
                    continue
                module.prepare(settings, graph=graph)
                future = pool.submit(fiber_normal_sign.launch_tensor_solve,
                                     settings, module.__name__, Path(module.__file__))
                jobs.append((backend, settings, future))
            for backend, settings, pending in jobs:
                result = pending if isinstance(pending, dict) else pending.result()
                rows.append(dict(variant=variant, backend=backend, stage=stage,
                    output_root=settings['output_root'], summary=result))
                atomic_write(Path(config['output_root'])/variant/'solver_comparison.json', rows, durable=False)
                print('SOLVER FINISHED '+variant+' '+backend+' '+stage, flush=True)
    return rows


def await_extraction(config, variant):
    """Join an already-running extraction without starting duplicate work."""
    root=Path(config['output_root'])
    name=variant+'_extract'+config.get('extraction_attempt_suffix','')
    record=root/(name+'.run.json')
    started=time.monotonic()
    while not record.exists():
        if time.monotonic()-started > config.get('dependency_timeout_seconds',21600):
            raise TimeoutError('Waiting for '+name)
        print('WAIT '+name+' '+str(round(time.monotonic()-started))+'s',flush=True)
        time.sleep(30)
    result=json.loads(record.read_text())
    if result['returncode']:
        raise RuntimeError('Extraction failed: '+name)
    completed=json.loads((root/(name+'.config.json')).read_text())
    if not completed.get('orient_normals'):
        raise ValueError('Completed extraction lacks the joint normal sweep; run the updated extract workflow')


def report_variant(config, variant):
    """Audit saved hard labels; length retention always uses the original source."""
    import numpy as np
    root=Path(config['output_root']);source=root/variant;rows=[]
    penalty=config['solver_parameters']['broken_penalties'][0]
    with np.load(source/'normal_tensors.npz') as data:
        ids=data['ids'];lengths=data['lengths'];i=data['i'];j=data['j'];dot=data['dot']
        for backend in ('adam','wrapped_irls'):
            pipeline=source/backend
            normal=json.loads((pipeline/'normal_sign/summary.json').read_text())
            hv=json.loads((pipeline/'classification/summary.json').read_text())
            normal_run=normal['runs'][0];hv_run=hv['runs'][0]
            with np.load(pipeline/'normal_sign'/f'solution_{penalty:g}.npz') as s:
                if not np.array_equal(ids,s['ids']):raise ValueError('Normal IDs changed')
                broken=s['broken'];sign=s['normal_sign']
            kept_events=wrong=0
            for start in range(0,len(i),1000000):
                stop=start+1000000;a=i[start:stop];b=j[start:stop]
                kept=~broken[a]&~broken[b]
                kept_events+=int(kept.sum())
                wrong+=int((kept&(dot[start:stop]*sign[a]*sign[b]<0)).sum())
            with np.load(pipeline/'classification'/('solution_'+hv_run['name']+'.npz')) as s:
                if not np.array_equal(ids,s['ids']):raise ValueError('H/V IDs changed')
                labels=s['labels']
                if not np.all(labels[broken]==2):raise ValueError('Normal exclusions changed')
            with np.load(pipeline/'hv_tensors.npz') as h:
                active_labels=labels[h['active']]
                hi,hj,relation=h['i'],h['j'],h['relation']
                counts={name:[0,0] for name in ('parallel','perpendicular')}
                for start in range(0,len(hi),1000000):
                    stop=start+1000000;a=active_labels[hi[start:stop]];b=active_labels[hj[start:stop]]
                    kept=(a!=2)&(b!=2);parallel=relation[start:stop]>0
                    bad=(a==b)!=parallel
                    for name,mask in [('parallel',parallel),('perpendicular',~parallel)]:
                        counts[name][0]+=int((kept&mask).sum())
                        counts[name][1]+=int((kept&mask&bad).sum())
                for name,(retained,violations) in counts.items():
                    metric=hv_run['final'][name]
                    if retained!=metric['retained'] or violations!=metric['violations']:
                        raise ValueError('Saved H/V labels disagree with reported '+name+' counts')
            row=dict(source=variant,backend=backend,fibers=len(ids),
                normal_broken=int(broken.sum()),final_broken=int((labels==2).sum()),
                normal_retained_arc_percent=float(100*lengths[~broken].sum()/lengths.sum()),
                final_retained_arc_percent=float(100*lengths[labels!=2].sum()/lengths.sum()),
                normal_violations=wrong,normal_retained_events=kept_events,
                normal_violation_percent=100*wrong/kept_events if kept_events else None,
                normal_seconds=normal_run['final']['seconds'],hv_seconds=hv_run['final']['seconds'],
                normal_metrics=normal_run['final'],hv_metrics=hv_run['final'],
                peak_allocated_bytes=max(normal['peak_allocated_bytes'],hv['peak_allocated_bytes']),
                normal_iterations=normal_run.get('iterations',normal_run['final']['step']),
                hv_iterations=hv_run.get('iterations',hv_run['final']['step']),
                gpu=hv['gpu'])
            rows.append(row)
    atomic_write(source/'audited_comparison.json',rows,durable=False)
    print('AUDITED '+variant+' '+json.dumps(rows),flush=True)
    return rows


def run(config):
    root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    for action in config['actions']:
        if action=='sources':prepare_sources(config)
        elif action=='import_pilot':
            # Private disposable pilot files are removed on completion/failure.
            with tempfile.TemporaryDirectory(prefix='fiber-import-pilot-',dir=config['pilot_root']) as temporary:
                name=config.get('pilot_name','import_pilot')
                native_step(config,name,'vc_object_import_afv',import_settings(config,
                    Path(temporary)/'fibers.vcobj',root/(name+'.json'),config['pilot_fibers']),timeout=28)
        elif action=='import':
            native_step(config,'import','vc_object_import_afv',import_settings(config,root/'fibers.vcobj',root/'import.json'),
                        timeout=config.get('import_timeout_seconds'))
        elif action=='extraction_pilot':
            with tempfile.TemporaryDirectory(prefix='fiber-extraction-pilot-',dir=config['pilot_root']) as temporary:
                geometry=Path(temporary)/'fibers.vcobj'
                name=config.get('extraction_pilot_name','extraction_pilot')
                native_step(config,name+'_import','vc_object_import_afv',import_settings(config,
                    geometry,root/(name+'_import.json'),config['pilot_fibers']),timeout=28)
                for variant in config.get('variants',['lasagna','compound']):
                    settings=extraction_settings(config,variant)
                    settings.update(fibers=str(geometry),database=str(Path(temporary)/(variant+'.vcobj')),
                        output=str(root/(name+'_'+variant+'.json')),
                        progress_file=str(root/(name+'_'+variant+'_progress.jsonl')))
                    native_step(config,name+'_'+variant,'vc_fiber_constraints',settings,timeout=28)
        elif action=='extract':
            for variant in config.get('variants',['lasagna','compound']):
                settings=extraction_settings(config,variant)
                name=variant+'_extract'+config.get('extraction_attempt_suffix','')
                native_step(config,name,'vc_fiber_constraints',settings,
                            timeout=config.get('extraction_timeout_seconds'))
        elif action=='orient':
            raise ValueError('Normal orientation is included in extract; no second normal sampling pass is needed')
        elif action=='features':
            identity=prepare_geometry(config)
            settings=extraction_settings(config,'lasagna',joint_normals=False);settings.pop('database')
            settings.update(axial_prior_output=str(root/'axial_features.json'),axial_prior_only=True,
                umbilicus=config['umbilicus'],fibers_digest=identity['fibers_digest'],
                umbilicus_sha256=identity['umbilicus_sha256'],output=str(root/'axial_report.json'),
                progress_file=str(root/'axial_progress.jsonl'))
            native_step(config,'features','vc_fiber_constraints',settings)
        elif action=='priors':
            raise ValueError('Outward priors are included in extract; no second normal sampling pass is needed')
        elif action=='solve':
            for variant in config.get('variants',['lasagna','compound']):
                solve_variant(config, variant)
        elif action=='complete':
            for variant in config.get('variants',['lasagna','compound']):
                await_extraction(config,variant)
                run(dict(config,variants=[variant],actions=['solve','report']))
        elif action=='report':
            for variant in config.get('variants',['lasagna','compound']):
                report_variant(config,variant)
        else:raise ValueError('Unknown full-source action: '+action)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    run(json.loads(parser.parse_args().config.read_text()))
