"""Experimental neural followers through the canonical native reference replay.

No model training or prediction-volume inference. Native CT readers share VC3D's
cache. Native reference selection, failure/grace/reset and report code are reused.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def implementation_identity(binary):
    """Prevent resumed cases from silently mixing evaluator/model revisions."""
    package = Path(__file__).resolve().parent
    files = list((package.parent / 'fiber_follow').rglob('*.py'))
    files += list(package.glob('neural_follow_*.py'))
    files += [package / 'replay_metrics.py', Path(binary).resolve()]
    native = Path(os.environ['FIBER_FOLLOW_NATIVE_DIR'])
    files += list(native.glob('*.so'))
    files += list(native.parents[1].rglob('*.so*'))
    return {str(path): digest(path) for path in sorted(set(files)) if path.is_file()}


def family_of(document):
    classification = document.get('hv_classification', {})
    family = classification.get('manual_tag') or classification.get('automatic_tag')
    if family not in ('H', 'V'):
        raise ValueError('reference annotation must have explicit H/V classification')
    return family


def prepare_cases(config, binary):
    source_config = json.loads(Path(config['reference_config']).read_text())
    plan = json.loads(subprocess.check_output([str(binary), 'replay-references', '--config',
                                             config['reference_config']], text=True))
    sources = {s['id']: s for s in source_config['sources']}
    cases = []
    for group in plan['sources']:
        source = sources[group['source_id']]
        for source_index, fiber in enumerate(group['fibers']):
            document = json.loads(Path(fiber['fiber_path']).read_text())
            family = family_of(document)
            for run_index, run in enumerate(fiber['runs']):
                for reverse in (False, True):
                    points = run['points_base_xyz']
                    cases.append(dict(index=len(cases), source_id=source['id'], source=source,
                        source_index=source_index, run_index=run_index, reverse=reverse,
                        family=family, fiber_path=fiber['fiber_path'], fiber_sha256=digest(fiber['fiber_path']),
                        reference_run={k: v for k, v in run.items() if k != 'points_base_xyz'},
                        points=points[::-1] if reverse else points))
    if config.get('expected_cases') is not None and len(cases) != config['expected_cases']:
        raise ValueError(f'expected {config["expected_cases"]} directed cases, found {len(cases)}')
    if config.get('case_ids') is not None:
        selected = set(config['case_ids'])
        cases = [case for case in cases if case['index'] in selected]
        if len(cases) != len(selected):
            raise ValueError('unknown requested pilot case')
    return source_config, cases


def worker(device, config, pending, completed):
    # Spawned only inside manager-owned process; no CUDA initialization before grant.
    import numpy as np
    import torch
    from .neural_follow_native import load_native
    load_native()
    from vc.volume import Volume, NormalSampler, set_chunk_cache_io_threads, set_chunk_cache_decode_threads, set_chunk_cache_budget
    from .neural_follow_reader import NativeArray
    from .neural_follow_batch import ForwardQueue
    if config.get('backend') == 'legacy':
        from ..fiber_follow.legacy.infer import load_checkpoint
    else:
        from ..fiber_follow.train.train import load_checkpoint
    from ..fiber_follow.data.volume import FiberVolume, FiberVolumeSpec
    from ..fiber_follow.data.observations import FiberTracer
    from ..fiber_follow.tracing.trace import TraceParams
    from ..fiber_follow.tracing.policy import checkpoint_policy
    from ..fiber_follow.tracing.heading import ct_seed_heading, SeedHeadingError
    from ..fiber_follow.data.ct_normalization import volume_key, ZSCORE_METHOD, ZSCORE_EPSILON
    from lasagna.inference_provenance import atomic_write

    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    cuda = f'cuda:{device}'
    root = Path(config['output'])
    try:
        model, crop, history, saved_spec, checkpoint = load_checkpoint(config['checkpoint'], cuda)
        expected_heading = model.cfg.frame_checkpoint_sha256
        if digest(config['heading_checkpoint']) != expected_heading:
            raise ValueError('heading model differs from follower training binding')
        model.cfg.frame_checkpoint = config['heading_checkpoint']
        policy = checkpoint_policy(checkpoint, model.cfg, **config.get('policy_overrides', {}))
        set_chunk_cache_budget(config.get('decoded_cache_bytes_per_gpu', 8 << 30))
        set_chunk_cache_io_threads(config.get('io_threads_per_gpu', 24))
        set_chunk_cache_decode_threads(8)
        pool = ThreadPoolExecutor(config.get('input_workers_per_gpu', 24))
        volumes = {}
        # Open each source once before case threads; only metadata, not voxel scans.
        for source in config['sources']:
            native = Volume.open_url(source['input_volume']) if '://' in source['input_volume'] else Volume.open(source['input_volume'])
            ct_level = config['ct_level']
            array = NativeArray(native, ct_level)
            spec = FiberVolumeSpec(**{**vars(saved_spec), 'ct_zarr': source['input_volume'],
                'ct_level': ct_level, 'grid_scale': config['trace_grid_base'],
                'ct_grid_scale': config['ct_grid_base'], 'ct_normalization':None})
            spec.ct_normalization = dict(method=ZSCORE_METHOD, epsilon=ZSCORE_EPSILON,
                volume=volume_key(spec), shape=list(array.shape), chunks=list(array.chunks), dtype=array.dtype.str)
            if checkpoint['ct_normalization']['method'] != ZSCORE_METHOD:
                raise ValueError('unsupported model CT normalization')
            volume = FiberVolume(spec, array_factory=lambda level, native=native: NativeArray(native, level))
            volumes[source['id']] = volume, NormalSampler(source['normal_manifest'])
        # One model, one forward queue per GPU; every case retains its own
        # unmodified upstream rollout, features/history and decision counter.
        first_volume = next(iter(volumes.values()))[0]
        params = TraceParams.from_policy(policy, precision=config.get('precision','bf16'), seed=config.get('sampling_seed', 0))
        executor = FiberTracer(model, first_volume, crop, history, params, device=cuda)
        forward = ForwardQueue(executor.forward, config.get('max_batch_size',8),cuda)

        def case_loop():
            with torch.inference_mode():
                while True:
                    case = pending.get()
                    if case is None:
                        return
                    started=time.monotonic()
                    try:
                        volume, normals = volumes[case['source_id']]
                        g = volume.spec.grid_scale
                        tracer = FiberTracer(model,volume,crop,history,copy.copy(params),device=cuda)
                        tracer.pool.shutdown()
                        tracer.pool = pool
                        tracer.forward = forward
                        points = np.ascontiguousarray(case['points'],dtype=np.float64)

                        def segment(start,tangent,remaining,max_steps,observe):
                            seed = np.asarray(start,dtype=np.float64)/g
                            try:
                                heading=ct_seed_heading(volume,seed,case['family'])
                            except SeedHeadingError:
                                return 'seed_heading_unavailable'
                            # GT provides only traversal sign, never frame/history.
                            if np.dot(heading,tangent)<0:
                                heading=-heading
                            tracer.p.max_len = config.get('max_len', 100000.)
                            sent=1
                            def abort(_index,path):
                                nonlocal sent
                                new=np.ascontiguousarray(np.asarray(path[sent:])*g,dtype=np.float64)
                                first=sent
                                sent=len(path)
                                if len(new) and not observe(new,first):
                                    return True
                                return sent>=max_steps
                            _, reasons=tracer._trace(np.asarray([seed]),np.asarray([heading]),
                                None,families=[case['family']],on_commit=abort)
                            return reasons[0]

                        settings=config['replay']
                        measured=json.loads(normals.replay(points,segment,
                            nominal_step_base=settings['nominal_step_base'],
                            threshold_base=settings['threshold_base'], grace_base=settings['grace_base'],
                            tangential_factor=settings['tangential_factor'],match_refine_steps=settings['match_refine_steps']))
                        outcome=measured['case']
                        outcome.update(source_index=case['source_index'],run_index=case['run_index'],
                            direction='reverse' if case['reverse'] else 'forward',reference_run=case['reference_run'])
                        # Native measurement uses a directed reference. Restore
                        # original-source arcs for reverse inspection only.
                        if case['reverse']:
                            length=outcome['reference_length_base_voxels']
                            for failure in outcome['failures']:
                                failure['source_reference_arc_base_voxels']=length-failure['directional_reference_arc_base_voxels']
                                failure['source_reference_arc_fraction']=1-failure['directional_reference_arc_fraction']
                        artifact=dict(format='vc_native_replay_artifact',version=1,
                            reference_points_base_xyz=case['points'],replay=measured['replay'],
                            provenance=dict(source_id=case['source_id'],fiber_path=case['fiber_path'],
                                reverse=case['reverse'],case_index=case['index'],reference_run=case['reference_run'],
                                normal_manifest=case['source']['normal_manifest'],
                                base_voxel_size_um=case['source']['base_voxel_size_um'],
                                ct_volume=case['source']['input_volume'],trace_to_base=1.,
                                neural_follow=config['identity'],family=case['family']))
                        folder=root/'cases'/str(case['index'])
                        atomic_write(folder/'trace.json',artifact,durable=False)
                        record=dict(index=case['index'],source_id=case['source_id'],outcome=outcome,
                            wall_seconds=time.monotonic()-started)
                        atomic_write(folder/'result.json',record,durable=False)
                        completed.put(record)
                    except BaseException as error:
                        completed.put(dict(index=case['index'],error=repr(error)))
                        raise
        try:
            with ThreadPoolExecutor(config.get('case_workers_per_gpu',16)) as threads:
                futures=[threads.submit(case_loop) for _ in range(config.get('case_workers_per_gpu',16))]
                for future in futures:
                    future.result()
        finally:
            forward.close()
            executor.close()
            pool.shutdown()
        completed.put(dict(worker=device,batch_sizes=forward.batch_sizes,finished=True))
    except BaseException as error:
        completed.put(dict(worker=device,error=repr(error)))
        raise


def run(config, devices):
    from lasagna.inference_provenance import atomic_write
    from .replay_metrics import correct_summary, complete_report
    root=Path(config['output'])
    root.mkdir(parents=True,exist_ok=True)
    source_config,cases=prepare_cases(config,config['native_binary'])
    config=copy.deepcopy(config)
    ids={case['source_id'] for case in cases}
    config['sources']=[s for s in source_config['sources'] if s['id'] in ids]
    config['identity']=dict(checkpoint_sha256=digest(config['checkpoint']),
        heading_sha256=digest(config['heading_checkpoint']),upstream='853dcf536',
        backend=config.get('backend', 'current'),sampling_seed=config.get('sampling_seed',0),
        policy_overrides=config.get('policy_overrides',{}),trace_grid_base=config['trace_grid_base'],
        ct_level=config['ct_level'],ct_grid_base=config['ct_grid_base'],precision=config.get('precision','bf16'),
        seed_heading='ct_hv_with_reference_traversal_sign',replay=config['replay'],
        implementation=implementation_identity(config['native_binary']))
    binding=dict(config=config,references=[{k:v for k,v in c.items() if k!='source'} for c in cases])
    path=root/'binding.json'
    if path.exists() and json.loads(path.read_text())!=binding:
        raise ValueError('evaluation binding changed; use a fresh output directory')
    atomic_write(path,binding,durable=False)
    records={}
    unfinished=[]
    for case in cases:
        result=root/'cases'/str(case['index'])/'result.json'
        if result.exists() and (result.parent/'trace.json').exists():
            records[case['index']]=json.loads(result.read_text())
        else:
            unfinished.append(case)
    started=time.monotonic()
    if unfinished:
        context=mp.get_context('spawn')
        pending,completed=context.Queue(),context.Queue()
        for case in sorted(unfinished,key=lambda c:-len(c['points'])):
            pending.put(case)
        for _ in range(len(devices)*config.get('case_workers_per_gpu',16)):
            pending.put(None)
        processes=[context.Process(target=worker,args=(device,config,pending,completed)) for device in devices]
        for process in processes: process.start()
        finished=0
        try:
            while finished<len(processes):
                from queue import Empty
                try: record=completed.get(timeout=5)
                except Empty:
                    if any(p.exitcode not in (None,0) for p in processes):
                        raise RuntimeError('neural worker exited; see worker traceback')
                    continue
                if 'error' in record: raise RuntimeError(str(record))
                if record.get('finished'):
                    finished+=1
                    atomic_write(root/f'worker{record["worker"]}.json',record,durable=False)
                else:
                    records[record['index']]=record
                    atomic_write(root/'progress.json',dict(completed=len(records),total=len(cases),
                        seconds=time.monotonic()-started),durable=False)
                    print(f'Neural replay {len(records)}/{len(cases)} case={record["index"]} '
                          f'events={len(record["outcome"]["failures"])} seconds={record["wall_seconds"]:.1f}',flush=True)
        finally:
            for process in processes:
                if finished<len(processes) and process.is_alive(): process.terminate()
            for process in processes: process.join()
    sources=[]
    for source in config['sources']:
        outcomes=[records[c['index']]['outcome'] for c in cases if c['source_id']==source['id']]
        sources.append(dict(source_id=source['id'],backend_config=source,cases=outcomes,
            summary=dict(base_voxel_size_um=source['base_voxel_size_um'],
                directed_reference_length_base_voxels=math.fsum(c['reference_length_base_voxels'] for c in outcomes),
                total_failures=sum(len(c['failures']) for c in outcomes))))
    summary=dict(completed=True,cases=len(cases),evaluated_cases=len(cases),sources=sources,
        failures=sum(len(r['outcome']['failures']) for r in records.values()),
        directed_reference_length_mm=math.fsum(c['reference_length_base_voxels']*s['summary']['base_voxel_size_um']/1000
            for s in sources for c in s['cases']),reference_selection_policy='cp-bounded-no-lasagna-v1',
        wall_seconds=time.monotonic()-started,neural_follow=config['identity'])
    summary=correct_summary(summary)
    atomic_write(root/'summary.json',summary,durable=False)
    atomic_write(root/'report.json',complete_report(summary),durable=False)
    print(json.dumps(complete_report(summary),indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--unmanaged',action='store_true')
    parser.add_argument('--devices',default='0,1,2,3,4,5')
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    if not args.unmanaged:
        from .managed_evaluation import launch_managed
        raise SystemExit(launch_managed([sys.executable,'-m',__spec__.name,'--config',args.config],
            devices=args.devices,settings=config['gpu_manager']))
    run(config,[int(x) for x in args.devices.split(',')])


if __name__=='__main__': main()
