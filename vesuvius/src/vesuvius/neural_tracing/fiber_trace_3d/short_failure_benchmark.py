"""Build a directed regression benchmark from saved beam failures.

This selects difficult examples; it is not an unbiased evaluation dataset.
All actual tracing uses the ordinary native replay implementation.
"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
import json
from pathlib import Path
from threading import Event
import time

import numpy as np

from lasagna.inference_provenance import atomic_write, json_digest, sha256_file
from .cached_trace_profile import initialize_trial_worker, execute_trial_replay, close_trial_worker
from .replay_metrics import _episode_indices

PROTOCOL = dict(version=1, start_step=100, start_padding=100, end_padding=100,
                match_tolerance=100, dedup_distance=200, first_failure_only=True,
                require_exhausted_grace=True)


class TrialJobs:
    """Bounded jobs with disjoint GPU lanes and cache-writer ownership.

    Only the coordinator submits/collects. Worker exceptions cancel every sibling
    before the outer manager grant can be released.
    """
    def __init__(self, devices, resources, cancel_event, affinity=None):
        self.devices = list(devices)
        self.resources = resources
        self.cancel_event = cancel_event
        self.affinity = affinity
        self.pool = ThreadPoolExecutor(max_workers=len(self.devices), thread_name_prefix='failure-trial')
        self.pending = {}
        self.last_model = {}

    def submit(self, model, function, *args):
        held = set().union(*(self.resources[m] for m, _ in self.pending.values()))
        if model in self.models or held.intersection(self.resources[model]):
            return False
        used = {lane for _, lane in self.pending.values()}
        lane = (self.affinity[model] if self.affinity is not None else
                next((i for i in range(len(self.devices)) if i not in used), None))
        if lane in used:
            return False
        if lane is None:
            return False
        future = self.pool.submit(function, model, *args, self.devices[lane])
        self.pending[future] = (model, lane)
        self.last_model[lane] = model
        return True

    def prefer_warm(self,model):
        if self.affinity is None:
            return False
        lane=self.affinity[model]
        return self.last_model.get(lane)==model

    @property
    def models(self):
        return {model for model, _ in self.pending.values()}

    def collect(self):
        if not self.pending:
            return
        done, _ = wait(self.pending, timeout=0.25, return_when=FIRST_COMPLETED)
        for future in done:
            del self.pending[future]
            future.result()

    def close(self):
        self.cancel_event.set()
        self.pool.shutdown(wait=True, cancel_futures=True)


def cache_affinity(models,resources,max_lanes):
    """Co-locate transitive shared-cache owners, balance components over lanes."""
    groups = []
    for model in models:
        members, caches = {model}, set(resources[model])
        remaining = []
        for old_members, old_caches in groups:
            if caches.intersection(old_caches):
                members.update(old_members); caches.update(old_caches)
            else:
                remaining.append((old_members,old_caches))
        groups = remaining+[(members,caches)]
    count = min(max_lanes,len(groups))
    loads, result = [0]*count, {}
    for members,_ in sorted(groups,key=lambda g:(-len(g[0]),sorted(g[0]))):
        lane = min(range(count),key=lambda i:loads[i])
        for model in members: result[model] = lane
        loads[lane]+=len(members)
    return result,count


def advance_search(candidate, record, positions, result):
    """Consume available trial audits in grid order, never infer missing results."""
    for direction in ('forward', 'reverse'):
        if direction in record:
            continue
        n = positions.get(direction, 100)
        a, b = candidate['run']['original_arcs_base']
        target = candidate['target']
        extent = target-a if direction == 'forward' else b-target
        tail = min(100, b-target if direction == 'forward' else target-a)
        while True:
            other = record.get('reverse' if direction == 'forward' else 'forward')
            if n+100 > extent or (other and n+100+tail >
                    other['padded']['original_arcs_base'][1]-other['padded']['original_arcs_base'][0]+1e-7):
                record[direction] = None
                break
            minimal, padded = result(candidate, direction, n), result(candidate, direction, n+100)
            if minimal is None or padded is None:
                break
            if minimal['valid'] and padded['valid']:
                record[direction] = dict(minimal=minimal, padded=padded)
                break
            n += 100
        positions[direction] = n


def slice_line(points, lo, hi):
    points = np.asarray(points, dtype=np.float64)
    arcs = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    keep = np.r_[True, np.diff(arcs) > 0]
    arcs, points = arcs[keep], points[keep]
    if not 0 <= lo < hi <= arcs[-1] + 1e-6:
        raise ValueError("slice outside eligible reference")
    endpoint = lambda s: [float(np.interp(s, arcs, points[:, k])) for k in range(3)]
    return [endpoint(lo), *points[(arcs > lo) & (arcs < hi)].tolist(), endpoint(hi)]


def same_location(a, b):
    return a['piece_key'] == b['piece_key'] and abs(a['target']-b['target']) <= PROTOCOL['dedup_distance']


def audit_failure(document, target, grace):
    """Audit first failure and actual outside travel, excluding terminal artifacts."""
    replay = document['replay']
    if not replay['failures']:
        return dict(valid=False, reason='no_failure')
    failure = replay['failures'][0]
    if failure['segment_index'] != 0 or abs(failure['reference_arc_base']-target) > PROTOCOL['match_tolerance']:
        return dict(valid=False, reason='different_first_failure', failure=failure)
    segment = replay['segments'][0]
    points = np.asarray(segment['trace_points_base_xyz'], dtype=np.float64)
    # Independent audit of the saved geometry, not a replacement tracer.
    outside = None
    for match in segment['matches']:
        i = match['trace_point_index']
        if match['threshold_error_ratio'] > 1:
            outside = 0. if outside is None else outside + float(np.linalg.norm(points[i]-points[i-1]))
        else:
            outside = None
    valid = outside is not None and outside + 1e-7 >= grace
    return dict(valid=valid, reason='reproduced' if valid else 'terminal_unresolved_excursion',
                outside_travel_base=outside, failure=failure)


def inventory(report, excluded=(), *, baseline_name='Classic', best_name=None, evaluation_names=None):
    rows = json.loads(Path(report).read_text())['rows']
    classic_row = next(r for r in rows if r['name'] == baseline_name)
    best_row = (next(r for r in rows if r['name']==best_name) if best_name else
                min(rows, key=lambda r: r['metrics']['global_result']['failures']))
    selected = [r for r in rows if r['metrics']['global_result']['failures'] <= classic_row['metrics']['global_result']['failures']
                and ('tuned' not in r['name'] if evaluation_names is None else r['name'] in evaluation_names)]
    if classic_row not in selected: selected.append(classic_row)
    if best_row not in selected: selected.append(best_row)
    summaries, candidates, classic_events = {}, [], defaultdict(list)
    classic_complete = defaultdict(set)
    verified = {}
    for row in selected:
        root = Path(row['path']).parent
        summary = json.loads((root/'summary.json').read_text())
        if not summary['completed']:
            raise ValueError(f'incomplete input {root}')
        config = deepcopy(summary['config']); config.pop('live_endpoint', None)
        for path, expected in config.get('annotation_sha256', {}).items():
            if path not in verified: verified[path] = sha256_file(path)
            if verified[path] != expected: raise ValueError(f'changed historical annotation: {path}')
        key = json_digest(dict(path=str(root), summary_sha256=sha256_file(root/'summary.json')))
        summaries[key] = dict(root=str(root), config=config, name=row['name'])
        index = 0
        for source in summary['sources']:
            for case in source['cases']:
                fiber = source['backend_config']['fibers'][case['source_index']]
                if str(Path(fiber).resolve()) in excluded:
                    index += 1
                    continue
                run = case['reference_run']
                piece = [str(Path(fiber).resolve()), run['run_index']]
                events = case['failures']
                if row is classic_row:
                    classic_events[tuple(piece)].extend(f['original_fiber_reference_arc_base_voxels'] for f in events)
                    if case['evaluation_complete'] and case['evaluated_through_base_voxels'] >= case['reference_length_base_voxels']-1e-6:
                        classic_complete[tuple(piece)].add(case['direction'])
                for i in _episode_indices([f['directional_reference_arc_base_voxels'] for f in events],
                                          summary.get('restart_episode_distance_base_voxels',80)):
                    failure = events[i]
                    c = dict(model=key, name=row['name'], source_id=source['source_id'],
                             piece_key=piece, fiber=fiber, run=run, case=index,
                             event_index=i, target=failure['original_fiber_reference_arc_base_voxels'],
                             historical_failure=failure, reverse=case['direction']=='reverse',
                             voxel_um=source['summary']['base_voxel_size_um'],
                             category='best' if row is best_row else 'classic' if row is classic_row else 'other')
                    c['id'] = json_digest(c)[:20]
                    candidates.append(c)
                index += 1
    # An unreported/suppressed Classic failure still disqualifies a local success.
    candidates = [c for c in candidates if c['category'] != 'other' or
                  (classic_complete[tuple(c['piece_key'])] == {'forward','reverse'} and
                   not any(abs(c['target']-x) <= 200 for x in classic_events[tuple(c['piece_key'])]))]
    candidates.sort(key=lambda c: (dict(best=0, other=1, classic=2)[c['category']],
                                  c['piece_key'], c['target'], c['name']))
    return summaries, candidates


def self_proximity_filter(settings, root, binary):
    """Geometry-only native preflight before examining individual failures."""
    import subprocess
    rows=json.loads(Path(settings['history_report']).read_text())['rows']
    best=(next(r for r in rows if r['name']==settings['best_name']) if settings.get('best_name') else
          min(rows,key=lambda r:r['metrics']['global_result']['failures']))
    summary_path=Path(best['path'])
    config=deepcopy(json.loads(summary_path.read_text())['config'])
    config.pop('live_endpoint',None)
    config.pop('reference_plan',None)
    config['self_proximity']=settings.get('self_proximity',dict(minimum_arc_separation_base=500,sample_spacing_base=4))
    config_path=root/'self_proximity_config.json'
    atomic_write(config_path,config)
    plan=json.loads(subprocess.check_output([str(binary),'replay-references','--config',str(config_path)],text=True))
    atomic_write(root/'self_proximity.json',plan)
    excluded={f['fiber_path'] for s in plan['sources'] for f in s['fibers'] if f['self_proximity']['excluded']}
    total=sum(len(s['fibers']) for s in plan['sources'])
    print(f"short-benchmark self-proximity excluded={len(excluded)} of{total}fibers",flush=True)
    return excluded


def run(workload, *, devices='0', time_limit_seconds=None):
    if time_limit_seconds is not None and (not np.isfinite(time_limit_seconds) or time_limit_seconds<=0):
        raise ValueError('extraction time limit must be finite and positive')
    settings = workload['short_failures']
    total_cases=settings.get('total_cases',200)
    best_quota=settings.get('best_quota',99)
    other_quota=settings.get('other_quota',50)
    if any(type(x) is not int or x<0 for x in (total_cases,best_quota,other_quota)) or total_cases<1:
        raise ValueError('case quotas must be nonnegative integers and total_cases positive')
    root = Path(workload['output_root']); root.mkdir(parents=True, exist_ok=True)
    binary = Path(workload['native_binary'])
    excluded = self_proximity_filter(settings,root,binary)
    models, candidates = inventory(settings['history_report'], excluded,
        baseline_name=settings.get('baseline_name','Classic'),best_name=settings.get('best_name'),
        evaluation_names=settings.get('evaluation_names'))
    fill_missing=settings.get('fill_missing',True)
    checkpoints={}
    if fill_missing:
        for key,model in models.items():
            override=settings.get('checkpoints',{}).get(model['name'])
            bindings=Path(model['root']).parent/'full_bindings.json'
            checkpoint=Path(override or json.loads(bindings.read_text())['checkpoint'])
            digest=sha256_file(checkpoint)
            for source in model['config']['sources']:
                cached=json.loads((Path(source['fiber_manifest']).parent/'inference.json').read_text())['model']
                if cached['checkpoint_sha256']!=digest: raise ValueError('producer snapshot does not match cache')
            checkpoints[key]=str(checkpoint)
    dependency_hashes = {}
    for model in models.values():
        for source in model['config']['sources']:
            for path in (source['normal_manifest'], source['fiber_manifest'],
                         str(Path(source['fiber_manifest']).parent/'cache_identity.json')):
                if path not in dependency_hashes: dependency_hashes[path] = sha256_file(path)
    identity = dict(protocol=PROTOCOL, native_sha256=sha256_file(binary),
                    implementation_sha256=sha256_file(__file__),
                    history_sha256=sha256_file(settings['history_report']),
                    models=models, settings=settings, dependency_sha256=dependency_hashes,
                    excluded_fibers=sorted(excluded))
    identity_hash = json_digest(identity)
    if settings.get('reuse_trial_root'):
        previous_root=Path(settings['reuse_trial_root'])
        previous=json.loads((previous_root/'protocol.json').read_text())
        for key in ('protocol','native_sha256','history_sha256','models','dependency_sha256','excluded_fibers'):
            if previous[key]!=identity[key]: raise ValueError(f'incompatible trial reuse: {key}')
        for path in (previous_root/'trials').glob('*.json'):
            audit=json.loads(path.read_text())
            if audit['reason']=='uncached_dependency': continue
            destination=root/'trials'/path.name
            if not destination.exists(): atomic_write(destination,audit)
    state_path = root/'state.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['identity'] != identity_hash:
            previous = json.loads((root/'protocol.json').read_text())
            scientific = lambda x: {k:v for k,v in x.items() if k != 'implementation_sha256'}
            if scientific(previous) != scientific(identity):
                raise ValueError('benchmark scientific identity changed; choose a new output directory')
            # PROTOCOL.version binds result semantics; implementation-only changes
            # (e.g. batching) can resume, with both revisions retained for audit.
            atomic_write(root/'revisions'/f"{state['identity']}.json", previous)
            state['identity'] = identity_hash
            atomic_write(root/'protocol.json', identity)
    else:
        state = dict(identity=identity_hash, candidates={}, selected=[])
        atomic_write(root/'protocol.json', identity)
    atomic_write(root/'candidates.json', candidates)
    point_cache = {}

    def geometry(c):
        key = (c['model'], c['case'])
        if key not in point_cache:
            artifact = Path(models[c['model']]['root'])/'cases'/str(c['case'])/'trace.json'
            doc = json.loads(artifact.read_text())
            hashes=state.setdefault('trace_sha256',{})
            digest=sha256_file(artifact)
            if str(artifact) in hashes and hashes[str(artifact)] != digest:
                raise ValueError('historical trace artifact changed')
            hashes[str(artifact)]=digest
            if str(Path(doc['provenance']['fiber_path']).resolve()) != c['piece_key'][0]:
                raise ValueError('historical case mapping mismatch')
            points = doc['reference_points_base_xyz']
            point_cache[key] = points[::-1] if c['reverse'] else points
        return point_cache[key]

    def execute_batch(model, requests, job_devices):
        if requests:
            if cancel_event.is_set():
                from concurrent.futures import CancelledError
                raise CancelledError('trace extraction cancelled')
            batch_key = json_digest([identity_hash, [r[3] for r in requests]])[:20]
            batch = root/'batches'/batch_key
            evaluation = deepcopy(models[model]['config'])
            for key in ('reference_plan', 'annotation_sha256', 'fiberlet'):
                evaluation.pop(key, None)
            attempt=0
            while (batch/f'attempt_{attempt}'/'full').exists() and not (batch/f'attempt_{attempt}'/'full/summary.json').exists():
                attempt+=1
            result_root=batch/f'attempt_{attempt}'/'full'
            evaluation.update(output=str(result_root), render=False, backend='beam',
                              workers=worker_budget, decoded_cache_bytes=(64 << 30)//lane_count)
            # Operational limits only: keep model/tile/precision/tracing semantics.
            inference = evaluation['inference']
            for key in ('input_workers', 'accumulator_workers', 'flush_workers'):
                inference[key] = min(int(inference.get(key) or worker_budget), worker_budget)
            inference['input_io_threads'] = min(int(inference.get('input_io_threads', 16)), worker_budget)
            inference['input_slots'] = max(int(inference.get('max_batch_size', 8)), 256//lane_count)
            inference['result_slots'] = max(int(inference.get('max_batch_size', 8)), 128//lane_count)
            sources = {s['id']: deepcopy(s) for s in evaluation['sources']}
            for s in sources.values(): s['fibers'] = []
            expected = {}
            for c, direction, distance, tag, cache in requests:
                a, b = c['run']['original_arcs_base']; target = c['target']
                lo, hi = ((target-distance, min(b, target+100)) if direction == 'forward'
                          else (max(a,target-100), target+distance))
                points = slice_line(geometry(c), lo-a, hi-a)
                if direction == 'reverse': points.reverse()
                annotation = batch/'references'/f'{tag}.json'
                source = sources[c['source_id']]
                atomic_write(annotation, dict(type='vc3d_fiber', version=1, line_points=points,
                    control_points=[points[0],points[-1]], evaluation_direction='forward',
                    optimization_mode='native_fiber_trace3d',
                    vc_open_data_coordinate_space=source['coordinate_space']))
                source['fibers'].append(str(annotation))
                expected[str(annotation)] = (c,direction,distance,cache,lo,hi)
            evaluation['sources'] = [s for s in sources.values() if s['fibers']]
            atomic_write(batch/'evaluation.json', evaluation)
            if not (result_root/'summary.json').exists():
                try:
                    replay_processes[job_devices].submit(execute_trial_replay,evaluation,binary,
                        None,job_devices,models[model]['config']['sources']).result()
                except RuntimeError as error:
                    if 'cached-only inference miss' not in str(error): raise
                    if fill_missing:
                        # Reuse the existing demand accumulator/loader/GPU manager.
                        # Only the output attempt changes; scientific inference cache
                        # identity and original producer parameters remain identical.
                        result_root=batch/f'attempt_{attempt}_inference'/'full'
                        retry=0
                        while result_root.exists():
                            retry+=1
                            result_root=batch/f'attempt_{attempt}_inference_{retry}'/'full'
                        evaluation['output']=str(result_root)
                        print(f"short-benchmark filling inference model={models[model]['name']} trials={len(requests)}",flush=True)
                        # The outer extraction runner owns one grant for its whole
                        # lifetime; never release/reacquire it between trial batches.
                        replay_processes[job_devices].submit(execute_trial_replay,evaluation,binary,
                            Path(checkpoints[model]),job_devices,models[model]['config']['sources']).result()
                    else:
                        if len(requests)>1:
                            middle=len(requests)//2
                            for part in (requests[:middle],requests[middle:]):
                                execute_batch(model, part, job_devices)
                        else:
                            c,d,n,tag,cache=requests[0]
                            atomic_write(cache,dict(valid=False,reason='uncached_dependency',distance=n,direction=d,error=str(error)))
                        return
            summary = json.loads((result_root/'summary.json').read_text())
            if not summary['completed'] or summary['cases'] != len(requests):
                raise ValueError('trial batch incomplete')
            for path in (result_root/'cases').glob('*/trace.json'):
                doc = json.loads(path.read_text())
                c,direction,distance,cache,lo,hi = expected[doc['provenance']['fiber_path']]
                audit = audit_failure(doc, distance, evaluation.get('excursion_grace_base_voxels',0))
                atomic_write(cache, dict(**audit, artifact=str(path), direction=direction,
                    distance=distance, original_arcs_base=[lo,hi]))
            print(f"short-benchmark batch={batch_key} trials={len(requests)} model={models[model]['name']}", flush=True)

    def result(c,d,n):
        path = root/'trials'/f"{c['id']}_{d}_{n}.json"
        return json.loads(path.read_text()) if path.exists() else None

    selected = state['selected']
    device_ids = str(devices).split(',')
    if not device_ids or len(set(device_ids)) != len(device_ids):
        raise ValueError('select distinct GPU device ordinals')
    resources = {key: {str(Path(s['fiber_manifest']).resolve().parent)
                       for s in model['config']['sources']} for key, model in models.items()}
    scheduler_started = time.monotonic()
    for category in ('best','other','classic'):
        count = sum(x['candidate']['category']==category for x in selected)
        limit = other_quota if category=='other' else best_quota if category=='best' else total_cases
        pool = [c for c in candidates if c['category']==category]
        if not pool or count>=limit or len(selected)>=total_cases:
            continue
        affinity,lane_count = cache_affinity(sorted({c['model'] for c in pool}),resources,len(device_ids))
        worker_budget = max(1, int(workload.get('evaluation',{}).get('gpu_manager',{}).get('cpu_cores') or 128)//lane_count)
        lanes = [','.join(device_ids[i::lane_count]) for i in range(lane_count)]
        import multiprocessing
        context = multiprocessing.get_context('spawn')
        cancel_event = context.Event()
        budget = workload.get('evaluation',{}).get('gpu_manager',{}).get('vram_bytes_per_gpu')
        session_limit = 2 if budget is not None and budget>=16<<30 else 1
        # CPU sessions may be retained together; their CUDA workers never are.
        # Prior batches reserve ~50GiB, so dividing62GiB between two active
        # models would violate the measured requirement of unchanged batch8.
        per_session_budget = budget
        replay_processes = {lane:ProcessPoolExecutor(max_workers=1,mp_context=context,
            initializer=initialize_trial_worker,initargs=(cancel_event,session_limit,per_session_budget)) for lane in lanes}
        jobs = TrialJobs(lanes, resources, cancel_event,affinity)
        positions, rotations, dispatched = {}, defaultdict(int), defaultdict(lambda: -1)
        dispatch_sequence = 0
        frontier = 0
        last_report = 0
        try:
            while frontier<len(pool) and count<limit and len(selected)<total_cases:
                if time_limit_seconds is not None and time.monotonic()-scheduler_started>=time_limit_seconds:
                    print(f'short-benchmark time limit reached; checkpoint retained selected={len(selected)}',flush=True)
                    return
                # Whole-category rolling queue: slow candidates do not prevent
                # independent later candidates/models from executing.
                active = defaultdict(list)
                unresolved = 0
                for c in pool[frontier:]:
                    if any(same_location(c,x['candidate']) for x in selected):
                        continue
                    record = state['candidates'].setdefault(c['id'],{})
                    pos = positions.setdefault(c['id'],{})
                    advance_search(c, record, pos, result)
                    if not all(d in record for d in ('forward','reverse')):
                        if unresolved>=128:
                            continue
                        unresolved+=1
                    for d in ('forward','reverse'):
                        if d not in record:
                            active[c['model']].append((c,d,pos[d]))
                # Commit only through a resolved original-order frontier. Worker
                # completion order never changes quotas, dedup or chosen geometry.
                while frontier<len(pool) and count<limit and len(selected)<total_cases:
                    c = pool[frontier]
                    if any(same_location(c,x['candidate']) for x in selected):
                        frontier+=1
                        continue
                    record = state['candidates'].get(c['id'],{})
                    if not all(d in record for d in ('forward','reverse')):
                        break
                    options = [v for v in record.values() if v]
                    if options:
                        best = min(options,key=lambda v:(v['padded']['original_arcs_base'][1]-v['padded']['original_arcs_base'][0],v['padded']['direction']))
                        selected.append(dict(candidate=c,**best));count+=1
                    frontier+=1
                atomic_write(state_path,state)
                if frontier==len(pool) or count>=limit or len(selected)>=total_cases:
                    break
                for model in sorted(active,key=lambda m:(not jobs.prefer_warm(m),dispatched[m])):
                    entries = active[model]
                    if model in jobs.models:
                        continue
                    # Round-robin directions prevent stubborn early candidates
                    # monopolizing every batch. At most128 trials per model job.
                    start = rotations[model] % len(entries)
                    entries = entries[start:]+entries[:start]
                    ahead = max(1,min(16,128//len(entries)-1))
                    requests = []
                    used = 0
                    for c,d,n in entries:
                        if any(same_location(c,x['candidate']) for x in selected):
                            continue
                        geometry(c)  # State/hash/cache writes only on coordinator.
                        extent = c['target']-c['run']['original_arcs_base'][0] if d=='forward' else c['run']['original_arcs_base'][1]-c['target']
                        for k in range(ahead+1):
                            distance = n+k*100
                            tag = f"{c['id']}_{d}_{distance}"
                            cache = root/'trials'/f'{tag}.json'
                            if distance<=extent and not cache.exists():
                                requests.append((c,d,distance,tag,cache))
                                if len(requests)>=128:
                                    break
                        used+=1
                        if len(requests)>=128:
                            break
                    if requests and jobs.submit(model,execute_batch,requests):
                        rotations[model] = start+used
                        dispatched[model] = dispatch_sequence
                        dispatch_sequence += 1
                if not jobs.pending:
                    raise RuntimeError('unresolved extraction candidates without runnable trials')
                if time.monotonic()-last_report>=5:
                    progress = dict(category=category,selected=len(selected),
                        pending_directions=sum(map(len,active.values())),
                        active_model_jobs=len(jobs.pending),model_job_capacity=lane_count,
                        workers_per_job=worker_budget,frontier=frontier,candidates=len(pool),
                        elapsed_seconds=time.monotonic()-scheduler_started)
                    atomic_write(root/'progress.json',progress)
                    print('short-benchmark '+json.dumps(progress),flush=True)
                    last_report=time.monotonic()
                jobs.collect()
        finally:
            try:
                jobs.close()
            finally:
                try:
                    closing = [process.submit(close_trial_worker) for process in replay_processes.values()]
                    for future in closing: future.result()
                finally:
                    for process in replay_processes.values():
                        process.shutdown(wait=True,cancel_futures=True)
                atomic_write(state_path,state)
                atomic_write(root/'progress.json',dict(status='stopped',selected=len(selected),
                    category=category,active_model_jobs=0,elapsed_seconds=time.monotonic()-scheduler_started))
    exported=[]
    for i, item in enumerate(selected):
        c=item['candidate']; trial=item['padded']; lo,hi=trial['original_arcs_base']; a=c['run']['original_arcs_base'][0]
        points=slice_line(geometry(c),lo-a,hi-a)
        if trial['direction']=='reverse': points.reverse()
        original=json.loads(Path(c['fiber']).read_text())
        annotation=dict(type='vc3d_fiber',version=1,line_points=points,control_points=[points[0],points[-1]],
            optimization_mode='native_fiber_trace3d', evaluation_direction='forward', benchmark_provenance=item)
        for key in ('vc_open_data_coordinate_space','vc_open_data_source_original_resolution'):
            if key in original: annotation[key]=original[key]
        path=root/'fibers'/f'{i:03d}_{c["id"]}.json';atomic_write(path,annotation)
        exported.append(dict(path=str(path),sha256=sha256_file(path),length_base_voxels=hi-lo,
                             length_mm=(hi-lo)*c['voxel_um']/1000,**item))
    atomic_write(root/'manifest.json',dict(version=1,completed=len(exported)==total_cases,case_count=len(exported),
        total_base_voxels=sum(x['length_base_voxels'] for x in exported),
        total_mm=sum(x['length_mm'] for x in exported),protocol=PROTOCOL,identity=identity_hash, cases=exported))
    print(json.dumps(dict(case_count=len(exported),total_mm=sum(x['length_mm'] for x in exported),output=str(root))),flush=True)
