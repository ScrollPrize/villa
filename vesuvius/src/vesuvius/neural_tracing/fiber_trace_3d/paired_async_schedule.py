"""Continuous cross-parameter scheduling; no candidate-batch barriers."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from .fiber_paired_weight_search import better
from .paired_search_schedule import can_improve, optimistic_rank


def candidate_queue(current, names, factors, bounds, absolute_values=None):
    """Interleave coordinates: one step per parameter before its next step."""
    by_name={}
    for name in names:
        lo,hi=bounds[name]
        values=[float(f'{current[name]*f:.12g}') for grid in factors for f in grid]
        values+=list((absolute_values or {}).get(name,[]))
        by_name[name]=[v for v in dict.fromkeys(values) if v!=current[name] and lo<=v<=hi]
    return deque((name,dict(current,**{name:by_name[name][step]}))
        for step in range(max(map(len,by_name.values()),default=0))
        for name in names if step<len(by_name[name]))


def asynchronous_refine(best,names,factors,bounds,devices,gt_names,key_for,
                        result_cache,stage_cache,work,finish,save,accepted,progress,
                        tolerance=1e-6,absolute_values=None):
    revision=0;cursor=0
    candidates=candidate_queue(best['parameters'],names,factors,bounds,absolute_values)
    ready=deque();idle=deque(devices);running={};states={};visited=set()
    stages=['joint',*gt_names]

    def publish():
        progress(dict(revision=revision,best_key=best['key'],queued_candidates=len(candidates),
            queued_stages=len(ready),completed=len(result_cache),
            running=[dict(key=j['key'],stage=j['stage'],device=j['device'],
                          parameter=j['name'],stale=j['revision']!=revision) for j in running.values()]))

    def consider(name,result):
        nonlocal best,revision,candidates,ready,visited
        if not better(result['ranking'],best['ranking'],tolerance):return False
        old=best;best=result;revision+=1
        accepted(name,best['parameters'],best,old)
        # Keep cyclic position, retry changed coordinate first, then pending
        # coordinates. Running work is not allowed to hold up the new queue.
        rotated=names[cursor:]+names[:cursor]
        order=[name]+[n for n in rotated if n!=name]
        candidates=candidate_queue(best['parameters'],order,factors,bounds,absolute_values)
        ready.clear();visited=set()
        return True

    def receive(job,report):
        key=job['key'];stage=job['stage']
        stage_cache[(key,stage)]=report
        # Finish/save already-running work for reuse, but do not continue or
        # accept a candidate from a superseded parameter vector.
        if job['revision']!=revision:return
        state=states[key];state['parts'][stage]=report
        if stage=='free':
            if not can_improve(report,best['ranking'],tolerance):
                result=dict(key=key,parameters=job['values'],large=dict(free=report),
                    screened_out=True,optimistic_ranking=optimistic_rank(report),
                    ranking=dict(optimistic_rank(report),passed=False),
                    screening_incumbent=best['ranking'],
                    skip_reason='Cannot win even with perfect GT-assisted result')
                result_cache[key]=result;save(result)
            else:
                for next_stage in stages:ready.append(dict(job,stage=next_stage))
        elif all(s in state['parts'] for s in ['free',*stages]):
            result=finish(key,job['values'],state['parts'])
            result_cache[key]=result;save(result)
            consider(job['name'],result)

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        while candidates or ready or running:
            # Only submit jobs that have a real free GPU slot. Pending work
            # stays in our queue, so reprioritization never waits on a pool backlog.
            blocked=[]
            while idle and (ready or candidates):
                if ready:
                    job=ready.popleft()
                    if job['revision']!=revision:continue
                    if job['stage']!='free' and not can_improve(states[job['key']]['parts']['free'],best['ranking'],tolerance):continue
                else:
                    name,values=candidates.popleft();cursor=(names.index(name)+1)%len(names)
                    key=key_for(values)
                    if key in visited:continue
                    visited.add(key)
                    if key in result_cache:
                        consider(name,result_cache[key]);continue
                    states[key]=dict(parts={})
                    job=dict(key=key,values=values,name=name,stage='free',revision=revision)
                cached=stage_cache.get((job['key'],job['stage']))
                if cached is not None:
                    receive(job,cached);continue
                # If an old-revision copy is still running, do not overwrite its
                # output. Revisit as soon as it completes instead of blocking.
                if any(j['key']==job['key'] and j['stage']==job['stage'] for j in running.values()):
                    blocked.append(job)
                    continue
                device=idle.popleft();job=dict(job,device=device)
                running[pool.submit(work,job['values'],job['stage'],device)]=job
            ready.extend(blocked)
            publish()
            if running:
                done,_=wait(running,return_when=FIRST_COMPLETED)
                for future in done:
                    job=running.pop(future);idle.append(job['device'])
                    receive(job,future.result())
            elif not candidates and not ready:break
    publish()
    return best
