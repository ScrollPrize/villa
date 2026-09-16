import pytest
from vesuvius.neural_tracing.fiber_trace_3d.short_failure_benchmark import slice_line, same_location, audit_failure


def test_exact_slice_and_duplicate_vertices():
    assert slice_line([[0,0,0],[0,0,0],[100,0,0],[200,0,0]],50,150) == [[50.,0.,0.],[100.,0.,0.],[150.,0.,0.]]
    with pytest.raises(ValueError): slice_line([[0,0,0],[10,0,0]],-1,5)


def test_location_dedup_respects_eligible_span():
    a=dict(piece_key=['f',0],target=1000)
    assert same_location(a,dict(piece_key=['f',0],target=1200))
    assert not same_location(a,dict(piece_key=['f',1],target=1000))
    assert not same_location(a,dict(piece_key=['f',0],target=1201))


def artifact(outside_steps=2, arc=200, segment=0):
    return dict(replay=dict(failures=[dict(segment_index=segment,reference_arc_base=arc)],segments=[dict(
        trace_points_base_xyz=[[i*80,0,0] for i in range(outside_steps+1)],
        matches=[dict(trace_point_index=i,threshold_error_ratio=2) for i in range(1,outside_steps+1)])]))


def test_short_endpoint_cannot_manufacture_failure():
    assert not audit_failure(artifact(2),200,160)['valid']
    assert audit_failure(artifact(3),200,160)['valid']
    assert not audit_failure(artifact(3,arc=301),200,160)['valid']
    assert not audit_failure(artifact(3,segment=1),200,160)['valid']


def test_reentry_resets_grace_audit():
    doc=artifact(4)
    doc['replay']['segments'][0]['matches'][2]['threshold_error_ratio']=0
    assert not audit_failure(doc,200,160)['valid']


def test_trial_jobs_overlap_and_reserve_cache_and_devices():
    from threading import Event, Lock
    from vesuvius.neural_tracing.fiber_trace_3d.short_failure_benchmark import TrialJobs
    cancel = Event()
    jobs = TrialJobs(['0','1'], {'a':{'cache1'},'b':{'cache2'},'alias':{'cache1'}}, cancel)
    started, release = {k:Event() for k in ('a','b')}, Event()
    seen = {}
    lock = Lock()
    def work(model, device):
        with lock: seen[model] = device
        started[model].set()
        assert release.wait(5)
    try:
        assert jobs.submit('a',work)
        assert jobs.submit('b',work)
        assert started['a'].wait(5) and started['b'].wait(5)
        assert len(set(seen.values())) == 2
        assert not jobs.submit('alias',work)
        assert not jobs.submit('a',work)
        release.set()
        while jobs.pending: jobs.collect()
    finally:
        release.set()
        jobs.close()
    assert cancel.is_set()


def test_trial_job_failure_cancels_and_drains_sibling():
    from threading import Event
    from vesuvius.neural_tracing.fiber_trace_3d.short_failure_benchmark import TrialJobs
    cancel, started, stopped = Event(), Event(), Event()
    jobs = TrialJobs(['0','1'], {'a':set(),'b':set()}, cancel)
    def work(model, device):
        if model == 'a':
            started.set()
            assert cancel.wait(5)
            stopped.set()
        else:
            assert started.wait(5)
            raise ValueError('worker failed')
    jobs.submit('a',work); jobs.submit('b',work)
    try:
        with pytest.raises(ValueError,match='worker failed'):
            while jobs.pending: jobs.collect()
    finally:
        jobs.close()
    assert stopped.is_set()


def test_rolling_search_preserves_grid_and_resumes_missing_confirmation():
    from vesuvius.neural_tracing.fiber_trace_3d.short_failure_benchmark import advance_search
    candidate = dict(target=500,run=dict(original_arcs_base=[0,1000]))
    audits = {}
    def result(c,d,n): return audits.get((d,n))
    def trial(n,valid): return dict(valid=valid,direction='forward',original_arcs_base=[500-n,600])
    audits[('forward',100)] = trial(100,False)
    audits[('forward',200)] = trial(200,True)
    # A later success cannot skip the missing300 confirmation.
    audits[('forward',400)] = trial(400,True)
    record,positions = {},{}
    advance_search(candidate,record,positions,result)
    assert 'forward' not in record and positions['forward']==200
    audits[('forward',300)] = trial(300,True)
    advance_search(candidate,record,positions,result)
    assert record['forward']['minimal']==trial(200,True)
    assert record['forward']['padded']==trial(300,True)
    resumed = {}
    advance_search(candidate,resumed,{},result)
    assert resumed == record


def test_cache_affinity_groups_transitive_aliases_and_uses_all_single_model_devices():
    from vesuvius.neural_tracing.fiber_trace_3d.short_failure_benchmark import cache_affinity
    affinity,count=cache_affinity(['a','b','c','d'],
        dict(a={'x'},b={'y'},c={'x','y'},d={'z'}),8)
    assert count==2 and affinity['a']==affinity['b']==affinity['c']
    assert affinity['d']!=affinity['a']
    affinity,count=cache_affinity(['only'],{'only':{'x'}},8)
    lanes=[','.join(str(d) for d in range(i,8,count)) for i in range(count)]
    assert lanes[affinity['only']]=='0,1,2,3,4,5,6,7'


def test_model_queue_keeps_resident_model_without_four_batch_rotation():
    from threading import Event
    from vesuvius.neural_tracing.fiber_trace_3d.short_failure_benchmark import TrialJobs
    jobs=TrialJobs(['0'],{'a':{'x'},'b':{'y'}},Event(),{'a':0,'b':0})
    try:
        for _ in range(6):
            assert jobs.submit('a',lambda model,device:None)
            while jobs.pending: jobs.collect()
        assert jobs.prefer_warm('a') and not jobs.prefer_warm('b')
        assert jobs.submit('b',lambda model,device:None)
        while jobs.pending: jobs.collect()
        assert jobs.prefer_warm('b') and not jobs.prefer_warm('a')
    finally:
        jobs.close()
