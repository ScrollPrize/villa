import pytest
import json
from pathlib import Path
from vesuvius.neural_tracing.fiber_trace_3d.short_failure_benchmark import (
    slice_line, same_location, audit_failure, merge_failure_intervals, cropped_control_metadata)


def test_inspection_model_can_be_independent_of_failure_sources(tmp_path, monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import short_failure_benchmark as m
    from vesuvius.neural_tracing.fiber_trace_3d import cached_trace_profile as cached
    evaluation = tmp_path / 'inspection'
    full = evaluation / 'full'
    full.mkdir(parents=True)
    cache = tmp_path / 'cache'
    checkpoint = tmp_path / 'snapshot.pt'
    checkpoint.write_bytes(b'fixture')
    summary = full / 'summary.json'
    summary.write_text(json.dumps({'completed': True, 'config': {
        'sources': [{'id': 's', 'fiber_manifest': str(cache / 'inference.lasagna.json')}],
        'inference': {'input_workers': 1}}}))
    (evaluation / 'full_bindings.json').write_text(json.dumps({
        'checkpoint': str(checkpoint), 'checkpoint_sha256': m.sha256_file(checkpoint)}))
    observed = {}
    class Session:
        def __init__(self, config, binary, checkpoint, **kwargs):
            observed.update(config=config, **kwargs)
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def fill_chunks(self, requests):
            observed['requests'] = requests
            return {'completed': True}
    monkeypatch.setattr(cached, 'CachedReplaySession', Session)
    workload = {'native_binary': 'unused', 'failure_union': {'evaluations': [
        {'name': 'lowres', 'summary': 'must-not-read'}]}}
    m._fill_union_corridor(workload, {'evaluation_name': 'fullres', 'summary': str(summary),
        'devices': '0,1,2,3,4,5,6,7', 'inference_overrides': {'input_workers': 128}},
        {str(cache): {(1,2,3)}}, tmp_path)
    assert observed['config']['inference']['input_workers'] == 128
    assert observed['devices'] == '0,1,2,3,4,5,6,7'
    assert observed['requests'] == {str(cache): [(1,2,3)]}


def test_training_mining_selects_only_lowres_failures():
    configs = Path(__file__).resolve().parents[2] / 'src/vesuvius/neural_tracing/fiber_trace_3d/configs'
    settings = json.loads((configs / 'failure_union_training_20260927.json').read_text())['failure_union']
    assert [e['name'] for e in settings['evaluations']] == ['lowres']
    assert settings['baseline_name'] == settings['best_name'] == 'lowres'
    assert settings['run_reproduction_search'] is True
    assert settings['context_base_voxels'] == 500
    assert settings['fill_missing_inference']['evaluation_name'] == 'fullres'


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('lo,hi', [(5, 25), (10, 20), (12, 18)])
def test_crop_preserves_source_controls_and_segment_constraints(lo, hi, reverse):
    from copy import deepcopy
    line = [[x, 0, 0] for x in range(31)]
    source = dict(version=3, line_points=line, optimization_mode='native_fiber_trace3d',
                  control_points=[dict(position=line[x], **({'segment_to_next': {
                      'interp_mode': 'trace', 'interp_goal': 'global',
                      'config': {'beam_width': x + 1}, 'metric': x}}
                      if x < 30 else {})) for x in (0, 10, 20, 30)])
    before = deepcopy(source)
    points = slice_line(line, lo, hi)
    if reverse:
        points.reverse()
    result = cropped_control_metadata(source, lo, hi, points, reverse=reverse)
    expected_x = [lo, *[x for x in (10, 20) if lo < x < hi], hi]
    widths = [int(((a+b)/2)//10)*10+1 for a,b in zip(expected_x, expected_x[1:])]
    if reverse:
        expected_x.reverse()
        widths.reverse()
    assert [c['position'][0] for c in result['control_points']] == expected_x
    assert [c['segment_to_next']['config']['beam_width']
            for c in result['control_points'][:-1]] == widths
    assert 'segment_to_next' not in result['control_points'][-1]
    result['control_points'][0]['segment_to_next']['config']['beam_width'] = 999
    assert source == before


def test_crop_does_not_invent_trace_metadata_for_legacy_or_extrapolation():
    source = dict(version=1, line_points=[[x, 0, 0] for x in range(31)],
                  control_points=[[5, 0, 0], [25, 0, 0]], optimization_mode='lasagna')
    result = cropped_control_metadata(source, 10, 20, [[10, 0, 0], [20, 0, 0]])
    assert result == dict(version=1, control_points=[[10, 0, 0], [20, 0, 0]],
                          optimization_mode='lasagna')
    with pytest.raises(ValueError, match='outside'):
        cropped_control_metadata(source, 0, 20, [[0, 0, 0], [20, 0, 0]])


def test_exact_slice_and_duplicate_vertices():
    assert slice_line([[0,0,0],[0,0,0],[100,0,0],[200,0,0]],50,150) == [[50.,0.,0.],[100.,0.,0.],[150.,0.,0.]]
    with pytest.raises(ValueError): slice_line([[0,0,0],[10,0,0]],-1,5)


def test_scroll_subsets_link_only_selected_chunks_and_preserve_cache(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.short_failure_benchmark import _export_scroll_subsets
    channels = ('fiber_presence.ome.zarr', 'fiber_nx.ome.zarr', 'fiber_ny.ome.zarr')
    roots = {}
    required = {}
    for scroll in ('PHercParis4', 'PHerc1299'):
        root = tmp_path / 'cache' / scroll
        roots[f's3://bucket/{scroll}/volumes/scan.zarr'] = root
        required[str(root.resolve())] = {(0, 0, 1)}
        for channel in channels:
            array = root / channel
            (array / '1/0/0').mkdir(parents=True)
            (array / '.zgroup').write_text('{}')
            (array / '1/.zarray').write_text('{}')
            (array / 'scratch.bin').write_bytes(b'not exported')
            for x in (1, 2):
                (array / f'1/0/0/{x}').write_bytes(bytes([x]))
    output = tmp_path / 'export'
    settings = {'export_chunk_subsets': {'chunk_mode': 'symlink'}}
    for _ in range(2):
        result = _export_scroll_subsets(settings, output, roots, required)
        assert set(result) == {'PHercParis4', 'PHerc1299'}
    for scroll in result:
        for channel in channels:
            array = output / 'vc3d_channels' / scroll / channel
            assert (array / '1/0/0/1').is_symlink()
            assert (array / '1/0/0/1').read_bytes() == b'\x01'
            assert not (array / '1/0/0/2').exists()
            assert not (array / 'scratch.bin').exists()


def test_all_failures_keeps_models_worse_than_classic(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.short_failure_benchmark import inventory
    rows = []
    for name, failures in [('classic', 1), ('best', 0), ('other', 2)]:
        root = tmp_path / name
        root.mkdir()
        summary = {'completed': True, 'config': {}, 'sources': []}
        (root / 'summary.json').write_text(json.dumps(summary))
        rows.append({'name': name, 'path': str(root / 'summary.json'),
                     'metrics': {'global_result': {'failures': failures}}})
    report = tmp_path / 'report.json'
    report.write_text(json.dumps({'rows': rows}))
    models, _ = inventory(report, baseline_name='classic', best_name='best',
                          evaluation_names=['classic', 'best', 'other'], include_all_failures=True)
    assert {m['name'] for m in models.values()} == {'classic', 'best', 'other'}


def test_location_dedup_respects_eligible_span():
    a=dict(piece_key=['f',0],target=1000)
    assert same_location(a,dict(piece_key=['f',0],target=1200))
    assert not same_location(a,dict(piece_key=['f',1],target=1000))
    assert not same_location(a,dict(piece_key=['f',0],target=1201))


def test_union_interval_merge_is_deterministic_and_keeps_separate_gaps():
    assert merge_failure_intervals([(1000, 1500), (1600, 1800), (2101, 2200)], 100) == [
        (1000, 1800), (2101, 2200)]


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
