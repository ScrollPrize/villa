from pathlib import Path
import pytest
from reviewed_profile_sweep import variant


def test_partial_run_cannot_be_timed_as_fresh(tmp_path):
    from reviewed_profile_sweep import check_fresh_or_complete
    assert not check_fresh_or_complete(tmp_path)
    (tmp_path/'full').mkdir()
    with pytest.raises(RuntimeError, match='Partial profiling run'):
        check_fresh_or_complete(tmp_path)


def test_completed_profile_requires_artifacts(tmp_path):
    from reviewed_profile_sweep import check_fresh_or_complete
    full = tmp_path/'full'
    full.mkdir()
    (full/'summary.json').write_text('{"completed":true}')
    with pytest.raises(RuntimeError, match='missing required profile artifact'):
        check_fresh_or_complete(tmp_path)
    for name in ('pipeline_profile.json', 'inference_summary.json', 'inference_queues.jsonl'):
        (full/name).write_text('{}')
    with pytest.raises(RuntimeError, match='missing GPU telemetry'):
        check_fresh_or_complete(tmp_path)
    (tmp_path/'gpu_utilization.csv').write_text('test')
    assert check_fresh_or_complete(tmp_path)


@pytest.mark.parametrize('chunk', [32, 64, 128])
def test_variant_keeps_payload_budget_and_inference(chunk):
    base = {'evaluation': {'gpu_manager': {'devices': '0,1,2,3,4,5'},
        'inference': {'accumulator_hot_slots': 114688, 'accumulator_spill_segment_slots': 2048,
                      'tile_size': 256, 'max_batch_size': 4, 'chunk_size': 64}}}
    cfg = variant(base, chunk, Path('/tmp/profile'))
    settings = cfg['evaluation']['inference']
    assert settings['accumulator_hot_slots'] * chunk**3 == 114688 * 64**3
    assert settings['accumulator_spill_segment_slots'] * chunk**3 == 2048 * 64**3
    assert settings['tile_size'] == 256 and settings['max_batch_size'] == 4
    assert settings['profile_pipeline'] is True
    assert base['evaluation']['inference']['chunk_size'] == 64


@pytest.mark.parametrize('enabled', [False, True])
def test_live_pool_profile_opt_in(monkeypatch, enabled):
    from vesuvius.neural_tracing.fiber_trace_3d import live_evaluate as live
    calls = []
    monkeypatch.setattr(live.demand, 'DemandWorkerPool', lambda *a, **kw: calls.append(kw))
    settings = {'profile_pipeline': True} if enabled else {}
    live.create_managed_demand_pool({}, {}, [], settings)
    assert calls[0]['profile_pipeline'] is enabled
    assert calls[0]['accumulator_workers'] == 128
