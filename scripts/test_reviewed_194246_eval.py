import json
from reviewed_194246_eval import BASE, SELECTIONS, configuration


def test_exact_winning_policy_and_resource_preservation():
    base=json.loads((BASE/'workload.json').read_text())
    cfg=configuration(base,'current','/tmp/snapshot.pt','a'*64)
    expected=json.loads(json.dumps(base))
    for k in ('checkpoint','checkpoint_sha256','output_root'):
        expected[k]=cfg[k]
    expected['evaluation']['inference_cache_namespace']=cfg['evaluation']['inference_cache_namespace']
    expected['evaluation']['gpu_manager']['label']=cfg['evaluation']['gpu_manager']['label']
    assert cfg==expected
    args=cfg['evaluation']['trace_args']
    for flag,value in {'step-voxels':'6','replay-presence-cutoff':'0.4','replay-presence-mode':'modulated','width-relative-tolerance':'0.4','width-weighting':'cosine','beam-width':'8'}.items():
        assert args[args.index('--'+flag)+1]==value
    assert not any(a.startswith('--replay-reverse') for a in args)
    assert cfg['devices']=='0,1,2,3,4,5'
    assert cfg['evaluation']['inference']['emit_width']
    assert cfg['evaluation']['inference_cache_namespace']!=base['evaluation']['inference_cache_namespace']
    assert len(cfg['evaluation']['inference_cache_namespace'])<=64


def test_selection_mapping():
    assert SELECTIONS[:3]==(('current','current.pt',None),('best_loss','best.pt','test/loss_total'),('best_restarts','best_reviewed_trace.pt','reviewed_trace/restarts_combined'))
    assert {s[1] for s in SELECTIONS[3:]} == {'current_ema_0p99.pt', 'best_ema_0p99.pt', 'current_ema_0p999.pt', 'best_ema_0p999.pt'}


def test_ema_validation():
    import pytest
    from reviewed_194246_eval import validate_selection
    saved = dict(snapshot_kind='ema', ema_metadata={'decay': 0.999}, metric_name='test_ema_0p999/loss_total')
    assert validate_selection(saved, 'best_ema_0p999.pt', saved['metric_name']) == 'EMA999'
    with pytest.raises(ValueError):
        validate_selection(saved, 'best_ema_0p99.pt', saved['metric_name'])
    with pytest.raises(ValueError):
        validate_selection(saved, 'best.pt', saved['metric_name'])
