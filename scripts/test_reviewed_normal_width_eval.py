import copy
import json
import pytest
from reviewed_normal_width_eval import configuration, validate_snapshot


def test_width_sweep_changes_only_gate_and_keeps_repeat_control():
    from reviewed_normal_width_eval import width_sweep_variants
    from vesuvius.neural_tracing.fiber_trace_3d.beam_parameter_sweep import apply_trace_args
    variants=width_sweep_variants()
    assert len(variants)==9
    assert variants[0]=={'name':'linear20','parameters':{}}
    original=['--beam-width','8','--width-relative-tolerance','0.2',
              '--replay-reverse-interval-base','512','--replay-reverse-to-start']
    for variant in variants:
        assert set(variant['parameters']) <= {'width-relative-tolerance','width-weighting'}
        args=apply_trace_args(original,variant['parameters'])
        assert args.count('--width-relative-tolerance')==1
        assert args[:2]==original[:2]
        assert args[4:7]==original[4:7]
    assert apply_trace_args(original,variants[1]['parameters'])[3]=='0'


@pytest.mark.parametrize('cosine',[False,True])
@pytest.mark.parametrize('tolerance',[.1,.4])
def test_no_reverse_width_keeps_inference_and_unrelated_tracing(cosine,tolerance):
    from reviewed_normal_width_eval import without_reverse_width_config
    kept=['--threads','1','--beam-width','8','--step-voxels','4','--embedding-weight','0']
    base={'output_root':'old','checkpoint':'frozen','evaluation':{
        'sources':[{'id':'scroll','fibers':['ref']}], 'workers':192,
        'inference_cache_namespace':'same-width-cache','inference':{'emit_width':True,'max_batch_size':4},
        'gpu_manager':{'devices':'0,1,2,3,4,5','label':'old'},
        'trace_args':kept+['--replay-reverse-interval-base','512','--replay-reverse-all-endpoints',
            '--replay-reverse-to-start','--replay-reverse-normal-base','20',
            '--replay-reverse-tangential-base','40','--replay-reverse-history-steps','0',
            '--width-relative-tolerance','0.2']}}
    before=copy.deepcopy(base)
    cfg=without_reverse_width_config(base,'test',cosine=cosine,tolerance=tolerance)
    assert base==before
    for key in ('sources','workers','inference_cache_namespace','inference'):
        assert cfg['evaluation'][key]==base['evaluation'][key]
    args=cfg['evaluation']['trace_args']
    assert args[:len(kept)]==kept
    assert not any('replay-reverse' in x for x in args)
    assert args[args.index('--width-relative-tolerance')+1]==(str(tolerance) if cosine else '0')
    assert args[args.index('--width-weighting')+1]==('cosine' if cosine else 'linear')
    base['evaluation']['trace_args'].append('--replay-reverse-unknown')
    with pytest.raises(ValueError,match='unhandled reverse'):
        without_reverse_width_config(base,'test',cosine=cosine)


def test_filtered_config_preserves_inference_and_parallelism():
    base=dict(evaluation=dict(inference={'max_batch_size':4,'input_workers':128},workers=192,
        gpu_manager={'devices':'0,1,2,3,4,5'},trace_args=['--beam-width','8','--embedding-weight','0']))
    before=copy.deepcopy(base)
    cfg=configuration(base,'test','/frozen.pt','abc')
    assert base==before
    assert cfg['evaluation']['inference']==base['evaluation']['inference']
    assert cfg['evaluation']['workers']==192
    args=cfg['evaluation']['trace_args']
    for key,value in [('--replay-reverse-interval-base','512'),('--replay-reverse-normal-base','20'),
                      ('--replay-reverse-tangential-base','40')]:
        assert args[args.index(key)+1]==value
    assert '--replay-reverse-all-endpoints' in args
    assert '--replay-reverse-to-start' in args


def test_selection_distinguishes_restart_best_from_dense_loss():
    raw={'snapshot_kind':'raw','metric_name':'reviewed_trace/restarts_combined',
         'reviewed_trace_selection':{'step':9800}}
    validate_snapshot(raw,'reviewed_trace/restarts_combined')
    with pytest.raises(ValueError): validate_snapshot(raw,'test/loss_total')
    with pytest.raises(ValueError): validate_snapshot({**raw,'snapshot_kind':'ema'},None)
    validate_snapshot({'metric_name':'test/loss_total'},'test/loss_total')


def test_width_sweep_report_only_does_not_trace(tmp_path,monkeypatch):
    import reviewed_normal_width_eval as module
    import lasagna.inference_provenance as provenance
    import vesuvius.neural_tracing.fiber_trace_3d.beam_parameter_sweep as sweep
    sha='2408ece6ffa2251655709cf3987357f1892a573828929934266145f09f9d2eba'
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module,'REPO',tmp_path)
    monkeypatch.setattr(provenance,'sha256_file',lambda _:sha)
    monkeypatch.setattr(sweep,'run_sweep',lambda _:pytest.fail('report reran tracing'))
    monkeypatch.setattr(module,'width_sweep_variants',lambda:[{'name':'linear20','parameters':{}}])
    base=tmp_path/module.WIDTH_NAME
    base.mkdir()
    (base/'workload.json').write_text(json.dumps({'checkpoint_sha256':sha,'checkpoint':'frozen'}))
    folder=tmp_path/'normalwidth_u11000_gate_sweep_demand_20261006_v1/linear20'
    attempt=folder/'attempt_001'
    (attempt/'full').mkdir(parents=True)
    (folder/'result.json').write_text(json.dumps({'attempt_root':str(attempt),'native_sha256':'native',
        'row':{'status':'completed','wall_seconds':12}}))
    (attempt/'full/inference_summary.json').write_text(json.dumps({'sources':{'scroll':{
        'inferred_tiles':0,'published_chunks':0}}}))
    row=dict(ordinary_restarts=10,filtered_stops=229,actual_stops=239,retained_percent=81.59,
             mean_mm=4.71,median_mm=1.41,p05_mm=0,p95_mm=9.06)
    monkeypatch.setattr(module,'measured_result',lambda _:dict(combined=row,scrolls={'scroll':row}))
    module.run_width_sweep({'report_only':True,'fill_missing':True})
    report=next(tmp_path.rglob('reviewed_u11000_width_sweep_20261006.md')).read_text()
    assert 'cos(pi*d/2)' in report
    assert '| linear20 | 10 | 229 | 239 |' in report
    assert '| linear20 | 12.00 | 0 | 0 |' in report


def test_report_rejects_changed_cohort(tmp_path,monkeypatch):
    import reviewed_normal_width_eval as module
    source=dict(id='scroll',coordinate_space='scroll@L0',base_voxel_size_um=2.4,fibers=['original.json'])
    baseline=tmp_path/'baseline'
    baseline.mkdir()
    (baseline/'full.json').write_text(json.dumps({'sources':[source]}))
    candidate=tmp_path/'candidate'
    candidate.mkdir()
    (candidate/'full.json').write_text(json.dumps({'sources':[{**source,'fibers':['changed.json']}]}))
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module,'BASELINE',baseline)
    monkeypatch.setattr(module,'SELECTIONS',[('candidate','','','')])
    with pytest.raises(ValueError,match='reference cohort'):
        module.document_results({'candidate':{}})
