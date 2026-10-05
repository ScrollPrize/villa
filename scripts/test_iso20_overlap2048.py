import json
from pathlib import Path
import pytest
import iso20_overlap2048 as experiment


def test_trace_settings_validation_rejects_real_changes(monkeypatch):
    small=['trace','-','--bbox','1','2','3','4','5','6','--output','small','--beam-width','8']
    large=list(small);large[3:9]=list(map(str,(*experiment.context.LO,*experiment.context.HI)))
    large[10]='large'
    c=lambda a:{'commands':[{'args':a}]}
    record=dict(returncode=0,command=['binary',*large],inputs={'manifest':'hash'})
    monkeypatch.setattr(experiment,'sha256_file',lambda _: 'hash')
    experiment.validate_source(c(large),c(small),record)
    small[-1]='1'
    with pytest.raises(AssertionError,match='settings differ'):
        experiment.validate_source(c(large),c(small),record)


def test_filter_binds_large_source(monkeypatch,tmp_path):
    overlap=experiment.overlap
    monkeypatch.setattr(overlap,'OUT',tmp_path)
    monkeypatch.setattr(overlap,'LOW',True)
    template={'commands':[{'args':['trace','-','--normal-manifest','n','--fiber-manifest','f']} ]}
    config=overlap.filter_config(template,'large','presence',10,20,0,source_trace=Path('/large.zarr'))
    args=config['commands'][0]['args']
    assert args[:2]==['deduplicate','/large.zarr']
    assert config['input_files']==['n','f','/large.zarr/.zattrs']
    assert args[args.index('--overlap-fraction')+1]=='0'
    assert '--overlap-cache' in args


def test_report_includes_all_metrics_and_missingness(monkeypatch,tmp_path):
    context=experiment.context
    monkeypatch.setattr(context,'REPORT',tmp_path/'report')
    row=dict(independent_percent=65.38,observed_ladder_shift={'ladder_shift_percent':91.67},
             independent_missing=1,tied_rows=0,unbounded_rows=0,
             ladder=[{'reference':'0','winding':'0','hard_sign_errors':'0'}],missing_references=['4'])
    data=dict(small=row,large=row,timings={},protocol='promoted test',oracle={
        '1024':{'constraint_score_percent':-55.12,'reference_score_percent':88.46},
        '2048':{'constraint_score_percent':-40,'reference_score_percent':90}})
    context.publish(data)
    report=(tmp_path/'report.md').read_text()
    assert 'promoted test' in report and '-40.0000%' in report and '90.0000%' in report
    assert '| 4 | Missing | Missing |' in report
