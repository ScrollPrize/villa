import copy
from iso20_minimum_sweep import configuration,MINIMUMS


def test_only_segmentation_flags_change():
    template=dict(commands=[dict(name='old',args=['direction-ablation','same-trace.zarr',
        '--output','old','--winding-hard-signs','both','--threads','128'])],input_files=['same/.zattrs'])
    before=copy.deepcopy(template)
    for minimum in MINIMUMS:
        config=configuration(template,minimum)
        args=config['commands'][0]['args']
        assert args[-3:]==['--local-constraint-events','--local-min-piece-length',str(minimum)]
        args[args.index('--output')+1]='old'
        assert args[:-3]==before['commands'][0]['args']
        assert config['input_files']==before['input_files']
        assert template==before
    assert MINIMUMS==(8,16,32,64,128,256)


def test_failed_oracle_reuse_requires_same_provenance(tmp_path,monkeypatch):
    import json
    import iso20_minimum_sweep as sweep
    monkeypatch.setattr(sweep,'OUT',tmp_path)
    cfg=dict(commands=[dict(name='bp-min8',args=['direction-ablation','trace'])])
    current=dict(command=['binary','independent'],binary_sha256='sha',shared_libraries={'lib':'sha'},inputs={'input':'sha'},references={'ref':'sha'})
    old=dict(current,command=['binary','direction-ablation','trace'],returncode=1)
    record=tmp_path/'bp-min8.run.json'
    record.write_text(json.dumps(old))
    (tmp_path/'bp-min8.log').write_text('Reference-connected BP gauge has no calibration')
    assert sweep.matching_failed_oracle(cfg,current)==old
    for key in ('binary_sha256','shared_libraries','inputs','references','command'):
        changed=dict(old);changed[key]='changed'
        record.write_text(json.dumps(changed))
        assert sweep.matching_failed_oracle(cfg,current) is None
    record.write_text(json.dumps(dict(old,returncode=0)))
    assert sweep.matching_failed_oracle(cfg,current) is None
    record.write_text(json.dumps(old))
    log=tmp_path/'bp-min8.log'
    log.write_text('different error')
    assert sweep.matching_failed_oracle(cfg,current) is None
    log.unlink()
    assert sweep.matching_failed_oracle(cfg,current) is None
