import json
from pathlib import Path
import iso20_best2048 as experiment


def test_trace_changes_only_bbox_and_output(tmp_path,monkeypatch):
    monkeypatch.setattr(experiment.shared,'OUT',tmp_path)
    args=['trace','-','--bbox','1','2','3','4','5','6','--output','old',
          '--reverse-normal','20','--reverse-tangential','20','--reverse-prefix-interval','512']
    template={'commands':[{'args':args,'name':'old'}]}
    changed=experiment.trace_config(template)['commands'][0]['args']
    expected=list(args);expected[3:9]=list(map(str,(*experiment.LO,*experiment.HI)))
    expected[10]=str(tmp_path/'beam2048.zarr')
    assert changed==expected
    assert args[3]=='1'


def test_bp_changes_only_input_output(tmp_path,monkeypatch):
    monkeypatch.setattr(experiment.shared,'OUT',tmp_path)
    args=['direction-ablation','/old.zarr','--output','old','--reference-independent',
          '--winding-message-iterations','500','--winding-fixed-phase','0.5',
          '--winding-fixed-scale','0.822','--orthogonal-probe-blend','1',
          '--partition-length-cost','--winding-defect-cost','100','--winding-weights','0,0,0,0.0375,0']
    config={'commands':[{'args':args,'name':'old'}],'input_files':['normal','/old.zarr/.zattrs']}
    changed=experiment.bp_config(config,Path('/new.zarr'))
    expected=list(args);expected[1]='/new.zarr';expected[3]=str(tmp_path/'independent2048')
    assert changed['commands'][0]['args']==expected
    assert changed['input_files']==['normal','/new.zarr/.zattrs']
    assert args[1]=='/old.zarr'


def test_shared_coverage_helper(tmp_path,monkeypatch):
    import reviewed_bp_prefix as original
    root=tmp_path/'root';(root/'full').mkdir(parents=True)
    cache=tmp_path/'cache';cache.mkdir()
    (cache/'cache_identity.json').write_text(json.dumps({'identity':{'source':{'level':0}},'geometry':{'scale':1,'chunk':1024}}))
    source={'id':'PHercParis4','fiber_manifest':str(cache/'inference.json')}
    (root/'workload.json').write_text(json.dumps({'checkpoint':'snapshot','checkpoint_sha256':'sha','native_binary':'binary','evaluation':{}}))
    (root/'full/summary.json').write_text(json.dumps({'config':{'sources':[source]}}))
    monkeypatch.setattr(original,'sha256_file',lambda _: 'sha')
    monkeypatch.setattr(original,'demand_gpu_manager_settings',lambda e,d: {'devices':d})
    observed={}
    class Session:
        def __init__(self,*args,**kwargs):observed.update(kwargs)
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def fill_chunks(self,requests):
            observed['requests']=requests
            return {'missing_after':0}
    monkeypatch.setattr(original,'CachedReplaySession',Session)
    original.prepare_coverage((1024,1024,1024),(2048,2048,2048),tmp_path/'out',root)
    assert observed['devices']=='0,1,2,3,4,5'
    assert len(observed['requests'][str(cache)])==27
    assert original.prepare_coverage.__defaults__[:2]==(original.LO,original.HI)
