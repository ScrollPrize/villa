import copy
import iso20_post_overlap as sweep


def test_filter_uses_frozen_completed_pieces_without_tracing():
    t={'commands':[{'args':['trace','-','--normal-manifest','normal','--fiber-manifest','fiber']}], 'input_files':[]}
    old=copy.deepcopy(t)
    config=sweep.filter_config(t,'test','presence',10,20,.8)
    args=config['commands'][0]['args']
    assert args[0]=='deduplicate'
    assert args[1]==str(sweep.SOURCE/'beam-isotropic20.zarr')
    assert '--stop-at-covered-traces' not in args and '--reverse-consistency' not in args
    assert t==old


def test_matrix_and_bp_preserve_protocol():
    assert len(sweep.VARIANTS)==12
    b={'commands':[{'args':['direction-ablation','old.zarr','--reference-independent','--winding-message-iterations','500','--output','old']}], 'input_files':['old.zarr/.zattrs']}
    config=sweep.bp_config(b,'test',sweep.OUT/'test.zarr')
    args=config['commands'][0]['args']
    assert '--reference-independent' in args
    assert args[args.index('--winding-message-iterations')+1]=='500'
    assert config['input_files']==[str(sweep.OUT/'test.zarr/.zattrs')]


def test_low_matrix_includes_zero_then_isotropic_and_shared_geometry_cache(monkeypatch):
    for name in ('OUT','REPORT','LOW','VARIANTS'):
        monkeypatch.setattr(sweep,name,getattr(sweep,name))
    sweep.setup_low()
    assert len(sweep.VARIANTS)==36
    assert {(n,t) for _,_,n,t,_ in sweep.VARIANTS[:18]}=={(10,20),(20,40)}
    assert {(n,t) for _,_,n,t,_ in sweep.VARIANTS[18:]}=={(10,10),(20,20)}
    assert {f for _,_,_,_,f in sweep.VARIANTS}=={0.,.1,.2}
    t={'commands':[{'args':['trace','-','--normal-manifest','normal','--fiber-manifest','fiber']}], 'input_files':[]}
    a=sweep.filter_config(t,'a','length',10,20,0)['commands'][0]['args']
    b=sweep.filter_config(t,'b','presence',10,20,.2)['commands'][0]['args']
    assert a[a.index('--overlap-cache')+1]==b[b.index('--overlap-cache')+1]
    assert float(a[a.index('--overlap-fraction')+1])==0
