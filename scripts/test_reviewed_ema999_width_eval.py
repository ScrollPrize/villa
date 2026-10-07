import pytest
from reviewed_ema999_width_eval import configs,validate_ema


def test_ema_kind():
    validate_ema({'snapshot_kind':'ema','ema_metadata':{'decay':.999}})
    for bad in ({'snapshot_kind':'raw'},{'snapshot_kind':'ema','ema_metadata':{'decay':.99}}):
        with pytest.raises(ValueError): validate_ema(bad)


def test_matched_configs():
    variants=configs('/tmp/frozen.pt','digest')
    inferences=[]
    for name,cfg in variants.items():
        assert cfg['checkpoint']=='/tmp/frozen.pt'
        inferences.append(cfg['evaluation']['inference'])
        args=cfg['evaluation']['trace_args']
        assert float(args[args.index('--width-relative-tolerance')+1])==(0 if name=='off_no_reverse' else .4)
        if name=='cosine40_fw512':
            assert args[args.index('--replay-reverse-interval-base')+1]=='512'
            assert args[args.index('--replay-reverse-normal-base')+1]=='20'
            assert args[args.index('--replay-reverse-tangential-base')+1]=='40'
            assert '--replay-reverse-to-start' in args and '--replay-reverse-all-endpoints' in args
        else:
            assert not any(a.startswith('--replay-reverse') for a in args)
        assert cfg['devices']=='0,1,2,3,4,5'
    assert inferences[0]==inferences[1]==inferences[2]
    assert len({c['evaluation']['inference_cache_namespace'] for c in variants.values()})==1
