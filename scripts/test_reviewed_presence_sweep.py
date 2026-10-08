import json
from reviewed_presence_sweep import MODELS,variants,THRESHOLDS


def test_complete_matched_factorial():
    names=set()
    for model,(folder,_) in MODELS.items():
        base=json.loads((folder/'workload.json').read_text())
        matrix=list(variants(base,model))
        assert len(matrix)==16
        for name,cfg,p in matrix:
            names.add(name)
            assert cfg['evaluation']['inference']==base['evaluation']['inference']
            assert cfg['evaluation']['inference_cache_namespace']==base['evaluation']['inference_cache_namespace']
            assert cfg['checkpoint']==base['checkpoint']
            args=cfg['evaluation']['trace_args']
            assert float(args[args.index('--replay-presence-cutoff')+1])==p['cutoff']
            assert float(args[args.index('--width-relative-tolerance')+1])==(.4 if p['width'] else 0)
            assert ('--replay-reverse-to-start' in args)==p['reverse']
            assert ('--replay-reverse-all-endpoints' in args)==p['reverse']
    assert len(names)==32 and THRESHOLDS==(0,.1,.25,.5)


def test_post_modulated_twenty_eight_rows():
    folder,_=MODELS['ema999_12100']
    base=json.loads((folder/'workload.json').read_text())
    matrix=list(variants(base,'ema999_12100',post=True))
    assert len(matrix)==28 and len({name for name,_,_ in matrix})==28
    assert {p['cutoff'] for _,_,p in matrix}=={.1,.2,.3,.4,.5,.6,.7}
    assert {(p['width'],p['reverse']) for _,_,p in matrix}=={(False,False),(False,True),(True,False),(True,True)}
    for _,cfg,p in matrix:
        assert p['mode']=='modulated'
        assert cfg['evaluation']['inference']==base['evaluation']['inference']
        assert cfg['evaluation']['inference_cache_namespace']==base['evaluation']['inference_cache_namespace']
        args=cfg['evaluation']['trace_args']
        assert args[args.index('--replay-presence-mode')+1]=='modulated'
