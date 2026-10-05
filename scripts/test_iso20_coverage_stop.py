import copy
import iso20_coverage_stop as sweep


def test_pilot_only_changes_stop_radius_not_seed_or_crop():
    t={'commands':[{'name':'old','args':['trace','-','--bbox','0','0','0','1024','1024','1024','--coverage','40','--coverage-tangential','80','--output','old.zarr']}], 'input_files':[]}
    b={'commands':[{'name':'bp','args':['direction-ablation','old.zarr','--reference-independent','--winding-message-iterations','500','--output','old']}], 'input_files':['old.zarr/.zattrs']}
    old=copy.deepcopy((t,b))
    tc,bc=sweep.configs('test',(10,20),t,b)
    args=tc['commands'][0]['args']
    assert args[args.index('--coverage')+1]=='40'
    assert args[args.index('--coverage-tangential')+1]=='80'
    assert args[args.index('--covered-trace-normal')+1]=='10'
    assert args[args.index('--covered-trace-tangential')+1]=='20'
    assert '--reference-independent' in bc['commands'][0]['args']
    assert (t,b)==old
    assert bc['input_files']==[str(sweep.OUT/'test.zarr/.zattrs')]


def test_control_does_not_enable_stopping():
    t={'commands':[{'args':['trace','-','--output','old']}], 'input_files':[]}
    b={'commands':[{'args':['direction-ablation','old','--output','old']}], 'input_files':[]}
    tc,_=sweep.configs('control',None,t,b)
    assert '--stop-at-covered-traces' not in tc['commands'][0]['args']
