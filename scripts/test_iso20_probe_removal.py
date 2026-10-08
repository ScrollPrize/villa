import pytest
from iso20_probe_removal import oracle_args,pruning_counts,ORACLE_FLAGS


def test_conversion_preserves_graph():
    args=['direction-ablation','trace','--reference-independent','--orthogonal-probe-span','64','--winding-weights','0,0,0,0.05,0']
    result=oracle_args(args)
    assert result[:-len(ORACLE_FLAGS)]==[a for a in args if a!='--reference-independent']
    assert result[-len(ORACLE_FLAGS):]==ORACLE_FLAGS
    with pytest.raises(ValueError):oracle_args(result)


def test_removal_accounting():
    r=pruning_counts('constraint pruning benchmark\nheader\n100 10 3 2 15 85 15 17.647\n')
    assert r['problematic_percent_initial']==15
    assert r['retained_fulfilled']==85
    with pytest.raises(AssertionError):pruning_counts('constraint pruning benchmark\nheader\n100 10 3 2 14 85\n')
