import pytest
from types import SimpleNamespace
from vesuvius.neural_tracing.fiber_trace_3d.fiber_plateau_lr import PlateauLR


def test_two_phases_all_rates_and_new_best_resets_patience():
    opt=SimpleNamespace(param_groups=[dict(lr=2.),dict(lr=.1)])
    s=PlateauLR(opt,2)
    assert s.observe(0,10) is None
    assert s.observe(1,9) is None
    assert s.observe(2,9) is None
    assert s.observe(3,10)=='reduce'
    assert [g['lr'] for g in opt.param_groups]==pytest.approx([.2,.01])
    assert s.observe(4,8) is None
    assert s.observe(5,8) is None
    assert s.observe(6,8)=='stop'
    assert [g['lr'] for g in opt.param_groups]==pytest.approx([.2,.01])


def test_full_patience_and_nonfinite():
    s=PlateauLR(SimpleNamespace(param_groups=[]),100)
    for it in range(100):assert s.observe(it,1) is None
    assert s.observe(100,1)=='reduce'
    with pytest.raises(FloatingPointError):s.observe(101,float('nan'))


@pytest.mark.parametrize('p',[0,-1,1.5])
def test_invalid_patience(p):
    with pytest.raises(ValueError):PlateauLR(None,p)
