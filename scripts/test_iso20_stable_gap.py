import pytest
import subprocess
from pathlib import Path
from iso20_stable_gap import blend_parameters


@pytest.mark.parametrize('alpha',[0,.25,.5,.75,1])
@pytest.mark.parametrize('cost',[150,300,600])
def test_blend_units(alpha,cost):
    p=blend_parameters(alpha,cost)
    for mass,length in [(0,128),(3.5,0),(17,345)]:
        actual=p['--winding-defect-cost']*((1-alpha)*mass+alpha*length/p['--winding-defect-length-unit'])
        assert actual==pytest.approx((1-alpha)*100*mass+alpha*cost*length/128)


@pytest.mark.parametrize('alpha,cost',[(-1,300),(2,300),(.5,0)])
def test_bad_blend(alpha,cost):
    with pytest.raises(ValueError):blend_parameters(alpha,cost)


@pytest.mark.parametrize('flag,value',[
    ('--winding-damping','0'),('--winding-damping','1.1'),
    ('--winding-defect-length-blend','-1'),('--winding-defect-length-blend','2')])
def test_native_rejects_invalid_options(flag,value):
    binary=Path(__file__).resolve().parents[1]/'volume-cartographer/build/fiber-evaluation/bin/vc_fiber_trace_chunk'
    result=subprocess.run([str(binary),'direction-ablation','unused',flag,value],capture_output=True,text=True)
    assert result.returncode!=0
    assert flag in result.stdout+result.stderr
