import numpy as np
import pytest
from reviewed_tile_repeatability import value_stats,compare,choose_tile


def test_exact_and_nonfinite():
    a=np.array([0,255],np.uint8)
    assert value_stats(a,a)['changed_percent']==0
    assert value_stats(a,a[::-1])['max']==255
    with pytest.raises(ValueError):
        value_stats(np.array([np.nan]),np.array([0.]))


def test_rounding_boundary():
    a=np.zeros((1,7,1,1,2),np.float32)
    a[:,6]=.5/255-1e-6
    b=a.copy()
    b[:,6,0,0,0]=.5/255+1e-6
    qa={'presence':np.round(a[:,6]*255).astype(np.uint8)}
    qb={'presence':np.round(b[:,6]*255).astype(np.uint8)}
    result=compare(a,b,qa,qb)
    assert result['rounded']['presence']['changed_percent']==50
    assert result['raw']['6']['max'] < 3e-6
    assert result['presence_rounding']['raw_equal_but_byte_changed']==0


def test_canonical_tile_covers_probe():
    from lasagna.demand_predict3d import DemandGeometry
    geometry=DemandGeometry((1000,1000,1000),256,96,32,1,64)
    xyz=[1100.,900.,1300.]
    tile=choose_tile(geometry,xyz,2)
    center=np.array(xyz[::-1])/2
    origin=geometry.origin(tuple((center//64).astype(int)))
    assert tile in geometry.contributors(origin)
    assert np.all(center>=np.array(tile)-32)
    assert np.all(center<np.array(tile)-32+256)
