"""Dense packed-polyline interval lookup preserves right-sided interpolation."""
import gc
import numpy as np
import pytest
from vc_object_store import PolylineArcIndex


def test_strided_arc_locations_match_numpy_upper_bound_and_survive_owner_release():
    records=np.zeros(9,dtype=[('source','i8'),('arc','f8')])
    records['source']=[0,0,0,0,0,1,1,1,1]
    records['arc']=[0.,0.,2.,4.,4.,0.,1.,3.,5.]
    offsets=np.array([0,5,9],np.int64)
    owners=np.array([0,0,0,0,1,1,1,1,0],np.int64)
    values=np.array([-1e-5,0.,1.,4.,0.,1.,2.,5.00001,4.00001])
    query=np.empty(len(values),records.dtype);query['source']=owners
    query['arc']=np.clip(values,0,records['arc'][offsets[owners+1]-1])
    j=np.minimum(np.searchsorted(records,query,side='right')-1,offsets[owners+1]-2)
    fraction=(query['arc']-records['arc'][j])/np.maximum(records['arc'][j+1]-records['arc'][j],1e-300)
    locator=PolylineArcIndex(offsets,records['arc'])
    del offsets,records;gc.collect()
    result=locator.locate(owners,values)
    np.testing.assert_array_equal(result['segment'],j)
    np.testing.assert_array_equal(result['fraction'],fraction)
    del locator;gc.collect()
    assert result['segment'].shape==(len(values),)
    assert not result['segment'].flags.writeable and not result['fraction'].flags.writeable


def test_arc_locations_validate_query_bounds():
    index=PolylineArcIndex(np.array([0,3],np.int64),np.array([0.,2.,5.]))
    for owner,arc in ((-1,1.),(1,1.),(0,np.nan),(0,np.inf),(0,-.001),(0,5.001)):
        with pytest.raises(ValueError):index.locate(np.array([owner],np.int64),np.array([arc]))
    with pytest.raises(ValueError):index.locate(np.array([0,0],np.int64),np.array([1.]))
    with pytest.raises(ValueError):PolylineArcIndex(np.array([0,1],np.int64),np.array([0.]))
