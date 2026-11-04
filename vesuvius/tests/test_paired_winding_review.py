import numpy as np
import pytest
from vesuvius.neural_tracing.fiber_trace_3d.paired_winding_review import identity,winding_shift


def test_compare_realized_winding_and_signed_steps():
    assert winding_shift([.05,.1,.2])==0
    assert winding_shift([.8,1.,1.2])==1
    assert winding_shift([-.8,-1.,-1.2])==-1


def test_piece_identity_not_array_position():
    a,b='00000000-0000-4000-8000-000000000012','00000000-0000-4000-8000-000000000010'
    d=dict(source_id=np.array([a,b]),start=np.array([0,0]),end=np.array([20,30]),family=np.array([0,1]))
    assert identity(d)[(b,0.,30.,1)]==1
    d={k:np.repeat(v[:1],2) for k,v in d.items()}
    with pytest.raises(ValueError):identity(d)
