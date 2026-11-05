import numpy as np
from vesuvius.neural_tracing.fiber_trace_3d.fiber_bucket_uv_visualization import visible_connection_segments


def test_all_visible_edges_without_confidence_or_same_sheet_gate():
    d=dict(i=np.array([0,0,1]),j=np.array([1,1,2]),start=np.zeros(3),end=np.ones(3)*10,
           ds_a=np.array([2.,8.,3.]),ds_b=np.array([4.,6.,5.]),node_ds=np.array([[0,10]]*3))
    g=(np.array([0,1]),[np.array([0,10]),np.array([100,110])], [np.zeros(2),np.ones(2)*20])
    result=visible_connection_segments(d,g,np.array([],dtype=int),np.array([]))
    assert result==[[[2.,0.],[104.,20.]],[[8.,0.],[106.,20.]]]
    assert visible_connection_segments(d,g,np.array([],dtype=int),np.array([]),excluded=[0])==result[1:]
