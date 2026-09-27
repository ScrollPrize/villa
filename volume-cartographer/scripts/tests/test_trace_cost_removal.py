import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('costs',Path(__file__).parents[1]/'analyze_trace_cost_removal.py')
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_boundary_proration_preserves_original_step_max():
    a=np.array([[0,8,4,4,2],[8,16,12,12,2]],dtype=float)
    x=mod.piece_costs(a,4,10)
    assert x['total_cost']==5
    assert x['max_original_step_cost']==12
    assert x['max_cost_per_base_voxel']==1.5
    assert x['mean_cost_per_prediction_voxel']==pytest.approx(5/3)
    assert mod.piece_costs(a,0,8)['max_original_step_cost']==4


def test_uncovered_piece_rejected():
    with pytest.raises(ValueError,match='fully covered'):
        mod.piece_costs(np.array([[0,8,4,4,2]],dtype=float),0,10)
