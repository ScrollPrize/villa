import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

directory=Path(__file__).parents[1]
sys.path.insert(0,str(directory))
spec=importlib.util.spec_from_file_location('density_inputs',directory/'prepare_defect_density_inputs.py')
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_weighted_statistics_and_shared_bounded_mapping():
    segments=np.array([[0,1,10,10,1],[1,10,18,18,1]],dtype=float)
    values=mod.densities(segments,0,10)
    assert values==dict(mean=2.8,median=2.,max=10.)
    assert mod.multiplier(0)==1
    assert mod.multiplier(.5)==.5
    assert mod.multiplier(100)==.1
    assert mod.multiplier(values['max']) <= mod.multiplier(values['mean'])
    with pytest.raises(ValueError): mod.multiplier(float('nan'))
    with pytest.raises(ValueError): mod.multiplier(-1.)
