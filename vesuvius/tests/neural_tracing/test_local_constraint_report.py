import importlib.util
from pathlib import Path
import pytest
spec = importlib.util.spec_from_file_location('local_report', Path(__file__).resolve().parents[2] / 'src/vesuvius/neural_tracing/fiber_trace_3d/local_constraint_report.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
geometry, validate_clean_geometry = module.geometry, module.validate_clean_geometry


def rows(second=10):
    return [dict(original_trace='7', source_piece='0', begin_arc_base='0', end_arc_base='10'),
            dict(original_trace='7', source_piece='1', begin_arc_base=str(second), end_arc_base='20')]


def test_clean_geometry():
    validate_clean_geometry(rows())
    assert geometry(rows())[1] == ('7', '1', 10., 20.)


@pytest.mark.parametrize('second', [9, 11, 20])
def test_reject_overlap_gap_or_empty(second):
    with pytest.raises(ValueError):
        validate_clean_geometry(rows(second))
