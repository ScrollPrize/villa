import numpy as np
from build_support import load_build
load_build()
from vc_reader import _reader


def test_display_conversion_and_palette():
    values = np.array([-1, 1.9, 255, np.nan, np.inf], np.float32)
    output = np.empty(5, np.uint8)
    coverage = np.empty_like(output)
    _reader.display_to_uint8(values, output, coverage)
    np.testing.assert_array_equal(output, [0, 1, 255, 0, 0])
    np.testing.assert_array_equal(coverage, [1, 1, 1, 0, 0])
    ramp = _reader.window_level_lut(10, 110)
    assert ramp[10] == 0xff000000 and ramp[60] == 0xff7f7f7f and ramp[110] == 0xffffffff


def test_pathfinding_entered_pixel_cost_and_ties():
    costs = np.array([[1, 100, 1], [1, 1, 1], [np.inf, np.inf, 1]], np.float32)
    path = _reader.shortest_path(costs, 0, 8)
    assert path == [0, 4, 8]
    assert _reader.shortest_path(costs, 4, 4) == [4]
    costs[1] = np.inf
    assert _reader.shortest_path(costs, 0, 8) == []
