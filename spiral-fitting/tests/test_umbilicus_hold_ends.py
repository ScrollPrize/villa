import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from umbilicus import json_umbilicus_z_to_yx  # noqa: E402


def _write(tmp_path, pts):
    p = tmp_path / 'umbilicus.json'
    p.write_text(json.dumps({'control_points': [{'x': x, 'y': y, 'z': z} for x, y, z in pts]}))
    return str(p)


def test_inside_is_linear_interpolation(tmp_path):
    f = json_umbilicus_z_to_yx(_write(tmp_path, [(100, 200, 10), (110, 220, 20), (130, 230, 40)]))
    np.testing.assert_allclose(f(15.0), [210.0, 105.0])
    np.testing.assert_allclose(f(30.0), [225.0, 120.0])


def test_past_the_ends_holds_the_end_points(tmp_path):
    # last two points 1 slice apart and 50 voxels apart sideways: a straight-line extension would put the
    # centre 5,000 voxels away 100 slices past the end
    f = json_umbilicus_z_to_yx(_write(tmp_path, [(100, 200, 10), (110, 220, 20), (160, 220, 21)]))
    np.testing.assert_allclose(f(0.0), [200.0, 100.0])
    np.testing.assert_allclose(f(121.0), [220.0, 160.0])
    np.testing.assert_allclose(f(np.array([-50.0, 500.0])), [[200.0, 100.0], [220.0, 160.0]])


def test_unsorted_control_points(tmp_path):
    f = json_umbilicus_z_to_yx(_write(tmp_path, [(130, 230, 40), (100, 200, 10), (110, 220, 20)]))
    np.testing.assert_allclose(f(5.0), [200.0, 100.0])
    np.testing.assert_allclose(f(45.0), [230.0, 130.0])
