"""json_umbilicus_z_to_yx holds the end control points past the ends of the line, like VC3D's
Umbilicus::interpolate_center, and stays linear between control points."""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from umbilicus import json_umbilicus_z_to_yx  # noqa: E402


def _write(tmp_path, points):
    path = tmp_path / 'umbilicus.json'
    path.write_text(json.dumps({'control_points': [{'x': x, 'y': y, 'z': z} for x, y, z in points]}))
    return str(path)


def test_holds_end_points_past_the_ends(tmp_path):
    # Unsorted on purpose: the reader sorts by z.
    path = _write(tmp_path, [(130, 210, 300), (100, 200, 100), (110, 190, 200)])
    umb = json_umbilicus_z_to_yx(path)
    np.testing.assert_allclose(umb([0, 50, 100]), [[200, 100]] * 3)
    np.testing.assert_allclose(umb([300, 400, 10_000]), [[210, 130]] * 3)


def test_linear_between_control_points(tmp_path):
    path = _write(tmp_path, [(100, 200, 100), (110, 190, 200), (130, 210, 300)])
    umb = json_umbilicus_z_to_yx(path)
    np.testing.assert_allclose(umb([150, 250]), [[195, 105], [200, 120]])


def test_coordinate_scale_applies_to_the_held_ends(tmp_path):
    path = _write(tmp_path, [(100, 200, 100), (130, 210, 300)])
    umb = json_umbilicus_z_to_yx(path, coordinate_scale=0.5)
    np.testing.assert_allclose(umb([0, 1_000]), [[100, 50], [105, 65]])
