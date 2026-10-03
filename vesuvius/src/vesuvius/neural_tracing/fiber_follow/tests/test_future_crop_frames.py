import numpy as np
import pytest

from vesuvius.neural_tracing.fiber_follow.evaluation.compare_future_crop_frames import containment
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec


def test_containment_does_not_hide_points_past_axial_bounds():
    crop = CropSpec(depth=11, width=9, behind=2, spacing=1.)
    result = containment(np.array([[0., 0., 5.], [0., 0., 9.]]), np.eye(3), crop)
    assert result['max_radial'] == 0
    assert not result['all_inside'] and not result['axial_all_inside']
    assert result['point_fraction'] == .5
    assert not any(result['width_sweep'].values())


def test_roll_changes_square_containment_but_not_radial_displacement():
    crop = CropSpec(depth=11, width=9, behind=2, spacing=1.)
    future = np.array([[5., 0., 5.]])
    a = np.pi/4
    frame = np.array([[np.cos(a), -np.sin(a), 0.], [np.sin(a), np.cos(a), 0.], [0., 0., 1.]])
    unrolled, rolled = [containment(future, f, crop) for f in (np.eye(3), frame)]
    assert not unrolled['all_inside'] and rolled['all_inside']
    assert rolled['max_radial'] == pytest.approx(unrolled['max_radial'])
    assert rolled['required_half_width'] == pytest.approx(5/np.sqrt(2))


def test_containment_includes_crop_boundary_and_checks_negative_forward():
    crop = CropSpec(depth=11, width=9, behind=2, spacing=1.)
    result = containment(np.array([[4., -4., -2.], [-4., 4., 8.]]), np.eye(3), crop)
    assert result['all_inside']
    assert not containment(np.array([[0., 0., -3.]]), np.eye(3), crop)['all_inside']
