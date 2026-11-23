"""Fiber classification and the radial normal stretch."""

import numpy as np
import torch

from sample_spiral import get_radial_normal_stretch
from spiral_helpers import classify_fiber_hv


DR = 12.0
WINDING = 40


def _vertical_fiber(theta_of_z, z_len=400.0, spacing=20.0, winding=WINDING):
    # Points on the continuous sheet r = DR * (winding + theta / 2pi) at
    # theta(z), z stepped at `spacing`. theta is deliberately not wrapped:
    # a fiber just past the seam sits on the next winding's radius.
    zs = np.arange(0.0, z_len + 1e-6, spacing)
    pts = []
    for z in zs:
        theta = float(theta_of_z(z))
        r = DR * (winding + theta / (2 * np.pi))
        pts.append([z, np.sin(theta) * r, np.cos(theta) * r])
    return np.asarray(pts, dtype=np.float32)


def _horizontal_fiber(wraps=0.5, theta0=0.3, spacing=20.0, winding=WINDING):
    pts = []
    theta = theta0
    while theta < theta0 + wraps * 2 * np.pi:
        r = DR * (winding + theta / (2 * np.pi))
        pts.append([0.0, np.sin(theta) * r, np.cos(theta) * r])
        theta += spacing / r
    return np.asarray(pts, dtype=np.float32)


def test_manual_tag_wins_over_geometry():
    horizontal = _horizontal_fiber()
    assert classify_fiber_hv(
        {'manual_tag': 'V', 'automatic_tag': 'H', 'automatic_certainty': 1.0},
        horizontal, min_z_fraction=0.8, min_auto_certainty=0.5) == 'V'
    assert classify_fiber_hv(
        {'manual_tag': 'h'}, _vertical_fiber(lambda z: 1.0),
        min_z_fraction=0.8, min_auto_certainty=0.5) == 'H'


def test_automatic_tag_requires_certainty_then_geometry_decides():
    vertical = _vertical_fiber(lambda z: 1.0)
    confident = {'automatic_tag': 'H', 'automatic_certainty': 0.9}
    unsure = {'automatic_tag': 'H', 'automatic_certainty': 0.2}
    assert classify_fiber_hv(
        confident, vertical, min_z_fraction=0.8, min_auto_certainty=0.5) == 'H'
    assert classify_fiber_hv(
        unsure, vertical, min_z_fraction=0.8, min_auto_certainty=0.5) == 'V'


def test_geometric_fallback_splits_vertical_horizontal_and_diagonal():
    assert classify_fiber_hv(
        None, _vertical_fiber(lambda z: 1.0),
        min_z_fraction=0.8, min_auto_certainty=0.5) == 'V'
    assert classify_fiber_hv(
        {}, _horizontal_fiber(),
        min_z_fraction=0.8, min_auto_certainty=0.5) == 'H'
    diagonal = np.asarray(
        [[0, 0, 0], [10, 10, 0], [20, 20, 0]], dtype=np.float32)
    assert classify_fiber_hv(
        {}, diagonal, min_z_fraction=0.8, min_auto_certainty=0.5) is None
    assert classify_fiber_hv(
        {}, diagonal[:1], min_z_fraction=0.8, min_auto_certainty=0.5) is None


class YxScale:
    """Scroll -> spiral map scaling y by `sy` and x by `sx` about the axis."""

    def __init__(self, sy, sx):
        self.sy, self.sx = float(sy), float(sx)

    def __call__(self, zyxs):
        out = zyxs.clone()
        out[..., 1] = out[..., 1] * self.sy
        out[..., 2] = out[..., 2] * self.sx
        return out


def _points_on_axes():
    # Points on the +x, +y axes and a diagonal, at several z.
    return torch.tensor([
        [10.0, 0.0, 50.0],
        [20.0, 50.0, 0.0],
        [30.0, 30.0, 40.0],
        [40.0, 0.0, -70.0],
        [50.0, -20.0, 0.0],
    ])


def test_stretch_is_unity_under_identity_and_the_scale_under_uniform_scaling():
    points = _points_on_axes()
    identity = torch.distributions.transforms.identity_transform
    torch.testing.assert_close(
        get_radial_normal_stretch(identity, points), torch.ones(5),
        rtol=0, atol=1e-5)
    torch.testing.assert_close(
        get_radial_normal_stretch(YxScale(2.5, 2.5), points),
        torch.full((5,), 2.5), rtol=0, atol=1e-4)


def test_stretch_follows_the_normal_direction_under_anisotropic_scaling():
    # Scaling x alone: the sheet normal at a point on the x axis is along x
    # (stretch 2), at a point on the y axis along y (stretch 1). Spiral-space
    # normals of the transformed points are still radial, so on the diagonal
    # the pulled-back covector is (0, y, 2x) normalised over the image.
    points = _points_on_axes()
    transform = YxScale(1.0, 2.0)
    stretch = get_radial_normal_stretch(transform, points)
    torch.testing.assert_close(stretch[[0, 3]], torch.full((2,), 2.0), rtol=0, atol=1e-4)
    torch.testing.assert_close(stretch[[1, 4]], torch.full((2,), 1.0), rtol=0, atol=1e-4)
    image = transform(points[2:3])
    n = torch.nn.functional.normalize(image[0, 1:], dim=0)
    expected = torch.sqrt((n[0] * 1.0) ** 2 + (n[1] * 2.0) ** 2)
    torch.testing.assert_close(stretch[2], expected, rtol=0, atol=1e-4)
