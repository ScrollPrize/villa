import numpy as np

from vesuvius.neural_tracing.fiber_follow.data.data import TracedFiber
from vesuvius.neural_tracing.fiber_follow.evaluation.compare_sheet_normals import (
    angle, crossing, normal_at, tensor_at,
)
from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength, normalize


def line(direction, offset, tag):
    points = np.arange(-80., 81.)[:, None]*normalize(np.array(direction))+offset
    return TracedFiber(tag, points, arclength(points), tag)


def test_crossing_normal_recovers_rotated_plane_and_rejects_separate_sheets():
    n = normalize(np.array([1., 2., 3.]))
    h = normalize(np.cross(n, [0., 0., 1.]))
    v = np.cross(n, h)
    a = line(h, np.zeros(3), 'H')
    b = line(v, .5*n, 'V')
    result = crossing(a, b, np.zeros(3))
    assert abs(result['gap_ct']-.5) < 1e-6
    assert angle(np.array(result['normals']['12']), n) < 1e-5
    assert crossing(a, line(v, 3*n, 'V'), np.zeros(3)) is None
    assert crossing(a, line(h, np.zeros(3), 'V'), np.zeros(3)) is None


def test_tensor_normal_uses_xyz_at_fractional_coordinates():
    xyz = np.indices((101,)*3).transpose(1, 2, 3, 0)[..., ::-1]
    n = normalize(np.array([1., 2., .4]))
    pos = np.array([50.2, 50.4, 50.1])
    image = 255*np.exp(-((xyz-pos) @ n)**2/8)
    actual, gap, energy = tensor_at(image, np.zeros(3), pos)
    assert angle(actual, n) < .05
    assert gap > .99 and energy > 0


def test_unsigned_interpolation_preserves_antipodal_normals_and_masks_missing():
    class Array:
        def __init__(self, data): self.data = data
        def read(self, start, size):
            np.testing.assert_array_equal(start, [1, 2, 3])
            return self.data
    nx = np.full((2, 2, 2), 255, dtype=np.uint8)
    nx[1] = 1  # +x and -x represent exactly the same sheet
    arrays = dict(nx=Array(nx), ny=Array(np.full_like(nx, 128)), grad_mag=Array(np.ones_like(nx)))
    manifest = dict(source_to_base=1., groups=dict(nx=dict(scaledown=4)))
    # CT is base/4: normal coordinates = CT coordinates / 4.
    n, support, coherence = normal_at(arrays, manifest, 4., np.array([3.5, 2.5, 1.5])*4)
    assert angle(n, [1., 0., 0.]) < 1e-5
    assert support == 1 and coherence == 1
    arrays['grad_mag'].data[1] = 0
    n, support, _ = normal_at(arrays, manifest, 4., np.array([3.5, 2.5, 1.5])*4)
    assert np.isnan(n).all() and support == .5


def test_threshold_zeros_only_values_below_60_before_tensor_computation(monkeypatch):
    from vesuvius.neural_tracing.fiber_follow.evaluation import compare_sheet_normals as comparison
    block = np.resize(np.array([0, 59, 60, 61, 255], dtype=np.uint8), (101, 101, 101))
    original = block.copy()
    captured = []
    def tensor(cube, center):
        captured.append(cube.copy())
        return np.diag([1., 2., 3.])
    monkeypatch.setattr(comparison, 'ct_structure_tensor', tensor)
    comparison.tensor_at(block, np.zeros(3), [50.]*3, threshold=60)
    expected = original[18:83, 18:83, 18:83]
    np.testing.assert_array_equal(captured[0], np.where(expected < 60, 0, expected))
    np.testing.assert_array_equal(block, original)
    assert set(np.unique(captured[0])) == {0, 60, 61, 255}
