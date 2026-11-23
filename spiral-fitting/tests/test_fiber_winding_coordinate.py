"""The fiber winding coordinate w (fiber_solve_inputs) is vc_object_store's
seam winding: W + turns + the fraction of a turn past the seam."""
import numpy as np
import pytest

from fiber_solve_inputs import (WindingSolveError, fiber_winding_coordinate,
                                packed_winding_coordinate)

vc_object_store = pytest.importorskip('vc_object_store')

# A tilted, piecewise-linear umbilicus (XYZ), deliberately not sorted by z.
AXIS = np.array([[120.0, 80.0, 300.0], [100.0, 100.0, 0.0], [110.0, 85.0, 150.0]])


def centre(z):
    order = np.argsort(AXIS[:, 2])
    return np.stack([np.interp(z, AXIS[order, 2], AXIS[order, 0]),
                     np.interp(z, AXIS[order, 2], AXIS[order, 1])], axis=-1)


def helix(turns, radius, z0, direction, start=0.37, steps_per_turn=60):
    """A helix about the axis rising from z0, ``direction`` +1 anticlockwise
    in XY; no vertex on the seam ray (-pi/2)."""
    angle = start + direction * np.linspace(0, 2 * np.pi * turns,
                                            int(turns * steps_per_turn) + 1)
    z = z0 + np.linspace(0, 40 * turns, len(angle))
    return np.concatenate([centre(z) + radius * np.stack(
        [np.cos(angle), np.sin(angle)], axis=-1), z[:, None]], axis=-1)


def native_winding(offsets, xyz, W, handedness):
    """W + turns + phi_seam from vc_object_store's WindingSeam."""
    turns = vc_object_store.pack_winding_geometry(
        offsets, xyz, AXIS, handedness=handedness)['turns']
    relative = vc_object_store.NativeWindingSeam(AXIS, handedness=handedness).relative(xyz)
    phase = np.mod((np.arctan2(relative[:, 1], relative[:, 0]) + np.pi / 2) / (2 * np.pi), 1.0)
    phi = phase if handedness == 1 else np.mod(-phase, 1.0)
    return np.repeat(W, np.diff(offsets)) + turns + phi


@pytest.mark.parametrize('handedness', [1, -1])
def test_w_is_the_native_seam_winding(handedness):
    fibers = [helix(3.4, 30.0, 10.0, 1), helix(2.7, 55.0, 60.0, -1, start=2.1),
              helix(0.6, 18.0, 250.0, 1, start=-1.2)]
    W = np.array([4, -7, 0])
    offsets = np.concatenate([[0], np.cumsum([len(fiber) for fiber in fibers])])
    xyz = np.ascontiguousarray(np.concatenate(fibers))
    expected = native_winding(offsets, xyz, W, handedness)
    np.testing.assert_allclose(packed_winding_coordinate(offsets, xyz, W, AXIS, handedness),
                               expected, rtol=0, atol=1e-9)
    for index, fiber in enumerate(fibers):
        np.testing.assert_allclose(
            fiber_winding_coordinate(fiber, W[index], AXIS, handedness),
            expected[offsets[index]:offsets[index + 1]], rtol=0, atol=1e-9)
    # The first helix gains one winding per turn in the scroll's direction.
    first = fiber_winding_coordinate(fibers[0], 4, AXIS, handedness)
    assert first[-1] - first[0] == pytest.approx(3.4 * handedness)


@pytest.mark.parametrize('change, message', [
    (lambda xyz: xyz.__setitem__((5, 1), np.nan), 'not finite'),
    (lambda xyz: xyz.__setitem__(5, [*centre(xyz[5, 2]), xyz[5, 2]]), 'within 1e-06'),
    (lambda xyz: _rotate_after(xyz, 5), 'turns apart'),
])
def test_the_load_guard_names_the_fiber(change, message):
    fibers = [helix(1.0, 30.0, 10.0, 1), helix(1.0, 30.0, 10.0, -1)]
    change(fibers[1])
    offsets = np.array([0, len(fibers[0]), len(fibers[0]) + len(fibers[1])])
    with pytest.raises(WindingSolveError, match=f'fiber second: .*{message}'):
        packed_winding_coordinate(offsets, np.concatenate(fibers), [0, 0], AXIS, -1,
                                  name=['first', 'second'].__getitem__)


def _rotate_after(xyz, index):
    """Turn every vertex after ``index`` half a turn about the axis (in
    place): the step from ``index`` sweeps about 0.5 turns."""
    after = slice(index + 1, None)
    xyz[after, :2] = 2 * centre(xyz[after, 2]) - xyz[after, :2]
