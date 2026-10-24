"""Cross-section planes stay on one global grid, including at fractional spacing."""
from types import SimpleNamespace

import numpy as np
import pytest

from vesuvius.neural_tracing.tracers.fibers.tracing import volume
from vesuvius.neural_tracing.tracers.fibers.tracing.volume import tile_dots


def fiber_presence(axis):
    y, x = np.mgrid[:32, :32]
    cross_section = (255*np.exp(-((x-16)**2+(y-16)**2)/8)).astype(np.uint8)
    return np.moveaxis(np.repeat(cross_section[None], 256, axis=0), 0, 2-axis)


def plane_seeds(presence, axis, spacing, start, stop):
    lo, hi = np.zeros(3, int), np.array(presence.shape[::-1])
    lo[axis], hi[axis] = start, stop
    args = SimpleNamespace(threshold=128., dot_radius=2., fill_spacing=4., plane_spacing=spacing)
    dots = tile_dots(presence, lo, hi, args)
    return dots['xyz'][dots['plane'] != 'fill']


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('spacing, expected', [
    (64., [64., 128., 192.]),
    (64*7.91/9.6, [52., 105., 158.]),
    (64*9.6/7.91, [77., 155.]),
])
def test_planes_follow_global_spacing_independently_of_tile_boundaries(axis, spacing, expected):
    presence = fiber_presence(axis)
    whole = plane_seeds(presence, axis, spacing, 16, 208)
    split = np.concatenate([plane_seeds(presence, axis, spacing, a, b)
                            for a, b in ((16, 112), (112, 208))])
    np.testing.assert_array_equal(whole[:, axis], expected)
    np.testing.assert_array_equal(split, whole)


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('spacing', [.75, 1e-9])
def test_fractional_spacing_below_one_visits_each_plane_once(axis, spacing, monkeypatch):
    original_arange = np.arange

    def bounded_arange(*args, **kwargs):
        if len(args) >= 2 and args[1]-args[0] > 1000:
            pytest.fail('Subvoxel spacing allocated duplicate nominal planes')
        return original_arange(*args, **kwargs)

    monkeypatch.setattr(volume.np, 'arange', bounded_arange)
    dots = plane_seeds(fiber_presence(axis), axis, spacing, 16, 32)
    np.testing.assert_array_equal(dots[:, axis], np.arange(16, 32))


@pytest.mark.parametrize('spacing', [0., -1., float('nan'), float('inf'), -float('inf')])
def test_invalid_plane_spacing_is_rejected(spacing):
    with pytest.raises(ValueError, match='plane spacing must be finite and positive'):
        plane_seeds(fiber_presence(2), 2, spacing, 16, 32)
