"""CT-only initialization, distance-based direction fitting and seed plumbing."""
import numpy as np
import pytest

from vesuvius.neural_tracing.fiber_follow.tracing.heading import ct_sheet_heading, ct_seed_heading, linear12_heading, SeedHeadingError
from vesuvius.neural_tracing.fiber_follow.tracing.inference import trace_bidirectional
from vesuvius.neural_tracing.fiber_follow.evaluation.legacy_evaluate import make_seeds
from vesuvius.neural_tracing.fiber_follow.data.data import TracedFiber
from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength


def sheet(shape=(97,)*3, normal=(1., 2., .2)):
    zyx = np.stack(np.meshgrid(*[np.arange(n)-(n-1)/2 for n in shape], indexing='ij'), -1)
    n = np.array(normal, dtype=float)
    n /= np.linalg.norm(n)
    return 255*np.exp(-(zyx[..., ::-1] @ n)**2/8), n


class CTReader:
    def __init__(self, image):
        self.image, self.shape, self.reads = image, image.shape, []

    def read(self, start, size):
        self.reads.append((np.array(start), size))
        return self.image[tuple(slice(int(a), int(a+b)) for a, b in zip(start, size))]


class CTOnlyVolume:
    input_scale = 2.

    def __init__(self, image):
        self.ct = CTReader(image)

    @property
    def presence(self):
        raise AssertionError('Seed heading must not access presence')

    def presence_for_seeding(self):
        raise AssertionError('Seed heading must not open presence')

    def direction_fields(self):
        raise AssertionError('Seed heading must not open directions')


def test_ct_seed_axis_uses_world_xyz_and_native_ct_scale():
    image, n = sheet()
    z = np.array([0., 0., 1.])
    for family, expected in (('H', np.cross(z, n)), ('V', z-(n@z)*n)):
        vol = CTOnlyVolume(image)
        pos = np.array([24., 24., 24.])
        axis = ct_seed_heading(vol, pos, family)
        assert abs(axis@expected)/np.linalg.norm(expected) > .99999
        np.testing.assert_array_equal(vol.ct.reads[0][0], [16, 16, 16])
        np.testing.assert_array_equal(pos, [24., 24., 24.])


def test_missing_or_ambiguous_ct_cannot_invent_a_heading():
    with pytest.raises(SeedHeadingError, match='normal'):
        ct_sheet_heading(np.zeros((65,)*3), [32]*3, 'H')
    image, _ = sheet((65,)*3, (0., 0., 1.))
    with pytest.raises(SeedHeadingError, match='ambiguous'):
        ct_sheet_heading(image, [32]*3, 'V')
    with pytest.raises(SeedHeadingError, match='family'):
        ct_sheet_heading(image, [32]*3, '')


def test_bidirectional_initialization_preserves_fractional_seed_positions():
    image, _ = sheet()
    vol = CTOnlyVolume(image)
    class Tracer:
        def __init__(self): self.calls = []
        def trace(self, positions, axes):
            self.calls.append((positions.copy(), axes.copy()))
            return [np.stack([p, p+h]) for p,h in zip(positions, axes)], ['test']*len(positions)
    tracer = Tracer()
    seeds = np.array([[24.1, 23.8, 24.3], [24., 24., 24.]])
    results = trace_bidirectional(tracer, vol, seeds, ['H', 'V'])
    assert len(results) == 2
    for positions, _ in tracer.calls: np.testing.assert_array_equal(positions, seeds)
    np.testing.assert_array_equal(tracer.calls[0][1], -tracer.calls[1][1])
    for i, (path, _) in enumerate(results): np.testing.assert_array_equal(path[1], seeds[i])


def test_collection_seeds_use_ct_axis_not_annotation_tangent():
    image, _ = sheet()
    vol = CTOnlyVolume(image)
    p = np.array([[24., 24., 18.], [24., 24., 30.]])
    fiber = TracedFiber('test', p, arclength(p), 'H')
    seeds = make_seeds([fiber], vol, per_fiber=1, margin=4., seed=4)
    assert len(seeds) == 2
    np.testing.assert_array_equal(seeds[0]['heading'], -seeds[1]['heading'])
    # The annotation points along z but H requests the sheet's horizontal axis.
    assert abs(seeds[0]['heading'][2]) < 1e-10
    assert len(vol.ct.reads) == 1


def test_linear12_has_a_physical_baseline_free_intercept_and_rigid_invariance():
    prior = np.array([[0., 0., 0.], [0., 0., 11.9]])
    assert linear12_heading(prior) is None
    path = np.array([[8., -3., 0.], [8., -3., 20.]])
    np.testing.assert_allclose(linear12_heading(path), [0., 0., 1.])
    assert linear12_heading(path, start=1) is None
    x = np.arange(30.)
    path = np.c_[x, np.sin(x/9), np.zeros(len(x))]
    dense = np.concatenate([a+(b-a)*np.arange(7)[:,None]/7 for a,b in zip(path[:-1],path[1:])]+[path[-1:]])
    a = linear12_heading(path)
    np.testing.assert_allclose(linear12_heading(dense), a, atol=1e-12)
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    np.testing.assert_allclose(linear12_heading(path@rotation.T+[100.,-100.,42.]), rotation@a, atol=1e-12)
