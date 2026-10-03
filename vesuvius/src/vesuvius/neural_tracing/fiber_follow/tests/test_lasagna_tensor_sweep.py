import numpy as np
import pytest

from vesuvius.neural_tracing.fiber_follow.evaluation.sweep_lasagna_tensor import Variant, estimate, variants, paired
from vesuvius.neural_tracing.fiber_follow.evaluation.compare_sheet_normals import angle
from vesuvius.neural_tracing.fiber_follow.shared.geometry import frame_from_heading, normalize
from vesuvius.neural_tracing.fiber_follow.tracing.heading import ct_structure_tensor, SeedHeadingError


def test_sweep_grid_rotation_and_smoothing_preserve_a_planar_normal():
    z, y, x = np.indices((101, 101, 101))
    n = normalize(np.array([.3, .8, -.4]))
    block = (128+30*np.sin(((x-50)*n[0]+(y-50)*n[1]+(z-50)*n[2])*.12)).astype(np.float32)
    frame = frame_from_heading(normalize(np.array([.7, -.1, .5])))
    for spacing in (1., 1.25, 2.5):
        for derivative, integration in ((1., 4.), (2., 6.)):
            v = Variant(spacing, derivative, integration, 4*integration+3*derivative)
            actual, gap, _, _ = estimate(block, np.zeros(3), np.array([50.2, 50.3, 50.1]), frame, v)
            assert angle(actual, n) < .5
            assert gap > .99
    with pytest.raises(ValueError, match='Insufficient CT context'):
        estimate(block, np.zeros(3), np.array([5., 5., 5.]), frame, variants()[0])


def test_sigmas_validate_and_default_tensor_is_unchanged():
    rng = np.random.default_rng(2)
    raw = rng.uniform(0, 255, (17, 18, 18))
    a = ct_structure_tensor(raw, [8, 8.5, 8.5], sample_spacing=2.5)
    b = ct_structure_tensor(raw, [8, 8.5, 8.5], sample_spacing=2.5, derivative_sigma=1., integration_sigma=4.)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, ct_structure_tensor(raw, [8, 8.5, 8.5], sample_spacing=2.5, integration_sigma=8.))
    for key in ('derivative_sigma', 'integration_sigma'):
        for bad in (0., -1., np.nan):
            with pytest.raises(SeedHeadingError, match='sigmas'):
                ct_structure_tensor(raw, [8, 8.5, 8.5], **{key: bad})


def test_paired_comparison_uses_regions_and_excludes_unmatched_values():
    baseline = np.full((32, 7), 10.)
    changed = baseline-2
    changed[1, 0] = np.nan
    result = paired(changed, baseline, np.ones_like(baseline, bool))
    assert result['n'] == 223 and result['mean_change'] == -2
    assert result['region_bootstrap_mean_change_ci95'] == [-2., -2.]
