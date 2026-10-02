"""Observed-path geometry samples beyond the crop, in the current frame."""
import numpy as np

from test_history_slabs import observation
from vesuvius.neural_tracing.fiber_follow.models.path_geometry import COUNT, OFFSETS, path_geometry_samples


def test_samples_follow_observed_path_without_crop_mask_and_differ_only_beyond_shared_tail():
    path = np.c_[np.r_[np.zeros(60), np.linspace(0, 6, 41)], np.zeros(101), np.arange(101.)]
    features, valid = path_geometry_samples(observation(path))
    offsets = np.asarray(OFFSETS)
    assert features.shape == (COUNT, 8) and valid.shape == (COUNT,)
    np.testing.assert_array_equal(valid[:-1], offsets <= 100)
    assert valid[-1] and features[-1, 7] == 1 and features[:-1, 7].max() == 0
    # Positions are relative to the head in its frame; samples far outside a crop stay valid.
    far = int(np.flatnonzero(offsets == 96)[0])
    # The head is 6 voxels lateral of the path's straight start: that offset must survive.
    assert valid[far] and features[far, 2] < -90 and abs(features[far, 0]+6) < 1e-6
    np.testing.assert_allclose(features[-1, :3], path[0]-path[-1], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(features[valid, 3:6], axis=-1), 1, atol=1e-6)
    assert features[~valid].max() == 0 == features[~valid].min()
    assert not path_geometry_samples(observation(np.zeros((1, 3))))[1].any()
    shared = np.c_[np.zeros(13), np.zeros(13), 88+np.arange(13.)]
    a = np.concatenate((np.c_[np.full(88, 4.), np.zeros(88), np.arange(88.)], shared))
    b = np.concatenate((np.c_[np.full(88, -4.), np.zeros(88), np.arange(88.)], shared))
    fa, va = path_geometry_samples(observation(a))
    fb, vb = path_geometry_samples(observation(b))
    near, older = offsets <= 8, offsets >= 24
    np.testing.assert_allclose(fa[:-1][near], fb[:-1][near], atol=1e-6)
    assert np.abs(fa[:-1][older & va[:-1], 0]-fb[:-1][older & vb[:-1], 0]).min() > 7
