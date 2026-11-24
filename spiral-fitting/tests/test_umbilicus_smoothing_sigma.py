"""One umbilicus curve: the transform centres on it and the losses read it, whatever the smoothing sigma."""
import json

import numpy as np
import pytest
import scipy.interpolate
import scipy.ndimage
import torch

from checkpoint_io import model_state_sha256
from config import Config
from geom_utils import interp1d
from transforms import UmbilicusTransform
from umbilicus import (LEGACY_UMBILICUS_SMOOTHING_SIGMA, smooth_umbilicus_z_to_yx,
                       umbilicus_zyx_table)

# (z, y, x): first nine control points of the published PHerc0139 umbilicus,
# PHerc0139/representations/umbilicus/20260102150214-umbilicus-20260826112529.json
POINTS = np.array([(3095, 14895, 14613), (3159, 14691, 14601), (3293, 14435, 14460), (3421, 14205, 14320),
                   (3549, 14026, 14154), (3677, 13924, 14345), (3785, 13899, 14409), (4009, 13833, 14461),
                   (4174, 13852, 14459)], dtype=np.float32)
RAW = scipy.interpolate.interp1d(POINTS[:, 0], POINTS[:, 1:], axis=0, fill_value='extrapolate')
ZS = np.arange(3100, 4100)


def centres(sigma):
    curve = smooth_umbilicus_z_to_yx(RAW, 3100, 4100, sigma)  # what the losses read
    zyx = torch.from_numpy(np.concatenate([ZS[:, None], curve(ZS)], axis=-1).astype(np.float32))
    origin = torch.zeros(len(ZS), 3)
    origin[:, 0] = torch.from_numpy(ZS.astype(np.float32))
    return curve(ZS), UmbilicusTransform(zyx)(origin)[:, 1:].double().numpy()


def test_sigma_zero_loss_centre_is_transform_centre():
    loss_centre, model_centre = centres(0.)
    assert np.abs(loss_centre - model_centre).max() < 2e-3  # float32 spacing at 1.5e4 is 1e-3


def test_default_sigma_matches_main_and_the_loss_centre():
    loss_centre, model_centre = centres(75.)
    # main smoothed the float32 table inside UmbilicusTransform: same table, same float32 interpolation
    table = RAW(ZS).astype(np.float32)
    old = scipy.ndimage.gaussian_filter1d(table, sigma=75., axis=0, mode='nearest')
    old_zyx = torch.from_numpy(np.concatenate([ZS[:, None].astype(np.float32), old], axis=-1))
    origin = torch.zeros(len(ZS), 3)
    origin[:, 0] = torch.from_numpy(ZS.astype(np.float32))
    assert np.abs(model_centre - UmbilicusTransform(old_zyx)(origin)[:, 1:].double().numpy()).max() < 1e-6
    assert np.abs(loss_centre - model_centre).max() < 1e-2


def _transform_centre(zyx, z):
    """The yx the model transform puts at height ``z``, from a table of the curve."""
    origin = torch.zeros(len(z), 3)
    origin[:, 0] = torch.from_numpy(np.asarray(z, dtype=np.float32))
    return UmbilicusTransform(torch.from_numpy(zyx))(origin)[:, 1:].double().numpy()


@pytest.mark.parametrize('sigma', [0., 75.])
def test_curve_is_the_transforms_between_slices_and_in_the_margin(sigma):
    """The shell polar table and the metrics read the curve where the transform does: at fractional slices
    and in the margin around the fit slab, where the transform extends the end segments of its table."""
    zyx = umbilicus_zyx_table(RAW, 3100, 4100, sigma)
    curve = smooth_umbilicus_z_to_yx(RAW, 3100, 4100, sigma)
    z = np.concatenate([np.arange(2940, 3100, 7.3), np.arange(3100, 4100, 11.37), np.arange(4100, 4260, 7.3)])
    assert np.abs(curve(z) - _transform_centre(zyx, z)).max() < 4e-3  # float32 spacing at 1.5e4 is 1e-3
    # the shell table is the curve at integer slices over the margin, read back with interp1d
    zs = np.arange(2940, 4260, dtype=np.float32)
    table = torch.from_numpy(np.concatenate([zs[:, None], curve(zs).astype(np.float32)], axis=-1))
    read = interp1d(torch.from_numpy(z.astype(np.float32)), table[:, :1], table[:, 1:]).double().numpy()
    assert np.abs(curve(z) - read).max() < 4e-3
    # beyond the slab the curve is not held at its end value
    assert np.abs(curve(np.array([3100 - 150.]))[0] - curve(np.array([3100.]))[0]).max() > 1.


def test_checkpoint_readers_and_ray_seeding_build_the_fitted_table():
    """The table the checkpoint readers and the ray seeder build equals what the model was fitted with."""
    legacy = scipy.ndimage.gaussian_filter1d(RAW(ZS).astype(np.float32), sigma=75., axis=0, mode='nearest')
    legacy_zyx = np.concatenate([ZS[:, None].astype(np.float32), legacy], axis=-1)
    assert np.array_equal(umbilicus_zyx_table(RAW, 3100, 4100, LEGACY_UMBILICUS_SMOOTHING_SIGMA), legacy_zyx)
    raw_zyx = np.concatenate([ZS[:, None].astype(np.float32), RAW(ZS).astype(np.float32)], axis=-1)
    assert np.array_equal(umbilicus_zyx_table(RAW, 3100, 4100, 0.), raw_zyx)
    assert np.abs(raw_zyx[:, 1:] - legacy_zyx[:, 1:]).max() > 5.  # the two axes are far apart, so a reader must pick the right one


def test_checkpoint_fitted_around_another_axis_is_refused():
    import test_checkpoint_load as load
    other = {**load.CONFIG, 'umbilicus_smoothing_sigma': 0.}
    verdict = load._inspect(load._checkpoint(cfg=other))  # the live fit smooths at the legacy width
    assert not verdict.accepted
    assert 'umbilicus_smoothing_sigma' in verdict.message()
    # a checkpoint from before the setting was fitted at the legacy width
    legacy = dict(load.CONFIG)
    del legacy['umbilicus_smoothing_sigma']
    assert load._inspect(load._checkpoint(cfg=legacy)).accepted
    unsmoothed = load._live_context()
    unsmoothed.config['umbilicus_smoothing_sigma'] = 0.
    assert not load._inspect(load._checkpoint(cfg=legacy), unsmoothed).accepted
    # a narrower window would smooth around another axis than the checkpoint's model domain
    narrower = load._live_context()
    narrower.z_begin = 1200
    assert load._inspect(load._checkpoint(), narrower).accepted
    narrower.config['umbilicus_smoothing_sigma'] = 30.
    assert 'umbilicus_smoothing_sigma' in load._inspect(load._checkpoint(cfg=narrower.config), narrower).message()


def test_surface_identity_follows_the_axis_but_the_legacy_digest_is_unchanged():
    state = {'weights': torch.arange(6.)}
    plain = model_state_sha256(state, (), 3100, 4100)
    assert model_state_sha256(state, (), 3100, 4100, umbilicus_smoothing_sigma=75.) == plain  # same surface, same digest
    assert model_state_sha256(state, (), 3100, 4100, umbilicus_smoothing_sigma=0.) != plain
    assert (model_state_sha256(state, (), 3100, 4100, umbilicus_smoothing_sigma=-0.)
            == model_state_sha256(state, (), 3100, 4100, umbilicus_smoothing_sigma=0))
    assert (model_state_sha256(state, (), 3100, 4100, umbilicus_smoothing_sigma=30.)
            != model_state_sha256(state, (), 3100, 4100, umbilicus_smoothing_sigma=0.))


@pytest.mark.parametrize('bad', [-1, -0.5, float('nan'), float('inf'), True, '5'])
def test_every_entry_refuses_the_same_sigma(bad, monkeypatch):
    import fit_spiral
    with pytest.raises(ValueError):
        Config({'umbilicus_smoothing_sigma': bad})  # checkpoint loading and the service
    monkeypatch.setenv('FIT_SPIRAL_CONFIG_OVERRIDES', json.dumps({'umbilicus_smoothing_sigma': bad}))
    with pytest.raises(ValueError):
        fit_spiral.get_env_config_overrides()  # a headless fit
    with pytest.raises(ValueError):
        smooth_umbilicus_z_to_yx(RAW, 3100, 4100, bad)  # the curve itself


@pytest.mark.parametrize('good', [0, 0., 30, 75.0, 120.5])
def test_every_entry_accepts_the_same_sigma(good, monkeypatch):
    import fit_spiral
    assert Config({'umbilicus_smoothing_sigma': good}).umbilicus_smoothing_sigma == good
    monkeypatch.setenv('FIT_SPIRAL_CONFIG_OVERRIDES', json.dumps({'umbilicus_smoothing_sigma': good}))
    assert fit_spiral.get_env_config_overrides() == {'umbilicus_smoothing_sigma': good}
    smooth_umbilicus_z_to_yx(RAW, 3100, 4100, good)
