"""One umbilicus curve: the transform centres on it and the losses read it, whatever the smoothing sigma."""
import numpy as np
import scipy.interpolate
import scipy.ndimage
import torch

from transforms import UmbilicusTransform
from umbilicus import smooth_umbilicus_z_to_yx

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
