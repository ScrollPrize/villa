"""The umbilicus centre the losses read is the centre UmbilicusTransform puts the model on."""
import numpy as np
import scipy.interpolate
import torch

from transforms import UmbilicusTransform

# (z, y, x): the first nine control points of the published PHerc0139 umbilicus,
# PHerc0139/representations/umbilicus/20260102150214-umbilicus-20260826112529.json
POINTS = np.array([(3095, 14895, 14613), (3159, 14691, 14601), (3293, 14435, 14460), (3421, 14205, 14320),
                   (3549, 14026, 14154), (3677, 13924, 14345), (3785, 13899, 14409), (4009, 13833, 14461),
                   (4174, 13852, 14459)], dtype=np.float32)


def test_the_losses_read_the_transforms_centre():
    raw = scipy.interpolate.interp1d(POINTS[:, 0], POINTS[:, 1:], axis=0, fill_value='extrapolate')
    zs, wide = np.arange(3100, 4100), np.arange(2900, 4300)  # the fit window and the shell table's margin
    transform = UmbilicusTransform(torch.from_numpy(np.concatenate([zs[:, None], raw(zs)], axis=-1).astype(np.float32)))
    origin = torch.zeros(len(wide), 3)
    origin[:, 0] = torch.from_numpy(wide.astype(np.float32))
    model_centre = transform(origin)[:, 1:].double().numpy()
    # The raw curve, which the losses read on main, is 84.652 voxels from the model's centre at z 3100.
    assert np.linalg.norm(raw(zs) - model_centre[200:1200], axis=-1).max() > 10
    from transforms import umbilicus_centre_z_to_yx
    gap = np.linalg.norm(umbilicus_centre_z_to_yx(raw, 3100, 4100)(wide) - model_centre, axis=-1)
    assert gap.max() < 1e-2, f'the losses read a centre {gap.max():.3f} voxels from the transform'
