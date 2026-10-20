"""Optional settings of the intensity transforms, checked against independent references."""
import math

import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

from vesuvius.models.augmentation.transforms.intensity.contrast import ContrastTransform
from vesuvius.models.augmentation.transforms.intensity.gamma import GammaTransform
from vesuvius.models.augmentation.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from vesuvius.models.augmentation.transforms.noise.gaussian_blur import GaussianBlurTransform


@pytest.mark.parametrize('padding_mode, scipy_mode', [('reflect', 'mirror'), ('replicate', 'nearest')])
@pytest.mark.parametrize('sigma', [.4, .8, 1.])
def test_blur_padding_mode_and_kernel_radius_match_scipy(padding_mode, scipy_mode, sigma):
    image = torch.from_numpy(np.random.default_rng(0).random((1, 12, 13, 14)))
    blur = GaussianBlurTransform(synchronize_channels=True, synchronize_axes=True, benchmark=False,
                                 padding_mode=padding_mode, kernel_radius=lambda s: math.ceil(3 * s))
    out = blur.apply({'image': image.clone()}, apply_to_channel=torch.tensor([True]), sigmas=[sigma] * 3)['image']
    ref = gaussian_filter(image[0].numpy(), sigma, mode=scipy_mode, radius=math.ceil(3 * sigma))
    np.testing.assert_allclose(out[0].numpy(), ref, rtol=0, atol=1e-12)


def test_gamma_value_range_is_a_plain_power():
    image = torch.rand(1, 8, 9, 10, generator=torch.Generator().manual_seed(1)) * .8 + .1
    gamma = GammaTransform(gamma=(.7, 1.5), p_invert_image=0, synchronize_channels=True, p_per_channel=1,
                           p_retain_stats=0, value_range=(0., 1.))
    out = gamma.apply({'image': image.clone()}, apply_to_channel=torch.tensor([0]), retain_stats=[False],
                      invert_image=[False], gamma=torch.tensor([1.3]))['image']
    assert torch.equal(out, torch.pow(image, torch.tensor(1.3)))


def test_contrast_mean_dtype_matches_a_float64_mean_reference():
    image = (torch.rand(1, 16, 16, 16, generator=torch.Generator().manual_seed(2)) * 1e-3 + 100.)
    contrast = ContrastTransform(contrast_range=1., preserve_range=False, synchronize_channels=True,
                                 mean_dtype=torch.float64)
    out = contrast.apply({'image': image.clone()}, apply_to_channel=torch.tensor([0]),
                         multipliers=torch.tensor([1.37]))['image']
    x = image.numpy()
    center = np.float32(x.mean(dtype=np.float64))
    np.testing.assert_array_equal(out.numpy(), (x - center) * np.float32(1.37) + center)


def test_noise_generator_replays_independently_of_the_global_rng():
    noise = GaussianNoiseTransform(synchronize_channels=True)

    def run(seed, global_seed):
        torch.manual_seed(global_seed)
        generator = torch.Generator().manual_seed(seed)
        return noise.apply({'image': torch.zeros(1, 32, 32, 32)}, apply_to_channel=torch.tensor([True]),
                           sigmas=.05, generator=generator)['image']

    first = run(7, 0)
    assert torch.equal(first, run(7, 123))
    assert not torch.equal(first, run(8, 0))
    assert float(first.std()) == pytest.approx(.05, rel=.03)
