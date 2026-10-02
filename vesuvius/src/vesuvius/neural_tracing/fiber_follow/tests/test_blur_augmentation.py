"""CT blur augmentation contracts for the CT-only input."""
import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_follow.data.observations import IdentitySampling, augment_image_pair, augment_ct


def test_ct_blur_conserves_material_mass_and_keeps_background_and_constants():
    image = torch.zeros(1, 17, 17, 17)
    image[0, 8, 8, 8] = 1
    augment_image_pair(image, (1., 0., 0.), np.random.default_rng(1), blur_sigma=1.)
    assert 0 < image[0, 8, 8, 9].item() and image[0, 8, 8, 8].item() < 1
    assert image.sum().item() == pytest.approx(1., abs=1e-5)
    assert image[0].argmax().item() == 8*17*17+8*17+8
    image = torch.full((1, 9, 9, 9), -8.)
    augment_image_pair(image, (1., 0., 0.), np.random.default_rng(2), blur_sigma=1.25)
    torch.testing.assert_close(image, torch.full_like(image, -8.))


def test_disabled_blur_is_exact_photometric_and_invalid_settings_are_rejected():
    image = torch.from_numpy(np.random.default_rng(0).random((1, 9, 9, 9), dtype=np.float32))
    expected = image.clone()
    a, b = np.random.default_rng(3), np.random.default_rng(3)
    params = (1.2, -.05, .02)
    augment_ct(expected[0].numpy(), params, a)
    augment_image_pair(image, params, b, blur_sigma=0.)
    torch.testing.assert_close(image, expected, rtol=0, atol=0)
    assert a.random() == b.random()
    assert IdentitySampling(blur_sigma=[.5, 1.25]).blur_sigma == (.5, 1.25)
    for kwargs in (dict(blur_probability=-.1), dict(blur_probability=1.1), dict(blur_probability=float('nan')),
                   dict(blur_sigma=(-1., 1.)), dict(blur_sigma=(2., 1.)), dict(blur_sigma=(0., float('inf')))):
        with pytest.raises(ValueError):
            IdentitySampling(**kwargs)
