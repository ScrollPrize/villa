import torch

from vesuvius.models.augmentation.helpers.scalar_type import RandomScalar, sample_scalar
from vesuvius.models.augmentation.transforms.base.basic_transform import ImageOnlyTransform


class MultiplicativeBrightnessTransform(ImageOnlyTransform):
    def __init__(self, multiplier_range: RandomScalar, synchronize_channels: bool, p_per_channel: float = 1):
        super().__init__()
        self.multiplier_range = multiplier_range
        self.synchronize_channels = synchronize_channels
        self.p_per_channel = p_per_channel

    def get_parameters(self, **data_dict) -> dict:
        shape = data_dict['image'].shape
        apply_to_channel = torch.where(torch.rand(shape[0]) < self.p_per_channel)[0]
        if self.synchronize_channels:
            multipliers = torch.Tensor([sample_scalar(self.multiplier_range, image=data_dict['image'], channel=None)] * len(apply_to_channel))
        else:
            multipliers = torch.Tensor([sample_scalar(self.multiplier_range, image=data_dict['image'], channel=c) for c in apply_to_channel])
        return {
            'apply_to_channel': apply_to_channel,
            'multipliers': multipliers
        }

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        if len(params['apply_to_channel']) == 0:
            return img
        # even though this is array notation it's a lot slower. Shame shame
        # img[params['apply_to_channel']] *= params['multipliers'].view(-1, *[1]*(img.ndim - 1))
        multipliers = params['multipliers'].to(img.device)
        for idx, c in enumerate(params['apply_to_channel']):
            m = multipliers[idx]
            img[c] *= m
        return img


class BrightnessAdditiveTransform(ImageOnlyTransform):
    """
    Adds Gaussian-sampled additive brightness to image channels.

    Complements MultiplicativeBrightnessTransform by adding rather than
    multiplying brightness values.

    Parameters:
        mu (RandomScalar): Mean of the Gaussian distribution for brightness shift.
        sigma (RandomScalar): Standard deviation of the Gaussian distribution.
        synchronize_channels (bool): If True, use same shift for all channels.
        p_per_channel (float): Probability to apply to each channel independently.
    """

    def __init__(
        self,
        mu: RandomScalar = 0,
        sigma: RandomScalar = 0.1,
        synchronize_channels: bool = False,
        p_per_channel: float = 1.0,
    ):
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.synchronize_channels = synchronize_channels
        self.p_per_channel = p_per_channel

    def get_parameters(self, **data_dict) -> dict:
        shape = data_dict['image'].shape
        apply_to_channel = torch.where(torch.rand(shape[0]) < self.p_per_channel)[0]

        if len(apply_to_channel) == 0:
            return {'apply_to_channel': apply_to_channel, 'shifts': torch.Tensor([])}

        # Sample mu and sigma
        mu = sample_scalar(self.mu, image=data_dict['image'])
        sigma = sample_scalar(self.sigma, image=data_dict['image'])

        if self.synchronize_channels:
            # Same shift for all channels
            shift = torch.normal(mean=mu, std=sigma, size=(1,)).item()
            shifts = torch.Tensor([shift] * len(apply_to_channel))
        else:
            # Different shift per channel
            shifts = torch.normal(mean=mu, std=sigma, size=(len(apply_to_channel),))

        return {
            'apply_to_channel': apply_to_channel,
            'shifts': shifts
        }

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        if len(params['apply_to_channel']) == 0:
            return img

        shifts = params['shifts'].to(img.device)
        for idx, c in enumerate(params['apply_to_channel']):
            img[c].add_(shifts[idx])

        return img


if __name__ == '__main__':
    from time import time
    import numpy as np
    import os

    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)

    mbt = MultiplicativeBrightnessTransform((0.5, 2.), False, 1)

    times_torch = []
    for _ in range(1000):
        data_dict = {'image': torch.ones((2, 128, 192, 64))}
        st = time()
        out = mbt(**data_dict)
        times_torch.append(time() - st)
    print('torch', np.mean(times_torch))

    from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform

    gnt_bg = BrightnessMultiplicativeTransform((0.5, 2), True, p_per_sample=1)
    times_bg = []
    for _ in range(1000):
        data_dict = {'data': np.ones((1, 2, 128, 192, 64))}
        st = time()
        out = gnt_bg(**data_dict)
        times_bg.append(time() - st)
    print('bg', np.mean(times_bg))
