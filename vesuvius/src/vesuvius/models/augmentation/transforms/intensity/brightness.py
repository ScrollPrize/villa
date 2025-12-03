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
