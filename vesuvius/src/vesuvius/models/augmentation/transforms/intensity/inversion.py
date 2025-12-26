import torch
import numpy as np

from vesuvius.models.augmentation.transforms.base.basic_transform import ImageOnlyTransform


class InvertImageTransform(ImageOnlyTransform):
    """
    Inverts image intensities around their mean.

    For each channel, computes: inverted = mean - (image - mean) = 2*mean - image

    Useful for training models where intensity polarity shouldn't matter.

    Parameters:
        p_invert_image (float): Probability of applying inversion to the sample.
        p_synchronize_channels (float): Probability that all channels are inverted together.
        p_per_channel (float): Probability to invert each channel independently
                               (when not synchronized).
    """

    def __init__(
        self,
        p_invert_image: float = 0.5,
        p_synchronize_channels: float = 1.0,
        p_per_channel: float = 1.0,
    ):
        super().__init__()
        self.p_invert_image = p_invert_image
        self.p_synchronize_channels = p_synchronize_channels
        self.p_per_channel = p_per_channel

    def get_parameters(self, **data_dict) -> dict:
        shape = data_dict['image'].shape

        # First check if we apply inversion at all
        apply = np.random.uniform() < self.p_invert_image

        if apply:
            if np.random.uniform() < self.p_synchronize_channels:
                # Apply to all channels
                apply_to_channel = torch.arange(0, shape[0])
            else:
                # Apply per-channel with probability
                apply_to_channel = torch.where(torch.rand(shape[0]) < self.p_per_channel)[0]
        else:
            apply_to_channel = torch.tensor([], dtype=torch.long)

        return {
            'apply_to_channel': apply_to_channel,
            'apply': apply,
        }

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        if not params['apply'] or len(params['apply_to_channel']) == 0:
            return img

        for ch in params['apply_to_channel']:
            mn = img[ch].mean()
            # Invert around mean: mean - (img - mean) = 2*mean - img
            img[ch] = 2 * mn - img[ch]

        return img
