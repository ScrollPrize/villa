import torch
import numpy as np
from typing import Union, Tuple
from scipy.ndimage import median_filter

from vesuvius.models.augmentation.transforms.base.basic_transform import ImageOnlyTransform


class MedianFilterTransform(ImageOnlyTransform):
    """
    Applies a median filter to selected image channels.

    Useful for simulating denoised data or removing salt-and-pepper noise.
    Uses scipy.ndimage.median_filter for efficient implementation.

    Parameters:
        filter_size (int or Tuple[int, int]): Either fixed filter size or
                                               range (min, max) for random sampling.
        p_same_for_each_channel (float): Probability that all channels share the same filter size.
        p_per_channel (float): Probability of applying the filter to a given channel.
    """

    def __init__(
        self,
        filter_size: Union[int, Tuple[int, int]] = (3, 7),
        p_same_for_each_channel: float = 0.5,
        p_per_channel: float = 1.0,
    ):
        super().__init__()
        self.filter_size = filter_size
        self.p_same_for_each_channel = p_same_for_each_channel
        self.p_per_channel = p_per_channel

    def get_parameters(self, image: torch.Tensor, **kwargs) -> dict:
        C = image.shape[0]
        use_same = np.random.rand() < self.p_same_for_each_channel

        if isinstance(self.filter_size, int):
            sizes = [self.filter_size] * C
        elif use_same:
            sampled_size = int(np.random.randint(self.filter_size[0], self.filter_size[1] + 1))
            sizes = [sampled_size] * C
        else:
            sizes = [
                int(np.random.randint(self.filter_size[0], self.filter_size[1] + 1))
                for _ in range(C)
            ]

        apply_channel = [np.random.rand() < self.p_per_channel for _ in range(C)]

        return {
            'filter_sizes': sizes,
            'apply_channel': apply_channel
        }

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        filter_sizes = params['filter_sizes']
        apply_channel = params['apply_channel']

        if not any(apply_channel):
            return img

        device = img.device
        dtype = img.dtype
        img_np = img.cpu().numpy()

        for c, (apply, size) in enumerate(zip(apply_channel, filter_sizes)):
            if apply:
                img_np[c] = median_filter(img_np[c], size=size)

        return torch.from_numpy(img_np).to(device=device, dtype=dtype)
