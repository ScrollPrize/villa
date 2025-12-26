import torch
import numpy as np

from vesuvius.models.augmentation.transforms.base.basic_transform import ImageOnlyTransform
from vesuvius.models.augmentation.transforms.local.local_transform import LocalTransform
from vesuvius.models.augmentation.helpers.scalar_type import RandomScalar, sample_scalar


class LocalContrastTransform(ImageOnlyTransform, LocalTransform):
    """
    Applies localized contrast modification using a spatial Gaussian mask.

    Creates a contrast-modified version of the image and blends it with the
    original using a kernel-based interpolation. Useful for simulating
    local contrast variations in medical or volumetric imaging.

    Parameters:
        scale (RandomScalar): Gaussian spread for the spatial weighting mask.
        loc (RandomScalar): Relative center position for the Gaussian (in % of image size).
        new_contrast (RandomScalar): Multiplicative factor for local contrast. 1.0 = no change.
        same_for_all_channels (bool): Whether to use one kernel/contrast for all channels.
        p_per_channel (float): Probability to apply to each channel independently.
    """

    def __init__(
        self,
        scale: RandomScalar,
        loc: RandomScalar = (-1, 2),
        new_contrast: RandomScalar = (0.5, 1.5),
        same_for_all_channels: bool = True,
        p_per_channel: float = 1.0,
    ):
        ImageOnlyTransform.__init__(self)
        LocalTransform.__init__(self, scale, loc)

        self.new_contrast = new_contrast
        self.same_for_all_channels = same_for_all_channels
        self.p_per_channel = p_per_channel

    def get_parameters(self, image: torch.Tensor, **kwargs) -> dict:
        C, *spatial = image.shape
        apply_channel = [np.random.rand() < self.p_per_channel for _ in range(C)]

        if not any(apply_channel):
            return {'kernels': [None] * C, 'contrasts': [None] * C}

        if self.same_for_all_channels:
            kernel = self._generate_kernel(tuple(spatial)).astype(np.float32)
            contrast = sample_scalar(self.new_contrast)

            kernels = [kernel if apply else None for apply in apply_channel]
            contrasts = [contrast if apply else None for apply in apply_channel]
        else:
            kernels, contrasts = [], []
            for apply in apply_channel:
                if not apply:
                    kernels.append(None)
                    contrasts.append(None)
                    continue

                kernel = self._generate_kernel(tuple(spatial)).astype(np.float32)
                contrast = sample_scalar(self.new_contrast)
                kernels.append(kernel)
                contrasts.append(contrast)

        return {'kernels': kernels, 'contrasts': contrasts}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        kernels = params['kernels']
        contrasts = params['contrasts']

        if all(k is None for k in kernels):
            return img

        device = img.device
        dtype = img.dtype
        img_np = img.cpu().numpy()

        for c, (kernel, contrast) in enumerate(zip(kernels, contrasts)):
            if kernel is None:
                continue

            channel = img_np[c]
            # Compute kernel-weighted local mean
            mean = (channel * kernel).sum() / (kernel.sum() + 1e-8)
            # Apply contrast modification around the mean
            modified = (channel - mean) * contrast + mean
            # Blend using the spatial kernel
            img_np[c] = self.run_interpolation(channel, modified, kernel)

        return torch.from_numpy(img_np).to(device=device, dtype=dtype)
